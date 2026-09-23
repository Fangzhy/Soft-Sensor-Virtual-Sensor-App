"""Verify inference, run isolation, and free-only explanations without live calls."""

from pathlib import Path
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app
from backend.schemas import ComparisonRequest, GenerationRequest
from backend.services.model_comparison import compare_models
from backend.services.synthetic_data import generate_dataset
from backend.services.run_store import RUNS, ModelRun, RunStore
from backend.services.explanations import explain_run, SYSTEM_PROMPT, SECTIONS


@pytest.fixture
def retained():
    """Create a fresh run for every test, so cached explanations cannot leak."""
    data = generate_dataset(GenerationRequest(n_samples=100))
    result = compare_models(ComparisonRequest(rows=data.rows, models=["Linear Regression"],
                                             include_diagnostics=True), persist=True)
    return data, result


def test_prediction_reuses_model_interval_and_ranges(retained):
    data, result = retained
    run_id = result.evaluation.run_id
    row = result.evaluation.predictions[0]
    inputs = data.rows[row.row_index].model_dump(exclude={"solid_concentration_pct"})
    run = RUNS.get(run_id)
    with patch.object(run.model, "fit", side_effect=AssertionError("Prediction must not fit")):
        with TestClient(app) as client:
            response = client.post("/models/predict", json={"run_id": run_id, "inputs": inputs})
    assert response.status_code == 200
    prediction = response.json()
    assert prediction["prediction"] == pytest.approx(row.predicted)
    assert prediction["upper"] - prediction["prediction"] == pytest.approx(result.diagnostics.half_width)
    inputs["temperature_c"] = 20
    with TestClient(app) as client:
        prediction = client.post("/models/predict", json={"run_id": run_id, "inputs": inputs}).json()
    assert any("temperature_c" in text for text in prediction["warnings"])


def test_linear_route_retains_uncalibrated_run(retained):
    data, _ = retained
    with TestClient(app) as client:
        result = client.post("/models/train", json={"rows": data.model_dump()["rows"]}).json()
        response = client.post("/models/predict", json={"run_id": result["run_id"],
            "inputs": data.rows[0].model_dump(exclude={"solid_concentration_pct"})})
    assert response.status_code == 200
    assert response.json()["lower"] is response.json()["upper"] is None


def test_run_store_expiry_capacity_and_isolation():
    store = RunStore(capacity=2, ttl=10)
    first = store.put(ModelRun(object(), "Linear Regression", {}, None, {}))
    second = store.put(ModelRun(object(), "Random Forest", {}, None, {}))
    assert first != second and store.get(first).model is not store.get(second).model
    third = store.put(ModelRun(object(), "XGBoost", {}, None, {}))
    with pytest.raises(KeyError):
        store.get(first)
    store.get(third).created -= 11
    with pytest.raises(KeyError):
        store.get(third)


def test_expired_and_invalid_requests(retained):
    data, result = retained
    with TestClient(app) as client:
        assert client.post("/models/explain", json={"run_id": "0" * 32}).status_code == 404
        assert client.post("/models/explain", json={"run_id": result.evaluation.run_id, "metrics": {}}).status_code == 422
        inputs = data.rows[0].model_dump(exclude={"solid_concentration_pct"})
        inputs["temperature_c"] = 1000
        assert client.post("/models/predict", json={"run_id": result.evaluation.run_id, "inputs": inputs}).status_code == 422


def test_free_request_evidence_and_cache(retained):
    _, result = retained
    response = Mock(status_code=200)
    response.json.return_value = {"model": "example/free-model:free", "choices": [
        {"finish_reason": "stop", "message": {"content": json.dumps({name: "Evidence-based explanation for this section." for name in SECTIONS})}}]}
    with patch("backend.services.explanations.openrouter_settings", return_value=("test-secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", return_value=response) as post:
        first = explain_run(result.evaluation.run_id)
        second = explain_run(result.evaluation.run_id)
    assert first.source == "openrouter" and second.cached
    assert post.call_count == 1
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
    assert kwargs["json"]["model"] == "openrouter/free" and kwargs["allow_redirects"] is False
    assert kwargs["json"]["response_format"]["type"] == "json_schema"
    assert kwargs["json"]["provider"]["require_parameters"] is True
    assert "test-secret" not in str(kwargs["json"])
    assert "predictions" not in str(first.evidence) and "rows" not in first.evidence
    assert first.evidence["split"] == {"training": 60, "test": 20, "seed": 42, "calibration": 20}
    assert "not causation" in SYSTEM_PROMPT and "percentage points" in SYSTEM_PROMPT


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 302])
def test_service_errors_use_uncached_template(retained, status):
    _, result = retained
    with patch("backend.services.explanations.openrouter_settings", return_value=("test-secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", return_value=Mock(status_code=status)):
        explanation = explain_run(result.evaluation.run_id)
    assert explanation.source == "template" and explanation.notice
    assert RUNS.get(result.evaluation.run_id).explanation is None
    assert "test-secret" not in explanation.model_dump_json()


@pytest.mark.parametrize("key,model", [("", "openrouter/free"), ("test-secret", "openrouter/auto")])
def test_missing_key_and_paid_model_never_call(retained, key, model):
    with patch("backend.services.explanations.openrouter_settings", return_value=(key, model)), \
         patch("backend.services.explanations.requests.post") as post:
        assert explain_run(retained[1].evaluation.run_id).source == "template"
    post.assert_not_called()


@pytest.mark.parametrize("payload", [{}, {"model": "free", "choices": []},
    {"model": "free", "choices": [{"finish_reason": "length", "message": {"content": "Partial"}}]},
    {"model": "free", "choices": [{"finish_reason": "stop", "message": {"content": " "}}]},
    {"model": "safety:free", "choices": [{"finish_reason": "stop", "message": {"content": "User Safety: safe"}}]},
    {"model": "free", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({name: "safe" for name in SECTIONS})}}]},
])
def test_bad_provider_response(retained, payload):
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", return_value=Mock(status_code=200, json=lambda: payload)):
        assert explain_run(retained[1].evaluation.run_id).source == "template"


def test_timeout_and_secret_safe_error(retained):
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", side_effect=requests.Timeout("secret")):
        result = explain_run(retained[1].evaluation.run_id)
    assert result.source == "template" and "secret" not in result.model_dump_json()


@pytest.mark.parametrize("error,expected", [
    (requests.ConnectTimeout("secret"), "10-second connection limit"),
    (requests.ReadTimeout("secret"), "60-second read timeout"),
    (requests.ConnectionError("secret"), "network, DNS, TLS, or proxy"),
    (requests.exceptions.JSONDecodeError("invalid", "x", 0), "invalid response JSON"),
])
def test_distinct_failure_messages_and_safe_logs(retained, error, expected, caplog):
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", side_effect=error):
        result = explain_run(retained[1].evaluation.run_id)
    assert expected in result.notice and "Elapsed:" in result.notice
    assert "secret" not in caplog.text


def test_output_limit_is_not_reported_as_timeout(retained):
    payload = {"model": "free", "choices": [{"finish_reason": "length", "message": {"content": "Partial"}}]}
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", return_value=Mock(status_code=200, json=lambda: payload)):
        result = explain_run(retained[1].evaluation.run_id)
    assert "output token limit" in result.notice and "not a timeout" in result.notice


def test_concurrent_explanation_does_not_wait_or_call_provider(retained):
    run_id = retained[1].evaluation.run_id
    with RUNS.get(run_id).explanation_lock:
        with patch("backend.services.explanations.requests.post") as post:
            result = explain_run(run_id)
    assert "already in progress" in result.notice
    post.assert_not_called()


def test_pinned_free_model_request(retained):
    from backend.config import DEFAULT_FREE_MODEL
    payload = {"model": DEFAULT_FREE_MODEL, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({name: "A substantive explanation of the measured evidence." for name in SECTIONS})}}]}
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", DEFAULT_FREE_MODEL)), \
         patch("backend.services.explanations.requests.post", return_value=Mock(status_code=200, json=lambda: payload)) as post:
        result = explain_run(retained[1].evaluation.run_id)
    assert result.source == "openrouter"
    body = post.call_args.kwargs["json"]
    assert body["model"] == DEFAULT_FREE_MODEL
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0}


@pytest.mark.parametrize("message,expected", [
    ("No endpoints found that support the requested parameters", "requested parameters"),
    ("No endpoints found matching your data policy", "data/privacy policy"),
    ("No endpoints found", "no eligible model/provider endpoint"),
])
def test_routing_404_diagnostics(retained, message, expected):
    response = Mock(status_code=404, json=lambda: {"error": {"message": message, "metadata": {"raw": "secret"}}})
    with patch("backend.services.explanations.openrouter_settings", return_value=("secret", "openrouter/free")), \
         patch("backend.services.explanations.requests.post", return_value=response):
        result = explain_run(retained[1].evaluation.run_id)
    assert expected in result.notice and "secret" not in result.model_dump_json()


def test_config_uses_only_root_env(monkeypatch):
    from backend.config import openrouter_settings
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API", raising=False)
    with patch("backend.config.dotenv_values", return_value={"OPENROUTER_API_KEY": "fake"}) as load:
        from backend.config import DEFAULT_FREE_MODEL
        assert openrouter_settings() == ("fake", DEFAULT_FREE_MODEL)
        assert load.call_args.args[0] == Path(__file__).resolve().parents[1] / ".env"
        monkeypatch.setenv("OPENROUTER_API_KEY", "override")
        assert openrouter_settings()[0] == "override"


def test_existing_key_alias(monkeypatch):
    from backend.config import openrouter_settings
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API", raising=False)
    with patch("backend.config.dotenv_values", return_value={"OPEN_ROUTER_API": "fake-alias"}):
        assert openrouter_settings()[0] == "fake-alias"


def test_prediction_explanation_ui_no_automatic_calls(retained):
    data, result = retained
    run_id = result.evaluation.run_id
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    inputs = dict(temperature_c=55.0, density_kg_m3=1150.0, flow_rate_l_min=55.0, pressure_bar=3.5, agitation_rpm=450.0)
    prediction = dict(run_id=run_id, model="Linear Regression", inputs=inputs, prediction=35.0,
                      lower=33.0, upper=37.0, nominal_coverage=0.9, warnings=[])
    explanation = dict(run_id=run_id, source="openrouter", model="example:free", text="Model explanation.",
                       evidence=RUNS.get(run_id).evidence, notice=None, cached=False)
    with patch("frontend.api_client.requests.post") as post:
        page = AppTest.from_file(str(path), default_timeout=30).run()
        page.session_state["dataset"] = data.model_dump()
        page.session_state["comparison_result"] = result.model_dump()
        page.run()
        post.assert_not_called()
        post.return_value = Mock(status_code=200, json=lambda: prediction)
        next(b for b in page.button if b.label == "Predict solid concentration").click().run()
        assert not page.exception and page.session_state["new_prediction"] == prediction
        post.return_value = Mock(status_code=200, json=lambda: explanation)
        next(b for b in page.button if b.label == "Explain this model").click().run()
        assert not page.exception and not page.error
        next(b for b in page.button if b.label == "Explain this model").click().run()
        assert post.call_count == 2
        page.selectbox[0].select("temperature_c").run()
        assert post.call_count == 2
        # A new run clears both outputs instead of displaying stale explanations.
        replacement = result.model_dump()
        replacement["evaluation"]["run_id"] = "f" * 32
        page.session_state["comparison_result"] = replacement
        page.run()
        assert not page.exception
        assert "model_explanation" not in page.session_state and "new_prediction" not in page.session_state
