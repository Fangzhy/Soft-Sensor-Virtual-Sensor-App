"""Exercise cloud deployment contracts without spending OpenRouter quota."""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from backend.main import app
from fastapi.testclient import TestClient
from backend.services.run_store import RUNS
from frontend import direct_backend as cloud
from frontend.execution import execution_mode


def train_session(session):
    """Create a real small fitted run for ownership and quota checks."""
    _, dataset = cloud.dispatch("post", "/data/generate", {"n_samples": 100}, session)
    status, result = cloud.dispatch("post", "/models/train", {"rows": dataset["rows"]}, session)
    assert status == 200
    return result["run_id"], dataset


def test_direct_matches_http_training_and_validation():
    session = {}
    run_id, dataset = train_session(session)
    with TestClient(app) as client:
        http = client.post("/models/train", json={"rows": dataset["rows"]}).json()
        _, direct = cloud.dispatch("post", "/models/train", {"rows": dataset["rows"]}, session)
        http.pop("run_id")
        direct.pop("run_id")
        assert direct == http
        invalid = {"n_samples": 2}
        assert cloud.dispatch("post", "/data/generate", invalid, session)[0] == 422
        assert client.post("/data/generate", json=invalid).status_code == 422


def test_run_ownership_and_expiry():
    owner, other = {}, {}
    run_id, dataset = train_session(owner)
    inputs = {k: v for k, v in dataset["rows"][0].items() if k != "solid_concentration_pct"}
    body = {"run_id": run_id, "inputs": inputs}
    assert cloud.dispatch("post", "/models/predict", body, owner)[0] == 200
    assert cloud.dispatch("post", "/models/predict", body, other)[0] == 404
    assert cloud.dispatch("post", "/models/explain", {"run_id": run_id}, other)[0] == 404
    with patch.object(cloud.RUNS, "get", side_effect=KeyError):
        assert cloud.dispatch("post", "/models/predict", body, owner)[0] == 404


def test_training_capacity_and_lock_release():
    session = {}
    _, dataset = cloud.dispatch("post", "/data/generate", {"n_samples": 100}, session)
    with cloud.TRAINING_LOCK:
        status, _ = cloud.dispatch("post", "/models/train", {"rows": dataset["rows"]}, session)
        assert status == 429
    with patch.object(cloud, "train_linear_regression", side_effect=RuntimeError("test failure")):
        with pytest.raises(RuntimeError):
            cloud.dispatch("post", "/models/train", {"rows": dataset["rows"]}, session)
    assert not cloud.TRAINING_LOCK.locked()


def test_explanation_cooldown_cache_and_busy(monkeypatch):
    session = {}
    run_id, _ = train_session(session)
    monkeypatch.setattr(cloud, "_next_explanation_at", 0)
    monkeypatch.setattr(cloud, "monotonic", lambda: 100)
    with patch("backend.services.explanations.openrouter_settings", return_value=("", "openrouter/free")), \
         patch("backend.services.explanations.requests.post") as post:
        first = cloud._explain(run_id)
        assert "missing" in first["notice"]
        second = cloud._explain(run_id)
        assert "cooling down" in second["notice"]
        # Another model/session shares the same quota gate.
        another, _ = train_session({})
        assert "cooling down" in cloud._explain(another)["notice"]
        monkeypatch.setattr(cloud, "monotonic", lambda: 131)
        assert "missing" in cloud._explain(run_id)["notice"]
        with cloud.EXPLANATION_LOCK:
            assert "busy" in cloud._explain(run_id)["notice"]
        RUNS.get(run_id).explanation = {**first, "source": "openrouter", "notice": None}
        assert cloud._explain(run_id)["cached"] is True
        post.assert_not_called()


def test_unknown_execution_mode_fails_clearly(monkeypatch):
    monkeypatch.setenv("APP_EXECUTION_MODE", "typo")
    with pytest.raises(ValueError, match="http.*direct"):
        execution_mode()


def test_cloud_secrets_bootstrap(monkeypatch, tmp_path):
    """Cloud configuration is loaded before dispatch; no secret is rendered."""
    from backend.config import openrouter_settings

    for name in ("APP_EXECUTION_MODE", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
        # Register restoration even if the variable was originally absent.
        monkeypatch.setenv(name, "")
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"))
    # AppTest's page.secrets bypasses TOML parsing/environment promotion.
    # Exercise a real Secrets parser using a temporary file with a fake key.
    import streamlit as st
    from streamlit import config
    from streamlit.runtime.secrets import Secrets

    secret_file = tmp_path / "secrets.toml"
    secret_file.write_text('APP_EXECUTION_MODE = "direct"\nOPENROUTER_API_KEY = "fake-cloud-test-key"\nOPENROUTER_MODEL = "openrouter/free"\n')
    get_option = config.get_option
    with patch.object(st, "secrets", Secrets()), \
         patch.object(config, "get_option", side_effect=lambda name: [str(secret_file)] if name == "secrets.files" else get_option(name)), \
         patch.object(Secrets, "_maybe_install_file_watchers"):
        page.run()
    assert not page.exception
    assert execution_mode() == "direct"
    assert openrouter_settings() == ("fake-cloud-test-key", "openrouter/free")
    assert not any(b.label == "Check backend connection" for b in page.button)
    assert "fake-cloud-test-key" not in str(page)


def click(page, label):
    """Submit a visible UI action, failing on a Streamlit exception."""
    next(button for button in page.button if button.label == label).click().run()
    assert not page.exception
    assert not page.error


def test_cloud_page_full_workflow_without_fastapi(monkeypatch):
    monkeypatch.setenv("APP_EXECUTION_MODE", "direct")
    monkeypatch.setattr(cloud, "_next_explanation_at", 0)
    entry = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    with patch("requests.post") as post, patch("requests.get") as get, \
         patch("backend.services.explanations.openrouter_settings", return_value=("", "openrouter/free")):
        page = AppTest.from_file(str(entry), default_timeout=60).run()
        assert not page.exception
        assert not any(b.label == "Check backend connection" for b in page.button)
        click(page, "Generate synthetic data")
        click(page, "Compare selected models")
        assert len(page.session_state["comparison_result"]["results"]) == 4
        click(page, "Predict solid concentration")
        assert page.session_state["new_prediction"]["nominal_coverage"] == 0.9
        click(page, "Explain this model")
        assert page.session_state["model_explanation"]["source"] == "template"
        click(page, "Prepare evaluation report")
        assert len(page.get("download_button")) >= 2
        # A separate page has no inherited dataset or selected model.
        other = AppTest.from_file(str(entry)).run()
        assert "dataset" not in other.session_state
        post.assert_not_called()
        get.assert_not_called()
