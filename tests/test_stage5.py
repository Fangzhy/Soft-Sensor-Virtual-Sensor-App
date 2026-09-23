"""Test finite-sample calibration, split isolation, diagnostics, and UI state."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from backend.main import app
from backend.schemas import ComparisonRequest, GenerationRequest
from backend.services.diagnostics import conformal_radius, concentration_bins
from backend.services.model_comparison import compare_models
from backend.services.synthetic_data import generate_dataset
from frontend.api_client import validate_diagnostics


@pytest.fixture
def request_data():
    return ComparisonRequest(rows=generate_dataset(GenerationRequest(n_samples=100)).rows,
                             models=["Linear Regression"], include_diagnostics=True)


def test_exact_finite_sample_rank():
    assert conformal_radius(np.arange(20)) == (18.0, 19)
    with pytest.raises(ValueError):
        conformal_radius(np.arange(3))


def test_calibration_and_test_labels_do_not_leak(request_data):
    before = compare_models(request_data)
    remaining, test_idx = train_test_split(np.arange(100), test_size=0.2, random_state=42)
    _, calibration_idx = train_test_split(remaining, test_size=20, random_state=42)
    changed = request_data.model_copy(deep=True)
    for index in calibration_idx:
        changed.rows[index].solid_concentration_pct = 99.0
    after = compare_models(changed)
    assert before.results == after.results and before.evaluation == after.evaluation
    assert after.diagnostics.half_width != before.diagnostics.half_width
    changed = request_data.model_copy(deep=True)
    for index in test_idx:
        changed.rows[index].solid_concentration_pct = 99.0
    after = compare_models(changed)
    assert before.results == after.results
    assert before.diagnostics.half_width == after.diagnostics.half_width
    assert [p.predicted for p in before.evaluation.predictions] == [p.predicted for p in after.evaluation.predictions]


def test_scaling_uses_only_proper_training(request_data):
    from sklearn.preprocessing import StandardScaler
    original = StandardScaler.fit
    fitted = []

    def record(self, x, y=None, **kwargs):
        fitted.append(set(x[:, 0]))
        return original(self, x, y, **kwargs)

    with patch.object(StandardScaler, "fit", record):
        compare_models(request_data)
    remaining, _ = train_test_split(np.arange(100), test_size=0.2, random_state=42)
    train_idx, _ = train_test_split(remaining, test_size=20, random_state=42)
    allowed = {request_data.rows[i].temperature_c for i in train_idx}
    assert len(fitted) == 6
    assert all(len(values) == 48 and values <= allowed for values in fitted[:5])
    assert fitted[-1] == allowed


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.4])
def test_interval_and_diagnostic_arithmetic(request_data, fraction):
    request_data.test_fraction = fraction
    with TestClient(app) as client:
        response = client.post("/models/compare", json=request_data.model_dump())
    assert response.status_code == 200
    result = response.json()
    d, e = result["diagnostics"], result["evaluation"]
    assert d["calibration_count"] == 20
    assert e["train_count"] + e["test_count"] + 20 == 100
    validate_diagnostics(d, e, 20)
    residuals = np.array([p["residual"] for p in e["predictions"]])
    assert d["mean_residual"] == pytest.approx(residuals.mean())
    assert d["residual_std"] == pytest.approx(residuals.std())
    assert sum(b["count"] for b in d["residual_bins"]) == len(residuals)
    expected = sum(interval["lower"] <= row["actual"] <= interval["upper"]
                   for interval, row in zip(d["intervals"], e["predictions"])) / len(residuals)
    assert d["empirical_coverage"] == expected
    assert d["mean_width"] == 2 * d["half_width"]


def test_constant_targets_and_bins(request_data):
    for row in request_data.rows:
        row.solid_concentration_pct = 35.0
    result = compare_models(request_data)
    assert result.diagnostics.half_width == 0
    assert result.diagnostics.empirical_coverage == 1
    assert result.diagnostics.abs_residual_prediction_correlation is None
    assert len(result.diagnostics.residual_bins) == 1
    bins = concentration_bins(np.array([1, 1, 1, 2, 2, 3]), np.arange(6))
    assert sum(b.count for b in bins) == 6


def test_corrupt_intervals_rejected(request_data):
    result = compare_models(request_data).model_dump()
    result["diagnostics"]["intervals"][0]["upper"] += 1
    with pytest.raises(ValueError):
        validate_diagnostics(result["diagnostics"], result["evaluation"], 20)


def test_diagnostics_ui_and_reset(request_data):
    result = compare_models(request_data).model_dump()
    dataset = generate_dataset(GenerationRequest(n_samples=100)).model_dump()
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    with patch("frontend.api_client.requests.post", return_value=Mock(status_code=200, json=lambda: result)) as post:
        page = AppTest.from_file(str(path), default_timeout=30).run()
        page.session_state["dataset"] = dataset
        page.run()
        page.multiselect[0].set_value(["Linear Regression"]).run()
        next(b for b in page.button if b.label == "Compare selected models").click().run()
        assert not page.exception and not page.error
        assert len(page.metric) == 8
        assert len(page.get("plotly_chart")) == 10
        assert post.call_args.kwargs["json"]["include_diagnostics"] is True
        page.selectbox[0].select("temperature_c").run()
        assert post.call_count == 1
        with patch("frontend.data_explorer.fetch_dataset", return_value=dataset):
            next(b for b in page.button if b.label == "Generate synthetic data").click().run()
        assert not page.exception and not page.metric
