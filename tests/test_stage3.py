"""Verify held-out evaluation, leakage prevention, and training UI lifecycle."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app
from backend.schemas import GenerationRequest, TrainingRequest
from backend.services.modeling import train_linear_regression
from backend.services.synthetic_data import generate_dataset
from frontend.api_client import BackendError, fetch_training


@pytest.fixture
def dataset():
    """Use real generated observations for evaluation and interface tests."""
    return generate_dataset(GenerationRequest(n_samples=100)).model_dump()


def test_metrics_split_and_reproducibility(dataset):
    """Recompute scores from returned predictions, independently of sklearn."""
    request = TrainingRequest(rows=dataset["rows"])
    result = train_linear_regression(request)
    assert result == train_linear_regression(request)
    assert (result.train_count, result.test_count) == (80, 20)
    indices = [row.row_index for row in result.predictions]
    assert len(set(indices)) == 20
    actual = np.array([row.actual for row in result.predictions])
    predicted = np.array([row.predicted for row in result.predictions])
    residual = actual - predicted
    assert np.allclose(residual, [row.residual for row in result.predictions])
    assert result.test_metrics.rmse == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert result.test_metrics.mae == pytest.approx(np.mean(abs(residual)))
    assert result.test_metrics.r2 == pytest.approx(1 - sum(residual**2) / sum((actual - actual.mean())**2))
    training_targets = [row["solid_concentration_pct"] for i, row in enumerate(dataset["rows"]) if i not in indices]
    assert result.baseline_test_metrics.mae == pytest.approx(np.mean(abs(actual - np.mean(training_targets))))


def test_test_targets_never_influence_fitting(dataset):
    """Changing only held-out labels must leave fitted predictions unchanged."""
    request = TrainingRequest(rows=dataset["rows"])
    before = train_linear_regression(request)
    changed = request.model_copy(deep=True)
    for row in before.predictions:
        changed.rows[row.row_index].solid_concentration_pct = 99.0
    after = train_linear_regression(changed)
    assert [row.predicted for row in before.predictions] == [row.predicted for row in after.predictions]
    assert before.train_metrics == after.train_metrics
    assert after.test_metrics.r2 is None


def test_scaler_only_fits_training_rows(dataset):
    """Inspect what reaches preprocessing, rather than relying on score quality."""
    from sklearn.preprocessing import StandardScaler
    original = StandardScaler.fit
    observed = []

    def record_fit(self, x, y=None, **kwargs):
        observed.append(x.copy())
        return original(self, x, y, **kwargs)

    with patch.object(StandardScaler, "fit", record_fit):
        result = train_linear_regression(TrainingRequest(rows=dataset["rows"]))
    assert len(observed) == 1 and observed[0].shape == (80, 5)
    held_out = {row.row_index for row in result.predictions}
    assert set(observed[0][:, 0]) == {
        row["temperature_c"] for i, row in enumerate(dataset["rows"]) if i not in held_out
    }


@pytest.mark.parametrize("update", [
    {"test_fraction": 0}, {"test_fraction": 0.5}, {"split_seed": -1},
    {"split_seed": 1.2}, {"rows": []}, {"extra": "invalid"},
])
def test_invalid_training_request(dataset, update):
    with TestClient(app) as client:
        response = client.post("/models/train", json={"rows": dataset["rows"], **update})
    assert response.status_code == 422


def test_endpoint_and_constant_target(dataset):
    """Constant targets yield valid JSON and explicitly undefined R²."""
    for row in dataset["rows"]:
        row["solid_concentration_pct"] = 35.0
    with TestClient(app) as client:
        response = client.post("/models/train", json={"rows": dataset["rows"]})
    assert response.status_code == 200
    assert response.json()["test_metrics"] == {"r2": None, "rmse": 0.0, "mae": 0.0}


def test_training_client_errors(dataset):
    with patch("frontend.api_client.requests.post", side_effect=requests.Timeout()):
        with pytest.raises(BackendError, match="Training failed"):
            fetch_training("http://localhost", dataset["rows"], 0.2, 42)
    with patch("frontend.api_client.requests.post", return_value=Mock(status_code=200, json=lambda: {})):
        with pytest.raises(BackendError, match="contract"):
            fetch_training("http://localhost", dataset["rows"], 0.2, 42)


def test_training_ui_and_dataset_invalidation():
    """Results survive reruns/failure but are removed after successful generation."""
    dataset = generate_dataset(GenerationRequest()).model_dump()
    result = train_linear_regression(TrainingRequest(rows=dataset["rows"])).model_dump()
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    with patch("frontend.api_client.requests.post") as post:
        page = AppTest.from_file(str(path), default_timeout=20).run()
        assert not any(button.label == "Train Linear Regression" for button in page.button)
        page.session_state["dataset"] = dataset
        page.run()
        post.return_value = Mock(status_code=200, json=lambda: result)
        next(b for b in page.button if b.label == "Train Linear Regression").click().run()
        assert not page.exception and len(page.metric) == 3
        assert len(page.get("plotly_chart")) == 5
        page.selectbox[0].select("temperature_c").run()
        assert post.call_count == 1
        post.side_effect = requests.ConnectionError()
        next(b for b in page.button if b.label == "Train Linear Regression").click().run()
        assert not page.exception and page.error and page.warning
        assert page.session_state["training_result"] == result
        post.side_effect = None
        post.return_value = Mock(status_code=200, json=lambda: dataset)
        next(b for b in page.button if b.label == "Generate synthetic data").click().run()
        assert not page.exception and not page.metric
        assert "training_result" not in page.session_state
