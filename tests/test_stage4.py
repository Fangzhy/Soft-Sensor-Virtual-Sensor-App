"""Exercise real model families, training-only folds, and comparison lifecycle."""

from pathlib import Path
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from backend.main import app
from backend.schemas import ComparisonRequest, GenerationRequest
from backend.services.model_comparison import compare_models, fit_with_notice
from backend.services.synthetic_data import generate_dataset
from frontend.api_client import BackendError, fetch_comparison


@pytest.fixture(scope="module")
def observations():
    return generate_dataset(GenerationRequest(n_samples=150))


@pytest.fixture(scope="module")
def comparison(observations):
    """Fit all real estimators once and reuse artifacts in contract/UI checks."""
    return compare_models(ComparisonRequest(rows=observations.rows))


def test_all_models_and_fold_statistics(comparison):
    assert [item.model for item in comparison.results] == [
        "Linear Regression", "Random Forest", "XGBoost", "Neural Network",
    ]
    for item in comparison.results:
        assert len(item.folds) == 5
        errors = [score.rmse for score in item.folds]
        assert np.isfinite(errors).all()
        assert item.mean.rmse == pytest.approx(np.mean(errors))
        assert item.std.rmse == pytest.approx(np.std(errors))
    winner = min(comparison.results, key=lambda item: item.mean.rmse).model
    assert comparison.selected_model == comparison.evaluation.model == winner
    assert len(comparison.evaluation.predictions) == 30


def test_test_labels_cannot_influence_cv_or_selection(observations, comparison):
    """Change only outer-test labels; CV, winner, and predictions must match."""
    changed = ComparisonRequest(rows=observations.rows).model_copy(deep=True)
    for row in comparison.evaluation.predictions:
        changed.rows[row.row_index].solid_concentration_pct = 99.0
    after = compare_models(changed)
    assert after.results == comparison.results
    assert after.selected_model == comparison.selected_model
    assert [p.predicted for p in after.evaluation.predictions] == [p.predicted for p in comparison.evaluation.predictions]
    assert after.evaluation.test_metrics.r2 is None


def test_identical_folds_and_fresh_estimators(observations):
    """Record estimator inputs to prove no test rows or fold state are reused."""
    records = []

    class Recorder:
        def __init__(self, name):
            self.name = name
            self.fitted = False

        def fit(self, x, y):
            assert not self.fitted
            self.fitted = True
            records.append((self.name, set(x[:, 0])))
            self.mean = y.mean()
            return self

        def predict(self, x):
            return np.full(len(x), self.mean)

    request = ComparisonRequest(rows=observations.rows)
    with patch("backend.services.model_comparison.build_model", side_effect=lambda name, seed: Recorder(name)):
        result = compare_models(request)
    train_idx, test_idx = train_test_split(np.arange(150), test_size=0.2, random_state=42)
    train_values = {observations.rows[i].temperature_c for i in train_idx}
    test_values = {observations.rows[i].temperature_c for i in test_idx}
    assert len(records) == 21  # 4 × 5 folds + one final refit.
    for model_index in range(4):
        for fold in range(5):
            fitted_values = records[model_index * 5 + fold][1]
            assert fitted_values == records[fold][1]
            assert len(fitted_values) == 96
            assert fitted_values <= train_values and not fitted_values & test_values
    assert records[-1][1] == train_values
    assert result.selected_model == request.models[0]  # Stable tie handling.


@pytest.mark.parametrize("models", [[], ["Unsupported"], ["Linear Regression"] * 5])
def test_api_rejects_invalid_candidates(observations, models):
    with TestClient(app) as client:
        response = client.post("/models/compare", json={"rows": observations.model_dump()["rows"], "models": models})
    assert response.status_code == 422


def test_single_candidate_constant_target_api(observations):
    rows = observations.model_dump()["rows"]
    for row in rows:
        row["solid_concentration_pct"] = 35.0
    with TestClient(app) as client:
        response = client.post("/models/compare", json={"rows": rows, "models": ["Linear Regression"]})
    assert response.status_code == 200
    result = response.json()
    assert result["selected_model"] == "Linear Regression"
    assert result["results"][0]["mean"]["r2"] is None


def test_convergence_notice_is_reported():
    class Nonconverging:
        def fit(self, x, y):
            warnings.warn("iteration limit", ConvergenceWarning)
    assert fit_with_notice(Nonconverging(), [], [])


def test_comparison_client_rejects_bad_response(observations):
    with patch("frontend.api_client.requests.post", return_value=Mock(status_code=200, json=lambda: {})):
        with pytest.raises(BackendError, match="contract"):
            fetch_comparison("http://localhost", observations.model_dump()["rows"], 0.2, 42, ["Linear Regression"])


def test_comparison_ui_and_invalidation(observations, comparison):
    """Show all candidates, keep results on failure, clear after new data."""
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    dataset = observations.model_dump()
    with patch("frontend.api_client.requests.post") as post:
        page = AppTest.from_file(str(path), default_timeout=30).run()
        page.session_state["dataset"] = dataset
        page.run()
        page.checkbox[0].uncheck().run()
        post.return_value = Mock(status_code=200, json=lambda: comparison.model_dump())
        next(b for b in page.button if b.label == "Compare selected models").click().run()
        assert not page.exception and not page.error
        assert len(page.metric) == 3 and len(page.get("plotly_chart")) == 6
        page.selectbox[0].select("temperature_c").run()
        assert post.call_count == 1
        post.side_effect = requests.Timeout()
        next(b for b in page.button if b.label == "Compare selected models").click().run()
        assert not page.exception and page.error
        assert page.session_state["comparison_result"] == comparison.model_dump()
        post.side_effect = None
        post.return_value = Mock(status_code=200, json=lambda: comparison.model_dump())
        next(b for b in page.button if b.label == "Compare selected models").click().run()
        assert not page.error
        page.multiselect[0].set_value([]).run()
        call_count = post.call_count
        next(b for b in page.button if b.label == "Compare selected models").click().run()
        assert page.error and post.call_count == call_count
        # Generation has its own contract; mock that boundary to isolate reset behavior.
        with patch("frontend.data_explorer.fetch_dataset", return_value=dataset):
            next(b for b in page.button if b.label == "Generate synthetic data").click().run()
        assert not page.exception and not page.metric
        assert "comparison_result" not in page.session_state
