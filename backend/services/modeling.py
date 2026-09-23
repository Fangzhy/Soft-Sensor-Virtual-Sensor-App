"""Stage 3: fit a linear baseline without leaking test data into preprocessing."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.schemas import RegressionMetrics, TestPrediction, TrainingRequest, TrainingResponse

FEATURES = (
    "temperature_c", "density_kg_m3", "flow_rate_l_min", "pressure_bar", "agitation_rpm",
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    """Calculate scores at full precision, leaving display rounding to the UI.

    R² divides by target variation. Return null when the target is constant
    instead of reporting a misleading finite score for this undefined case.
    """
    return RegressionMetrics(
        r2=None if np.all(actual == actual[0]) else float(r2_score(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        mae=float(mean_absolute_error(actual, predicted)),
    )


def train_linear_regression(request: TrainingRequest, persist: bool = False) -> TrainingResponse:
    """Split row indices, fit only training rows, and predict held-out targets.

    HTTP routes request retention under a unique run ID for Stage 6 inference.
    Direct callers can keep the original stateless behavior with persist=False.
    """
    # An explicit feature list prevents accidental use of the target as an input.
    x = np.array([[getattr(row, name) for name in FEATURES] for row in request.rows])
    y = np.array([row.solid_concentration_pct for row in request.rows])
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=request.test_fraction, random_state=request.split_seed,
    )
    # Scaling is not essential for ordinary least squares, but the pipeline
    # teaches a reusable pattern for later scale-sensitive models. Both steps
    # learn from training data ONLY; predict reuses those fitted parameters.
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x[train_idx], y[train_idx])
    predicted = model.predict(x[test_idx])
    # A simple reference: always predict the training-set mean concentration.
    baseline = np.full(len(test_idx), y[train_idx].mean())
    result = TrainingResponse(
        train_count=len(train_idx), test_count=len(test_idx),
        test_fraction=request.test_fraction, split_seed=request.split_seed,
        train_metrics=regression_metrics(y[train_idx], model.predict(x[train_idx])),
        test_metrics=regression_metrics(y[test_idx], predicted),
        baseline_test_metrics=regression_metrics(y[test_idx], baseline),
        predictions=[TestPrediction(
            row_index=int(index), actual=float(y[index]), predicted=float(value),
            residual=float(y[index] - value),
        ) for index, value in zip(test_idx, predicted)],
    )
    if persist:
        from backend.services.run_store import retain_run
        result.run_id = retain_run(model, x[train_idx], FEATURES, result)
    return result
