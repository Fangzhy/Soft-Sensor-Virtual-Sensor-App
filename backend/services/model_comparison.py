"""Stage 4 model factory and training-only five-fold comparison.

Construct a fresh estimator for every fold so preprocessing, fitted weights,
and target scaling cannot carry information between folds.
"""

import warnings
import math

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from backend.schemas import (
    ComparisonRequest, ComparisonResponse, CrossValidationResult,
    RegressionMetrics, TestPrediction, TrainingResponse,
)
from backend.services.modeling import FEATURES, regression_metrics
from backend.services.diagnostics import inspect_model


def build_model(name: str, seed: int):
    """Return a small deterministic demo preset, with appropriate preprocessing.

    Trees use native sensor units. Linear and neural models standardize inputs.
    The neural network also scales the target within each fit, then converts
    predictions back to percentage points automatically.
    """
    if name == "Linear Regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=2,
            random_state=seed, n_jobs=1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            objective="reg:squarederror", tree_method="hist",
            random_state=seed, n_jobs=1,
        )
    if name == "Neural Network":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32, 16), activation="tanh", solver="lbfgs",
                alpha=0.1, max_iter=800, max_fun=15000, tol=1e-3, random_state=seed,
            )), transformer=StandardScaler(),
        )
    raise ValueError(f"Unsupported model: {name}")


def fit_with_notice(model, x, y) -> bool:
    """Fit and report nonconvergence instead of silently hiding the warning."""
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return any(issubclass(item.category, ConvergenceWarning) for item in notices)


def summarize_folds(scores: list[RegressionMetrics], use_std: bool = False) -> RegressionMetrics:
    """Aggregate five scores; leave R² undefined if any fold has constant y."""
    aggregate = np.std if use_std else np.mean
    return RegressionMetrics(**{
        metric: None if any(getattr(score, metric) is None for score in scores)
        else float(aggregate([getattr(score, metric) for score in scores]))
        for metric in ("r2", "rmse", "mae")
    })


def compare_models(request: ComparisonRequest, persist: bool = False) -> ComparisonResponse:
    """Rank by training CV RMSE, then refit and test only the selected model.

    The outer test rows never appear in a CV fold. Precomputing folds once
    guarantees identical observations for all candidates. This is a comparison
    of fixed presets, not a hyperparameter search.
    """
    x = np.array([[getattr(row, name) for name in FEATURES] for row in request.rows])
    y = np.array([row.solid_concentration_pct for row in request.rows])
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=request.test_fraction, random_state=request.split_seed,
    )
    calibration_idx = None
    if request.include_diagnostics:
        # Reserve 20% of ALL observations, after preserving the outer test split.
        # Only the remaining proper-training rows may enter CV or the final fit.
        train_idx, calibration_idx = train_test_split(
            train_idx, test_size=math.ceil(len(y) * 0.2), random_state=request.split_seed,
        )
    x_train, y_train = x[train_idx], y[train_idx]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=request.split_seed).split(x_train))
    results = []
    # Deduplicate API input without changing user order; ties use that order.
    for name in dict.fromkeys(request.models):
        scores, notices = [], []
        for number, (fit_idx, validation_idx) in enumerate(folds, start=1):
            model = build_model(name, request.split_seed)
            if fit_with_notice(model, x_train[fit_idx], y_train[fit_idx]):
                notices.append(f"Fold {number}: optimizer reached its limit; convergence was not confirmed.")
            scores.append(regression_metrics(
                y_train[validation_idx], model.predict(x_train[validation_idx]),
            ))
        results.append(CrossValidationResult(
            model=name, folds=scores, mean=summarize_folds(scores),
            std=summarize_folds(scores, use_std=True), warnings=notices,
        ))
    winner = min(results, key=lambda result: result.mean.rmse).model
    final_model = build_model(winner, request.split_seed)
    final_notices = []
    if fit_with_notice(final_model, x_train, y_train):
        final_notices.append("Selected model: optimizer reached its limit on the full training set.")
    predicted = final_model.predict(x[test_idx])
    evaluation = TrainingResponse(
        model=winner, train_count=len(train_idx), test_count=len(test_idx),
        test_fraction=request.test_fraction, split_seed=request.split_seed,
        train_metrics=regression_metrics(y_train, final_model.predict(x_train)),
        test_metrics=regression_metrics(y[test_idx], predicted),
        baseline_test_metrics=regression_metrics(y[test_idx], np.full(len(test_idx), y_train.mean())),
        predictions=[TestPrediction(
            row_index=int(index), actual=float(y[index]), predicted=float(value),
            residual=float(y[index] - value),
        ) for index, value in zip(test_idx, predicted)],
    )
    result = ComparisonResponse(
        results=results, selected_model=winner, evaluation=evaluation, warnings=final_notices,
        diagnostics=None if calibration_idx is None else inspect_model(
            final_model, x[calibration_idx], y[calibration_idx],
            x[test_idx], y[test_idx], test_idx, request.split_seed,
        ),
    )
    if persist:
        from backend.services.run_store import retain_run
        result.evaluation.run_id = retain_run(final_model, x_train, FEATURES, evaluation, result)
    return result
