"""Stage 5: split-conformal intervals and post-evaluation model inspection."""

import math

import numpy as np
from sklearn.inspection import permutation_importance

from backend.schemas import FeatureImportance, IntervalPrediction, ModelDiagnostics, ResidualBin
from backend.services.modeling import FEATURES


def conformal_radius(errors: np.ndarray) -> tuple[float, int]:
    """Use the finite-sample 90% rank, without interpolating quantiles.

    The kth smallest absolute calibration error defines a constant half-width.
    Model fitting/selection must be complete before these errors are examined.
    """
    rank = math.ceil((len(errors) + 1) * 0.9)
    if rank > len(errors) or len(errors) == 0:
        raise ValueError("Insufficient calibration observations for finite 90% intervals")
    return float(np.sort(errors)[rank - 1]), rank


def concentration_bins(actual: np.ndarray, residual: np.ndarray) -> list[ResidualBin]:
    """Summarize three quantile ranges, merging repeated boundaries for ties."""
    edges = np.unique(np.quantile(actual, [0, 1/3, 2/3, 1]))
    if len(edges) == 1:
        edges = np.repeat(edges, 2)
    bins = []
    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (actual >= lower) & ((actual <= upper) if i == len(edges) - 2 else (actual < upper))
        values = residual[mask]
        if len(values):
            bins.append(ResidualBin(
                lower=float(lower), upper=float(upper), count=len(values),
                mean_residual=float(values.mean()), mae=float(np.abs(values).mean()),
                rmse=float(np.sqrt(np.mean(values**2))),
            ))
    return bins


def inspect_model(model, x_cal, y_cal, x_test, y_test, test_indices, seed: int) -> ModelDiagnostics:
    """Calibrate on untouched rows, then inspect the frozen model on test data.

    Test importance and residual summaries are descriptive only. They never
    feed back into selection, fitting, or interval calibration.
    """
    radius, rank = conformal_radius(np.abs(y_cal - model.predict(x_cal)))
    predicted = model.predict(x_test)
    residual = y_test - predicted
    lower, upper = predicted - radius, predicted + radius
    covered = (y_test >= lower) & (y_test <= upper)
    # Negative RMSE is a sklearn score (higher is better). Score decrease after
    # shuffling is therefore an INCREASE in RMSE, measured in percentage points.
    importance = permutation_importance(
        model, x_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=10, random_state=seed, n_jobs=1,
    )
    correlation = None
    if np.std(np.abs(residual)) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(np.abs(residual), predicted)[0, 1])
    return ModelDiagnostics(
        calibration_count=len(y_cal), quantile_rank=rank, half_width=radius,
        empirical_coverage=float(covered.mean()), mean_width=2 * radius,
        intervals=[IntervalPrediction(row_index=int(index), lower=float(lo), upper=float(hi), covered=bool(hit))
                   for index, lo, hi, hit in zip(test_indices, lower, upper, covered)],
        importance=[FeatureImportance(feature=name, mean=float(mean), std=float(std))
                    for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        mean_residual=float(residual.mean()), residual_std=float(residual.std()),
        abs_residual_prediction_correlation=correlation,
        residual_bins=concentration_bins(y_test, residual),
    )
