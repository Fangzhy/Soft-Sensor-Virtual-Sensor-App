"""Validated request and response contracts for synthetic datasets."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["Linear Regression", "Random Forest", "XGBoost", "Neural Network"]


class GenerationRequest(BaseModel):
    """Bound dataset size and random settings to keep the demo responsive."""

    model_config = ConfigDict(extra="forbid")
    n_samples: int = Field(default=1000, ge=100, le=5000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    noise_std: float = Field(default=1.0, ge=0, le=5, allow_inf_nan=False)


class SensorRow(BaseModel):
    """One independent simulated observation; column names encode units."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature_c: float = Field(ge=20, le=90)
    density_kg_m3: float = Field(ge=1000, le=1300)
    flow_rate_l_min: float = Field(ge=10, le=100)
    pressure_bar: float = Field(ge=1, le=6)
    agitation_rpm: float = Field(ge=100, le=800)
    solid_concentration_pct: float = Field(ge=0, le=100)


class DatasetResponse(BaseModel):
    """Return the actual settings alongside the data for reproducible exploration."""

    source: Literal["synthetic"] = "synthetic"
    generator_version: Literal["1.0"] = "1.0"
    target_basis: Literal["mass percent (w/w)"] = "mass percent (w/w)"
    settings: GenerationRequest
    clipped_target_count: int
    rows: list[SensorRow]


class TrainingRequest(BaseModel):
    """Train on the displayed observations, with a reproducible random split."""

    model_config = ConfigDict(extra="forbid")
    rows: list[SensorRow] = Field(min_length=100, max_length=5000)
    test_fraction: float = Field(default=0.2, ge=0.1, le=0.4, allow_inf_nan=False)
    split_seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)


class RegressionMetrics(BaseModel):
    """Errors are in percentage points; R² is undefined for a constant target."""

    model_config = ConfigDict(allow_inf_nan=False)
    r2: float | None
    rmse: float = Field(ge=0)
    mae: float = Field(ge=0)


class TestPrediction(BaseModel):
    """A held-out prediction linked to the original zero-based dataset row."""

    model_config = ConfigDict(allow_inf_nan=False)
    row_index: int
    actual: float
    predicted: float
    residual: float


class TrainingResponse(BaseModel):
    """Evaluation artifacts, not a serialized executable model."""

    model: ModelName = "Linear Regression"
    train_count: int
    test_count: int
    test_fraction: float
    split_seed: int
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    baseline_test_metrics: RegressionMetrics
    predictions: list[TestPrediction]


class ComparisonRequest(TrainingRequest):
    """Compare one to four model families using the same training folds."""

    models: list[ModelName] = Field(
        default_factory=lambda: ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"],
        min_length=1, max_length=4,
    )
    include_diagnostics: bool = Field(default=False, strict=True)


class FeatureImportance(BaseModel):
    """Increase in test RMSE after shuffling one feature; not a percentage share."""

    model_config = ConfigDict(allow_inf_nan=False)
    feature: str
    mean: float
    std: float = Field(ge=0)


class IntervalPrediction(BaseModel):
    """Unclipped split-conformal bounds for one held-out observation."""

    model_config = ConfigDict(allow_inf_nan=False)
    row_index: int
    lower: float
    upper: float
    covered: bool


class ResidualBin(BaseModel):
    """Descriptive residual statistics in a range of actual concentrations."""

    model_config = ConfigDict(allow_inf_nan=False)
    lower: float
    upper: float
    count: int
    mean_residual: float
    mae: float
    rmse: float


class ModelDiagnostics(BaseModel):
    """Separate calibration information, test-only diagnostics, and importances."""

    model_config = ConfigDict(allow_inf_nan=False)
    nominal_coverage: Literal[0.9] = 0.9
    calibration_count: int
    quantile_rank: int
    half_width: float = Field(ge=0)
    empirical_coverage: float = Field(ge=0, le=1)
    mean_width: float = Field(ge=0)
    intervals: list[IntervalPrediction]
    importance_repeats: Literal[10] = 10
    importance: list[FeatureImportance]
    mean_residual: float
    residual_std: float = Field(ge=0)
    abs_residual_prediction_correlation: float | None
    residual_bins: list[ResidualBin]


class CrossValidationResult(BaseModel):
    """Fold scores and their unweighted mean/population standard deviation."""

    model: ModelName
    folds: list[RegressionMetrics]
    mean: RegressionMetrics
    std: RegressionMetrics
    warnings: list[str]


class ComparisonResponse(BaseModel):
    """Only the CV-selected model is evaluated on the held-out test set."""

    selection_metric: Literal["mean CV RMSE"] = "mean CV RMSE"
    cv_folds: Literal[5] = 5
    results: list[CrossValidationResult]
    selected_model: ModelName
    evaluation: TrainingResponse
    warnings: list[str]
    diagnostics: ModelDiagnostics | None = None
