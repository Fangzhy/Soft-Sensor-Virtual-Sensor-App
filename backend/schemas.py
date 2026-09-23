"""Validated request and response contracts for synthetic datasets."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    model: Literal["Linear Regression"] = "Linear Regression"
    train_count: int
    test_count: int
    test_fraction: float
    split_seed: int
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    baseline_test_metrics: RegressionMetrics
    predictions: list[TestPrediction]
