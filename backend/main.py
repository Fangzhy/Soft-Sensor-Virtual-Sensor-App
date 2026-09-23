"""FastAPI routes for health, synthetic data, training, and model comparison.

Run from the project root with: python -m uvicorn backend.main:app --reload
Here, ``backend.main`` identifies this module and ``app`` is the object below.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from backend.schemas import DatasetResponse, GenerationRequest, TrainingRequest, TrainingResponse
from backend.services.modeling import train_linear_regression
from backend.schemas import ComparisonRequest, ComparisonResponse
from backend.services.model_comparison import compare_models
from backend.services.synthetic_data import generate_dataset


class HealthResponse(BaseModel):
    """Describe the JSON contract that the frontend can expect from this API."""

    # Literal restricts these fields to known values and documents them in /docs.
    status: Literal["ok"] = "ok"
    service: Literal["sensor-fusion-api"] = "sensor-fusion-api"
    version: str = "0.1.0"
    target: Literal["Solid concentration"] = "Solid concentration"
    target_unit: Literal["%"] = "%"


app = FastAPI(
    title="SensorData-FusionPredtionExplaination API",
    description="A learning demo for predicting solid concentration (%) from sensors.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Confirm that the API can respond; this does not indicate model readiness.

    FastAPI serializes this Pydantic object into JSON and validates its shape
    against HealthResponse. Later stages will add separate model endpoints.
    """
    return HealthResponse()


@app.post("/data/generate", response_model=DatasetResponse, tags=["Data"])
def create_dataset(settings: GenerationRequest) -> DatasetResponse:
    """Validate JSON settings and return a fresh synthetic dataset.

    FastAPI returns 422 for invalid inputs before calling this function. The
    service owns the numerical recipe; this route only connects it to HTTP.
    Datasets are returned to the caller, not stored globally on the server.
    """
    return generate_dataset(settings)


@app.post("/models/train", response_model=TrainingResponse, tags=["Models"])
def train_model(request: TrainingRequest) -> TrainingResponse:
    """Evaluate Linear Regression on the submitted dataset's held-out rows."""
    return train_linear_regression(request)


@app.post("/models/compare", response_model=ComparisonResponse, tags=["Models"])
def compare_model_candidates(request: ComparisonRequest) -> ComparisonResponse:
    """Compare fixed presets using training-only CV and evaluate the winner."""
    return compare_models(request)
