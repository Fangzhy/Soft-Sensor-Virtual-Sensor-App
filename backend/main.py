"""Stage 1 API: expose a health check before adding data or model endpoints.

Run from the project root with: python -m uvicorn backend.main:app --reload
Here, ``backend.main`` identifies this module and ``app`` is the object below.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


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
