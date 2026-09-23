"""Predict with a retained model without fitting or recalibrating it."""

import numpy as np

from backend.schemas import PredictionRequest, PredictionResponse
from backend.services.modeling import FEATURES
from backend.services.run_store import RUNS


def predict_observation(request: PredictionRequest) -> PredictionResponse:
    """Reuse the pipeline's preprocessing and existing calibration half-width."""
    run = RUNS.get(request.run_id)
    values = request.inputs.model_dump()
    prediction = float(run.model.predict(np.array([[values[name] for name in FEATURES]]))[0])
    notices = [f"{name} is outside the observed training range ({limits['min']:.3f}–{limits['max']:.3f})."
               for name, limits in run.ranges.items()
               if not limits["min"] <= values[name] <= limits["max"]]
    if not 0 <= prediction <= 100:
        notices.append("The raw prediction lies outside 0–100%; it has not been clipped.")
    return PredictionResponse(
        run_id=request.run_id, model=run.model_name, inputs=request.inputs, prediction=prediction,
        lower=None if run.radius is None else prediction - run.radius,
        upper=None if run.radius is None else prediction + run.radius,
        nominal_coverage=None if run.radius is None else 0.9, warnings=notices,
    )
