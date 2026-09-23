"""Cloud adapter: validated backend services with per-session run ownership.

    Module-level locks survive Streamlit reruns and are shared by visitors in
    this process. They bound work without caching datasets across sessions.
"""

from threading import Lock
from time import monotonic

from pydantic import ValidationError

from backend.schemas import GenerationRequest, TrainingRequest, ComparisonRequest, PredictionRequest, RunRequest
from backend.services.synthetic_data import generate_dataset
from backend.services.modeling import train_linear_regression
from backend.services.model_comparison import compare_models
from backend.services.inference import predict_observation
from backend.services.explanations import explain_run, template_text
from backend.services.run_store import RUNS

TRAINING_LOCK = Lock()
EXPLANATION_LOCK = Lock()
EXPLANATION_COOLDOWN = 30.0
_next_explanation_at = 0.0
SESSION_RUNS = "_cloud_owned_runs"


def _explain(run_id: str) -> dict:
    """Reuse cached explanations; throttle uncached attempts across visitors.

    A busy service returns an explicitly labeled template, without consuming
    OpenRouter quota. The cooldown also applies to failed provider attempts.
    """
    global _next_explanation_at
    run = RUNS.get(run_id)
    if run.explanation is not None:
        return {**run.explanation, "cached": True}
    acquired = EXPLANATION_LOCK.acquire(blocking=False)
    try:
        if not acquired or monotonic() < _next_explanation_at:
            return dict(run_id=run_id, source="template", model=None, cached=False,
                        text=template_text(run.evidence), evidence=run.evidence,
                        notice="The shared demo explanation service is busy or cooling down. Wait 30 seconds and retry; no OpenRouter call was made.")
        _next_explanation_at = monotonic() + EXPLANATION_COOLDOWN
        return explain_run(run_id).model_dump(mode="json")
    finally:
        if acquired:
            EXPLANATION_LOCK.release()


def dispatch(method: str, path: str, body: dict, session) -> tuple[int, dict]:
    """Validate inputs just as FastAPI does, then serialize the service result.

    Only run IDs created in this browser session can be used here. A reconnect
    may lose that session; retraining then creates a new owned run. The bounded
    global store still expires and evicts models independently of the browser.
    """
    try:
        if method == "get" and path == "/health":
            return 200, dict(status="ok", service="sensor-fusion-api", version="0.1.0",
                             target="Solid concentration", target_unit="%")
        if method != "post":
            return 404, {}
        if path == "/data/generate":
            result = generate_dataset(GenerationRequest.model_validate(body))
        elif path in {"/models/train", "/models/compare"}:
            comparing = path == "/models/compare"
            schema = ComparisonRequest if comparing else TrainingRequest
            settings = schema.model_validate(body)
            if not TRAINING_LOCK.acquire(blocking=False):
                return 429, {"detail": "Another visitor is training. Please retry shortly."}
            try:
                service = compare_models if comparing else train_linear_regression
                result = service(settings, persist=True)
            finally:
                TRAINING_LOCK.release()
            run_id = result.evaluation.run_id if comparing else result.run_id
            # At most 16 models exist globally; bound session bookkeeping too.
            session[SESSION_RUNS] = (list(session.get(SESSION_RUNS, [])) + [run_id])[-16:]
        elif path in {"/models/predict", "/models/explain"}:
            schema = PredictionRequest if path.endswith("predict") else RunRequest
            settings = schema.model_validate(body)
            if settings.run_id not in session.get(SESSION_RUNS, []):
                return 404, {}
            if path.endswith("explain"):
                return 200, _explain(settings.run_id)
            result = predict_observation(settings)
        else:
            return 404, {}
        return 200, result.model_dump(mode="json")
    except ValidationError:
        # Do not echo submitted values or exception internals to public users.
        return 422, {"detail": "Invalid request settings."}
    except KeyError:
        return 404, {"detail": "Model run unavailable. Train or compare again."}
