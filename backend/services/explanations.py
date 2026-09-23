"""Explain server-owned model evidence through a free-only OpenRouter route."""

import json
import logging
from contextlib import contextmanager
from time import monotonic

import requests

from backend.config import DEFAULT_FREE_MODEL, openrouter_settings
from backend.schemas import ExplanationResponse
from backend.services.run_store import RUNS

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
logger = logging.getLogger(__name__)


@contextmanager
def try_explanation_lock(lock):
    """Never queue a second HTTP request behind a slow provider call."""
    acquired = lock.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


class InvalidExplanation(ValueError):
    """A fixed, secret-free reason a provider response cannot be displayed."""


def routing_error_label(response) -> str:
    """Classify routing errors without logging arbitrary provider text or metadata."""
    labels = {401: "API key rejected", 402: "account or credit restriction",
              403: "provider access or privacy restriction", 429: "free-service rate limit reached"}
    if response.status_code != 404:
        return labels.get(response.status_code, "service error")
    try:
        message = response.json().get("error", {}).get("message", "").lower()
    except (ValueError, AttributeError, TypeError):
        message = ""
    if "data policy" in message or "privacy" in message:
        return "no endpoint matches your account data/privacy policy; review OpenRouter privacy settings"
    if "parameter" in message or "structured" in message:
        return "no endpoint supports the requested parameters/structured output"
    return "no eligible model/provider endpoint; check model availability, required parameters, and account routing settings"

SECTIONS = {
    "performance": "Performance",
    "influential_sensors": "Influential sensors",
    "error_patterns": "Error patterns",
    "uncertainty_and_limitations": "Uncertainty and limitations",
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "model_explanation", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {key: {"type": "string", "description": f"A short evidence-based paragraph about {label.lower()}."}
                       for key, label in SECTIONS.items()},
        "required": list(SECTIONS),
    }},
}
SYSTEM_PROMPT = """You explain a synthetic sensor-regression demo to a learner.
Use ONLY the supplied evidence. Write four short sections: Performance,
Influential sensors, Error patterns, and Uncertainty and limitations.
Keep the whole answer under 350 words. Quote only supplied numerical facts.
RMSE/MAE and importance increases are percentage points, not percentages.
Permutation importance measures model reliance, not causation or a percent share.
Correlated flow and pressure can distort rankings. Do not infer importance if absent.
Describe residual patterns only using the supplied summaries; do not claim
statistical significance or a physical mechanism. R2=null means undefined.
90% conformal intervals have marginal coverage under exchangeability, not a
per-observation or per-range guarantee. Distinguish nominal from test coverage.
If diagnostics are absent, state that intervals/importance were not calculated.
Mention optimizer warnings when present. Results on synthetic data do not
establish real-process performance. Never invent predictions or change metrics.
Return the required JSON object with four explanatory paragraph strings,
without HTML, links, or executable code."""


def template_text(evidence: dict) -> str:
    """Return a useful deterministic fallback clearly distinguished from AI."""
    scores = evidence["test_metrics"]
    r2 = "undefined" if scores["r2"] is None else f"{scores['r2']:.3f}"
    text = (f"Performance: {evidence['model']} achieved test R² {r2}, "
            f"RMSE {scores['rmse']:.3f} and MAE {scores['mae']:.3f} percentage points.\n\n")
    d = evidence["diagnostics"]
    if d:
        strongest = max(d["importance"], key=lambda item: item["mean"])
        text += (f"Influential sensors: The largest measured permutation RMSE increase was "
                 f"{strongest['feature']} ({strongest['mean']:.3f} percentage points). "
                 "This is model reliance, not causation; correlated sensors can distort rankings.\n\n"
                 f"Error patterns: Mean residual was {d['mean_residual']:.3f} percentage points "
                 "(positive means underprediction). Group summaries are descriptive, not formal tests.\n\n"
                 f"Uncertainty and limitations: Nominal coverage is 90%; observed test coverage "
                 f"was {d['empirical_coverage']:.1%}, with mean width {d['mean_width']:.3f} percentage points. "
                 "Coverage is marginal under exchangeability, not guaranteed for each input.")
    else:
        text += ("Influential sensors: Permutation importance was not calculated.\n\n"
                 "Error patterns: Detailed residual summaries were not calculated.\n\n"
                 "Uncertainty and limitations: No calibrated interval is available for this run.")
    notices = evidence["optimizer_warnings"] + [w for item in evidence["cv"] or [] for w in item["warnings"]]
    if notices:
        text += " Optimizer convergence warnings were reported; inspect them before relying on the results."
    return text + " Synthetic-demo results do not establish real-process performance."


def explain_run(run_id: str) -> ExplanationResponse:
    """Make at most one outbound call per click; cache successful AI responses.

    No redirects, paid fallback, automatic retry, or raw provider-error logging.
    The per-run lock rejects concurrent requests immediately instead of queuing.
    Failed calls return an uncached local template so the user can retry later.
    """
    run = RUNS.get(run_id)
    started = monotonic()
    with try_explanation_lock(run.explanation_lock) as acquired:
        if not acquired:
            return ExplanationResponse(
                run_id=run_id, source="template", model=None, evidence=run.evidence,
                text=template_text(run.evidence),
                notice="An explanation request for this run is already in progress. No additional OpenRouter call was made. Wait before retrying.",
            )
        if run.explanation is not None:
            return ExplanationResponse(**{**run.explanation, "cached": True})
        key, model = openrouter_settings()
        notice = None
        if not key or key == "your-key-here":
            notice = "OPENROUTER_API_KEY is missing. Showing a local template."
        elif model not in {"openrouter/free", DEFAULT_FREE_MODEL}:
            notice = "This demo permits only its verified free-model allowlist; no external request was made."
        else:
            try:
                response = requests.post(
                    ENDPOINT, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(run.evidence, allow_nan=False)},
                    ], "max_tokens": 1600, "stream": False,
                        "response_format": RESPONSE_FORMAT,
                        "provider": {"require_parameters": True, "max_price": {"prompt": 0, "completion": 0}},
                        **({"reasoning": {"enabled": False}} if model == DEFAULT_FREE_MODEL else {})},
                    timeout=(10, 60), allow_redirects=False,
                )
                if response.status_code != 200:
                    notice = f"OpenRouter HTTP {response.status_code}: {routing_error_label(response)}. Showing a local template."
                else:
                    payload = response.json()
                    choice = payload["choices"][0]
                    content, actual_model = choice["message"]["content"], payload["model"]
                    if choice.get("finish_reason") == "length":
                        raise InvalidExplanation("OpenRouter stopped at the output token limit before completing the explanation; this was not a timeout.")
                    if (not isinstance(content, str) or not content.strip() or len(content) > 20000
                            or not isinstance(actual_model, str) or not actual_model.strip()
                            or choice.get("finish_reason") != "stop"):
                        raise InvalidExplanation("OpenRouter returned empty, oversized, or unfinished explanation content; this was not a timeout.")
                    # A live free-router check returned a safety classifier's
                    # verdict instead of an explanation. Require all four
                    # substantive sections; never cache that kind of non-answer.
                    sections = json.loads(content)
                    if (not isinstance(sections, dict) or set(sections) != set(SECTIONS)
                            or any(not isinstance(sections[name], str) or len(sections[name].strip()) < 20
                                   for name in SECTIONS)):
                        raise InvalidExplanation("OpenRouter returned JSON without all four substantive explanation sections; this was not a timeout.")
                    content = "\n\n".join(f"{label}\n{sections[name].strip()}" for name, label in SECTIONS.items())
                    # Never return a secret even if a provider unexpectedly echoes it.
                    result = ExplanationResponse(run_id=run_id, source="openrouter", model=actual_model.replace(key, "[redacted]"),
                                                 text=content.strip().replace(key, "[redacted]"), evidence=run.evidence)
                    run.explanation = result.model_dump()
                    return result
            except InvalidExplanation as exc:
                notice = f"{exc} Showing a local template."
            except requests.ConnectTimeout:
                notice = "Connection to OpenRouter timed out (10-second connection limit). Showing a local template."
            except requests.ReadTimeout:
                notice = "OpenRouter did not deliver response data within the 60-second read timeout. Showing a local template."
            except requests.Timeout:
                notice = "The OpenRouter request timed out. Showing a local template."
            except requests.exceptions.JSONDecodeError:
                # Requests' JSONDecodeError is also a RequestException; catch it first.
                notice = "OpenRouter returned invalid response JSON; this was not a timeout. Showing a local template."
            except requests.ConnectionError:
                notice = "Could not connect to OpenRouter (network, DNS, TLS, or proxy failure). Showing a local template."
            except requests.RequestException:
                notice = "OpenRouter transport failed. Showing a local template."
            except (ValueError, KeyError, IndexError, TypeError):
                notice = "OpenRouter returned malformed JSON or an unexpected response structure; this was not a timeout. Showing a local template."
        # Only our fixed diagnostic text is logged, never exception strings,
        # provider bodies, headers, prompts, credentials, or evidence.
        elapsed = monotonic() - started
        logger.warning("Explanation fallback after %.1fs: %s", elapsed, notice)
        notice = f"{notice} Elapsed: {elapsed:.1f}s."
        return ExplanationResponse(run_id=run_id, source="template", model=None,
                                   text=template_text(run.evidence), evidence=run.evidence, notice=notice)
