# Stage 6: predict new observations and explain model results

This stage adds two actions below a trained model's evaluation:

- **Predict solid concentration** uses the retained regression model.
- **Explain this model** asks OpenRouter's free LLM service to interpret a
  compact summary of measured model results.

The LLM never computes the concentration prediction, retrains the regression
model, or changes metrics. Its text can contain errors; compare it with the
evidence displayed underneath.

## Setup

With the virtual environment active, install the added dotenv dependency:

```powershell
python -m pip install -r requirements-dev.txt
```

The backend explicitly reads the **repository-root `.env`**, not `docs/.env`.
Use the variable names shown in `.env.example`:

```dotenv
OPENROUTER_API_KEY=your-real-key-here
OPENROUTER_MODEL=nex-agi/nex-n2.5-mini:free
```

Keep real keys in `.env`, which is ignored by Git. `.env.example` contains
placeholders only. Existing process environment variables take precedence.
Your existing `OPEN_ROUTER_API` variable is also supported as an alias;
you do not need to rename it. The canonical name wins within the same source.
Do not put your key into Streamlit widgets, source files, or browser code.

Restart FastAPI and Streamlit using the README commands. Train or compare
again: previous results do not have a retained Stage 6 run. Start with the
default comparison and Stage 5 diagnostics enabled for calibrated intervals.

## Prediction: retain and reuse, do not retrain

Previously each model existed only for the duration of a training request.
The backend now retains the final fitted model and returns an opaque `run_id`.
It also stores observed training ranges, the calibration half-width when
available, and a compact results summary. Requests use the run ID to select
the exact fitted model; there is no shared "latest model" slot.

The five input fields use the synthetic operating limits from Stage 2. FastAPI
validates those limits independently of the UI. Inputs outside the narrower
observed training range produce warnings. Being inside each individual range
does not establish that a combination of measurements is familiar to the model.

The prediction pipeline applies its existing preprocessing and returns the
raw prediction. For a calibrated run, bounds are `prediction ± existing q`.
There is no new calibration, fitting, or clipping. Stage 3 and comparisons
without Stage 5 provide point predictions with no fabricated interval.
Conformal coverage assumptions still apply; a 90% marginal guarantee does not
mean a 90% conditional guarantee for a particular user-chosen sensor setting.

The store holds at most 16 runs for one hour each, evicting the oldest at
capacity. Backend restarts (including auto-reload) discard them. An unavailable
run produces HTTP 404 with instructions to retrain. This is a **local,
single-worker** demo: run IDs act as access capabilities, not user authentication.
Do not expose it as a public multi-user service without authentication and
appropriate shared storage. No fitted model is serialized to disk.

## Exactly how the free LLM call works

```text
Streamlit button
  → POST /models/explain with run_id
  → backend retrieves its own stored evidence
  → HTTPS POST https://openrouter.ai/api/v1/chat/completions
  → OpenRouter routes to the pinned free model
  → backend validates the reply and returns text + model name + evidence
```

The backend sends an Authorization Bearer header containing the key and JSON
with `model: "nex-agi/nex-n2.5-mini:free"`, a system instruction, and a user message
containing the evidence summary. It requests a nonstreamed response with a
maximum of 1,600 output tokens. It disables redirects and uses a 10-second
connection timeout and 60-second read timeout. There is one external call per
uncached click, with no automatic retry. Timeout values are socket-operation
limits rather than a strict end-to-end deadline.

It also requests a strict JSON schema with four paragraph fields and sets
`provider.require_parameters=true`. This requests a compatible free endpoint
instead of silently ignoring the output format. The backend checks every
section, then renders it as plain text. A safety-classification verdict such
as "User Safety: safe" is rejected as an unusable explanation. If no compatible
free provider is available, the labeled local template is used.

The default is now **`nex-agi/nex-n2.5-mini:free`**, verified against the
OpenRouter catalog for zero prompt/completion prices and structured-output
support. Reasoning is disabled for this short explanatory task. A small
allowlist permits this model and the legacy `openrouter/free` router; arbitrary
paid model IDs are rejected. Provider `max_price` limits for prompt and
completion are both zero, with no paid fallback.

The root `.env` model setting has been updated; the API key is unchanged.
Existing process environment variables still take precedence. Free model
availability, latency, provider permissions, and quotas can vary. OpenRouter
and its chosen provider receive the summary under their applicable policies.
The actual returned model name is displayed.

### Evidence sent

The prompt contains the selected model, split counts, CV summaries, training
and test metrics, baseline metrics, optimizer warnings, permutation importance,
residual summaries, and interval coverage/width when present. It excludes raw
dataset rows, individual test predictions/intervals, new sensor inputs, run ID,
and API key. The backend accepts no user-supplied metrics in the explanation
request, preventing accidental mismatch with the trained model.

The system prompt requests four short sections: performance, influential
sensors, error patterns, and uncertainty/limitations. It explicitly requires
the correct units, distinguishes importance from causation, and prohibits
unsupported residual or physical claims. Prompt instructions reduce errors
but are not a formal factuality guarantee. Output is displayed as plain text.

### Failure handling and caching

Missing credentials, an unsupported model setting, rejected keys, rate limits,
service failures, timeouts, and incomplete responses lead to a clearly labeled
**local template explanation — not LLM-generated**. Existing evaluation and
prediction functionality remains available. Raw provider errors and credentials
are not logged or sent to the UI.

Successful AI explanations are cached per run on the backend and retained in
Streamlit state. Clicking again reuses the answer; concurrent backend requests
for the same run return an immediate in-progress message rather than queueing.
Fallbacks are not cached on the backend, so a
later click can retry after fixing settings or waiting for service recovery.
Ordinary chart interactions never trigger LLM calls. New training runs clear
the visible prediction and explanation; data regeneration clears all results.

## Code reading order

1. `backend/config.py`: root `.env` loading and environment precedence.
2. `backend/services/run_store.py`: bounded retention and evidence construction.
3. `backend/services/inference.py`: prediction and training-range checks.
4. `backend/services/explanations.py`: prompt, HTTP request, response checks,
   per-run cache, and deterministic fallback.
5. `backend/main.py`: `/models/predict` and `/models/explain` endpoints.
6. `frontend/prediction_panel.py`: forms, explicit actions, state invalidation,
   and evidence display.

The training/comparison service functions remain usable without retention for
unit tests. HTTP training routes enable retention and return a run ID.

## Learning checkpoint

1. Compare models with diagnostics enabled. Enter one set of sensor values and
   predict. Confirm the interval has the same width as the evaluated intervals.
2. Change an input and submit again. The model remains fixed; only its input
   changes. Try a boundary value to see a training-range warning.
3. Click Explain and compare the resulting text with the expandable evidence.
   Check whether it correctly distinguishes nominal from observed coverage.
4. Switch to the Stage 3 walkthrough and train. Confirm that predictions work
   and the app states that no calibrated interval is available.
5. Restart FastAPI and try predicting with the old run. Retrain when prompted.

Run `python -m pytest -q` for local checks. Automated tests mock OpenRouter and
never use your real key or free quota. A live verification is a separate,
explicit action.

The initial live verification authenticated successfully and verified real
prediction/interval reuse, but the free router selected a safety classifier
that returned only a safety verdict. That finding prompted the structured
output requirement and regression tests above. That earlier router configuration was initially checked only with mocked
responses. During the later user-authorized 404 investigation, the pinned
`nex-agi/nex-n2.5-mini:free` model returned a valid four-section structured
explanation in 7.6 seconds. Future availability is not guaranteed.

References: [OpenRouter free router](https://openrouter.ai/docs/guides/routing/routers/free-router),
[chat-completion quickstart](https://openrouter.ai/docs/quickstart), and
[service limits](https://openrouter.ai/docs/api_reference/limits).
See also [structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

## Diagnosing a fast HTTP 404

A 404 from OpenRouter is distinct from a missing local model run. It can
indicate that no provider satisfies the model, capability requirements, or
account routing/data policy. The backend now classifies those causes from the
error message, without exposing raw provider metadata. It reports a generic
routing explanation when the cause cannot be identified.

FastAPI returns HTTP 200 when it successfully supplies the labeled local
fallback, even if OpenRouter returned 404. A request rejected before generation
may not appear as a completed generation in the OpenRouter activity view;
absence there alone does not identify the cause. The original router returned
200 during a later diagnostic request, so the historical 404 was not reproduced.
