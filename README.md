# SensorData-FusionPredtionExplaination

[Launch live demo](https://sensor-fusion-lab.streamlit.app) | [Cloud deployment guide](docs/deployment.md)

A step-by-step learning project using **FastAPI** for the backend and
**Streamlit** for the frontend. The prediction target is **solid concentration
(%)**, using temperature, density, flow rate, pressure, and agitation speed.

## Complete portfolio demo ? all seven stages

The app generates reproducible synthetic sensor data through shared backend services and
explores it in Streamlit. Choose 100–5000 rows, a random seed, and target noise;
inspect summaries, histograms, scatter plots, and correlations; download all
rows as CSV. Solid concentration is mass percent (w/w).
Train Linear Regression on the displayed data and inspect held-out R², RMSE,
MAE, actual-versus-predicted and residual plots, plus a training-mean baseline.
Compare Linear Regression, Random Forest, XGBoost, and a neural network using
five identical training-only folds. Select by mean CV RMSE and evaluate the
selected model on held-out test rows. The Stage 1 health check and Stage 3
walkthrough remain available. Predict new sensor observations with a retained
model, and request an AI explanation through a pinned free OpenRouter model. The
server reads `OPENROUTER_API_KEY` from Cloud secrets or the local root `.env`; without it, a labeled
local explanation is available. No company data is required.

Stage 5 adds permutation feature importance, residual summaries, and 90%
split-conformal prediction intervals. The default comparison now uses 60%
training, 20% calibration, and 20% test data. Calibration rows are excluded
from fitting and model selection. Disable the Stage 5 checkbox to reproduce
the earlier Stage 4 two-way split.

## Community Cloud deployment

Cloud mode runs the existing backend services inside Streamlit; local mode
retains the separate FastAPI server. Set `APP_EXECUTION_MODE = "direct"` in
Streamlit's private Secrets settings, alongside the OpenRouter key and model.
Deploy `frontend/app.py` from `main` with Python 3.12. Development stays on `dev`;
merge through a pull request to release. See the [deployment walkthrough](docs/deployment.md)
for configuration, testing, model availability, and GitHub website settings.

## Setup (Windows PowerShell)

Use Python 3.12. Open a terminal in this repository:

```powershell
cd C:\Users\fang\demoApps\Soft-Sensor-Virtual-Sensor-App
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

If `python` is not recognized, the Python installation found on this machine
can create the environment instead:

```powershell
& 'C:\Users\fang\anaconda3\envs\codeEnv\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Create the environment only once. If `.venv` is already set up, skip creation.
These commands use its interpreter directly, so activation and changes to
PowerShell's execution policy are unnecessary. In VS Code, select
`.venv\Scripts\python.exe` using **Python: Select Interpreter**.

## Run the app

Keep **two terminals** open in the repository root.

**Terminal 1 — FastAPI backend:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Visit <http://127.0.0.1:8000/health> to see JSON, or
<http://127.0.0.1:8000/docs> to try the endpoint interactively.

**Terminal 2 — Streamlit frontend:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501> and click **Check backend connection**.
You should see a success message and this response:

```json
{
  "status": "ok",
  "service": "sensor-fusion-api",
  "version": "0.1.0",
  "target": "Solid concentration",
  "target_unit": "%"
}
```

Press **Ctrl+C** in each terminal to stop that server. The connection result
reflects the last button click; the app does not continuously monitor the API.

For Stage 2, scroll to **Explore synthetic sensor data** and click **Generate
synthetic data**. Changing chart controls keeps the current data. Click Generate
again to apply new settings. See the [Stage 2 walkthrough](docs/stage-2.md) for
the exact data recipe, assumptions, and exercises.

For Stage 3, scroll to **Train and evaluate Linear Regression** after generating
data, then click **Train Linear Regression**. The default split is 80% training
and 20% test. See the [Stage 3 walkthrough](docs/stage-3.md) for metrics and
leakage prevention. Install updated dependencies and restart servers when
moving from an earlier stage. Select **Linear Regression walkthrough** to use
the original Stage 3 form.

For Stage 4, choose **Compare models**, select candidates, and click **Compare
selected models**. The leaderboard and fold chart show validation performance;
the test plots below belong to the CV-selected model. See the
[Stage 4 walkthrough](docs/stage-4.md) for model presets, cross-validation,
optimizer notices, and exercises. XGBoost requires the updated dependencies.

For Stage 5, leave **Include Stage 5 diagnostics and 90% intervals** checked,
run the comparison, and inspect **Explainability and uncertainty** below the
test plots. Read the [Stage 5 walkthrough](docs/stage-5.md) for calibration,
coverage assumptions, feature-importance interpretation, and exercises.

For Stage 6, train or compare again, then use **Predict and explain**. Predictions
reuse the fitted model; **Explain this model** sends only a compact evidence
summary to `nex-agi/nex-n2.5-mini:free` on click. Configure the key as shown in
[.env.example](.env.example) and read the [Stage 6 walkthrough](docs/stage-6.md).
Runs expire after one hour, at store capacity, or on backend restart. Use one
backend worker for this local demo.

### Optional backend address

If you run FastAPI on a different port, set the matching URL in the frontend
terminal **before** launching Streamlit:

```powershell
$env:BACKEND_URL = 'http://127.0.0.1:8001'
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The frontend reads `BACKEND_URL` from its process environment. Separately, the
backend loads OpenRouter settings from the repository-root `.env` (not
`docs/.env`), with process environment variables taking precedence.

## How the files fit together

| File | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI application, health response schema, and `/health` route. |
| `backend/__init__.py` | Marks the backend as a Python package. |
| `frontend/app.py` | Streamlit layout and connection-check interaction. |
| `frontend/api_client.py` | HTTP request, timeout, response validation, and readable errors. |
| `frontend/__init__.py` | Marks the frontend as a Python package. |
| `requirements.txt` | Runtime libraries, with version bounds. |
| `requirements-dev.txt` | Runtime libraries plus test dependencies. |
| `tests/test_stage1.py` | API contract and UI connection/recovery checks. |
| `docs/stage-1.md` | Guided explanation and a hands-on exercise. |
| `backend/schemas.py` | Dataset request validation and response structure. |
| `backend/services/synthetic_data.py` | Reproducible synthetic process recipe. |
| `frontend/data_explorer.py` | Generation form, session state, charts, and CSV export. |
| `tests/test_stage2.py` | Reproducibility, validation, ranges, and UI lifecycle checks. |
| `docs/stage-2.md` | Data assumptions, equation, and exploration exercises. |
| `backend/services/modeling.py` | Train-only preprocessing, linear fitting, and held-out metrics. |
| `frontend/model_trainer.py` | Training form and evaluation plots. |
| `tests/test_stage3.py` | Metric correctness, leakage checks, validation, and UI lifecycle. |
| `docs/stage-3.md` | Linear Regression, split strategy, metrics, and exercises. |
| `backend/services/model_comparison.py` | Four model presets, shared training folds, and CV selection. |
| `frontend/model_comparison.py` | Candidate selection, leaderboard, fold chart, and selected-model evaluation. |
| `tests/test_stage4.py` | Model-family, fold isolation, warning, and UI lifecycle checks. |
| `docs/stage-4.md` | Cross-validation, hyperparameters, comparison interpretation, and exercises. |
| `backend/services/diagnostics.py` | Conformal calibration, permutation importance, and residual statistics. |
| `frontend/model_diagnostics.py` | Feature importance, residual diagnostics, and interval visualization. |
| `tests/test_stage5.py` | Calibration isolation, finite-sample ranks, coverage, and UI checks. |
| `docs/stage-5.md` | Interpretation, assumptions, and hands-on exercises. |
| `backend/config.py` | Backend-only OpenRouter configuration from root `.env`. |
| `backend/services/run_store.py` | Expiring model retention and compact evidence. |
| `backend/services/inference.py` | New-observation predictions with existing models/intervals. |
| `backend/services/explanations.py` | Free OpenRouter request, cache, and local fallback. |
| `frontend/prediction_panel.py` | Sensor-input form and explanation/evidence display. |
| `tests/test_stage6.py` | Inference, expiry, secret handling, free-only routing, and UI tests. |
| `docs/stage-6.md` | Configuration, request flow, limitations, and exercises. |

The readable dependency ranges allow compatible updates. `requirements-lock.txt`
records the tested Python 3.12 Windows environment, including test tools.
The existing `.gitignore` excludes the virtual environment and local secrets.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests exercise the real FastAPI routes in-process and the Streamlit page using
AppTest. Network responses are mocked in the UI tests so no running server is
required. Try the two-terminal workflow above to verify the actual HTTP link.

## Troubleshooting

- **Cannot reach the backend:** start Terminal 1, then click the check again.
- **Port already in use:** stop the previous server or choose another port.
  If you change the backend port, update `BACKEND_URL` too.
- **Module not found:** run from the repository root and use the `.venv`
  interpreter shown above; install `requirements-dev.txt` in that environment.
- **Unexpected response:** ensure port 8000 belongs to this FastAPI app.
- **Connection timeout:** inspect the backend terminal, then retry the check.

## Learning roadmap

1. App foundation (complete).
2. Reproducible synthetic sensor data and exploration (complete).
3. Linear Regression and evaluation charts (complete).
4. Random Forest, XGBoost, neural network, and cross-validation (complete).
5. Feature importance, residual diagnostics, and prediction intervals (complete).
6. Interactive prediction and evidence-based AI explanations (complete).
7. Portfolio polish, exports, and final validation (complete).

Continue with the [Stage 1 walkthrough](docs/stage-1.md).
Then work through the [Stage 2 walkthrough](docs/stage-2.md).

## Portfolio walkthrough and exports

Use **Save your results ? Prepare evaluation report** after training/comparison
to download a ZIP of synthetic data, metrics, predictions, CV results, and
available diagnostics, or a standalone Markdown summary. No API call is made
when preparing a report.

See the [Stage 7 demo guide](docs/stage-7.md) and
[architecture diagram](docs/architecture.md) for a presentation script, export
contents, reproducibility notes, and local-demo limitations.
