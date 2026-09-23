# SensorData-FusionPredtionExplaination

A step-by-step learning project using **FastAPI** for the backend and
**Streamlit** for the frontend. The prediction target is **solid concentration
(%)**, using temperature, density, flow rate, pressure, and agitation speed.

## Current milestone: Stage 1

The app has a health endpoint, interactive API documentation, and a Streamlit
page that checks the backend connection and displays its JSON response.
Data generation, model training, predictions, and AI explanations are future
stages. No company data or API key is required.

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

### Optional backend address

If you run FastAPI on a different port, set the matching URL in the frontend
terminal **before** launching Streamlit:

```powershell
$env:BACKEND_URL = 'http://127.0.0.1:8001'
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

This stage reads the process environment directly; it does not load `.env` files.

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

The dependency ranges allow compatible updates; they are not an exact lockfile.
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

1. App foundation (this milestone).
2. Reproducible synthetic sensor data and exploration.
3. Linear Regression and evaluation charts.
4. Random Forest, XGBoost, neural network, and cross-validation.
5. Feature importance, residual diagnostics, and prediction intervals.
6. Interactive prediction and evidence-based AI explanations.
7. Portfolio polish, exports, and final validation.

Continue with the [Stage 1 walkthrough](docs/stage-1.md).
