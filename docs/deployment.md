# Deploy Sensor Fusion Lab on Streamlit Community Cloud

## What changes in the cloud?

Locally, the default `APP_EXECUTION_MODE=http` uses Streamlit -> FastAPI ->
backend services. With `APP_EXECUTION_MODE=direct`, Streamlit calls those same
Python services inside its server process. Community Cloud does not need a
separate FastAPI server, and this deployment does not expose API endpoints.
Model presets, input validation, evaluation, and explanation logic are shared.

`frontend/execution.py` selects the mode. `frontend/direct_backend.py` validates
requests with the existing Pydantic schemas and returns serialized results to
the existing frontend validators. Service imports happen only in direct mode.
`frontend/app.py` loads Cloud secrets before choosing the execution mode.

## 1. Review and publish through GitHub

Work on `dev`, review the changes, commit and push them yourself, then open a
pull request from `dev` into `main`. Merge when ready. The deployed app follows
`main`, so pushing only to `dev` does not release these changes.

Include the new Python files, documentation, `.streamlit/config.toml`, and
`.streamlit/secrets.example.toml`. Do not commit `.env` or
`.streamlit/secrets.toml`; both are ignored. The example contains no credential.

Run the checks before merging:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

## 2. Configure the deployment

| Setting | Value |
| --- | --- |
| Repository | `Fangzhy/Soft-Sensor-Virtual-Sensor-App` |
| Branch | `main` |
| Main file path | `frontend/app.py` |
| App URL | `sensor-fusion-lab` |
| Advanced settings: Python | `3.12` |

Cloud installs the root `requirements.txt`. `requirements-lock.txt` is a local
environment snapshot, not the cloud installation file. Configuration lives in
the root `.streamlit/config.toml`. The app entry point adds the repository root
to Python's import path so sibling backend/frontend packages can be imported.

See Streamlit's [file organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
and [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## 3. Configure OpenRouter privately in Streamlit

Before deployment, click **Advanced settings** above **Deploy**, then use the
**Secrets** field. For an existing app, open your workspace at
https://share.streamlit.io, choose the app's **three-dot menu -> Settings -> Secrets**.
These are Streamlit settings, not GitHub Actions secrets.

Paste these values as TOML, without a section header:

```toml
APP_EXECUTION_MODE = "direct"
OPENROUTER_API_KEY = "YOUR_ACTUAL_KEY"
OPENROUTER_MODEL = "nex-agi/nex-n2.5-mini:free"
```

The mode and model name are configuration, not credentials. The key must stay
private. Root-level Streamlit secrets become environment variables, which the
existing backend configuration reads before considering the local root `.env`.
Your local `.env` is not uploaded or required on Cloud. Do not paste your key
into source code, README, screenshots, or GitHub's About field.

On **Explain this model**, server-side code sends a compact evaluation summary
to `https://openrouter.ai/api/v1/chat/completions` with the key in its Authorization
header. Raw training rows and the credential are not sent to the browser.
The model allowlist and zero-price provider filters remain active. There is no
automatic paid fallback. Failed calls yield a labeled local template.

**Model availability checked September 23, 2026:** OpenRouter's
[current model page](https://openrouter.ai/nex-agi/nex-n2.5-mini:free) announces
retirement on September 25, 2026. Keep the working setting while available.
The existing supported alternative is `openrouter/free`, though its selected
model varies and may not produce a valid explanation. For a different pinned
free model, verify structured-output support and zero pricing, update the
allowlist in `backend/services/explanations.py`, test it, then update the Cloud
secret. Merely entering an arbitrary model name will not bypass the allowlist.

See [Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
and [root-level environment variables](https://docs.streamlit.io/develop/concepts/connections/secrets-management).

## 4. Deploy and check the app

After merging into `main`, deploy or wait for the existing deployment to update.
Open https://sensor-fusion-lab.streamlit.app and check:

1. The sidebar says Cloud demo and links to the GitHub repository.
2. Generate 1,000 rows and compare the four models with diagnostics enabled.
3. Inspect metrics, residuals, feature importance, and intervals.
4. Predict a new observation and request an explanation. Check whether it is
   labeled OpenRouter or a local template; the notice explains a fallback.
5. Prepare and download the evaluation report.
6. Open another browser session and confirm it starts without your dataset.

To set GitHub's About website manually: repository home -> About gear icon ->
Website -> `https://sensor-fusion-lab.streamlit.app` -> Save changes. The README
also links to the demo. The README link is prepared before release; it becomes
usable once your deployment succeeds and the subdomain is assigned.

## Try cloud mode locally

From the repository root, without starting FastAPI:

```powershell
$env:APP_EXECUTION_MODE = "direct"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The existing root `.env` supplies OpenRouter configuration locally. To return
to the original two-process setup, stop Streamlit and run:

```powershell
$env:APP_EXECUTION_MODE = "http"
```

Then start FastAPI and Streamlit using the README commands. If you created a
local `.streamlit/secrets.toml`, update its mode too because Streamlit loads
root-level secrets into the environment.

## Shared resources and troubleshooting

- A browser session owns its dataset and permitted model-run IDs. Model storage
  is bounded to 16 runs across visitors, with a one-hour expiry. Restart, session
  loss, expiration, or eviction requires retraining. No durable accounts exist.
- One training/comparison job runs at a time. Other attempts show a retry
  message. Requests remain bounded to 100–5,000 rows and the four fixed presets.
- One uncached explanation runs at a time, with at least 30 seconds between
  attempts across visitors. Cached successful results remain available. Busy
  attempts show a local template and make no provider call. This in-process
  throttle resets on restart; it is not a durable quota or authentication system.
- If the app asks you to start FastAPI, check `APP_EXECUTION_MODE = "direct"`
  in Cloud secrets and reboot after saving. `BACKEND_URL` is unused in direct mode.
- Missing key: check the exact `OPENROUTER_API_KEY` spelling. Provider 404:
  check retirement/availability and provider privacy settings. Provider 429:
  wait for your account's quota to recover. All visitors share your account.
- Read build/runtime logs through Manage app. Start with 1,000 rows if training
  is slow. Cloud resource limits differ from your local machine; the tests here
  cannot guarantee the hosted environment's available memory or response time.
