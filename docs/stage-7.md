# Stage 7: portfolio finish and reproducible results

The final stage adds a consistent Streamlit theme, a concise workflow guide,
evaluation downloads, architecture documentation, and export validation.
The statistical workflow and working OpenRouter configuration are preserved.

## Run and demonstrate

Use the two-terminal launch commands in the README. Restart Streamlit to pick
up `.streamlit/config.toml`. The app is branded **Sensor Fusion Lab** while
retaining the original project name in the subtitle.

A five-minute portfolio demonstration:

1. Generate 1,000 rows with seed 42 and noise 1. Explain that the data are
   synthetic and solid concentration means mass percent, not volume percent.
2. Show density versus concentration and the flow/pressure correlation.
3. Compare all four models with Stage 5 enabled. Explain the 600/200/200 split,
   CV-based selection, and the distinction between CV and test metrics.
4. Inspect the winner's residuals, feature importance, and observed interval
   coverage. Discuss at least one limitation rather than promising real-world
   accuracy based on synthetic results.
5. Enter new measurements and predict. Request an explanation if you want to
   use the free service; otherwise the modeling workflow works without a key.
6. Scroll to **Save your results**, click **Prepare evaluation report**, and
   download the evaluation bundle or Markdown summary.

## What gets exported

| File | Contents |
| --- | --- |
| `README.md` | Human-readable metrics, settings, warnings, and interpretation limits |
| `results.json` | Full measured evaluation, comparison/diagnostics where available, and software versions |
| `dataset.csv` | All generated rows, including units in column names |
| `test_predictions.csv` | Original zero-based row indices, actuals, predictions, residuals, and optional interval bounds/coverage |
| `cross_validation.csv` | All candidate fold scores, when comparison was used |
| `feature_importance.csv` | Permutation mean/SD values, when diagnostics were enabled |
| `residual_bins.csv` | Concentration-range error summaries, when diagnostics were enabled |

An explanation is included only when its run ID matches the evaluation being
exported. Its source (LLM or template) and model are recorded. Preparation uses
stored results, not unsubmitted form controls, and does not call the backend or
LLM. Downloads are assembled in memory. Preparing another report captures the
latest state. Run IDs, `.env` files, and fitted model binaries are excluded.

The exported dependency versions describe the frontend's environment, which
is also the backend environment in the documented local setup. They are not
remote-backend version attestation. The exact dataset is included because
seeds alone cannot guarantee identical numerical behavior across library
versions or hardware. Presets are documented in Stage 4.

## Learn from the implementation

Read `frontend/exports.py`: `build_report()` is independent of Streamlit and
returns bytes/text, which makes export behavior easy to test. `render_exports()`
handles the UI. The code checks explanation ownership before including it and
removes ephemeral run IDs recursively from nested evaluation objects.

Read [architecture.md](architecture.md) for the complete request flow and the
reasons for separating frontend, backend, model storage, and the LLM service.

## Verification and sharing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests cover data, fitting, CV isolation, calibration, inference, mocked LLM
success/failure paths, UI actions, CSV round-tripping, report completeness,
and exclusion of stale explanations/run tokens. They do not spend API quota.

For the tested Python 3.12 Windows package versions, install
`requirements-lock.txt` in a fresh virtual environment. `requirements-dev.txt`
remains the readable dependency list permitting compatible updates. The lock
snapshot is not a guarantee of support on every operating system.

Share source and synthetic result bundles, never your `.env`. This final stage
does not deploy a public service. The local single-worker limitations described
in the architecture document still apply.

You have completed all seven stages. Useful future extensions include public
datasets, batch/time-aware validation, durable authenticated model storage,
background jobs, and real sensor ingestion with an explicit data contract.
