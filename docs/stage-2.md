# Stage 2: synthetic data and exploration

This stage adds `POST /data/generate` and a Streamlit explorer. All observations
are synthetic, independent operating snapshots. There are no timestamps,
production records, trained models, or proprietary data.

## Definition and operating ranges

Solid concentration is **mass percent (w/w)**: 100 × solid mass / total mixture
mass. For example, 35% means 35 kg of solids per 100 kg of mixture.

| Column | Unit | Generation rule |
| --- | --- | --- |
| `temperature_c` | °C | Uniform between 20 and 90 |
| `density_kg_m3` | kg/m³ | Uniform between 1000 and 1300 |
| `flow_rate_l_min` | L/min | Uniform between 10 and 100 |
| `pressure_bar` | bar | 1 + 0.04 × (flow − 10) + Gaussian noise with SD 0.4, clipped to 1–6 |
| `agitation_rpm` | rpm | Uniform between 100 and 800 |
| `solid_concentration_pct` | % w/w | Recipe below plus target noise, clipped to 0–100 |

These ranges and relationships are invented teaching assumptions, not a
validated mass-balance or process model. Clipping pressure can create a pileup
at its lower bound. Pressure intentionally shares information with flow.

## The target recipe

First center and scale the measurements:

```text
d = (density − 1150) / 150
t = (temperature − 55) / 35
f = (flow − 55) / 45
p = (pressure − 3.5) / 2.5
a = (agitation − 450) / 350

clean concentration = 35 + 14d + 5t − 3f + 2p + 1.5a
                         + 4dt + 3t² − 2a²
observed concentration = clip(clean concentration + noise_std × Z, 0, 100)
Z follows a standard normal distribution.
```

The coefficients are in percentage points because the inputs are scaled.
Density has the largest direct coefficient. The `dt` interaction means that
the effect of density varies with temperature. The squared terms introduce
curvature for future nonlinear models to learn. These coefficients are not
model feature-importance scores.

`noise_std=1` means a standard deviation of **one percentage point**, not one
percent of the current concentration. Noise represents unobserved variability
in the target; the app does not separately simulate sensor measurement error.
The response reports how many targets required clipping. Values are rounded
to four decimal places for export. The hidden clean target is never returned
as a feature, which avoids handing a future model the answer.

## Read the code in this order

1. `backend/schemas.py`: Pydantic bounds the row count (100–5000), seed
   (0–4294967295), and noise (0–5). Bad input receives an HTTP 422 response.
2. `backend/services/synthetic_data.py`: a local NumPy random generator makes
   observations and wraps them in a validated response. The same settings and
   software environment produce the same data. Changing noise alone preserves
   sensor readings; changing row count need not preserve a dataset prefix.
3. `backend/main.py`: the POST route accepts JSON settings and calls the service.
4. `frontend/api_client.py`: sends settings using HTTP and checks the response.
5. `frontend/data_explorer.py`: uses a form to batch settings, stores a successful
   response in `st.session_state`, and renders the tables and Plotly charts.

Session state survives widget reruns within the current browser session. It is
not durable storage: reloading or restarting can lose the dataset. A failed
generation leaves the previous data visible with a warning. The caption always
identifies the settings of the displayed data, even if form inputs change.

## Try it

Install the additional dependencies with your virtual environment active:

```powershell
python -m pip install -r requirements-dev.txt
```

Start the two servers using the README commands. Restart Streamlit if needed.

1. Generate 1,000 rows with seed 42 and noise 1. Inspect the first 50 rows and
   the statistics calculated from **all** rows.
2. Download the CSV. It includes all six columns and all rows, without an index.
   The filename records count, seed, and noise; the generator version is shown
   on the page. This export is synthetic even though the values look plausible.
3. Generate again with identical settings: the data should be identical.
4. Increase noise to 5 with the same row count and seed. Sensors stay the same;
   the scatter around the concentration relationship becomes wider.
5. Switch the scatter plot between density and temperature. Look for stronger
   linear association versus curvature. The other inputs also contribute to
   vertical spread, so even noise-free points need not lie on one curve.
6. Inspect the correlation heatmap. Its scale runs from −1 to +1; zero means
   little linear association, not necessarily independence. Correlation does
   not establish causation or measure trained-model importance.
7. Stop FastAPI and try generating again. Your previous data stays available
   for exploration and download. Restart FastAPI to generate a new dataset.

You can also try `POST /data/generate` in <http://127.0.0.1:8000/docs>:

```json
{"n_samples": 1000, "seed": 42, "noise_std": 1.0}
```

## Check your understanding

- What does a random seed control, and why record it?
- Why is noise measured in percentage points?
- Why do we store the response rather than regenerate whenever a chart changes?
- Why might a curved relationship have weak Pearson correlation?

Stage 3 will split these observations into training and evaluation sets and fit
Linear Regression. Later modeling stages must keep evaluation data separate
from fitting and model selection.

References: [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/)
and [Streamlit Plotly charts](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).
