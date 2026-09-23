# Stage 1: understanding the connection

## What you have built

Two Python processes cooperate to display one web app:

```text
Browser → Streamlit server (port 8501) → FastAPI server (port 8000)
                                             ↓
Browser ← Streamlit renders the result ← JSON health response
```

The frontend organizes the user experience. The backend owns the API and will
later own the data and models. The HTTP client lets them communicate without
the frontend importing backend functions. Because this request comes from
Streamlit's Python server, browser CORS configuration is not needed here.

## Read the code in this order

### 1. `backend/main.py`

`app = FastAPI(...)` creates the application. Uvicorn is the server process that
listens for HTTP requests and passes them to that application.

`@app.get("/health")` associates an HTTP GET request at `/health` with
`health_check()`. A GET request asks for information. The function returns a
`HealthResponse` object; FastAPI converts it to JSON.

`HealthResponse` is a Pydantic model: it describes the fields and their types.
`response_model=HealthResponse` makes this structure part of the API contract
and generated documentation. `Literal["ok"]` means only that exact value is
allowed, while `str` allows any string.

This endpoint only proves the API responds. It says nothing about training
quality, prediction accuracy, or whether a model has been loaded.

### 2. `frontend/api_client.py`

`fetch_health()` builds the URL, sends the request with `requests.get()`, and
checks both the HTTP status and the response content.

These are different kinds of failures:

- A **connection error** means no usable connection was established.
- A **timeout** means the request waited too long.
- An **HTTP error** means a server replied with an error status such as 404.
- **Invalid JSON or an unexpected schema** means the reply is not the expected
  sensor API response, even if its HTTP status was 200.

`BackendError` translates these failures into messages the page can display.
Keeping this logic outside the page makes it easier to test and reuse when we
add training and prediction requests.

### 3. `frontend/app.py`

Streamlit executes the page from top to bottom. Widget interaction triggers
another execution. On the run caused by the connection button, `st.button()`
returns true and the page calls `fetch_health()`.

The request is deliberately not cached: a new click should check the backend's
current state. `st.spinner()` gives feedback while waiting, and `st.json()`
displays the returned fields. A failed request displays a readable error.

`BACKEND_URL` is a configuration value, not a secret. Reading it from the
environment lets you change the backend address without editing source code.

## Hands-on checkpoint

1. Start both servers using the README commands.
2. Open `/docs` on port 8000, expand `GET /health`, click **Try it out**, and
   then **Execute**. Inspect the 200 status and response body.
3. Open Streamlit on port 8501 and check the connection. Compare the JSON.
4. Stop only FastAPI with Ctrl+C and click the connection button again.
   The page should display a connection error and remain usable.
5. Restart FastAPI and click again. The success message should return.

Try explaining these questions in your own words:

- Why do we run two servers on different ports?
- Which file sends the HTTP request, and which function handles it?
- Why can the Streamlit page open even when FastAPI is stopped?
- Why does a healthy API not mean that a prediction model is ready?

## Next checkpoint

Stage 2 will generate reproducible synthetic measurements and solid
concentration (%) values. Before generating data we will define the percentage
basis, operating ranges, and relationships, clearly labeling them as demo
assumptions rather than validated process physics.

## References

- [FastAPI: first steps and interactive documentation](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [Streamlit: execution model](https://docs.streamlit.io/develop/concepts/architecture)
- [Streamlit: AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
