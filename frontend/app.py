"""Streamlit entry point: connect, explore data, and choose a modeling workflow.

Run from the project root with: python -m streamlit run frontend/app.py
Streamlit re-executes this script when a user interacts with a widget.
"""

import os

import streamlit as st

from frontend.api_client import BackendError, fetch_health
from frontend.data_explorer import render_data_explorer
from frontend.model_trainer import render_model_trainer
from frontend.model_comparison import render_model_comparison


def main() -> None:
    """Render the learning dashboard and handle an explicit connection check."""
    st.set_page_config(page_title="Sensor Fusion | Stage 6", page_icon="🔬", layout="wide")

    # Streamlit needs only the backend address; LLM credentials stay in FastAPI.
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    with st.sidebar:
        st.title("Sensor Fusion")
        st.caption("Stage 6 of 7 · Predict and explain")
        st.markdown("**Prediction target**\n\nSolid concentration (%)")
        st.divider()
        st.caption("Backend URL")
        st.code(backend_url, language=None)
        st.caption("Set BACKEND_URL before starting Streamlit to change this address.")

    st.title("Sensor data → solid concentration")
    st.caption("SensorData-FusionPredtionExplaination")
    st.write(
        "Learn how process measurements can be combined to estimate solid "
        "concentration. Generate synthetic observations and explore their relationships."
    )

    sensors, target = st.columns([2, 1])
    with sensors:
        st.subheader("Sensor inputs")
        st.table([
            {"Measurement": "Temperature", "Unit": "°C"},
            {"Measurement": "Density", "Unit": "kg/m³"},
            {"Measurement": "Flow rate", "Unit": "L/min"},
            {"Measurement": "Pressure", "Unit": "bar"},
            {"Measurement": "Agitation speed", "Unit": "rpm"},
        ])
    with target:
        st.subheader("Prediction target")
        st.markdown("### Solid concentration (%)")
        st.info("Target: mass percent (w/w). Compare four model families using training-only cross-validation below.")

    st.divider()
    st.subheader("Check the backend connection")
    st.write("Start FastAPI, then use the button below to request its health status.")

    # Only call the API on a click. Merely opening the page does not make a request.
    # Results describe this check, rather than pretending to monitor continuously.
    if st.button("Check backend connection", type="primary"):
        with st.spinner("Contacting FastAPI…"):
            try:
                health = fetch_health(backend_url)
            except BackendError as exc:
                st.error(str(exc))
                st.caption("See README.md for the command to start the backend.")
            else:
                st.success("Connection check passed: FastAPI is responding.")
                st.caption(f"API version {health['version']} · Target: {health['target']} (%)")
                with st.expander("Inspect the JSON response", expanded=True):
                    st.json(health)

    with st.expander("What happens when I click?"):
        st.markdown(
            "1. Streamlit runs this page again and detects the button click.\n"
            "2. The Python HTTP client sends `GET /health` to FastAPI.\n"
            "3. FastAPI returns a JSON response describing the service.\n"
            "4. Streamlit displays the result or a helpful connection error."
        )
        st.caption("Your browser talks to Streamlit; the Streamlit server talks to FastAPI.")

    st.divider()
    render_data_explorer(backend_url)
    st.divider()
    workflow = st.radio("Modeling workflow", ["Compare models", "Linear Regression walkthrough"])
    if workflow == "Compare models":
        render_model_comparison(backend_url)
    else:
        render_model_trainer(backend_url)


if __name__ == "__main__":
    main()
