"""Stage 6: request predictions and evidence-based explanations on demand."""

import streamlit as st

from frontend.api_client import BackendError, fetch_run_action

SENSORS = [
    ("temperature_c", "Temperature (°C)", 20.0, 90.0, 55.0),
    ("density_kg_m3", "Density (kg/m³)", 1000.0, 1300.0, 1150.0),
    ("flow_rate_l_min", "Flow rate (L/min)", 10.0, 100.0, 55.0),
    ("pressure_bar", "Pressure (bar)", 1.0, 6.0, 3.5),
    ("agitation_rpm", "Agitation speed (rpm)", 100.0, 800.0, 450.0),
]


def render_prediction_panel(backend_url: str, evaluation: dict) -> None:
    """Display only results tied to the active model run, never call on reruns."""
    run_id = evaluation.get("run_id")
    st.header("Predict and explain")
    if not run_id:
        st.info("Train or compare again to enable predictions and explanations for this run.")
        return
    if st.session_state.get("interactive_run") != run_id:
        st.session_state["interactive_run"] = run_id
        st.session_state.pop("new_prediction", None)
        st.session_state.pop("model_explanation", None)
    st.caption(f"Using retained model: {evaluation['model']}. Runs expire after one hour, server restart, or eviction when the demo is busy.")
    with st.form("new_sensor_inputs"):
        values = {}
        columns = st.columns(2)
        for i, (name, label, low, high, initial) in enumerate(SENSORS):
            values[name] = columns[i % 2].number_input(label, min_value=low, max_value=high, value=initial, key=f"prediction_{name}")
        submitted = st.form_submit_button("Predict solid concentration", type="primary")
    if submitted:
        # Clear old output on failure so it cannot look like a new prediction.
        st.session_state.pop("new_prediction", None)
        try:
            st.session_state["new_prediction"] = fetch_run_action(backend_url, "predict", run_id, values)
        except BackendError as exc:
            st.error(str(exc))
    if "new_prediction" in st.session_state:
        prediction = st.session_state["new_prediction"]
        st.metric("Predicted solid concentration (% w/w)", f"{prediction['prediction']:.3f}")
        if prediction["lower"] is not None:
            st.write(f"90% prediction interval: {prediction['lower']:.3f} to {prediction['upper']:.3f}% w/w.")
            st.caption("Reuses the existing calibration width. Marginal coverage assumes exchangeability; this is not a 90% guarantee for this particular input.")
        else:
            st.info("No calibrated interval for this run. Enable Stage 5 diagnostics and compare again to obtain one.")
        for notice in prediction["warnings"]:
            st.warning(notice)
        with st.expander("Inputs used for the displayed prediction"):
            st.json(prediction["inputs"])
        st.caption("Inputs within individual training ranges may still form an unfamiliar combination. The fitted regression model produces predictions; the LLM does not.")

    st.subheader("Explain this model")
    st.caption("Sends a compact results summary to OpenRouter's free service on click. No raw dataset or new sensor inputs are sent. Successful explanations are reused for this run.")
    if st.button("Explain this model", key="explain_model"):
        existing = st.session_state.get("model_explanation")
        if not existing or existing["source"] != "openrouter":
            with st.spinner("Requesting a free-model explanation…"):
                try:
                    st.session_state["model_explanation"] = fetch_run_action(backend_url, "explain", run_id)
                except BackendError as exc:
                    st.session_state.pop("model_explanation", None)
                    st.error(str(exc))
    if "model_explanation" in st.session_state:
        explanation = st.session_state["model_explanation"]
        if explanation["source"] == "template":
            st.warning(explanation["notice"] or "Local template explanation — not LLM-generated.")
            st.caption("Local template explanation — not LLM-generated. Click again to retry the free service.")
        else:
            st.caption(f"AI explanation · OpenRouter model: {explanation['model']}")
        # Render provider output as text, not executable HTML or clickable links.
        st.text(explanation["text"])
        st.caption("Compare this explanation with the measured evidence; LLM text can contain mistakes.")
        with st.expander("Evidence supplied to the explanation service"):
            st.json(explanation["evidence"])
