"""Stage 2 controls and plots; datasets survive widget reruns in session state."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import BackendError, DATA_COLUMNS, fetch_dataset

LABELS = dict(zip(DATA_COLUMNS, [
    "Temperature (°C)", "Density (kg/m³)", "Flow rate (L/min)",
    "Pressure (bar)", "Agitation speed (rpm)", "Solid concentration (% w/w)",
]))
TARGET = "solid_concentration_pct"


def render_data_explorer(backend_url: str) -> None:
    """Generate on submit, then explore the exact stored dataset on each rerun."""
    st.header("Explore synthetic sensor data")
    st.info("Synthetic teaching data · Independent snapshots · Solid concentration is mass percent (w/w).")
    with st.form("generation_settings"):
        count, seed_col, noise = st.columns(3)
        n_samples = count.number_input("Number of rows", 100, 5000, 1000, step=100)
        seed = seed_col.number_input("Random seed", 0, 2**32 - 1, 42, step=1)
        noise_std = noise.slider("Target noise (percentage points)", 0.0, 5.0, 1.0, step=0.25)
        st.caption("The seed makes results repeatable. Noise is the standard deviation of simulated target error.")
        submitted = st.form_submit_button("Generate synthetic data", type="primary")

    if submitted:
        with st.spinner("Generating sensor observations…"):
            try:
                result = fetch_dataset(backend_url, int(n_samples), int(seed), float(noise_std))
            except BackendError as exc:
                st.error(str(exc))
                if "dataset" in st.session_state:
                    st.warning("Generation failed. The previous dataset remains displayed below.")
            else:
                # Replace only after success. Each browser session owns its dataset.
                st.session_state["dataset"] = result
                # Evaluation belongs to one dataset; never show stale model scores.
                st.session_state.pop("training_result", None)
                st.session_state.pop("comparison_result", None)
                for key in ("interactive_run", "new_prediction", "model_explanation"):
                    st.session_state.pop(key, None)

    if "dataset" not in st.session_state:
        st.caption("Generate a dataset to view its measurements, distributions, and correlations.")
        return

    dataset = st.session_state["dataset"]
    settings = dataset["settings"]
    df = pd.DataFrame(dataset["rows"], columns=DATA_COLUMNS)
    st.caption(
        f"Displayed dataset: {len(df):,} rows · seed {settings['seed']} · "
        f"noise SD {settings['noise_std']} percentage points · generator {dataset['generator_version']}"
    )
    st.caption("Changing form settings takes effect only after Generate. Chart controls do not regenerate data.")
    if dataset["clipped_target_count"]:
        st.warning(f"{dataset['clipped_target_count']} target values were clipped to the physical 0–100% range.")

    preview, distributions, relationships = st.tabs(["Preview & summary", "Distributions", "Relationships"])
    with preview:
        st.subheader("First 50 observations")
        st.dataframe(df.head(50).rename(columns=LABELS), hide_index=True)
        st.subheader("Summary of all observations")
        st.dataframe(df.rename(columns=LABELS).describe().T.round(3))
        # Export the full stored dataset, not just the preview or a fresh draw.
        st.download_button(
            "Download all rows as CSV", data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"synthetic_sensors_n{len(df)}_seed{settings['seed']}_noise{settings['noise_std']}.csv",
            mime="text/csv",
        )
        st.caption("CSV column names include units; solid_concentration_pct is mass percent. No row index is exported.")
    with distributions:
        variable = st.selectbox("Distribution variable", DATA_COLUMNS, format_func=LABELS.get, index=5)
        st.plotly_chart(px.histogram(df, x=variable, nbins=35, labels=LABELS), key="distribution")
        st.caption("A histogram counts observations in value ranges. Counts use the full dataset.")
    with relationships:
        feature = st.selectbox("Sensor versus concentration", DATA_COLUMNS[:-1], format_func=LABELS.get, index=1)
        st.plotly_chart(px.scatter(df, x=feature, y=TARGET, labels=LABELS, opacity=0.5), key="scatter")
        corr = df.rename(columns=LABELS).corr(method="pearson")
        st.plotly_chart(px.imshow(
            corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
            text_auto=".2f", aspect="auto", title="Pearson correlation (all rows)",
        ), key="correlations")
        st.caption("Correlation measures linear association, not causation or model feature importance. Nonlinear relationships can have low correlation.")
    with st.expander("How the synthetic process works"):
        st.write(
            "Density has the largest direct coefficient in our invented target recipe. "
            "Temperature, flow, pressure, and agitation also contribute, with nonlinear "
            "temperature/agitation terms and a density–temperature interaction. Pressure "
            "partly follows flow. Gaussian noise represents unobserved target variability."
        )
        st.caption("These are demo assumptions, not validated process physics. See docs/stage-2.md for ranges and the exact equation.")
