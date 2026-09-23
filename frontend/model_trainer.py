"""Training controls and honest held-out evaluation for the current dataset."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import BackendError, fetch_training


def render_model_trainer(backend_url: str) -> None:
    """Train on submit and preserve results across ordinary Streamlit reruns."""
    st.header("Train and evaluate Linear Regression")
    if "dataset" not in st.session_state:
        st.info("Generate synthetic data above before training a model.")
        return
    with st.form("training_settings"):
        fraction = st.slider("Test set (%)", 10, 40, 20, step=5)
        seed = st.number_input("Split seed", 0, 2**32 - 1, 42, step=1)
        st.caption("The test set is held out from both scaling and model fitting.")
        submitted = st.form_submit_button("Train Linear Regression", type="primary")
    if submitted:
        with st.spinner("Fitting on training rows and evaluating held-out rows…"):
            try:
                result = fetch_training(
                    backend_url, st.session_state["dataset"]["rows"], fraction / 100, int(seed),
                )
            except BackendError as exc:
                st.error(str(exc))
                if "training_result" in st.session_state:
                    st.warning("The previous successful evaluation is still displayed below.")
            else:
                st.session_state["training_result"] = result
    if "training_result" not in st.session_state:
        return

    render_evaluation(st.session_state["training_result"])


def render_evaluation(result: dict) -> None:
    """Reuse held-out metrics and diagnostic plots for any model family."""
    st.caption(
        f"Displayed evaluation: {result['train_count']} training rows / {result['test_count']} test rows · "
        f"test fraction {result['test_fraction']:.0%} · split seed {result['split_seed']}"
    )
    st.caption("Form changes apply on Train. Generating a new dataset clears this evaluation.")
    st.subheader("Held-out test performance")
    scores = result["test_metrics"]
    r2, rmse, mae = st.columns(3)
    r2.metric("R²", "Undefined" if scores["r2"] is None else f"{scores['r2']:.3f}")
    rmse.metric("RMSE (percentage points)", f"{scores['rmse']:.3f}")
    mae.metric("MAE (percentage points)", f"{scores['mae']:.3f}")
    st.caption("R² can be negative; it is undefined for a constant target. Lower RMSE and MAE are better.")
    with st.expander("Compare training, test, and a simple baseline"):
        st.dataframe(pd.DataFrame({
            f"{result['model']} — training": result["train_metrics"],
            f"{result['model']} — test": scores,
            "Training-mean baseline — test": result["baseline_test_metrics"],
        }).T.rename(columns={"r2": "R²", "rmse": "RMSE (pp)", "mae": "MAE (pp)"}))
        st.caption("The baseline predicts the training-set mean for every test row. Training scores describe fit, not unseen-data performance.")

    predictions = pd.DataFrame(result["predictions"])
    labels = {"actual": "Actual concentration (% w/w)", "predicted": "Predicted concentration (% w/w)",
              "residual": "Residual (percentage points)", "row_index": "Original row index (0-based)"}
    left, right = st.columns(2)
    with left:
        actual_plot = px.scatter(predictions, x="actual", y="predicted", labels=labels,
                                 hover_data=["row_index"], title="Actual vs predicted — test rows", opacity=0.6)
        low = float(predictions[["actual", "predicted"]].min().min())
        high = float(predictions[["actual", "predicted"]].max().max())
        actual_plot.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                              line={"dash": "dash", "color": "gray"})
        actual_plot.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(actual_plot, key="actual_predicted")
        st.caption("The dashed line indicates perfect predictions.")
    with right:
        residual_plot = px.scatter(predictions, x="predicted", y="residual", labels=labels,
                                   hover_data=["row_index"], title="Residuals — test rows", opacity=0.6)
        residual_plot.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(residual_plot, key="residuals")
        st.caption("Residual = actual − predicted. Positive values mean underprediction. Look for curves or changing spread.")
    st.caption("Predictions are not clipped to 0–100%, so errors reflect the fitted model's raw output.")
    with st.expander("Inspect held-out predictions"):
        st.dataframe(predictions.rename(columns=labels), hide_index=True)
    st.info("Test scores estimate unseen-data performance. Use training cross-validation for model selection, and avoid choosing seeds for better test scores.")
