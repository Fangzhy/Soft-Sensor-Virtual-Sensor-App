"""Stage 4 comparison UI: choose candidates by CV, then inspect the winner."""

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import BackendError, MODEL_NAMES, fetch_comparison
from frontend.model_trainer import render_evaluation
from frontend.model_diagnostics import render_diagnostics
from frontend.prediction_panel import render_prediction_panel


def render_model_comparison(backend_url: str) -> None:
    """Submit fixed model presets and retain one dataset-specific comparison."""
    st.header("Compare prediction models")
    if "dataset" not in st.session_state:
        st.info("Generate synthetic data above before comparing models.")
        return
    with st.form("comparison_settings"):
        models = st.multiselect("Models to compare", MODEL_NAMES, default=MODEL_NAMES)
        fraction = st.slider("Held-out test set (%)", 10, 40, 20, step=5)
        seed = st.number_input("Comparison split seed", 0, 2**32 - 1, 42, step=1)
        diagnostics = st.checkbox("Include Stage 5 diagnostics and 90% intervals", value=True)
        st.caption("With Stage 5 enabled, another 20% of all rows is reserved for calibration. Defaults: 60% training, 20% calibration, 20% test. CV uses only the training portion.")
        st.caption("Five identical training-only folds per model. The lowest mean CV RMSE selects the model for test evaluation.")
        submitted = st.form_submit_button("Compare selected models", type="primary")
    with st.expander("Model presets and hyperparameters"):
        st.markdown(
            "- **Linear Regression:** scaled inputs and ordinary least squares.\n"
            "- **Random Forest:** 100 trees, maximum depth 10, minimum leaf size 2.\n"
            "- **XGBoost:** 150 trees, maximum depth 3, learning rate 0.05.\n"
            "- **Neural Network:** scaled inputs and target, hidden layers 32 and 16, "
            "tanh activation, L-BFGS solver, regularization alpha 0.1, tolerance 0.001, maximum 800 iterations."
        )
        st.caption("These are fixed teaching presets, not tuned winners. A comparison fits each model five times, then refits the selected model.")
    if submitted:
        if not models:
            st.error("Select at least one model. Any previous comparison remains below.")
        else:
            with st.spinner("Comparing models across five folds; this may take a minute…"):
                try:
                    result = fetch_comparison(
                        backend_url, st.session_state["dataset"]["rows"], fraction / 100, int(seed), models, diagnostics,
                    )
                except BackendError as exc:
                    st.error(str(exc))
                    if "comparison_result" in st.session_state:
                        st.warning("The previous successful comparison remains displayed below.")
                else:
                    st.session_state["comparison_result"] = result
    if "comparison_result" not in st.session_state:
        return
    result = st.session_state["comparison_result"]
    evaluation = result["evaluation"]
    analysis = result.get("diagnostics")
    st.caption(f"Displayed split: {evaluation['train_count']} training / {analysis['calibration_count'] if analysis else 0} calibration / {evaluation['test_count']} test rows.")
    st.caption(f"Displayed comparison: split seed {evaluation['split_seed']}, {evaluation['test_fraction']:.0%} held out. Form changes apply only on Compare.")
    table = pd.DataFrame([{
        "Model": item["model"], "CV RMSE mean (pp)": item["mean"]["rmse"],
        "CV RMSE SD (pp)": item["std"]["rmse"], "CV MAE mean (pp)": item["mean"]["mae"],
        "CV R² mean": item["mean"]["r2"], "Optimizer notices": len(item["warnings"]),
    } for item in result["results"]]).sort_values("CV RMSE mean (pp)", kind="stable")
    st.subheader("Training cross-validation leaderboard")
    st.dataframe(table, hide_index=True)
    st.caption("SD describes variability across five validation folds; it is not a prediction interval. R² is undefined if any fold has a constant target.")
    fold_rows = [{"Model": item["model"], "Fold": i, "RMSE (pp)": score["rmse"],
                  "MAE (pp)": score["mae"], "R²": score["r2"]}
                 for item in result["results"] for i, score in enumerate(item["folds"], 1)]
    st.plotly_chart(px.line(pd.DataFrame(fold_rows), x="Fold", y="RMSE (pp)", color="Model",
                           markers=True, title="Validation RMSE across shared folds"), key="cv_folds")
    with st.expander("Inspect all fold scores"):
        st.dataframe(pd.DataFrame(fold_rows), hide_index=True)
    for item in result["results"]:
        if item["warnings"]:
            st.warning(f"{item['model']}: " + " ".join(item["warnings"]))
    for notice in result["warnings"]:
        st.warning(notice)
    st.success(f"Selected by mean CV RMSE: {result['selected_model']}")
    st.caption("The selected model was refitted on all training rows. Only its held-out evaluation is shown below.")
    render_evaluation(evaluation)
    if analysis is not None:
        render_diagnostics(analysis, evaluation)
    render_prediction_panel(backend_url, evaluation)
    from frontend.exports import render_exports
    render_exports(st.session_state["dataset"], evaluation, result)
