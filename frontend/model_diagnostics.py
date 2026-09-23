"""Stage 5 visual explanations for a frozen model, without automatic claims."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.data_explorer import LABELS


def render_diagnostics(data: dict, evaluation: dict) -> None:
    """Display uncertainty, feature reliance, and measured residual patterns."""
    st.header("Explainability and uncertainty")
    importance_tab, residual_tab, interval_tab = st.tabs([
        "Feature importance", "Residual diagnostics", "90% prediction intervals",
    ])
    predictions = pd.DataFrame(evaluation["predictions"])
    with importance_tab:
        importance = pd.DataFrame(data["importance"]).sort_values("mean")
        importance["Sensor"] = importance["feature"].map(LABELS)
        st.plotly_chart(px.bar(
            importance, x="mean", y="Sensor", error_x="std", orientation="h",
            labels={"mean": "Increase in test RMSE (percentage points)", "std": "Shuffle SD"},
            title="Permutation importance — 10 shuffles per sensor",
        ), key="permutation_importance")
        st.caption("Larger increases indicate greater model reliance. Error bars show shuffle SD, not confidence intervals. Negative values are retained.")
        st.info("Importance describes this model on this test set, not causation. Correlated flow and pressure can share information and distort individual rankings. Do not use these test results to tune the model.")
    with residual_tab:
        bias, spread = st.columns(2)
        bias.metric("Mean residual (pp)", f"{data['mean_residual']:.3f}")
        spread.metric("Residual SD (pp)", f"{data['residual_std']:.3f}")
        st.caption("Positive mean residual indicates underprediction; negative indicates overprediction.")
        st.plotly_chart(px.histogram(predictions, x="residual", nbins=30,
                                     labels={"residual": "Residual (percentage points)"},
                                     title="Test residual distribution"), key="residual_histogram")
        st.plotly_chart(px.scatter(predictions, x="actual", y="residual", opacity=0.6,
                                   labels={"actual": "Actual concentration (% w/w)", "residual": "Residual (pp)"},
                                   title="Residuals versus actual concentration"), key="residual_actual")
        st.caption("Residuals contain the actual target mathematically; patterns in this plot alone do not prove misspecification. Also inspect residuals versus predictions above.")
        st.subheader("Errors by actual concentration range")
        st.dataframe(pd.DataFrame(data["residual_bins"]).rename(columns={
            "lower": "Range lower (% w/w)", "upper": "Range upper (% w/w)",
            "count": "Test rows", "mean_residual": "Mean residual (pp)", "mae": "MAE (pp)", "rmse": "RMSE (pp)",
        }), hide_index=True)
        st.caption("Ranges use actual-target tertiles; repeated boundaries are merged. Upper bounds are exclusive except in the final range. Small groups give noisy summaries.")
        corr = data["abs_residual_prediction_correlation"]
        st.write("Correlation of absolute residual with predicted concentration: " + ("undefined (constant values)." if corr is None else f"{corr:.3f}."))
        st.caption("This is a descriptive linear association, not a statistical test of changing variance or an automatic diagnosis.")
    with interval_tab:
        nominal, coverage, width = st.columns(3)
        nominal.metric("Nominal coverage", "90%")
        coverage.metric("Observed test coverage", f"{data['empirical_coverage']:.1%}")
        width.metric("Mean interval width (pp)", f"{data['mean_width']:.3f}")
        st.caption(f"Calibration: {data['calibration_count']} rows · absolute-error rank {data['quantile_rank']} · half-width {data['half_width']:.3f} pp.")
        combined = predictions.merge(pd.DataFrame(data["intervals"]), on="row_index", validate="one_to_one")
        ordered = combined.sort_values("predicted").reset_index(drop=True)
        position = list(range(len(ordered)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=position, y=ordered["upper"], mode="lines", line={"width": 0}, name="Upper bound"))
        fig.add_trace(go.Scatter(x=position, y=ordered["lower"], mode="lines", line={"width": 0}, fill="tonexty", name="90% interval"))
        fig.add_trace(go.Scatter(x=position, y=ordered["predicted"], mode="lines", name="Prediction"))
        fig.add_trace(go.Scatter(x=position, y=ordered["actual"], mode="markers", name="Actual",
                                customdata=ordered["row_index"], hovertemplate="Original row %{customdata}<br>Actual %{y:.3f}<extra></extra>",
                                marker={"size": 5}))
        fig.update_layout(xaxis_title="Test observations sorted by prediction (not time)", yaxis_title="Concentration (% w/w)")
        st.plotly_chart(fig, key="conformal_intervals")
        st.info("90% is a marginal coverage target under exchangeability with calibration data, not a guarantee for each observation or concentration range. Finite test coverage can differ from 90%. Distribution shifts can invalidate coverage.")
        st.caption("These predict individual outcomes, not the mean response. This basic method gives constant-width, unclipped intervals; bounds can extend outside 0–100%.")
        with st.expander("Inspect every interval"):
            st.dataframe(combined, hide_index=True)
