"""Create portable result artifacts from the displayed run, without new API calls."""

import io
import json
from datetime import datetime, timezone
from importlib.metadata import version
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import streamlit as st


def remove_run_ids(value):
    """Exclude ephemeral backend access tokens from shareable artifacts."""
    if isinstance(value, dict):
        return {key: remove_run_ids(item) for key, item in value.items() if key != "run_id"}
    if isinstance(value, list):
        return [remove_run_ids(item) for item in value]
    return value


def build_report(dataset: dict, evaluation: dict, comparison: dict | None = None,
                 explanation: dict | None = None) -> tuple[bytes, str]:
    """Package data, measured results, and a human-readable summary in memory.

    Use the stored settings, never the unsent form controls. Explanations are
    included only if they belong to this exact run. No environment files,
    credentials, serialized models, or backend run IDs enter the archive.
    """
    diagnostics = comparison.get("diagnostics") if comparison else None
    report = {
        "report_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": {key: dataset[key] for key in ("source", "generator_version", "target_basis", "settings")},
        "evaluation": remove_run_ids(evaluation),
        "comparison": remove_run_ids(comparison) if comparison else None,
        "export_environment_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "streamlit")},
    }
    scores = evaluation["test_metrics"]
    r2 = "undefined" if scores["r2"] is None else f"{scores['r2']:.4f}"
    summary = (
        "# Sensor Fusion — evaluation report\n\n"
        "Synthetic teaching data; target: solid concentration (% w/w).\n\n"
        f"Model: **{evaluation['model']}**\n\n"
        f"Generation settings: `{json.dumps(dataset['settings'])}`\n\n"
        f"Split seed: {evaluation['split_seed']}; training: {evaluation['train_count']}; "
        f"calibration: {diagnostics['calibration_count'] if diagnostics else 0}; test: {evaluation['test_count']}.\n\n"
        "| Test metric | Value |\n| --- | --- |\n"
        f"| R² | {r2} |\n| RMSE (percentage points) | {scores['rmse']:.4f} |\n"
        f"| MAE (percentage points) | {scores['mae']:.4f} |\n\n"
    )
    if comparison:
        summary += "Selection: lowest mean training-only five-fold CV RMSE.\n\n"
        notices = comparison["warnings"] + [f"{item['model']}: {w}" for item in comparison["results"] for w in item["warnings"]]
        if notices:
            summary += "Optimizer notices:\n\n" + "\n".join(f"- {w}" for w in notices) + "\n\n"
    if diagnostics:
        summary += (f"Nominal interval coverage: 90%; observed test coverage: {diagnostics['empirical_coverage']:.1%}; "
                    f"mean interval width: {diagnostics['mean_width']:.4f} percentage points.\n\n")
    summary += ("## Interpretation limits\n\n"
                "Synthetic results do not establish real-process performance. Permutation importance is not causation. "
                "Conformal coverage is marginal under exchangeability, not guaranteed for every input. "
                "Do not tune against the test diagnostics. Free LLM explanations can contain mistakes.\n\n"
                "Reproduction: use dataset.csv with the saved model presets, split settings and software versions. "
                "See docs/stage-4.md and docs/stage-5.md in the source repository.\n")
    if explanation and evaluation.get("run_id") and explanation.get("run_id") == evaluation["run_id"]:
        report["explanation"] = {key: explanation.get(key) for key in ("source", "model", "text", "notice")}
        summary += f"\n## Explanation ({explanation['source']})\n\nModel: {explanation.get('model') or 'local template'}\n\n"
        if explanation.get("notice"):
            summary += explanation["notice"] + "\n\n"
        # Quote external text as text so it remains distinguished from measured facts.
        summary += "\n".join("> " + line for line in explanation["text"].splitlines()) + "\n"

    predictions = pd.DataFrame(evaluation["predictions"])
    if diagnostics:
        predictions = predictions.merge(pd.DataFrame(diagnostics["intervals"]), on="row_index", validate="one_to_one")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("README.md", summary)
        archive.writestr("results.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        archive.writestr("dataset.csv", pd.DataFrame(dataset["rows"]).to_csv(index=False))
        archive.writestr("test_predictions.csv", predictions.to_csv(index=False))
        if comparison:
            folds = [{"model": item["model"], "fold": i, **score}
                     for item in comparison["results"] for i, score in enumerate(item["folds"], 1)]
            archive.writestr("cross_validation.csv", pd.DataFrame(folds).to_csv(index=False))
        if diagnostics:
            archive.writestr("feature_importance.csv", pd.DataFrame(diagnostics["importance"]).to_csv(index=False))
            archive.writestr("residual_bins.csv", pd.DataFrame(diagnostics["residual_bins"]).to_csv(index=False))
    return buffer.getvalue(), summary


def render_exports(dataset: dict, evaluation: dict, comparison: dict | None = None) -> None:
    """Offer explicit report preparation so ordinary widget reruns stay cheap."""
    st.header("Save your results")
    st.caption("Export the displayed dataset, split settings, scores, predictions, and available diagnostics. An explanation is included only for this model run.")
    if st.button("Prepare evaluation report", key="prepare_report"):
        archive, summary = build_report(dataset, evaluation, comparison, st.session_state.get("model_explanation"))
        st.download_button("Download evaluation bundle (.zip)", archive, "sensor_fusion_evaluation.zip", "application/zip", on_click="ignore")
        st.download_button("Download summary (.md)", summary, "sensor_fusion_summary.md", "text/markdown", on_click="ignore")
        st.caption("This is a snapshot of the current results. Prepare again after new training or an explanation.")
