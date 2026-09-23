"""Keep HTTP communication separate from the Streamlit display code."""

import requests

import math

# This is the public data contract, not an import of backend implementation.
DATA_COLUMNS = (
    "temperature_c", "density_kg_m3", "flow_rate_l_min", "pressure_bar",
    "agitation_rpm", "solid_concentration_pct",
)


class BackendError(Exception):
    """An API problem that the UI can explain without displaying a traceback."""


def fetch_health(base_url: str) -> dict:
    """Fetch and validate API metadata, or raise a readable BackendError.

    A timeout prevents an unavailable backend from leaving the page waiting
    indefinitely. We do not cache this request: every check should be fresh.
    """
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=5)
        # An HTTP response can arrive successfully but still contain a 404/500.
        response.raise_for_status()
    except requests.Timeout as exc:
        raise BackendError("The backend took too long to respond. Try again.") from exc
    except requests.ConnectionError as exc:
        raise BackendError(
            "Cannot reach the backend. Start FastAPI and check the backend URL."
        ) from exc
    except requests.RequestException as exc:
        raise BackendError(
            "The health request failed. Check the backend URL and FastAPI terminal."
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError("The backend returned invalid JSON.") from exc

    # A different service may also expose /health. Check identity and the fields
    # used by the page before reporting a successful connection.
    if (
        not isinstance(data, dict)
        or data.get("status") != "ok"
        or data.get("service") != "sensor-fusion-api"
        or data.get("target") != "Solid concentration"
        or data.get("target_unit") != "%"
        or not isinstance(data.get("version"), str)
    ):
        raise BackendError("The response does not match the expected sensor API.")
    return data


def fetch_training(base_url: str, rows: list[dict], test_fraction: float, split_seed: int) -> dict:
    """Submit the current dataset and validate evaluation artifacts for the UI."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/models/train",
            json={"rows": rows, "test_fraction": test_fraction, "split_seed": split_seed},
            timeout=60,
        )
        if response.status_code == 422:
            raise BackendError("Training settings or dataset are invalid. Regenerate data and try again.")
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BackendError("Training failed. Check the FastAPI terminal and try again.") from exc
    try:
        result = response.json()
        validate_evaluation(result, rows, test_fraction, split_seed, "Linear Regression")
    except (ValueError, KeyError, TypeError) as exc:
        raise BackendError("The training response does not match the expected evaluation contract.") from exc
    return result


def require(condition: bool) -> None:
    """Check HTTP contracts even when Python assertions are disabled."""
    if not condition:
        raise ValueError("Unexpected response contract")


def validate_evaluation(result: dict, rows: list[dict], test_fraction: float, split_seed: int, model: str, calibration_count: int = 0) -> None:
    """Validate shared held-out artifacts for training and model comparison."""
    try:
        expected_test = math.ceil(len(rows) * test_fraction)
        require(result["model"] == model)
        require(result["split_seed"] == split_seed and result["test_fraction"] == test_fraction)
        require(result["test_count"] == expected_test)
        require(result["train_count"] == len(rows) - expected_test - calibration_count)
        for name in ("train_metrics", "test_metrics", "baseline_test_metrics"):
            scores = result[name]
            for metric in ("r2", "rmse", "mae"):
                value = scores[metric]
                if metric == "r2" and value is None:
                    continue
                require(type(value) in (int, float) and math.isfinite(value))
                require(metric == "r2" or value >= 0)
        predictions = result["predictions"]
        require(len(predictions) == expected_test)
        indices = set()
        for row in predictions:
            index = row["row_index"]
            require(type(index) is int and 0 <= index < len(rows) and index not in indices)
            indices.add(index)
            for name in ("actual", "predicted", "residual"):
                require(type(row[name]) in (int, float) and math.isfinite(row[name]))
            require(row["actual"] == rows[index]["solid_concentration_pct"])
            require(math.isclose(row["residual"], row["actual"] - row["predicted"], abs_tol=1e-9))
    except (ValueError, KeyError, TypeError) as exc:
        raise BackendError("The training response does not match the expected evaluation contract.") from exc


MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"]


def fetch_run_action(base_url: str, action: str, run_id: str, inputs: dict | None = None) -> dict:
    """Call an existing run without ever passing an LLM key through Streamlit."""
    if action not in ("predict", "explain"):
        raise BackendError("Unsupported model action.")
    body = {"run_id": run_id}
    if inputs is not None:
        body["inputs"] = inputs
    try:
        response = requests.post(f"{base_url.rstrip('/')}/models/{action}", json=body, timeout=(5, 90))
        if response.status_code == 404:
            raise BackendError("This model run expired or the backend restarted. Train or compare again.")
        if response.status_code == 422:
            raise BackendError("Invalid sensor inputs or model run. Check the values and retrain if needed.")
        response.raise_for_status()
        data = response.json()
        require(data["run_id"] == run_id)
        if action == "predict":
            require(data["model"] in MODEL_NAMES and data["inputs"] == inputs)
            require(type(data["prediction"]) in (int, float) and math.isfinite(data["prediction"]))
            require(isinstance(data["warnings"], list) and all(isinstance(w, str) for w in data["warnings"]))
            if data["nominal_coverage"] is None:
                require(data["lower"] is None and data["upper"] is None)
            else:
                require(data["nominal_coverage"] == 0.9)
                require(all(type(data[n]) in (int, float) and math.isfinite(data[n]) for n in ("lower", "upper")))
                require(data["lower"] <= data["prediction"] <= data["upper"])
        else:
            require(data["source"] in ("template", "openrouter"))
            require(isinstance(data["text"], str) and bool(data["text"].strip()))
            require(isinstance(data["evidence"], dict) and type(data["cached"]) is bool)
            require(data["notice"] is None or isinstance(data["notice"], str))
            require(data["model"] is None or isinstance(data["model"], str))
        return data
    except requests.Timeout as exc:
        raise BackendError("Streamlit timed out waiting for FastAPI (90-second read limit). Check the backend terminal; an explanation may still be running.") from exc
    except requests.exceptions.JSONDecodeError as exc:
        raise BackendError("FastAPI returned invalid JSON, not a timeout. Check the backend terminal.") from exc
    except requests.RequestException as exc:
        raise BackendError("Could not complete the request to FastAPI. Check its connection and backend terminal.") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise BackendError("The model response was incomplete or invalid.") from exc


def fetch_comparison(base_url: str, rows: list[dict], test_fraction: float, split_seed: int, models: list[str], include_diagnostics: bool = False) -> dict:
    """Run one comparison request; retain readable failures for the UI."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/models/compare",
            json={"rows": rows, "test_fraction": test_fraction, "split_seed": split_seed, "models": models,
                  "include_diagnostics": include_diagnostics},
            timeout=180,
        )
        if response.status_code == 422:
            raise BackendError("Invalid comparison settings. Select at least one supported model.")
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BackendError("Model comparison failed or timed out. Check FastAPI and try fewer models or rows.") from exc
    try:
        result = response.json()
        require(result["cv_folds"] == 5 and result["selection_metric"] == "mean CV RMSE")
        require([item["model"] for item in result["results"]] == list(dict.fromkeys(models)))
        for item in result["results"]:
            require(len(item["folds"]) == 5)
            for scores in [*item["folds"], item["mean"], item["std"]]:
                for metric in ("r2", "rmse", "mae"):
                    value = scores[metric]
                    if metric == "r2" and value is None:
                        continue
                    require(type(value) in (int, float) and math.isfinite(value))
                    require(metric == "r2" or value >= 0)
            require(isinstance(item["warnings"], list) and all(isinstance(w, str) for w in item["warnings"]))
        winner = min(result["results"], key=lambda item: item["mean"]["rmse"])["model"]
        require(result["selected_model"] == winner)
        require(isinstance(result["warnings"], list) and all(isinstance(w, str) for w in result["warnings"]))
        calibration_count = math.ceil(len(rows) * 0.2) if include_diagnostics else 0
        validate_evaluation(result["evaluation"], rows, test_fraction, split_seed, winner, calibration_count)
        if include_diagnostics:
            validate_diagnostics(result["diagnostics"], result["evaluation"], calibration_count)
        else:
            require(result.get("diagnostics") is None)
    except (ValueError, KeyError, TypeError) as exc:
        raise BackendError("The comparison response does not match the expected contract.") from exc
    return result


def validate_diagnostics(data: dict, evaluation: dict, calibration_count: int) -> None:
    """Validate finite diagnostic numbers and interval alignment before plotting."""
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    require(data["calibration_count"] == calibration_count and data["nominal_coverage"] == 0.9)
    require(data["quantile_rank"] == math.ceil((calibration_count + 1) * 0.9))
    for name in ("half_width", "mean_width", "residual_std"):
        require(finite(data[name]) and data[name] >= 0)
    require(math.isclose(data["mean_width"], 2 * data["half_width"]))
    require(finite(data["mean_residual"]))
    corr = data["abs_residual_prediction_correlation"]
    require(corr is None or (finite(corr) and -1 <= corr <= 1))
    require(data["importance_repeats"] == 10)
    require([row["feature"] for row in data["importance"]] == list(DATA_COLUMNS[:-1]))
    for row in data["importance"]:
        require(finite(row["mean"]) and finite(row["std"]) and row["std"] >= 0)
    require(len(data["intervals"]) == evaluation["test_count"])
    hits = 0
    for interval, prediction in zip(data["intervals"], evaluation["predictions"]):
        require(interval["row_index"] == prediction["row_index"])
        require(finite(interval["lower"]) and finite(interval["upper"]))
        require(math.isclose(interval["lower"], prediction["predicted"] - data["half_width"], abs_tol=1e-9))
        require(math.isclose(interval["upper"], prediction["predicted"] + data["half_width"], abs_tol=1e-9))
        hit = interval["lower"] <= prediction["actual"] <= interval["upper"]
        require(type(interval["covered"]) is bool and interval["covered"] == hit)
        hits += hit
    require(finite(data["empirical_coverage"]) and math.isclose(data["empirical_coverage"], hits / evaluation["test_count"]))
    require(sum(row["count"] for row in data["residual_bins"]) == evaluation["test_count"])
    for row in data["residual_bins"]:
        require(type(row["count"]) is int and row["count"] > 0)
        require(all(finite(row[name]) for name in ("lower", "upper", "mean_residual", "mae", "rmse")))
        require(row["lower"] <= row["upper"] and row["mae"] >= 0 and row["rmse"] >= 0)


def fetch_dataset(base_url: str, n_samples: int, seed: int, noise_std: float) -> dict:
    """Request synthetic observations and reject malformed data before plotting."""
    settings = {"n_samples": n_samples, "seed": seed, "noise_std": noise_std}
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/data/generate", json=settings, timeout=30,
        )
        if response.status_code == 422:
            raise BackendError("Invalid settings: use 100–5000 rows, a valid seed, and noise from 0–5.")
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BackendError(
            "Could not generate data. Check that FastAPI is running, then try again."
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError("The data endpoint returned invalid JSON.") from exc
    if (
        not isinstance(data, dict)
        or data.get("source") != "synthetic"
        or data.get("generator_version") != "1.0"
        or data.get("target_basis") != "mass percent (w/w)"
        or data.get("settings") != settings
        or not isinstance(data.get("clipped_target_count"), int)
        or not 0 <= data["clipped_target_count"] <= n_samples
        or not isinstance(data.get("rows"), list)
        or len(data["rows"]) != n_samples
    ):
        raise BackendError("The data response does not match the expected dataset contract.")
    for row in data["rows"]:
        if not isinstance(row, dict) or set(row) != set(DATA_COLUMNS):
            raise BackendError("The dataset has unexpected columns.")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row.values()):
            raise BackendError("The dataset contains missing or nonnumeric values.")
    return data
