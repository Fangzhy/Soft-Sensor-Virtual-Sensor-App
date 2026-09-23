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
    def require(condition: bool) -> None:
        """Validate even when Python runs with assertions disabled (-O)."""
        if not condition:
            raise ValueError("Unexpected training response")

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
        expected_test = math.ceil(len(rows) * test_fraction)
        require(result["model"] == "Linear Regression")
        require(result["split_seed"] == split_seed and result["test_fraction"] == test_fraction)
        require(result["test_count"] == expected_test)
        require(result["train_count"] == len(rows) - expected_test)
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
    return result


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
