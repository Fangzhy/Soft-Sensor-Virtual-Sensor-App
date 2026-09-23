"""Keep HTTP communication separate from the Streamlit display code."""

import requests


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
