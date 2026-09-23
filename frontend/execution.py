"""Select the deployment mode without changing the UI's data contracts."""

import os


def execution_mode() -> str:
    """Fail clearly on typos rather than silently connecting to localhost."""
    mode = os.getenv("APP_EXECUTION_MODE", "http").strip().lower()
    if mode not in {"http", "direct"}:
        raise ValueError("APP_EXECUTION_MODE must be 'http' or 'direct'.")
    return mode


def request(method: str, url: str, **kwargs):
    """Use HTTP locally or validated services within the Streamlit server.

    The lightweight response wrapper lets both modes use the same existing
    frontend response validators. It never sends a loopback HTTP request.
    Backend imports are lazy so the HTTP frontend stays independently runnable.
    """
    import requests

    if execution_mode() == "http":
        return getattr(requests, method)(url, **kwargs)

    import json
    from urllib.parse import urlsplit
    import streamlit as st
    from frontend.direct_backend import dispatch

    status, payload = dispatch(method, urlsplit(url).path, kwargs.get("json", {}), st.session_state)
    response = requests.Response()
    response.status_code = status
    response.encoding = "utf-8"
    response._content = json.dumps(payload, allow_nan=False).encode("utf-8")
    return response
