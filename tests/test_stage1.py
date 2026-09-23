"""Exercise the API contract and the connection states a learner will encounter."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app
from frontend.api_client import BackendError, fetch_health


def test_health_endpoint_and_documentation():
    """Exercise real FastAPI routing and serialization without opening a port."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "service": "sensor-fusion-api", "version": "0.1.0",
            "target": "Solid concentration", "target_unit": "%",
        }
        assert client.get("/docs").status_code == 200
        assert "/health" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("error", [requests.ConnectionError(), requests.Timeout(), requests.HTTPError()])
def test_client_translates_network_errors(error):
    """Expected network problems should produce displayable application errors."""
    with patch("frontend.api_client.requests.get", side_effect=error):
        with pytest.raises(BackendError):
            fetch_health("http://127.0.0.1:8000")


@pytest.mark.parametrize("payload", [[], {"status": "ok"}, {"status": "unhealthy"}])
def test_client_rejects_unexpected_service(payload):
    """A successful HTTP status alone must not be displayed as a healthy API."""
    response = Mock()
    response.json.return_value = payload
    with patch("frontend.api_client.requests.get", return_value=response):
        with pytest.raises(BackendError, match="expected sensor API"):
            fetch_health("http://127.0.0.1:8000")


def test_client_rejects_invalid_json():
    """An HTML response (such as a proxy error page) should not crash the UI."""
    response = Mock()
    response.json.side_effect = ValueError("invalid JSON")
    with patch("frontend.api_client.requests.get", return_value=response):
        with pytest.raises(BackendError, match="invalid JSON"):
            fetch_health("http://127.0.0.1:8000")


def test_page_connection_and_recovery():
    """Click the actual UI through success, disconnection, and recovery."""
    with TestClient(app) as client:
        payload = client.get("/health").json()
    response = Mock()
    response.json.return_value = payload
    page_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    with patch("frontend.api_client.requests.get", return_value=response) as get:
        page = AppTest.from_file(str(page_path)).run()
        assert not page.exception
        get.assert_not_called()
        page.button[0].click().run()
        assert not page.exception
        assert len(page.success) == 1
        assert len(page.json) == 1

        get.side_effect = requests.ConnectionError()
        page.button[0].click().run()
        assert not page.exception
        assert len(page.error) == 1
        assert not page.success

        get.side_effect = None
        page.button[0].click().run()
        assert not page.exception
        assert len(page.success) == 1
        assert not page.error
