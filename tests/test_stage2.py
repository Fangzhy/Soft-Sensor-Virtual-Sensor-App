"""Check reproducibility, API boundaries, and the explorer's dataset lifecycle."""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app
from backend.schemas import GenerationRequest
from backend.services.synthetic_data import generate_dataset
from frontend.api_client import BackendError, fetch_dataset


def test_reproducibility_and_noise():
    """Changing noise preserves sensor readings; changing the seed changes data."""
    settings = GenerationRequest(n_samples=300, seed=42, noise_std=0)
    clean = generate_dataset(settings)
    assert clean == generate_dataset(settings)
    assert clean != generate_dataset(settings.model_copy(update={"seed": 43}))
    noisy = generate_dataset(settings.model_copy(update={"noise_std": 2.0}))
    clean_df = pd.DataFrame([row.model_dump() for row in clean.rows])
    noisy_df = pd.DataFrame([row.model_dump() for row in noisy.rows])
    pd.testing.assert_frame_equal(clean_df.iloc[:, :-1], noisy_df.iloc[:, :-1])
    assert not np.allclose(clean_df.iloc[:, -1], noisy_df.iloc[:, -1])
    assert np.isfinite(noisy_df.to_numpy()).all()


@pytest.mark.parametrize("settings", [
    {"n_samples": 99}, {"n_samples": 5001}, {"n_samples": 100.5},
    {"seed": -1}, {"seed": 2**32}, {"noise_std": -0.1},
    {"noise_std": 5.1}, {"unknown": 2},
])
def test_invalid_generation_request(settings):
    """Bad settings are rejected at the HTTP boundary with a validation error."""
    with TestClient(app) as client:
        assert client.post("/data/generate", json=settings).status_code == 422


@pytest.mark.parametrize("n_samples", [100, 5000])
def test_api_bounds_and_csv_round_trip(n_samples):
    """Both supported dataset sizes return bounded numeric rows and export cleanly."""
    with TestClient(app) as client:
        response = client.post("/data/generate", json={"n_samples": n_samples})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "synthetic"
    assert data["target_basis"] == "mass percent (w/w)"
    df = pd.DataFrame(data["rows"])
    assert df.shape == (n_samples, 6)
    for column, low, high in [
        ("temperature_c", 20, 90), ("density_kg_m3", 1000, 1300),
        ("flow_rate_l_min", 10, 100), ("pressure_bar", 1, 6),
        ("agitation_rpm", 100, 800), ("solid_concentration_pct", 0, 100),
    ]:
        assert df[column].between(low, high).all()
    pd.testing.assert_frame_equal(df, pd.read_csv(StringIO(df.to_csv(index=False))))


def test_data_client_rejects_bad_payload_and_handles_network_error():
    """Do not hand incomplete JSON or a failed request to the plotting layer."""
    response = Mock(status_code=200)
    response.json.return_value = {"rows": []}
    with patch("frontend.api_client.requests.post", return_value=response):
        with pytest.raises(BackendError, match="contract"):
            fetch_dataset("http://localhost:8000", 100, 42, 1.0)
    with patch("frontend.api_client.requests.post", side_effect=requests.Timeout()):
        with pytest.raises(BackendError, match="Could not generate"):
            fetch_dataset("http://localhost:8000", 100, 42, 1.0)


def test_explorer_persists_data_and_recovers_from_failed_generation():
    """Chart reruns must not regenerate; a failed request must keep old data."""
    payload = generate_dataset(GenerationRequest()).model_dump()
    response = Mock(status_code=200)
    response.json.return_value = payload
    path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    with patch("frontend.api_client.requests.post", return_value=response) as post:
        page = AppTest.from_file(str(path), default_timeout=20).run()
        assert not page.exception
        post.assert_not_called()
        next(button for button in page.button if button.label == "Generate synthetic data").click().run()
        assert not page.exception
        assert len(page.dataframe) == 2
        assert len(page.get("plotly_chart")) == 3
        assert page.session_state["dataset"] == payload
        page.selectbox[0].select("temperature_c").run()
        assert not page.exception
        assert post.call_count == 1
        post.side_effect = requests.ConnectionError()
        next(button for button in page.button if button.label == "Generate synthetic data").click().run()
        assert not page.exception
        assert page.error and page.warning
        assert page.session_state["dataset"] == payload
        post.side_effect = None
        next(button for button in page.button if button.label == "Generate synthetic data").click().run()
        assert not page.exception
        assert not page.error
