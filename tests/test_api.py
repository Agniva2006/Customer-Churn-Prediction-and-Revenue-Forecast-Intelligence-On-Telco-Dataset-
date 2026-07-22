"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info_or_503():
    response = client.get("/model-info")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "default_threshold" in data
