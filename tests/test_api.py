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
    assert "baseline_loaded" in data
    assert "metadata_loaded" in data
    assert "version" in data


def test_model_info_or_503():
    response = client.get("/model-info")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "default_threshold" in data
        assert "pipeline_steps" in data


def test_forecast_monte_carlo():
    response = client.get("/forecast/monte-carlo", params={
        "n_customers": 1000,
        "avg_revenue": 6000,
        "churn_rate_mean": 0.27,
        "churn_rate_std": 0.05,
        "n_simulations": 100,
    })
    assert response.status_code == 200
    data = response.json()
    assert "simulation_parameters" in data
    assert "results" in data
    assert "value_at_risk_5pct" in data["results"]
    assert "histogram_bins" in data
    assert "histogram_counts" in data


def test_forecast_revenue_or_503():
    """Revenue forecast requires baseline data — test it returns 200 or 503."""
    response = client.post("/forecast/revenue", params={"steps": 3})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "historical_monthly_revenue" in data
        assert "forecast" in data
        assert len(data["forecast"]) == 3
