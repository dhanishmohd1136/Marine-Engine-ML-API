"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.main import app


@pytest.fixture(scope="module")
def client():
    # Use context manager so lifespan runs
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    """Test /health endpoint returns healthy status and model flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "MAN B&W" in data["engine_type"]


def test_model_info_endpoint(client):
    """Test /api/v1/model/info endpoint."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert len(data["features"]) == 15
    assert len(data["condition_classes"]) == 5
    assert "fuel_consumption_r2" in data["metrics"]


def test_predict_single_endpoint(client):
    """Test single telemetry prediction endpoint with normal payload."""
    payload = {
        "engine_rpm": 98.0,
        "engine_load_pct": 78.0,
        "scavenge_air_pressure_bar": 2.65,
        "scavenge_air_temp_c": 44.0,
        "turbocharger_rpm": 16200.0,
        "exhaust_gas_temp_avg_c": 380.0,
        "exhaust_gas_temp_deviation_c": 12.0,
        "fuel_oil_inlet_pressure_bar": 7.5,
        "fuel_oil_inlet_temp_c": 135.0,
        "lub_oil_pressure_bar": 3.9,
        "lub_oil_temp_c": 49.0,
        "cooling_water_temp_out_c": 85.0,
        "cooling_water_pressure_bar": 3.4,
        "ambient_temp_c": 27.0,
        "vessel_speed_knots": 18.5,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_fuel_consumption_kg_h" in data
    assert "predicted_sfoc_g_kwh" in data
    assert "engine_condition" in data
    assert "health_index" in data
    assert "severity" in data
    assert isinstance(data["recommendations"], list)


def test_predict_batch_endpoint(client):
    """Test batch telemetry prediction endpoint."""
    reading = {
        "engine_rpm": 98.0,
        "engine_load_pct": 78.0,
        "scavenge_air_pressure_bar": 2.65,
        "scavenge_air_temp_c": 44.0,
        "turbocharger_rpm": 16200.0,
        "exhaust_gas_temp_avg_c": 380.0,
        "exhaust_gas_temp_deviation_c": 12.0,
        "fuel_oil_inlet_pressure_bar": 7.5,
        "fuel_oil_inlet_temp_c": 135.0,
        "lub_oil_pressure_bar": 3.9,
        "lub_oil_temp_c": 49.0,
        "cooling_water_temp_out_c": 85.0,
        "cooling_water_pressure_bar": 3.4,
        "ambient_temp_c": 27.0,
        "vessel_speed_knots": 18.5,
    }
    payload = {"readings": [reading, reading]}
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert len(data["predictions"]) == 2


def test_simulate_telemetry_endpoint(client):
    """Test /api/v1/telemetry/simulate returns realistic payload."""
    response = client.get("/api/v1/telemetry/simulate?scenario=normal")
    assert response.status_code == 200
    data = response.json()
    assert "engine_rpm" in data
    assert 30.0 <= data["engine_rpm"] <= 140.0


def test_scenarios_endpoint(client):
    """Test /api/v1/scenarios returns predefined testing presets."""
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) >= 4
    scenario_ids = [s["id"] for s in scenarios]
    assert "normal" in scenario_ids
    assert "tc_fouling" in scenario_ids
