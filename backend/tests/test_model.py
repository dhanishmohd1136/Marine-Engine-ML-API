"""
Unit and integration tests for Marine Engine ML models and preprocessing.
"""

import pytest
import os
import joblib
import pandas as pd
import numpy as np

from backend.src.preprocessing import EnginePreprocessor, FEATURE_COLUMNS, ENGINE_CONDITIONS
from backend.src.train import generate_synthetic_marine_data, train_and_save_pipeline, MarineEnginePipeline


@pytest.fixture(scope="module")
def sample_data():
    return generate_synthetic_marine_data(n_samples=200, random_state=123)


@pytest.fixture(scope="module")
def pipeline():
    model_path = "backend/models/marine_engine_model.joblib"
    if not os.path.exists(model_path):
        pipe, _ = train_and_save_pipeline(output_model_path=model_path, n_samples=500)
        return pipe
    return joblib.load(model_path)


def test_data_generation_columns(sample_data):
    """Verify generated dataset contains all mandatory feature and target columns."""
    for col in FEATURE_COLUMNS:
        assert col in sample_data.columns, f"Missing feature: {col}"
    assert "fuel_consumption_kg_h" in sample_data.columns
    assert "sfoc_g_kwh" in sample_data.columns
    assert "condition" in sample_data.columns
    assert len(sample_data) == 200


def test_preprocessor_transformation(sample_data):
    """Ensure preprocessor transforms raw features into engineered matrix with correct dimensions."""
    preprocessor = EnginePreprocessor()
    X = sample_data[FEATURE_COLUMNS]
    X_trans = preprocessor.fit_transform(X)

    expected_features_len = len(FEATURE_COLUMNS) + len(preprocessor.engineered_feature_names)
    assert X_trans.shape[1] == expected_features_len
    assert not np.isnan(X_trans).any()


def test_sensor_bounds_validation():
    """Verify out-of-range sensor detection."""
    valid_data = {
        "engine_rpm": 95.0,
        "engine_load_pct": 75.0,
        "scavenge_air_pressure_bar": 2.5,
    }
    assert len(EnginePreprocessor.validate_sensor_data(valid_data)) == 0

    invalid_data = {
        "engine_rpm": 250.0,  # Physically impossible for low-speed 2-stroke
        "engine_load_pct": 140.0,
    }
    warnings = EnginePreprocessor.validate_sensor_data(invalid_data)
    assert len(warnings) >= 2


def test_pipeline_prediction_output(pipeline):
    """Verify single telemetry inference format and value ranges."""
    sample_input = {
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

    result = pipeline.predict_telemetry(sample_input)

    assert "predicted_fuel_consumption_kg_h" in result
    assert result["predicted_fuel_consumption_kg_h"] > 0
    assert "predicted_sfoc_g_kwh" in result
    assert 140.0 <= result["predicted_sfoc_g_kwh"] <= 260.0
    assert result["engine_condition"] in ENGINE_CONDITIONS
    assert 0.0 <= result["health_index"] <= 100.0
    assert result["severity"] in ["normal", "warning", "critical"]
    assert len(result["recommendations"]) > 0


def test_pipeline_batch_prediction(pipeline, sample_data):
    """Verify batch dataframe inference."""
    result_df = pipeline.predict_batch_df(sample_data.head(10))
    assert "pred_fuel_kg_h" in result_df.columns
    assert "pred_sfoc_g_kwh" in result_df.columns
    assert "pred_condition" in result_df.columns
    assert len(result_df) == 10
