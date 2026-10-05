"""
Main FastAPI application for Marine Engine ML API.
Provides endpoints for real-time inference, batch predictions, model diagnostics,
telemetry simulation, and sensor sanity checks.
"""

from contextlib import asynccontextmanager
import os
import random
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.src.preprocessing import EnginePreprocessor, FEATURE_COLUMNS, ENGINE_CONDITIONS
from backend.src.train import MarineEnginePipeline, train_and_save_pipeline
from backend.api.schemas import (
    EngineTelemetryInput,
    EngineTelemetryBatchInput,
    PredictionResult,
    BatchPredictionResult,
    BatchPredictionItem,
    HealthCheckResponse,
    ModelMetadataResponse,
    ScenarioPreset,
)
from backend.api.exceptions import (
    MarineEngineAPIException,
    ModelNotLoadedException,
    marine_engine_exception_handler,
)

MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/marine_engine_model.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: ensure model is trained and loaded into memory on startup."""
    print("Initializing Marine Engine ML API...")
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Training baseline pipeline...")
        pipeline, _ = train_and_save_pipeline(output_model_path=MODEL_PATH)
        app.state.pipeline = pipeline
    else:
        try:
            print(f"Loading trained pipeline from {MODEL_PATH}...")
            app.state.pipeline = joblib.load(MODEL_PATH)
            print("Model pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading model ({e}). Retraining fallback...")
            pipeline, _ = train_and_save_pipeline(output_model_path=MODEL_PATH)
            app.state.pipeline = pipeline

    yield

    print("Shutting down Marine Engine ML API.")


app = FastAPI(
    title="Marine Engine ML API",
    description=(
        "Production-grade Machine Learning API for Marine Propulsion Diesel Engines.\n\n"
        "Features real-time fuel consumption prediction, specific fuel oil consumption (SFOC) estimation, "
        "and condition-based monitoring / fault detection (Turbocharger Fouling, Scavenge Fire, Injector issues)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(MarineEngineAPIException, marine_engine_exception_handler)


def get_pipeline(request: Request) -> MarineEnginePipeline:
    """Dependency to retrieve loaded ML pipeline from application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ModelNotLoadedException()
    return pipeline


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root requests to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health check & system status",
)
async def health_check(request: Request):
    """Verify API availability and ML model pipeline readiness."""
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None
    engine_type = "MAN B&W 6S50ME Two-Stroke Diesel"
    if pipeline_loaded:
        engine_type = request.app.state.pipeline.metadata.get("engine_type", engine_type)

    return HealthCheckResponse(
        status="healthy" if pipeline_loaded else "degraded",
        service="Marine Engine ML API",
        version="1.0.0",
        model_loaded=pipeline_loaded,
        engine_type=engine_type,
    )


@app.get(
    "/api/v1/model/info",
    response_model=ModelMetadataResponse,
    tags=["Model Diagnostics"],
    summary="Get model parameters and accuracy metrics",
)
async def get_model_info(pipeline: MarineEnginePipeline = Depends(get_pipeline)):
    """Retrieve model training metadata, engineered features, and validation performance."""
    metadata = pipeline.metadata
    return ModelMetadataResponse(
        model_version=metadata.get("model_version", "1.0.0"),
        engine_type=metadata.get("engine_type", "MAN B&W 6S50ME Two-Stroke Diesel"),
        features=metadata.get("feature_columns", FEATURE_COLUMNS),
        engineered_features=metadata.get("engineered_features", []),
        condition_classes=metadata.get("condition_classes", ENGINE_CONDITIONS),
        metrics=metadata.get("metrics", {}),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Real-time engine telemetry prediction",
)
async def predict_telemetry(
    telemetry: EngineTelemetryInput,
    pipeline: MarineEnginePipeline = Depends(get_pipeline),
):
    """
    Predict Fuel Consumption (kg/h), SFOC (g/kWh), and Diagnostic Condition
    given real-time marine engine sensor parameters.
    """
    sensor_dict = telemetry.model_dump()
    warnings = EnginePreprocessor.validate_sensor_data(sensor_dict)

    result = pipeline.predict_telemetry(sensor_dict)
    result["warnings"] = warnings

    return PredictionResult(**result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResult,
    tags=["Inference"],
    summary="Batch telemetry prediction",
)
async def predict_batch(
    batch_input: EngineTelemetryBatchInput,
    pipeline: MarineEnginePipeline = Depends(get_pipeline),
):
    """Run high-throughput inference across multiple marine sensor records."""
    records = [reading.model_dump() for reading in batch_input.readings]
    results: List[BatchPredictionItem] = []

    for idx, record in enumerate(records):
        pred = pipeline.predict_telemetry(record)
        results.append(
            BatchPredictionItem(
                index=idx,
                predicted_fuel_consumption_kg_h=pred["predicted_fuel_consumption_kg_h"],
                predicted_sfoc_g_kwh=pred["predicted_sfoc_g_kwh"],
                engine_condition=pred["engine_condition"],
                condition_confidence=pred["condition_confidence"],
                health_index=pred["health_index"],
                severity=pred["severity"],
            )
        )

    return BatchPredictionResult(
        total_samples=len(results),
        predictions=results,
    )


@app.get(
    "/api/v1/telemetry/simulate",
    response_model=EngineTelemetryInput,
    tags=["Simulation"],
    summary="Simulate live sensor reading",
)
async def simulate_telemetry(scenario: str = "normal"):
    """
    Generate realistic fluctuating telemetry based on marine operational profiles:
    - `normal`: Optimal steady-state cruising
    - `tc_fouling`: Turbocharger fouling
    - `scavenge_fire`: High scavenge trunk temperature
    - `injector_clog`: Cylinder imbalance and injector fouling
    - `cooling_issue`: Jacket cooling water overheat
    """
    jitter = lambda base, scale: round(base + random.uniform(-scale, scale), 2)

    if scenario == "tc_fouling":
        return EngineTelemetryInput(
            engine_rpm=jitter(95.0, 1.0),
            engine_load_pct=jitter(78.0, 1.5),
            scavenge_air_pressure_bar=jitter(2.1, 0.05),  # Lower scavenge pressure
            scavenge_air_temp_c=jitter(46.0, 1.0),
            turbocharger_rpm=jitter(18800.0, 300.0),      # High TC RPM
            exhaust_gas_temp_avg_c=jitter(425.0, 5.0),     # Elevated exhaust temp
            exhaust_gas_temp_deviation_c=jitter(14.0, 2.0),
            fuel_oil_inlet_pressure_bar=jitter(7.4, 0.2),
            fuel_oil_inlet_temp_c=jitter(134.5, 1.0),
            lub_oil_pressure_bar=jitter(3.8, 0.1),
            lub_oil_temp_c=jitter(49.0, 0.8),
            cooling_water_temp_out_c=jitter(85.0, 1.0),
            cooling_water_pressure_bar=jitter(3.4, 0.1),
            ambient_temp_c=jitter(28.0, 1.0),
            vessel_speed_knots=jitter(17.2, 0.3),
        )
    elif scenario == "scavenge_fire":
        return EngineTelemetryInput(
            engine_rpm=jitter(92.0, 1.5),
            engine_load_pct=jitter(75.0, 2.0),
            scavenge_air_pressure_bar=jitter(2.4, 0.1),
            scavenge_air_temp_c=jitter(78.5, 3.0),         # Severe scavenge temp spike!
            turbocharger_rpm=jitter(16000.0, 400.0),
            exhaust_gas_temp_avg_c=jitter(415.0, 8.0),
            exhaust_gas_temp_deviation_c=jitter(48.0, 5.0), # Extreme deviation
            fuel_oil_inlet_pressure_bar=jitter(7.5, 0.2),
            fuel_oil_inlet_temp_c=jitter(135.0, 1.0),
            lub_oil_pressure_bar=jitter(3.7, 0.1),
            lub_oil_temp_c=jitter(51.0, 1.0),
            cooling_water_temp_out_c=jitter(87.0, 1.2),
            cooling_water_pressure_bar=jitter(3.3, 0.1),
            ambient_temp_c=jitter(31.0, 1.0),
            vessel_speed_knots=jitter(16.5, 0.4),
        )
    elif scenario == "injector_clog":
        return EngineTelemetryInput(
            engine_rpm=jitter(96.0, 1.0),
            engine_load_pct=jitter(77.0, 1.5),
            scavenge_air_pressure_bar=jitter(2.5, 0.08),
            scavenge_air_temp_c=jitter(44.0, 1.0),
            turbocharger_rpm=jitter(16200.0, 250.0),
            exhaust_gas_temp_avg_c=jitter(395.0, 6.0),
            exhaust_gas_temp_deviation_c=jitter(52.0, 4.0), # Severe cylinder delta
            fuel_oil_inlet_pressure_bar=jitter(5.6, 0.3),  # Drop in fuel rail pressure
            fuel_oil_inlet_temp_c=jitter(132.0, 1.5),
            lub_oil_pressure_bar=jitter(3.9, 0.1),
            lub_oil_temp_c=jitter(49.2, 0.8),
            cooling_water_temp_out_c=jitter(85.5, 1.0),
            cooling_water_pressure_bar=jitter(3.4, 0.1),
            ambient_temp_c=jitter(26.0, 1.0),
            vessel_speed_knots=jitter(17.8, 0.3),
        )
    elif scenario == "cooling_issue":
        return EngineTelemetryInput(
            engine_rpm=jitter(97.0, 1.0),
            engine_load_pct=jitter(80.0, 1.5),
            scavenge_air_pressure_bar=jitter(2.6, 0.08),
            scavenge_air_temp_c=jitter(45.0, 1.0),
            turbocharger_rpm=jitter(16500.0, 250.0),
            exhaust_gas_temp_avg_c=jitter(390.0, 5.0),
            exhaust_gas_temp_deviation_c=jitter(14.0, 1.5),
            fuel_oil_inlet_pressure_bar=jitter(7.5, 0.2),
            fuel_oil_inlet_temp_c=jitter(135.0, 1.0),
            lub_oil_pressure_bar=jitter(3.7, 0.1),
            lub_oil_temp_c=jitter(54.0, 1.0),
            cooling_water_temp_out_c=jitter(95.5, 1.5),    # High cooling temp!
            cooling_water_pressure_bar=jitter(2.1, 0.15),  # Low cooling pressure!
            ambient_temp_c=jitter(32.0, 1.0),
            vessel_speed_knots=jitter(18.0, 0.3),
        )
    else:
        # Default: Normal sea cruising
        return EngineTelemetryInput(
            engine_rpm=jitter(98.5, 0.8),
            engine_load_pct=jitter(78.0, 1.2),
            scavenge_air_pressure_bar=jitter(2.65, 0.04),
            scavenge_air_temp_c=jitter(43.5, 0.8),
            turbocharger_rpm=jitter(16300.0, 150.0),
            exhaust_gas_temp_avg_c=jitter(380.0, 4.0),
            exhaust_gas_temp_deviation_c=jitter(11.5, 1.2),
            fuel_oil_inlet_pressure_bar=jitter(7.5, 0.15),
            fuel_oil_inlet_temp_c=jitter(135.0, 0.8),
            lub_oil_pressure_bar=jitter(3.9, 0.08),
            lub_oil_temp_c=jitter(49.0, 0.6),
            cooling_water_temp_out_c=jitter(84.8, 0.8),
            cooling_water_pressure_bar=jitter(3.4, 0.08),
            ambient_temp_c=jitter(27.0, 0.8),
            vessel_speed_knots=jitter(18.5, 0.2),
        )


@app.get(
    "/api/v1/scenarios",
    response_model=List[ScenarioPreset],
    tags=["Simulation"],
    summary="List predefined operating scenarios",
)
async def list_scenarios():
    """Retrieve operational scenario presets for manual testing or demonstration."""
    return [
        ScenarioPreset(
            id="normal",
            name="Normal Open Sea Cruising",
            description="Optimal steady-state propulsion profile at 78% continuous service rating (CSR).",
            telemetry=await simulate_telemetry("normal"),
        ),
        ScenarioPreset(
            id="tc_fouling",
            name="Turbocharger Fouling",
            description="Turbine nozzle ring fouling resulting in elevated TC RPM and high exhaust backpressure.",
            telemetry=await simulate_telemetry("tc_fouling"),
        ),
        ScenarioPreset(
            id="scavenge_fire",
            name="Scavenge Trunk Overheat Risk",
            description="Excessive scavenge receiver temperature with cylinder blow-by risk.",
            telemetry=await simulate_telemetry("scavenge_fire"),
        ),
        ScenarioPreset(
            id="injector_clog",
            name="Fuel Injector Clogged",
            description="Cylinder thermal imbalance with large exhaust temperature delta and elevated SFOC.",
            telemetry=await simulate_telemetry("injector_clog"),
        ),
        ScenarioPreset(
            id="cooling_issue",
            name="Jacket Cooling System Degradation",
            description="Cooling water pressure drop and thermal dissipation deficiency.",
            telemetry=await simulate_telemetry("cooling_issue"),
        ),
    ]
