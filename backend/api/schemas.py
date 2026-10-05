"""
Pydantic Schemas for Marine Engine ML API.
Includes input validation, documentation examples, and response schemas.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class EngineTelemetryInput(BaseModel):
    """Real-time marine diesel engine operational telemetry reading."""

    engine_rpm: float = Field(
        default=98.5,
        ge=30.0,
        le=140.0,
        description="Main engine crankshaft speed (RPM)",
        examples=[98.5],
    )
    engine_load_pct: float = Field(
        default=78.0,
        ge=10.0,
        le=115.0,
        description="Engine load percentage based on MCR (%)",
        examples=[78.0],
    )
    scavenge_air_pressure_bar: float = Field(
        default=2.65,
        ge=0.2,
        le=5.0,
        description="Scavenge air manifold receiver pressure (bar)",
        examples=[2.65],
    )
    scavenge_air_temp_c: float = Field(
        default=44.2,
        ge=15.0,
        le=95.0,
        description="Scavenge air temperature after cooler (°C)",
        examples=[44.2],
    )
    turbocharger_rpm: float = Field(
        default=16200.0,
        ge=2000.0,
        le=26000.0,
        description="Exhaust turbocharger rotor speed (RPM)",
        examples=[16200.0],
    )
    exhaust_gas_temp_avg_c: float = Field(
        default=382.0,
        ge=200.0,
        le=580.0,
        description="Average cylinder exhaust gas temperature (°C)",
        examples=[382.0],
    )
    exhaust_gas_temp_deviation_c: float = Field(
        default=12.4,
        ge=0.0,
        le=120.0,
        description="Max deviation between cylinder exhaust temperatures (°C)",
        examples=[12.4],
    )
    fuel_oil_inlet_pressure_bar: float = Field(
        default=7.5,
        ge=3.0,
        le=16.0,
        description="Heavy fuel oil supply pressure to engine rail (bar)",
        examples=[7.5],
    )
    fuel_oil_inlet_temp_c: float = Field(
        default=134.8,
        ge=60.0,
        le=165.0,
        description="Fuel oil pre-heater outlet temperature (°C)",
        examples=[134.8],
    )
    lub_oil_pressure_bar: float = Field(
        default=3.9,
        ge=1.5,
        le=7.5,
        description="Main bearing lubricating oil inlet pressure (bar)",
        examples=[3.9],
    )
    lub_oil_temp_c: float = Field(
        default=49.5,
        ge=25.0,
        le=75.0,
        description="Main bearing lubricating oil inlet temperature (°C)",
        examples=[49.5],
    )
    cooling_water_temp_out_c: float = Field(
        default=85.2,
        ge=50.0,
        le=110.0,
        description="Jacket cooling water outlet temperature (°C)",
        examples=[85.2],
    )
    cooling_water_pressure_bar: float = Field(
        default=3.4,
        ge=1.0,
        le=6.0,
        description="Jacket cooling water inlet pressure (bar)",
        examples=[3.4],
    )
    ambient_temp_c: float = Field(
        default=27.0,
        ge=5.0,
        le=55.0,
        description="Engine room ambient temperature (°C)",
        examples=[27.0],
    )
    vessel_speed_knots: float = Field(
        default=18.5,
        ge=0.0,
        le=30.0,
        description="Ship speed over ground (knots)",
        examples=[18.5],
    )


class EngineTelemetryBatchInput(BaseModel):
    """Batch of engine sensor readings for high-throughput inference."""
    readings: List[EngineTelemetryInput]


class PredictionResult(BaseModel):
    """Engine condition and fuel performance inference response."""
    predicted_fuel_consumption_kg_h: float = Field(
        ..., description="Predicted fuel mass flow rate (kg/hour)"
    )
    predicted_sfoc_g_kwh: float = Field(
        ..., description="Specific Fuel Oil Consumption (grams per kWh)"
    )
    engine_condition: str = Field(
        ..., description="Classified condition (Normal, Turbocharger Fouling, Scavenge Fire Warning, etc.)"
    )
    condition_confidence: float = Field(
        ..., description="Confidence score for the predicted condition class (0.0 to 1.0)"
    )
    health_index: float = Field(
        ..., description="Overall engine health score from 0 (critical) to 100 (pristine)"
    )
    severity: str = Field(
        ..., description="Health status severity: 'normal', 'warning', or 'critical'"
    )
    class_probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all diagnosed conditions"
    )
    recommendations: List[str] = Field(
        ..., description="Operational and maintenance recommendations for marine engineers"
    )
    warnings: Optional[List[str]] = Field(
        default=[], description="Sensor boundary anomalies or out-of-range sensor alerts"
    )


class BatchPredictionItem(BaseModel):
    index: int
    predicted_fuel_consumption_kg_h: float
    predicted_sfoc_g_kwh: float
    engine_condition: str
    condition_confidence: float
    health_index: float
    severity: str


class BatchPredictionResult(BaseModel):
    total_samples: int
    predictions: List[BatchPredictionItem]


class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    engine_type: str


class ModelMetadataResponse(BaseModel):
    model_version: str
    engine_type: str
    features: List[str]
    engineered_features: List[str]
    condition_classes: List[str]
    metrics: Dict[str, float]


class ScenarioPreset(BaseModel):
    id: str
    name: str
    description: str
    telemetry: EngineTelemetryInput
