export interface EngineTelemetry {
  engine_rpm: number;
  engine_load_pct: number;
  scavenge_air_pressure_bar: number;
  scavenge_air_temp_c: number;
  turbocharger_rpm: number;
  exhaust_gas_temp_avg_c: number;
  exhaust_gas_temp_deviation_c: number;
  fuel_oil_inlet_pressure_bar: number;
  fuel_oil_inlet_temp_c: number;
  lub_oil_pressure_bar: number;
  lub_oil_temp_c: number;
  cooling_water_temp_out_c: number;
  cooling_water_pressure_bar: number;
  ambient_temp_c: number;
  vessel_speed_knots: number;
}

export type SeverityLevel = "normal" | "warning" | "critical";

export interface PredictionResponse {
  predicted_fuel_consumption_kg_h: number;
  predicted_sfoc_g_kwh: number;
  engine_condition: string;
  condition_confidence: number;
  health_index: number;
  severity: SeverityLevel;
  class_probabilities: Record<string, number>;
  recommendations: string[];
  warnings: string[];
}

export interface ScenarioPreset {
  id: string;
  name: string;
  description: string;
  telemetry: EngineTelemetry;
}

export interface ModelMetrics {
  fuel_consumption_mae_kg_h?: number;
  fuel_consumption_r2?: number;
  sfoc_mae_g_kwh?: number;
  sfoc_r2?: number;
  condition_accuracy?: number;
}

export interface ModelInfo {
  model_version: string;
  engine_type: string;
  features: string[];
  engineered_features: string[];
  condition_classes: string[];
  metrics: ModelMetrics;
}

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  engine_type: string;
}
