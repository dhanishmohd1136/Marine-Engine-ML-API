import {
  EngineTelemetry,
  PredictionResponse,
  ScenarioPreset,
  ModelInfo,
  SystemHealth,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/api/v1/model/info`);
  if (!res.ok) throw new Error(`Failed to load model info: ${res.statusText}`);
  return res.json();
}

export async function fetchScenarios(): Promise<ScenarioPreset[]> {
  const res = await fetch(`${API_BASE}/api/v1/scenarios`);
  if (!res.ok) throw new Error(`Failed to load scenarios: ${res.statusText}`);
  return res.json();
}

export async function simulateTelemetry(scenario: string = "normal"): Promise<EngineTelemetry> {
  const res = await fetch(`${API_BASE}/api/v1/telemetry/simulate?scenario=${scenario}`);
  if (!res.ok) throw new Error(`Simulation failed: ${res.statusText}`);
  return res.json();
}

export async function predictTelemetry(telemetry: EngineTelemetry): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(telemetry),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.message || `Prediction failed with status ${res.status}`);
  }
  return res.json();
}
