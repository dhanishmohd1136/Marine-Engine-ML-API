import React, { useState, useEffect, useRef } from "react";
import {
  EngineTelemetry,
  PredictionResponse,
  ScenarioPreset,
  ModelInfo,
} from "./types";
import {
  fetchHealth,
  fetchModelInfo,
  fetchScenarios,
  simulateTelemetry,
  predictTelemetry,
} from "./api/client";

const NOMINAL_TELEMETRY: EngineTelemetry = {
  engine_rpm: 98.5,
  engine_load_pct: 78.0,
  scavenge_air_pressure_bar: 2.65,
  scavenge_air_temp_c: 44.2,
  turbocharger_rpm: 16200.0,
  exhaust_gas_temp_avg_c: 382.0,
  exhaust_gas_temp_deviation_c: 12.4,
  fuel_oil_inlet_pressure_bar: 7.5,
  fuel_oil_inlet_temp_c: 134.8,
  lub_oil_pressure_bar: 3.9,
  lub_oil_temp_c: 49.5,
  cooling_water_temp_out_c: 85.2,
  cooling_water_pressure_bar: 3.4,
  ambient_temp_c: 27.0,
  vessel_speed_knots: 18.5,
};

const DEFAULT_PREDICTION: PredictionResponse = {
  predicted_fuel_consumption_kg_h: 1982.5,
  predicted_sfoc_g_kwh: 169.4,
  engine_condition: "Normal",
  condition_confidence: 0.985,
  health_index: 96.2,
  severity: "normal",
  class_probabilities: {
    Normal: 0.985,
    "Turbocharger Fouling": 0.008,
    "Scavenge Fire Warning": 0.001,
    "Fuel Injector Clogged": 0.004,
    "Cooling System Issue": 0.002,
  },
  recommendations: [
    "Engine operation within optimal continuous service rating (CSR).",
    "All thermodynamic parameters are balanced and nominal.",
  ],
  warnings: [],
};

export default function App() {
  const [telemetry, setTelemetry] = useState<EngineTelemetry>(NOMINAL_TELEMETRY);
  const [prediction, setPrediction] = useState<PredictionResponse>(DEFAULT_PREDICTION);
  const [scenarios, setScenarios] = useState<ScenarioPreset[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("normal");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [showModelModal, setShowModelModal] = useState<boolean>(false);

  const streamIntervalRef = useRef<number | null>(null);

  // Initial connection and data fetching
  useEffect(() => {
    async function init() {
      try {
        const health = await fetchHealth();
        setBackendConnected(health.model_loaded);

        const scenariosData = await fetchScenarios();
        setScenarios(scenariosData);

        const info = await fetchModelInfo();
        setModelInfo(info);

        // Run initial prediction
        const initialPred = await predictTelemetry(NOMINAL_TELEMETRY);
        setPrediction(initialPred);
      } catch (err) {
        console.warn("Backend not yet connected or initializing:", err);
        setBackendConnected(false);
      }
    }
    init();
  }, []);

  // Handle telemetry streaming simulation
  useEffect(() => {
    if (isStreaming) {
      streamIntervalRef.current = window.setInterval(async () => {
        try {
          const simulated = await simulateTelemetry(selectedScenarioId);
          setTelemetry(simulated);
          const pred = await predictTelemetry(simulated);
          setPrediction(pred);
          setBackendConnected(true);
        } catch (e) {
          console.error("Stream update failed:", e);
        }
      }, 2500);
    } else {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
        streamIntervalRef.current = null;
      }
    }

    return () => {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
    };
  }, [isStreaming, selectedScenarioId]);

  const handleInputChange = (field: keyof EngineTelemetry, value: string) => {
    const num = parseFloat(value);
    setTelemetry((prev) => ({
      ...prev,
      [field]: isNaN(num) ? 0 : num,
    }));
  };

  const handleRunDiagnostics = async () => {
    setLoading(true);
    try {
      const result = await predictTelemetry(telemetry);
      setPrediction(result);
      setBackendConnected(true);
    } catch (err: any) {
      alert(`Diagnosis error: ${err.message || "Could not complete prediction."}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectScenario = async (scenarioId: string) => {
    setSelectedScenarioId(scenarioId);
    try {
      const simulated = await simulateTelemetry(scenarioId);
      setTelemetry(simulated);
      const pred = await predictTelemetry(simulated);
      setPrediction(pred);
      setBackendConnected(true);
    } catch (err) {
      console.warn("Could not fetch scenario telemetry:", err);
    }
  };

  const handleResetNominal = () => {
    setSelectedScenarioId("normal");
    setTelemetry(NOMINAL_TELEMETRY);
    predictTelemetry(NOMINAL_TELEMETRY).then(setPrediction).catch(console.error);
  };

  // Cylinder simulated temperature readings (deviation spread)
  const cylinderCount = 6;
  const avgTemp = telemetry.exhaust_gas_temp_avg_c || 380;
  const dev = telemetry.exhaust_gas_temp_deviation_c || 10;
  const cylTemps = [
    avgTemp - dev * 0.45,
    avgTemp + dev * 0.5,
    avgTemp - dev * 0.2,
    avgTemp + (prediction.engine_condition === "Fuel Injector Clogged" ? dev * 1.1 : dev * 0.1),
    avgTemp - dev * 0.35,
    avgTemp + dev * 0.3,
  ];

  return (
    <div className="dashboard-container">
      {/* Top Header */}
      <header className="top-header">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">Marine Engine ML Telemetry & Diagnostics</h1>
            <p className="brand-subtitle">
              MAN B&W 6S50ME Two-Stroke Propulsion Engine • Real-time AI Analytics
            </p>
          </div>
        </div>

        <div className="header-status-group">
          <button
            className={`btn-secondary ${isStreaming ? "active-stream" : ""}`}
            onClick={() => setIsStreaming(!isStreaming)}
            title="Toggle continuous telemetry streaming"
          >
            <span className="pulse-dot" style={{ backgroundColor: isStreaming ? "#10b981" : "#94a3b8" }} />
            {isStreaming ? "Live Feed Streaming..." : "Simulate Live Feed"}
          </button>

          <button
            className="btn-secondary"
            onClick={() => setShowModelModal(true)}
            title="View Model Metrics & Details"
          >
            Model Specs
          </button>

          <div className={`status-badge ${backendConnected ? "" : "offline"}`}>
            <span className="pulse-dot" />
            {backendConnected ? "FastAPI Online" : "Connecting..."}
          </div>
        </div>
      </header>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="glass-card kpi-card">
          <div className="kpi-header">
            <span>Fuel Oil Mass Flow</span>
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{prediction.predicted_fuel_consumption_kg_h.toLocaleString()}</span>
            <span className="kpi-unit">kg/h</span>
          </div>
          <div className="kpi-footnote">
            <span>Propeller law regression</span>
          </div>
        </div>

        <div className="glass-card kpi-card">
          <div className="kpi-header">
            <span>Specific Fuel Consumption</span>
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10" strokeWidth="2" />
              <path strokeWidth="2" d="M12 6v6l4 2" />
            </svg>
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{prediction.predicted_sfoc_g_kwh}</span>
            <span className="kpi-unit">g/kWh</span>
          </div>
          <div className="kpi-footnote">
            <span>Optimal CSR: 165-172 g/kWh</span>
          </div>
        </div>

        <div className="glass-card kpi-card">
          <div className="kpi-header">
            <span>Health Index</span>
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="kpi-value-row">
            <span
              className="kpi-value"
              style={{
                color:
                  prediction.health_index > 80
                    ? "#10b981"
                    : prediction.health_index > 50
                    ? "#f59e0b"
                    : "#f43f5e",
              }}
            >
              {prediction.health_index}%
            </span>
            <span className="kpi-unit">Composite</span>
          </div>
          <div className="kpi-footnote">
            <span>Condition confidence: {(prediction.condition_confidence * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="glass-card kpi-card">
          <div className="kpi-header">
            <span>Engine Speed & Load</span>
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{telemetry.engine_rpm}</span>
            <span className="kpi-unit">RPM @ {telemetry.engine_load_pct}%</span>
          </div>
          <div className="kpi-footnote">
            <span>Vessel Speed: {telemetry.vessel_speed_knots} kts</span>
          </div>
        </div>
      </div>

      {/* Condition Status Banner */}
      <div className={`condition-banner ${prediction.severity}`}>
        <div className="condition-info">
          <div className="condition-icon-badge">
            {prediction.severity === "normal" && "⚓"}
            {prediction.severity === "warning" && "⚠️"}
            {prediction.severity === "critical" && "🚨"}
          </div>
          <div>
            <div className="condition-title">
              <span>Engine Status: {prediction.engine_condition}</span>
              <span
                style={{
                  fontSize: "0.75rem",
                  padding: "0.2rem 0.6rem",
                  borderRadius: "999px",
                  background: "rgba(255,255,255,0.12)",
                }}
              >
                Severity: {prediction.severity.toUpperCase()}
              </span>
            </div>
            <div className="condition-desc">
              ML diagnostic assessment based on thermodynamic balance and sensor cross-correlations.
            </div>
          </div>
        </div>

        <div>
          <button className="btn-primary" onClick={handleRunDiagnostics} disabled={loading}>
            {loading ? "Analyzing..." : "Re-evaluate Model"}
          </button>
        </div>
      </div>

      {/* Main Workspace: Telemetry Inputs + Diagnostics Center */}
      <div className="workspace-grid">
        {/* Left Column: Interactive Telemetry Controls */}
        <div className="glass-card">
          <div className="section-header">
            <div className="section-title">
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
              <span>Live Sensor Telemetry Matrix</span>
            </div>
            <button className="btn-secondary" style={{ padding: "0.35rem 0.75rem", fontSize: "0.78rem" }} onClick={handleResetNominal}>
              Reset Nominal
            </button>
          </div>

          {/* Quick Scenario Buttons */}
          <div style={{ marginBottom: "0.5rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Quick Fault Injection / Operating Scenarios:
          </div>
          <div className="scenario-selector">
            {[
              { id: "normal", label: "Open Sea Cruising" },
              { id: "tc_fouling", label: "Turbocharger Fouling" },
              { id: "scavenge_fire", label: "Scavenge Trunk Overheat" },
              { id: "injector_clog", label: "Injector Clogged" },
              { id: "cooling_issue", label: "Cooling Degradation" },
            ].map((sc) => (
              <button
                key={sc.id}
                className={`scenario-btn ${selectedScenarioId === sc.id ? "active" : ""}`}
                onClick={() => handleSelectScenario(sc.id)}
              >
                {sc.label}
              </button>
            ))}
          </div>

          {/* Sensor Input Fields */}
          <div className="sensor-input-grid">
            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Engine Speed</span>
                <span>(30 - 140)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.engine_rpm}
                  onChange={(e) => handleInputChange("engine_rpm", e.target.value)}
                />
                <span className="sensor-unit-tag">RPM</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Engine Load</span>
                <span>(10 - 115)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.engine_load_pct}
                  onChange={(e) => handleInputChange("engine_load_pct", e.target.value)}
                />
                <span className="sensor-unit-tag">%</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Scavenge Pressure</span>
                <span>(0.2 - 5.0)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.05"
                  className="sensor-input"
                  value={telemetry.scavenge_air_pressure_bar}
                  onChange={(e) => handleInputChange("scavenge_air_pressure_bar", e.target.value)}
                />
                <span className="sensor-unit-tag">bar</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Scavenge Air Temp</span>
                <span>(15 - 95)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.scavenge_air_temp_c}
                  onChange={(e) => handleInputChange("scavenge_air_temp_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Turbocharger Speed</span>
                <span>(2k - 26k)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="100"
                  className="sensor-input"
                  value={telemetry.turbocharger_rpm}
                  onChange={(e) => handleInputChange("turbocharger_rpm", e.target.value)}
                />
                <span className="sensor-unit-tag">RPM</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Exhaust Gas Temp (Avg)</span>
                <span>(200 - 580)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="1"
                  className="sensor-input"
                  value={telemetry.exhaust_gas_temp_avg_c}
                  onChange={(e) => handleInputChange("exhaust_gas_temp_avg_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Exhaust Deviation (ΔT)</span>
                <span>(0 - 120)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.exhaust_gas_temp_deviation_c}
                  onChange={(e) => handleInputChange("exhaust_gas_temp_deviation_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Fuel Rail Pressure</span>
                <span>(3 - 16)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.1"
                  className="sensor-input"
                  value={telemetry.fuel_oil_inlet_pressure_bar}
                  onChange={(e) => handleInputChange("fuel_oil_inlet_pressure_bar", e.target.value)}
                />
                <span className="sensor-unit-tag">bar</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Fuel Inlet Temp</span>
                <span>(60 - 165)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.fuel_oil_inlet_temp_c}
                  onChange={(e) => handleInputChange("fuel_oil_inlet_temp_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Lube Oil Pressure</span>
                <span>(1.5 - 7.5)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.1"
                  className="sensor-input"
                  value={telemetry.lub_oil_pressure_bar}
                  onChange={(e) => handleInputChange("lub_oil_pressure_bar", e.target.value)}
                />
                <span className="sensor-unit-tag">bar</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Lube Oil Temp</span>
                <span>(25 - 75)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.lub_oil_temp_c}
                  onChange={(e) => handleInputChange("lub_oil_temp_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Cooling Temp Outlet</span>
                <span>(50 - 110)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.cooling_water_temp_out_c}
                  onChange={(e) => handleInputChange("cooling_water_temp_out_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Cooling Pressure</span>
                <span>(1.0 - 6.0)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.1"
                  className="sensor-input"
                  value={telemetry.cooling_water_pressure_bar}
                  onChange={(e) => handleInputChange("cooling_water_pressure_bar", e.target.value)}
                />
                <span className="sensor-unit-tag">bar</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Ambient Temp</span>
                <span>(5 - 55)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.5"
                  className="sensor-input"
                  value={telemetry.ambient_temp_c}
                  onChange={(e) => handleInputChange("ambient_temp_c", e.target.value)}
                />
                <span className="sensor-unit-tag">°C</span>
              </div>
            </div>

            <div className="sensor-field">
              <div className="sensor-label-row">
                <span>Vessel Speed</span>
                <span>(0 - 30)</span>
              </div>
              <div className="sensor-input-wrapper">
                <input
                  type="number"
                  step="0.1"
                  className="sensor-input"
                  value={telemetry.vessel_speed_knots}
                  onChange={(e) => handleInputChange("vessel_speed_knots", e.target.value)}
                />
                <span className="sensor-unit-tag">kts</span>
              </div>
            </div>
          </div>

          <div className="controls-bar">
            <span style={{ fontSize: "0.82rem", color: "var(--text-dim)" }}>
              Changes take effect immediately on manual trigger or stream cycle.
            </span>
            <button className="btn-primary" onClick={handleRunDiagnostics} disabled={loading}>
              Run AI Diagnostic
            </button>
          </div>
        </div>

        {/* Right Column: AI Condition Probabilities & Cylinder Health */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Classification Probabilities */}
          <div className="glass-card">
            <div className="section-header">
              <div className="section-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Fault Classification Confidence</span>
              </div>
            </div>

            <div>
              {Object.entries(prediction.class_probabilities).map(([cond, prob]) => {
                const isSelected = cond === prediction.engine_condition;
                const pct = (prob * 100).toFixed(1);
                let barColor = "var(--accent-cyan)";
                if (cond === "Normal") barColor = "var(--accent-emerald)";
                else if (cond === "Scavenge Fire Warning") barColor = "var(--accent-rose)";
                else if (cond === "Turbocharger Fouling" || cond === "Cooling System Issue") barColor = "var(--accent-amber)";

                return (
                  <div key={cond} className="prob-item">
                    <div className="prob-label-row">
                      <span style={{ fontWeight: isSelected ? 600 : 400, color: isSelected ? "#ffffff" : "var(--text-muted)" }}>
                        {cond}
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{pct}%</span>
                    </div>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: barColor,
                          boxShadow: isSelected ? `0 0 10px ${barColor}` : "none",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Cylinder Exhaust Balance Simulator */}
          <div className="glass-card">
            <div className="section-header">
              <div className="section-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span>Cylinder Exhaust Temperature Balance</span>
              </div>
              <span style={{ fontSize: "0.78rem", color: dev > 25 ? "var(--accent-rose)" : "var(--accent-emerald)" }}>
                ΔT: {dev}°C
              </span>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-end", height: "130px", padding: "0.5rem 0" }}>
              {cylTemps.map((temp, i) => {
                const heightPct = Math.min(100, Math.max(20, ((temp - 250) / 250) * 100));
                const isOutlier = Math.abs(temp - avgTemp) > 25;
                return (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "0.35rem",
                      height: "100%",
                      justifyContent: "flex-end",
                    }}
                  >
                    <span style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: isOutlier ? "#f43f5e" : "#cbd5e1" }}>
                      {Math.round(temp)}°
                    </span>
                    <div
                      style={{
                        width: "100%",
                        height: `${heightPct}%`,
                        borderRadius: "4px",
                        background: isOutlier
                          ? "linear-gradient(180deg, #f43f5e, #be123c)"
                          : "linear-gradient(180deg, #06b6d4, #1d4ed8)",
                        transition: "all 0.4s ease",
                      }}
                    />
                    <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>Cyl {i + 1}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Marine Engineer Recommendations */}
          <div className="glass-card">
            <div className="section-header">
              <div className="section-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span>Marine Engineer Advisory</span>
              </div>
            </div>

            <div className="rec-list">
              {prediction.recommendations.map((rec, i) => (
                <div key={i} className={`rec-item ${prediction.severity}`}>
                  <span>•</span>
                  <span>{rec}</span>
                </div>
              ))}
              {prediction.warnings && prediction.warnings.length > 0 && (
                <div style={{ marginTop: "0.5rem" }}>
                  <div style={{ fontSize: "0.8rem", color: "var(--accent-amber)", marginBottom: "0.25rem" }}>
                    Sensor Limits Notice:
                  </div>
                  {prediction.warnings.map((w, i) => (
                    <div key={i} style={{ fontSize: "0.78rem", color: "#fcd34d", marginBottom: "0.2rem" }}>
                      {w}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Model Information Modal */}
      {showModelModal && modelInfo && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0,0,0,0.75)",
            backdropFilter: "blur(6px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 999,
            padding: "1rem",
          }}
          onClick={() => setShowModelModal(false)}
        >
          <div
            className="glass-card"
            style={{ maxWidth: "600px", width: "100%", maxHeight: "90vh", overflowY: "auto" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="section-header">
              <div className="brand-title">ML Model Architecture & Verification</div>
              <button
                className="btn-secondary"
                style={{ padding: "0.2rem 0.6rem" }}
                onClick={() => setShowModelModal(false)}
              >
                ✕
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", fontSize: "0.88rem" }}>
              <div>
                <strong>Engine Target:</strong> {modelInfo.engine_type}
              </div>
              <div>
                <strong>Model Version:</strong> {modelInfo.model_version} (Dual RandomForest Regressor & Classifier)
              </div>
              <div>
                <strong>Validation Metrics:</strong>
                <ul style={{ paddingLeft: "1.2rem", marginTop: "0.4rem", color: "var(--text-muted)" }}>
                  <li>Fuel Consumption MAE: {modelInfo.metrics.fuel_consumption_mae_kg_h?.toFixed(2)} kg/h</li>
                  <li>Fuel Consumption R² Score: {modelInfo.metrics.fuel_consumption_r2?.toFixed(4)}</li>
                  <li>SFOC MAE: {modelInfo.metrics.sfoc_mae_g_kwh?.toFixed(2)} g/kWh</li>
                  <li>Diagnostic Classifier Accuracy: {((modelInfo.metrics.condition_accuracy || 1) * 100).toFixed(1)}%</li>
                </ul>
              </div>

              <div>
                <strong>Engineered Thermodynamic Features:</strong>
                <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginTop: "0.4rem" }}>
                  {modelInfo.engineered_features.map((f) => (
                    <span
                      key={f}
                      style={{
                        padding: "0.2rem 0.5rem",
                        background: "rgba(56,189,248,0.15)",
                        borderRadius: "4px",
                        fontSize: "0.75rem",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <strong>Supported Fault Conditions:</strong>
                <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", marginTop: "0.4rem" }}>
                  {modelInfo.condition_classes.map((c) => (
                    <span
                      key={c}
                      style={{
                        padding: "0.2rem 0.5rem",
                        background: "rgba(16,185,129,0.15)",
                        borderRadius: "4px",
                        fontSize: "0.75rem",
                      }}
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="dashboard-footer">
        Marine Engine ML System • IMO CII & EEXI Compliance Telemetry Pipeline • FastAPI & React Vite
      </footer>
    </div>
  );
}
