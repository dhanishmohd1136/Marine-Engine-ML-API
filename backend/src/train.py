"""
Marine Engine Model Training Pipeline.
Generates realistic marine telemetry data based on propeller law and diesel cycle thermodynamics,
trains dual ML models (Fuel Consumption Regression + Condition Classification),
and saves the serialized artifact.
"""

from typing import Dict, Any, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

from backend.src.preprocessing import (
    EnginePreprocessor,
    FEATURE_COLUMNS,
    ENGINE_CONDITIONS,
)


class MarineEnginePipeline:
    """
    Combined ML pipeline for Marine Propulsion Diesel Engines.
    Provides simultaneous fuel consumption regression and condition monitoring classification.
    """

    def __init__(self, preprocessor: EnginePreprocessor, regressor: RandomForestRegressor, classifier: RandomForestClassifier, metadata: Dict[str, Any]):
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.classifier = classifier
        self.metadata = metadata

    def predict_telemetry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a single engine sensor reading.
        """
        df_input = pd.DataFrame([data])[FEATURE_COLUMNS]
        X_scaled = self.preprocessor.transform(df_input)

        reg_preds = self.regressor.predict(X_scaled)[0]
        fuel_consumption_kg_h = float(np.round(reg_preds[0], 2))
        sfoc_g_kwh = float(np.round(reg_preds[1], 2))

        condition_idx = int(self.classifier.predict(X_scaled)[0])
        condition_label = ENGINE_CONDITIONS[condition_idx]
        probabilities = self.classifier.predict_proba(X_scaled)[0]
        prob_dict = {
            ENGINE_CONDITIONS[i]: float(np.round(probabilities[i], 4))
            for i in range(len(ENGINE_CONDITIONS))
        }

        # Calculate composite health index (0 to 100)
        # Normal probability is primary driver, penalized by high exhaust deviation or over-temp
        base_health = probabilities[0] * 100.0
        scavenge_temp = float(data.get("scavenge_air_temp_c", 40.0))
        exhaust_dev = float(data.get("exhaust_gas_temp_deviation_c", 15.0))
        cooling_temp = float(data.get("cooling_water_temp_out_c", 82.0))

        penalty = 0.0
        if scavenge_temp > 65.0:
            penalty += min(35.0, (scavenge_temp - 65.0) * 1.5)
        if exhaust_dev > 25.0:
            penalty += min(30.0, (exhaust_dev - 25.0) * 1.2)
        if cooling_temp > 88.0:
            penalty += min(25.0, (cooling_temp - 88.0) * 2.0)

        health_index = float(np.clip(base_health - penalty, 5.0, 99.5))

        # Determine engineering recommendations
        recommendations = []
        status_severity = "normal"

        if condition_label == "Normal":
            if health_index > 85.0:
                recommendations.append("Engine operation within optimal continuous service rating (CSR).")
            else:
                recommendations.append("Minor thermal deviation observed; verify cylinder balance.")
        elif condition_label == "Turbocharger Fouling":
            status_severity = "warning"
            recommendations.append("Initiate turbocharger turbine and compressor dry/wet wash sequence.")
            recommendations.append("Inspect air filter intake differential pressure.")
        elif condition_label == "Scavenge Fire Warning":
            status_severity = "critical"
            recommendations.append("CRITICAL: Scavenge trunk temperature abnormal. Reduce RPM immediately.")
            recommendations.append("Inspect cylinder drain valves and check for piston ring blow-by.")
        elif condition_label == "Fuel Injector Clogged":
            status_severity = "warning"
            recommendations.append("Significant exhaust deviation detected across cylinders.")
            recommendations.append("Schedule injector overhaul and verify fuel viscosity / temperature control.")
        elif condition_label == "Cooling System Issue":
            status_severity = "warning"
            recommendations.append("Jacket cooling water outlet temperature elevated.")
            recommendations.append("Check fresh water cooler bypass valve and deaeration tank pressure.")

        return {
            "predicted_fuel_consumption_kg_h": fuel_consumption_kg_h,
            "predicted_sfoc_g_kwh": sfoc_g_kwh,
            "engine_condition": condition_label,
            "condition_confidence": float(np.round(probabilities[condition_idx], 4)),
            "health_index": round(health_index, 1),
            "severity": status_severity,
            "class_probabilities": prob_dict,
            "recommendations": recommendations,
        }

    def predict_batch_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch predictions on a DataFrame."""
        X_scaled = self.preprocessor.transform(df[FEATURE_COLUMNS])
        reg_preds = self.regressor.predict(X_scaled)
        clf_preds = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)

        result_df = df.copy()
        result_df["pred_fuel_kg_h"] = np.round(reg_preds[:, 0], 2)
        result_df["pred_sfoc_g_kwh"] = np.round(reg_preds[:, 1], 2)
        result_df["pred_condition"] = [ENGINE_CONDITIONS[i] for i in clf_preds]
        result_df["pred_confidence"] = [
            np.round(probabilities[idx, i], 4) for idx, i in enumerate(clf_preds)
        ]
        return result_df


def generate_synthetic_marine_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic marine propulsion engine telemetry following physics relationships
    (Propeller curve, thermodynamic combustion cycles, and failure mode injections).
    """
    np.random.seed(random_state)

    # Base operating load distribution (realistic operating profile at sea)
    engine_load_pct = np.random.beta(5, 2, n_samples) * 70.0 + 30.0  # 30% to 100% load
    engine_load_pct = np.clip(engine_load_pct, 25.0, 100.0)

    # Engine RPM follows propeller law: RPM ~ RPM_max * (Load / 100)^(1/3)
    max_rpm = 108.0  # Nominal MCR for large two-stroke engine
    engine_rpm = max_rpm * ((engine_load_pct / 100.0) ** (1.0 / 3.0)) + np.random.normal(0, 1.2, n_samples)
    engine_rpm = np.clip(engine_rpm, 45.0, 115.0)

    # Scavenge air pressure follows engine load
    scavenge_air_pressure_bar = 0.4 + (engine_load_pct / 100.0) * 2.8 + np.random.normal(0, 0.08, n_samples)

    # Scavenge air temp (°C) after air cooler
    scavenge_air_temp_c = 38.0 + (engine_load_pct / 100.0) * 8.0 + np.random.normal(0, 1.5, n_samples)

    # Turbocharger RPM follows load (8,000 - 19,000 RPM)
    turbocharger_rpm = 6500.0 + (engine_load_pct / 100.0) * 12500.0 + np.random.normal(0, 220.0, n_samples)

    # Exhaust gas temp average (°C)
    exhaust_gas_temp_avg_c = 280.0 + (engine_load_pct / 100.0) * 125.0 + np.random.normal(0, 8.0, n_samples)

    # Exhaust gas deviation (°C) across cylinders (normally tightly balanced 8 - 18°C)
    exhaust_gas_temp_deviation_c = np.random.gamma(3, 3, n_samples) + 5.0

    # Fuel oil inlet pressure (bar)
    fuel_oil_inlet_pressure_bar = 7.5 + np.random.normal(0, 0.35, n_samples)

    # Fuel oil inlet temp (°C) (heated to maintain ~12-14 cSt viscosity)
    fuel_oil_inlet_temp_c = 135.0 + np.random.normal(0, 2.5, n_samples)

    # Lubricating oil pressure (bar)
    lub_oil_pressure_bar = 4.2 - (engine_load_pct / 100.0) * 0.4 + np.random.normal(0, 0.15, n_samples)

    # Lubricating oil temp (°C)
    lub_oil_temp_c = 46.0 + (engine_load_pct / 100.0) * 4.5 + np.random.normal(0, 1.0, n_samples)

    # Cooling water (jacket) outlet temp (°C)
    cooling_water_temp_out_c = 82.0 + (engine_load_pct / 100.0) * 4.0 + np.random.normal(0, 1.2, n_samples)

    # Cooling water pressure (bar)
    cooling_water_pressure_bar = 3.4 + np.random.normal(0, 0.18, n_samples)

    # Ambient conditions
    ambient_temp_c = np.random.uniform(18.0, 38.0, n_samples)

    # Vessel speed (knots)
    vessel_speed_knots = 11.0 + (engine_load_pct / 100.0) * 9.5 + np.random.normal(0, 0.6, n_samples)

    # Baseline Specific Fuel Oil Consumption (SFOC in g/kWh) has a 'bath-tub' curve with minimum around 75-85% load
    sfoc_g_kwh = 168.0 + 0.015 * ((engine_load_pct - 80.0) ** 2) + np.random.normal(0, 2.0, n_samples)

    # Engine Brake Power estimation (kW): Power = MCR_Power * (Load / 100)
    # E.g. 15,000 kW engine
    engine_power_kw = 15000.0 * (engine_load_pct / 100.0)
    fuel_consumption_kg_h = (sfoc_g_kwh * engine_power_kw) / 1000.0 + np.random.normal(0, 15.0, n_samples)

    # Default condition: 0 = Normal
    condition = np.zeros(n_samples, dtype=int)

    # Inject failure anomalies
    n_anomalies = int(n_samples * 0.28)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    split_chunks = np.array_split(anomaly_indices, 4)

    # Mode 1: Turbocharger Fouling (Condition 1)
    tc_fouling_idx = split_chunks[0]
    scavenge_air_pressure_bar[tc_fouling_idx] *= np.random.uniform(0.72, 0.85, len(tc_fouling_idx))
    turbocharger_rpm[tc_fouling_idx] *= np.random.uniform(1.08, 1.18, len(tc_fouling_idx))
    exhaust_gas_temp_avg_c[tc_fouling_idx] += np.random.uniform(30.0, 65.0, len(tc_fouling_idx))
    fuel_consumption_kg_h[tc_fouling_idx] *= np.random.uniform(1.04, 1.09, len(tc_fouling_idx))
    sfoc_g_kwh[tc_fouling_idx] *= np.random.uniform(1.04, 1.09, len(tc_fouling_idx))
    condition[tc_fouling_idx] = 1

    # Mode 2: Scavenge Fire Warning (Condition 2)
    scavenge_fire_idx = split_chunks[1]
    scavenge_air_temp_c[scavenge_fire_idx] += np.random.uniform(30.0, 55.0, len(scavenge_fire_idx))
    exhaust_gas_temp_deviation_c[scavenge_fire_idx] += np.random.uniform(25.0, 55.0, len(scavenge_fire_idx))
    condition[scavenge_fire_idx] = 2

    # Mode 3: Fuel Injector Clogged (Condition 3)
    injector_idx = split_chunks[2]
    exhaust_gas_temp_deviation_c[injector_idx] += np.random.uniform(32.0, 75.0, len(injector_idx))
    fuel_oil_inlet_pressure_bar[injector_idx] -= np.random.uniform(1.2, 2.5, len(injector_idx))
    sfoc_g_kwh[injector_idx] *= np.random.uniform(1.06, 1.14, len(injector_idx))
    fuel_consumption_kg_h[injector_idx] *= np.random.uniform(1.06, 1.14, len(injector_idx))
    condition[injector_idx] = 3

    # Mode 4: Cooling System Issue (Condition 4)
    cooling_idx = split_chunks[3]
    cooling_water_temp_out_c[cooling_idx] += np.random.uniform(11.0, 22.0, len(cooling_idx))
    cooling_water_pressure_bar[cooling_idx] -= np.random.uniform(0.8, 1.6, len(cooling_idx))
    condition[cooling_idx] = 4

    data = {
        "engine_rpm": np.round(engine_rpm, 2),
        "engine_load_pct": np.round(engine_load_pct, 2),
        "scavenge_air_pressure_bar": np.round(scavenge_air_pressure_bar, 2),
        "scavenge_air_temp_c": np.round(scavenge_air_temp_c, 2),
        "turbocharger_rpm": np.round(turbocharger_rpm, 1),
        "exhaust_gas_temp_avg_c": np.round(exhaust_gas_temp_avg_c, 2),
        "exhaust_gas_temp_deviation_c": np.round(exhaust_gas_temp_deviation_c, 2),
        "fuel_oil_inlet_pressure_bar": np.round(fuel_oil_inlet_pressure_bar, 2),
        "fuel_oil_inlet_temp_c": np.round(fuel_oil_inlet_temp_c, 2),
        "lub_oil_pressure_bar": np.round(lub_oil_pressure_bar, 2),
        "lub_oil_temp_c": np.round(lub_oil_temp_c, 2),
        "cooling_water_temp_out_c": np.round(cooling_water_temp_out_c, 2),
        "cooling_water_pressure_bar": np.round(cooling_water_pressure_bar, 2),
        "ambient_temp_c": np.round(ambient_temp_c, 2),
        "vessel_speed_knots": np.round(vessel_speed_knots, 2),
        "fuel_consumption_kg_h": np.round(fuel_consumption_kg_h, 2),
        "sfoc_g_kwh": np.round(sfoc_g_kwh, 2),
        "condition_label": [ENGINE_CONDITIONS[c] for c in condition],
        "condition": condition,
    }

    return pd.DataFrame(data)


def train_and_save_pipeline(
    output_model_path: str = "backend/models/marine_engine_model.joblib",
    raw_data_path: str = "backend/data/raw/marine_engine_data.csv",
    processed_data_path: str = "backend/data/processed/processed_features.csv",
    n_samples: int = 3500,
) -> Tuple[MarineEnginePipeline, Dict[str, Any]]:
    """
    Execute full training workflow: dataset generation, preprocessing,
    dual-model fitting (regression + classification), validation, and serialization.
    """
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    print("Generating marine engine physics telemetry dataset...")
    df = generate_synthetic_marine_data(n_samples=n_samples, random_state=42)
    df.to_csv(raw_data_path, index=False)

    X = df[FEATURE_COLUMNS]
    y_reg = df[["fuel_consumption_kg_h", "sfoc_g_kwh"]]
    y_clf = df["condition"]

    preprocessor = EnginePreprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Save processed features sample
    df_processed = pd.DataFrame(X_transformed, columns=preprocessor.all_feature_names)
    df_processed["fuel_consumption_kg_h"] = y_reg["fuel_consumption_kg_h"].values
    df_processed["condition"] = y_clf.values
    df_processed.to_csv(processed_data_path, index=False)

    # Train/Test Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_transformed, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("Training Fuel Consumption Regressor (RandomForest)...")
    regressor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_reg_train)

    y_reg_pred = regressor.predict(X_test)
    fuel_mae = float(mean_absolute_error(y_reg_test.iloc[:, 0], y_reg_pred[:, 0]))
    fuel_r2 = float(r2_score(y_reg_test.iloc[:, 0], y_reg_pred[:, 0]))
    sfoc_mae = float(mean_absolute_error(y_reg_test.iloc[:, 1], y_reg_pred[:, 1]))
    sfoc_r2 = float(r2_score(y_reg_test.iloc[:, 1], y_reg_pred[:, 1]))

    print(f"Regression -> Fuel MAE: {fuel_mae:.2f} kg/h, R2: {fuel_r2:.4f} | SFOC MAE: {sfoc_mae:.2f} g/kWh, R2: {sfoc_r2:.4f}")

    print("Training Engine Condition Classifier (RandomForest)...")
    classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    classifier.fit(X_train, y_clf_train)

    y_clf_pred = classifier.predict(X_test)
    clf_acc = float(accuracy_score(y_clf_test, y_clf_pred))
    print(f"Classification -> Accuracy: {clf_acc * 100:.2f}%")

    metadata = {
        "model_version": "1.0.0",
        "engine_type": "MAN B&W 6S50ME Two-Stroke Diesel",
        "metrics": {
            "fuel_consumption_mae_kg_h": fuel_mae,
            "fuel_consumption_r2": fuel_r2,
            "sfoc_mae_g_kwh": sfoc_mae,
            "sfoc_r2": sfoc_r2,
            "condition_accuracy": clf_acc,
        },
        "feature_columns": FEATURE_COLUMNS,
        "engineered_features": preprocessor.engineered_feature_names,
        "condition_classes": ENGINE_CONDITIONS,
    }

    pipeline = MarineEnginePipeline(preprocessor, regressor, classifier, metadata)
    joblib.dump(pipeline, output_model_path)
    print(f"Successfully saved trained model pipeline to {output_model_path}")

    return pipeline, metadata


if __name__ == "__main__":
    train_and_save_pipeline()
