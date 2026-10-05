"""
Marine Engine Data Preprocessing and Feature Engineering.
Handles sensor validation, boundary checks, feature engineering, and scaling.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: List[str] = [
    "engine_rpm",
    "engine_load_pct",
    "scavenge_air_pressure_bar",
    "scavenge_air_temp_c",
    "turbocharger_rpm",
    "exhaust_gas_temp_avg_c",
    "exhaust_gas_temp_deviation_c",
    "fuel_oil_inlet_pressure_bar",
    "fuel_oil_inlet_temp_c",
    "lub_oil_pressure_bar",
    "lub_oil_temp_c",
    "cooling_water_temp_out_c",
    "cooling_water_pressure_bar",
    "ambient_temp_c",
    "vessel_speed_knots",
]

ENGINE_CONDITIONS: List[str] = [
    "Normal",
    "Turbocharger Fouling",
    "Scavenge Fire Warning",
    "Fuel Injector Clogged",
    "Cooling System Issue",
]

# Physical plausible limits for Marine Two-Stroke Propulsion Diesel Engines
SENSOR_LIMITS: Dict[str, Tuple[float, float]] = {
    "engine_rpm": (30.0, 140.0),
    "engine_load_pct": (10.0, 115.0),
    "scavenge_air_pressure_bar": (0.2, 5.0),
    "scavenge_air_temp_c": (15.0, 95.0),
    "turbocharger_rpm": (2000.0, 26000.0),
    "exhaust_gas_temp_avg_c": (200.0, 580.0),
    "exhaust_gas_temp_deviation_c": (0.0, 120.0),
    "fuel_oil_inlet_pressure_bar": (3.0, 16.0),
    "fuel_oil_inlet_temp_c": (60.0, 165.0),
    "lub_oil_pressure_bar": (1.5, 7.5),
    "lub_oil_temp_c": (25.0, 75.0),
    "cooling_water_temp_out_c": (50.0, 110.0),
    "cooling_water_pressure_bar": (1.0, 6.0),
    "ambient_temp_c": (5.0, 55.0),
    "vessel_speed_knots": (0.0, 30.0),
}


class EnginePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies feature engineering
    and standardization to marine engine telemetry.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_in_: List[str] = FEATURE_COLUMNS
        self.engineered_feature_names: List[str] = [
            "tc_pressure_ratio",
            "exhaust_to_load_ratio",
            "cooling_temp_delta",
            "power_demand_proxy",
            "fuel_temp_deviation",
        ]
        self.all_feature_names: List[str] = []

    @staticmethod
    def validate_sensor_data(data: Dict[str, Any]) -> List[str]:
        """
        Validate input sensor values against physical operating boundaries.
        Returns a list of validation warning/error messages.
        """
        warnings = []
        for feature, (min_val, max_val) in SENSOR_LIMITS.items():
            if feature in data:
                val = float(data[feature])
                if val < min_val or val > max_val:
                    warnings.append(
                        f"Sensor '{feature}' value {val} is outside normal marine operating envelope [{min_val}, {max_val}]."
                    )
        return warnings

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute domain-specific marine engineering indicators:
        - tc_pressure_ratio: Scavenge pressure per 10k TC RPM
        - exhaust_to_load_ratio: Exhaust temperature relative to engine load
        - cooling_temp_delta: Cooling water outlet temperature over ambient
        - power_demand_proxy: Load percentage scaled by RPM
        - fuel_temp_deviation: Deviation from optimal heavy fuel oil viscosity temp (~135°C)
        """
        engineered = df.copy()

        tc_rpm_safe = engineered["turbocharger_rpm"].clip(lower=1000.0)
        load_safe = engineered["engine_load_pct"].clip(lower=5.0)

        engineered["tc_pressure_ratio"] = (
            engineered["scavenge_air_pressure_bar"] / (tc_rpm_safe / 10000.0)
        )
        engineered["exhaust_to_load_ratio"] = (
            engineered["exhaust_gas_temp_avg_c"] / load_safe
        )
        engineered["cooling_temp_delta"] = (
            engineered["cooling_water_temp_out_c"] - engineered["ambient_temp_c"]
        )
        engineered["power_demand_proxy"] = (
            engineered["engine_load_pct"] * engineered["engine_rpm"] / 100.0
        )
        engineered["fuel_temp_deviation"] = (
            (engineered["fuel_oil_inlet_temp_c"] - 135.0).abs()
        )

        return engineered

    def fit(self, X: pd.DataFrame, y=None):
        """Fit scaler on raw and engineered features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        engineered_df = self._engineer_features(X)
        self.all_feature_names = list(engineered_df.columns)
        self.scaler.fit(engineered_df)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw engine features into scaled matrix."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        engineered_df = self._engineer_features(X)
        return self.scaler.transform(engineered_df)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
