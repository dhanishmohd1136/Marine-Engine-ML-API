"""
Model Evaluation and Diagnostics Module for Marine Engine Telemetry.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from backend.src.preprocessing import FEATURE_COLUMNS, ENGINE_CONDITIONS


def evaluate_pipeline(pipeline, test_data_path: str = "backend/data/raw/marine_engine_data.csv") -> Dict[str, Any]:
    """
    Evaluates regression and classification components on evaluation data.
    """
    df = pd.read_csv(test_data_path)
    X = df[FEATURE_COLUMNS]
    y_reg = df[["fuel_consumption_kg_h", "sfoc_g_kwh"]]
    y_clf = df["condition"]

    X_trans = pipeline.preprocessor.transform(X)

    # Regression evaluation
    reg_preds = pipeline.regressor.predict(X_trans)
    fuel_actual = y_reg["fuel_consumption_kg_h"].values
    fuel_pred = reg_preds[:, 0]
    sfoc_actual = y_reg["sfoc_g_kwh"].values
    sfoc_pred = reg_preds[:, 1]

    reg_metrics = {
        "fuel_consumption": {
            "mae": float(np.round(mean_absolute_error(fuel_actual, fuel_pred), 3)),
            "rmse": float(np.round(np.sqrt(mean_squared_error(fuel_actual, fuel_pred)), 3)),
            "r2": float(np.round(r2_score(fuel_actual, fuel_pred), 4)),
        },
        "sfoc": {
            "mae": float(np.round(mean_absolute_error(sfoc_actual, sfoc_pred), 3)),
            "rmse": float(np.round(np.sqrt(mean_squared_error(sfoc_actual, sfoc_pred)), 3)),
            "r2": float(np.round(r2_score(sfoc_actual, sfoc_pred), 4)),
        }
    }

    # Classification evaluation
    clf_preds = pipeline.classifier.predict(X_trans)
    clf_acc = float(np.round(accuracy_score(y_clf, clf_preds), 4))
    cm = confusion_matrix(y_clf, clf_preds).tolist()
    report = classification_report(
        y_clf,
        clf_preds,
        target_names=ENGINE_CONDITIONS,
        output_dict=True,
        zero_division=0,
    )

    # Feature Importances
    all_features = pipeline.preprocessor.all_feature_names
    reg_importances = dict(zip(all_features, [float(v) for v in pipeline.regressor.feature_importances_]))
    clf_importances = dict(zip(all_features, [float(v) for v in pipeline.classifier.feature_importances_]))

    return {
        "regression_metrics": reg_metrics,
        "classification_metrics": {
            "accuracy": clf_acc,
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "feature_importances": {
            "regression_fuel": sorted(reg_importances.items(), key=lambda x: x[1], reverse=True)[:8],
            "classification_condition": sorted(clf_importances.items(), key=lambda x: x[1], reverse=True)[:8],
        }
    }
