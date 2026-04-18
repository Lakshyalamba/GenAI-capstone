from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def extract_model_coefficients(model: Any, preprocessor: Any) -> dict[str, Any]:
    """Return coefficient metadata for visualization and interpretation."""
    feature_names = list(preprocessor.get_feature_names_out())
    coefficients = np.asarray(model.coef_).ravel()

    rows = [
        {"feature": feature_name, "coefficient": float(coefficient)}
        for feature_name, coefficient in zip(feature_names, coefficients, strict=False)
    ]
    ranked = sorted(rows, key=lambda item: item["coefficient"], reverse=True)

    return {
        "intercept": float(np.asarray(model.intercept_).ravel()[0]),
        "all_coefficients": rows,
        "top_positive_coefficients": ranked[:10],
        "top_negative_coefficients": sorted(rows, key=lambda item: item["coefficient"])[:10],
        "transformed_feature_names": feature_names,
    }


def evaluate_model(
    model: Any,
    preprocessor: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute core evaluation metrics and chart-ready artifacts."""
    transformed_test = preprocessor.transform(x_test)
    y_pred = model.predict(transformed_test)
    y_prob = model.predict_proba(transformed_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    coefficients = extract_model_coefficients(model, preprocessor)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        },
        "confusion_matrix": matrix.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "prediction_breakdown": {
            "test_rows": int(len(y_test)),
            "predicted_positive": int((y_pred == 1).sum()),
            "predicted_negative": int((y_pred == 0).sum()),
            "actual_positive": int((y_test == 1).sum()),
            "actual_negative": int((y_test == 0).sum()),
        },
        "coefficients": coefficients,
    }
