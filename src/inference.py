from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    build_feature_frame,
    coerce_and_validate_patient_payload,
)
from src.utils import MODELS_DIR, humanize_slug, load_json


MODEL_FILENAME = "logistic_regression_model.joblib"
PREPROCESSOR_FILENAME = "scaler.joblib"
FEATURE_FILENAME = "feature_columns.json"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "evaluation_summary.json"


def load_artifact_bundle(models_dir: str | Path = MODELS_DIR) -> dict[str, Any]:
    """Load the trained model artifacts required for inference."""
    base_dir = Path(models_dir)
    required_paths = {
        "model": base_dir / MODEL_FILENAME,
        "preprocessor": base_dir / PREPROCESSOR_FILENAME,
        "feature_config": base_dir / FEATURE_FILENAME,
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run `python train.py` first. Missing: " + ", ".join(missing)
        )

    return {
        "model": joblib.load(required_paths["model"]),
        "preprocessor": joblib.load(required_paths["preprocessor"]),
        "feature_config": load_json(required_paths["feature_config"], default={}),
        "metadata": load_json(base_dir / METADATA_FILENAME, default={}),
        "evaluation": load_json(base_dir / EVALUATION_FILENAME, default={}),
    }


def get_risk_category(probability: float) -> str:
    """Map a probability score into a user-friendly risk tier."""
    if probability < 0.35:
        return "Low"
    if probability < 0.70:
        return "Moderate"
    return "High"


def _humanize_transformed_feature(feature_name: str, patient_data: Mapping[str, Any]) -> str:
    if "__" not in feature_name:
        return humanize_slug(feature_name)

    transformer_name, raw_feature = feature_name.split("__", maxsplit=1)
    if transformer_name == "numeric":
        label = FEATURE_METADATA.get(raw_feature, {}).get("label", humanize_slug(raw_feature))
        value = patient_data.get(raw_feature)
        if value is None:
            return label
        if float(value).is_integer():
            value_label = str(int(value))
        else:
            value_label = f"{float(value):.1f}"
        return f"{label} ({value_label})"

    if transformer_name == "categorical":
        for candidate in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{candidate}_"
            if raw_feature.startswith(prefix):
                category = raw_feature[len(prefix) :]
                label = FEATURE_METADATA[candidate]["label"]
                return f"{label}: {category}"

    return humanize_slug(raw_feature)


def explain_top_risk_factors(
    input_data: Mapping[str, Any],
    model: Any,
    preprocessor: Any,
    features: list[str] | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Approximate the strongest logistic-regression contributors for one prediction."""
    patient = coerce_and_validate_patient_payload(input_data)
    frame = build_feature_frame(patient)
    transformed = preprocessor.transform(frame)
    dense_vector = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    coefficients = np.asarray(model.coef_).ravel()
    transformed_features = features or list(preprocessor.get_feature_names_out())

    rows = []
    for feature_name, value, coefficient in zip(
        transformed_features, dense_vector, coefficients, strict=False
    ):
        contribution = float(value * coefficient)
        if feature_name.startswith("categorical__") and abs(value) < 1e-12:
            continue
        rows.append(
            {
                "feature": _humanize_transformed_feature(feature_name, patient),
                "transformed_feature": feature_name,
                "coefficient": float(coefficient),
                "contribution": contribution,
                "direction": "raises risk" if contribution >= 0 else "reduces risk",
            }
        )

    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return rows[:top_n]


def predict_single(
    patient_data: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    """Run inference for a single patient profile."""
    artifacts = dict(bundle) if bundle is not None else load_artifact_bundle(models_dir=models_dir)
    patient = coerce_and_validate_patient_payload(patient_data)
    frame = build_feature_frame(patient)
    transformed = artifacts["preprocessor"].transform(frame)
    probability = float(artifacts["model"].predict_proba(transformed)[0, 1])
    predicted_class = int(artifacts["model"].predict(transformed)[0])

    feature_config = artifacts.get("feature_config", {})
    transformed_features = feature_config.get("transformed_feature_columns")

    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "risk_category": get_risk_category(probability),
        "important_features": explain_top_risk_factors(
            patient,
            artifacts["model"],
            artifacts["preprocessor"],
            features=transformed_features,
        ),
        "validated_input": patient,
    }


def predict_batch(
    dataframe: pd.DataFrame,
    bundle: Mapping[str, Any] | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> pd.DataFrame:
    """Run inference for a batch of patient records."""
    artifacts = dict(bundle) if bundle is not None else load_artifact_bundle(models_dir=models_dir)
    records = [coerce_and_validate_patient_payload(record) for record in dataframe.to_dict(orient="records")]
    frame = pd.DataFrame(records)[FEATURE_COLUMNS]
    transformed = artifacts["preprocessor"].transform(frame)
    probabilities = artifacts["model"].predict_proba(transformed)[:, 1]
    predictions = artifacts["model"].predict(transformed)

    results = frame.copy()
    results["predicted_class"] = predictions.astype(int)
    results["probability"] = probabilities.astype(float)
    results["risk_category"] = [get_risk_category(score) for score in probabilities]
    return results
