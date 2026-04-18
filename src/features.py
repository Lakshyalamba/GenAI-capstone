from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.utils import humanize_slug


TARGET_COLUMN = "risk"

NUMERIC_FEATURES = [
    "age",
    "systolic_bp",
    "cholesterol",
    "max_heart_rate",
    "bmi",
]

CATEGORICAL_FEATURES = [
    "sex",
    "chest_pain",
    "smoker",
    "diabetes",
    "exercise_angina",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALLOWED_CATEGORIES = {
    "sex": ["Male", "Female"],
    "chest_pain": ["typical", "atypical", "non-anginal", "asymptomatic"],
    "smoker": ["Yes", "No"],
    "diabetes": ["Yes", "No"],
    "exercise_angina": ["Yes", "No"],
}

FEATURE_METADATA = {
    "age": {
        "label": "Age",
        "kind": "numeric",
        "min": 18,
        "max": 100,
        "step": 1.0,
        "default": 55.0,
        "unit": "years",
    },
    "systolic_bp": {
        "label": "Systolic Blood Pressure",
        "kind": "numeric",
        "min": 80,
        "max": 250,
        "step": 1.0,
        "default": 132.0,
        "unit": "mmHg",
    },
    "cholesterol": {
        "label": "Total Cholesterol",
        "kind": "numeric",
        "min": 100,
        "max": 400,
        "step": 1.0,
        "default": 220.0,
        "unit": "mg/dL",
    },
    "max_heart_rate": {
        "label": "Max Heart Rate",
        "kind": "numeric",
        "min": 60,
        "max": 220,
        "step": 1.0,
        "default": 145.0,
        "unit": "bpm",
    },
    "bmi": {
        "label": "BMI",
        "kind": "numeric",
        "min": 10,
        "max": 60,
        "step": 0.1,
        "default": 28.0,
        "unit": "kg/m²",
    },
    "sex": {
        "label": "Sex",
        "kind": "categorical",
        "options": ALLOWED_CATEGORIES["sex"],
        "default": "Male",
    },
    "chest_pain": {
        "label": "Chest Pain Type",
        "kind": "categorical",
        "options": ALLOWED_CATEGORIES["chest_pain"],
        "default": "non-anginal",
    },
    "smoker": {
        "label": "Current Smoker",
        "kind": "categorical",
        "options": ALLOWED_CATEGORIES["smoker"],
        "default": "No",
    },
    "diabetes": {
        "label": "Diabetes",
        "kind": "categorical",
        "options": ALLOWED_CATEGORIES["diabetes"],
        "default": "No",
    },
    "exercise_angina": {
        "label": "Exercise-induced Angina",
        "kind": "categorical",
        "options": ALLOWED_CATEGORIES["exercise_angina"],
        "default": "No",
    },
}


_CATEGORY_NORMALIZATION = {
    feature: {option.lower(): option for option in options}
    for feature, options in ALLOWED_CATEGORIES.items()
}


def normalize_categorical_value(feature_name: str, value: Any) -> str | None:
    """Normalize categorical values into the project vocabulary."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    candidate = str(value).strip().lower()
    return _CATEGORY_NORMALIZATION.get(feature_name, {}).get(candidate, str(value).strip())


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the selected feature columns in model order."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    return frame[FEATURE_COLUMNS].copy()


def coerce_and_validate_patient_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a single patient payload."""
    missing_fields = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required patient fields: {', '.join(missing_fields)}")

    cleaned: dict[str, Any] = {}
    errors: list[str] = []

    for field in NUMERIC_FEATURES:
        value = payload.get(field)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{FEATURE_METADATA[field]['label']} must be numeric.")
            continue

        minimum = FEATURE_METADATA[field]["min"]
        maximum = FEATURE_METADATA[field]["max"]
        if numeric_value < minimum or numeric_value > maximum:
            errors.append(
                f"{FEATURE_METADATA[field]['label']} must be between {minimum} and {maximum}."
            )
            continue

        cleaned[field] = numeric_value

    for field in CATEGORICAL_FEATURES:
        value = normalize_categorical_value(field, payload.get(field))
        options = ALLOWED_CATEGORIES[field]
        if value not in options:
            errors.append(
                f"{FEATURE_METADATA[field]['label']} must be one of: {', '.join(options)}."
            )
            continue
        cleaned[field] = value

    if errors:
        raise ValueError(" ".join(errors))

    return cleaned


def build_feature_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a validated patient payload into a model-ready dataframe."""
    cleaned = coerce_and_validate_patient_payload(payload)
    return select_model_features(pd.DataFrame([cleaned]))


def format_feature_value(feature_name: str, value: Any) -> str:
    """Format raw feature values for UI rendering."""
    metadata = FEATURE_METADATA[feature_name]
    if metadata["kind"] == "numeric":
        unit = metadata.get("unit", "")
        if float(value).is_integer():
            rendered = f"{int(value)}"
        else:
            rendered = f"{float(value):.1f}"
        return f"{rendered} {unit}".strip()
    return str(value)


def derive_risk_signals(patient_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create human-readable clinical-style signals to drive retrieval and guidance."""
    patient = coerce_and_validate_patient_payload(patient_data)
    signals: list[dict[str, Any]] = []

    def add_signal(identifier: str, label: str, severity: str, value: Any, topics: list[str]) -> None:
        signals.append(
            {
                "id": identifier,
                "label": label,
                "severity": severity,
                "value": value,
                "topics": topics,
            }
        )

    if patient["age"] >= 60:
        add_signal("age", "Older age profile", "moderate", patient["age"], ["preventive", "follow-up"])
    if patient["systolic_bp"] >= 140:
        add_signal(
            "blood_pressure",
            "Elevated systolic blood pressure",
            "high",
            patient["systolic_bp"],
            ["blood", "pressure", "bp", "warning", "follow-up"],
        )
    elif patient["systolic_bp"] >= 130:
        add_signal(
            "blood_pressure_borderline",
            "Borderline high systolic blood pressure",
            "moderate",
            patient["systolic_bp"],
            ["blood", "pressure", "bp", "monitoring"],
        )

    if patient["cholesterol"] >= 240:
        add_signal(
            "cholesterol",
            "High total cholesterol",
            "high",
            patient["cholesterol"],
            ["cholesterol", "diet", "exercise"],
        )
    elif patient["cholesterol"] >= 200:
        add_signal(
            "cholesterol_borderline",
            "Borderline cholesterol",
            "moderate",
            patient["cholesterol"],
            ["cholesterol", "diet"],
        )

    if patient["bmi"] >= 30:
        add_signal("bmi", "Obesity-range BMI", "high", patient["bmi"], ["diet", "exercise", "weight"])
    elif patient["bmi"] >= 25:
        add_signal(
            "bmi_overweight",
            "Above-ideal BMI",
            "moderate",
            patient["bmi"],
            ["diet", "exercise", "weight"],
        )

    if patient["smoker"] == "Yes":
        add_signal("smoking", "Current smoker", "high", "Yes", ["lifestyle", "warning", "preventive"])

    if patient["diabetes"] == "Yes":
        add_signal("diabetes", "Diabetes present", "high", "Yes", ["diet", "follow-up", "preventive"])

    if patient["exercise_angina"] == "Yes":
        add_signal(
            "exercise_angina",
            "Exercise-induced angina",
            "high",
            "Yes",
            ["warning", "preventive", "follow-up"],
        )

    if patient["chest_pain"] in {"typical", "asymptomatic"}:
        add_signal(
            "chest_pain",
            f"Chest pain pattern: {humanize_slug(patient['chest_pain'])}",
            "moderate" if patient["chest_pain"] == "typical" else "high",
            patient["chest_pain"],
            ["warning", "preventive"],
        )

    if patient["max_heart_rate"] < 120:
        add_signal(
            "heart_rate",
            "Low recorded max heart rate",
            "moderate",
            patient["max_heart_rate"],
            ["exercise", "follow-up"],
        )

    return signals
