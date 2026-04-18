from __future__ import annotations

import pandas as pd
import pytest

from src.inference import load_artifact_bundle, predict_batch, predict_single


def sample_patient() -> dict[str, object]:
    return {
        "age": 58,
        "systolic_bp": 148,
        "cholesterol": 245,
        "max_heart_rate": 132,
        "bmi": 31.4,
        "sex": "Male",
        "chest_pain": "asymptomatic",
        "smoker": "Yes",
        "diabetes": "No",
        "exercise_angina": "No",
    }


def test_model_bundle_loads() -> None:
    bundle = load_artifact_bundle()
    assert bundle["model"] is not None
    assert bundle["preprocessor"] is not None
    assert "selected_feature_columns" in bundle["feature_config"]


def test_predict_single_returns_complete_payload() -> None:
    result = predict_single(sample_patient())
    assert result["predicted_class"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_category"] in {"Low", "Moderate", "High"}
    assert len(result["important_features"]) > 0


def test_predict_batch_scores_multiple_rows() -> None:
    rows = pd.DataFrame(
        [
            sample_patient(),
            {**sample_patient(), "age": 43, "smoker": "No", "systolic_bp": 122, "bmi": 24.0},
        ]
    )
    result = predict_batch(rows)
    assert len(result) == 2
    assert {"predicted_class", "probability", "risk_category"} <= set(result.columns)


def test_bad_input_raises_validation_error() -> None:
    patient = sample_patient()
    patient["age"] = "bad-value"
    with pytest.raises(ValueError):
        predict_single(patient)


def test_missing_model_artifact_behavior(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_artifact_bundle(models_dir=tmp_path)
