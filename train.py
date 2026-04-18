from __future__ import annotations

from datetime import datetime, timezone

import joblib
from sklearn.linear_model import LogisticRegression

from src.data_processing import (
    build_preprocessor,
    clean_dataset,
    load_raw_dataset,
    save_processed_dataset,
    split_dataset,
)
from src.evaluation import evaluate_model
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.utils import MODELS_DIR, ensure_project_directories, save_json


def train() -> dict[str, object]:
    """Train the production logistic regression model and persist deployment artifacts."""
    ensure_project_directories()

    raw_frame = load_raw_dataset()
    cleaned_frame, cleaning_report = clean_dataset(raw_frame)
    processed_path = save_processed_dataset(cleaned_frame)

    x_train, x_test, y_train, y_test = split_dataset(cleaned_frame, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(x_train)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(transformed_train, y_train)

    evaluation_summary = evaluate_model(model, preprocessor, x_test, y_test)
    transformed_features = list(preprocessor.get_feature_names_out())

    feature_config = {
        "selected_feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "transformed_feature_columns": transformed_features,
    }

    metadata = {
        "model_name": "LogisticRegression",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_dataset_path": str(processed_path),
        "raw_rows": int(len(raw_frame)),
        "clean_rows": int(len(cleaned_frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cleaning_report": cleaning_report,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_regression_model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "scaler.joblib")
    save_json(feature_config, MODELS_DIR / "feature_columns.json")
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    save_json(evaluation_summary, MODELS_DIR / "evaluation_summary.json")

    return {
        "metadata": metadata,
        "evaluation": evaluation_summary,
    }


if __name__ == "__main__":
    result = train()
    metrics = result["evaluation"]["metrics"]
    print("Training complete.")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
