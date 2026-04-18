from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    ALLOWED_CATEGORIES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    normalize_categorical_value,
    select_model_features,
)
from src.utils import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


RAW_DATA_FILENAME = "synthetic_health.csv"
PROCESSED_DATA_FILENAME = "cardio_clean.csv"


def resolve_raw_data_path(path: str | Path | None = None) -> Path:
    """Resolve the raw dataset location."""
    if path is not None:
        resolved = Path(path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Raw data file not found: {resolved}")

    candidates = [
        RAW_DATA_DIR / RAW_DATA_FILENAME,
        DATA_DIR / RAW_DATA_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Raw dataset not found. Expected data/raw/synthetic_health.csv or data/synthetic_health.csv."
    )


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw synthetic cardiovascular dataset."""
    return pd.read_csv(resolve_raw_data_path(path))


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw dataset while retaining feature-level missing values for imputation."""
    df = frame.copy()
    report: dict[str, Any] = {"original_rows": int(len(df))}

    df.columns = [str(column).strip() for column in df.columns]
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    rows_missing_target = int(df[TARGET_COLUMN].isna().sum())
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    invalid_category_rows = 0
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].apply(lambda value: normalize_categorical_value(column, value))
        valid_mask = df[column].isin(ALLOWED_CATEGORIES[column])
        invalid_category_rows += int((~valid_mask).sum())
        df = df.loc[valid_mask].copy()

    df = df.reset_index(drop=True)

    report.update(
        {
            "duplicates_removed": duplicates_removed,
            "rows_missing_target_removed": rows_missing_target,
            "invalid_category_rows_removed": invalid_category_rows,
            "final_rows": int(len(df)),
            "missing_feature_counts_after_cleaning": {
                column: int(df[column].isna().sum()) for column in FEATURE_COLUMNS
            },
            "target_distribution": {
                str(label): int(count) for label, count in df[TARGET_COLUMN].value_counts().sort_index().items()
            },
        }
    )
    return df, report


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing graph used by training and inference."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the cleaned dataset for model training and evaluation."""
    model_frame = select_model_features(frame)
    target = frame[TARGET_COLUMN].copy()
    return train_test_split(
        model_frame,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def save_processed_dataset(frame: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist the cleaned dataset for the dashboard and repeatable experiments."""
    output_path = Path(path) if path else PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned dataset, generating it from raw data when needed."""
    resolved_path = Path(path) if path else PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
    if resolved_path.exists():
        return pd.read_csv(resolved_path)

    frame, _ = clean_dataset(load_raw_dataset())
    return frame


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Provide dataset-level metrics for the dashboard overview."""
    return {
        "records": int(len(frame)),
        "features": len(FEATURE_COLUMNS),
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "avg_age": float(frame["age"].mean()),
        "avg_bp": float(frame["systolic_bp"].mean()),
        "avg_cholesterol": float(frame["cholesterol"].mean()),
    }
