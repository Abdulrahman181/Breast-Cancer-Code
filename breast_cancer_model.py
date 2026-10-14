"""Train and evaluate the breast-cancer classifier without inventing local data.

The repository intentionally does not ship the dataset. Run this module with a path to a
CSV containing the five documented feature columns and a binary ``diagnosis`` target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = (
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
)
TARGET_COLUMN = "diagnosis"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the requested CSV or fail with an actionable message."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Provide a CSV via --data; "
            "the repository does not include patient data."
        )
    try:
        dataset = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV {csv_path}: {exc}") from exc
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Validate schema and labels before any split or model fitting."""
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if dataset.empty:
        raise ValueError("Dataset is empty; provide rows with measured features and diagnosis.")
    non_numeric = [
        column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {', '.join(non_numeric)}")
    if dataset[list(required)].isna().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns; impute explicitly first."
        )
    labels = set(dataset[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN} must contain only binary 0/1 labels; found {sorted(labels)}"
        )
    if dataset[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Dataset must contain both diagnosis classes for a stratified evaluation.")
    if dataset[TARGET_COLUMN].value_counts().min() < 2:
        raise ValueError(
            "Each diagnosis class must contain at least two rows for a stratified split."
        )


def build_pipeline(random_state: int = 0) -> Pipeline:
    """Build the estimator; preprocessing is fit only on the training partition."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    criterion="entropy",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Return test metrics only; callers must fit the model on training data first."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": int(len(y_test)),
    }


def train_and_evaluate(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
) -> tuple[Pipeline, dict[str, Any]]:
    """Split with stratification, fit the pipeline, and evaluate once on held-out data."""
    validate_dataset(dataset)
    x = dataset.loc[:, FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model = build_pipeline(random_state=random_state)
    model.fit(x_train, y_train)
    return model, evaluate_model(model, x_test, y_test)


def save_model(model: Pipeline, path: str | Path) -> None:
    """Persist a fitted pipeline, creating its parent directory when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV path; not bundled with this repository")
    parser.add_argument(
        "--model-out",
        default="artifacts/breast_cancer_pipeline.joblib",
        help="Output path for the fitted pipeline",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    dataset = load_dataset(args.data)
    model, metrics = train_and_evaluate(
        dataset,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(model, args.model_out)
    print(json.dumps({"metrics": metrics, "model_path": str(args.model_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
