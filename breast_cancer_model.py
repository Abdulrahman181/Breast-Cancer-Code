"""Train and evaluate the breast-cancer classifier without inventing local data.

The repository intentionally does not ship the dataset. Run this module with a path to a
CSV containing the five documented feature columns and a binary ``diagnosis`` target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
from sklearn.utils.validation import check_is_fitted

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
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    if not dataset.columns.is_unique:
        duplicates = dataset.columns[dataset.columns.duplicated()].unique().tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")
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
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if dataset[required_columns].isna().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns; impute explicitly first."
        )
    feature_values = dataset.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature columns must contain only finite numeric values.")
    invalid_labels = dataset.loc[
        ~dataset[TARGET_COLUMN].isin((0, 1)), TARGET_COLUMN
    ].unique()
    if len(invalid_labels):
        found = ", ".join(repr(value) for value in invalid_labels)
        raise ValueError(
            f"{TARGET_COLUMN} must contain only binary 0/1 labels; found [{found}]"
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
    check_is_fitted(model)
    if len(x_test) != len(y_test):
        raise ValueError("x_test and y_test must contain the same number of rows")
    if len(y_test) == 0 or y_test.nunique() != 2:
        raise ValueError("y_test must contain both diagnosis classes for ROC AUC evaluation")
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
    if isinstance(test_size, bool) or not isinstance(test_size, Real):
        raise ValueError("test_size must be a numeric value strictly between 0 and 1")
    test_size = float(test_size)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a numeric value strictly between 0 and 1")
    test_rows = math.ceil(len(dataset) * test_size)
    train_rows = len(dataset) - test_rows
    if test_rows < 2 or train_rows < 2:
        raise ValueError(
            "test_size must leave at least two rows in both train and test partitions "
            "for binary stratified evaluation"
        )
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
    check_is_fitted(model)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def _fraction(value: str) -> float:
    """Parse a CLI fraction and reject values that cannot form a valid split."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number strictly between 0 and 1") from exc
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be a number strictly between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV path; not bundled with this repository")
    parser.add_argument(
        "--model-out",
        default="artifacts/breast_cancer_pipeline.joblib",
        help="Output path for the fitted pipeline",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--test-size", type=_fraction, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_dataset(args.data)
        model, metrics = train_and_evaluate(
            dataset,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        save_model(model, args.model_out)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"metrics": metrics, "model_path": str(args.model_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
