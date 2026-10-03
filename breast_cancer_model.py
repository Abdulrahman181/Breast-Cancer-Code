"""Small, reproducible utilities for the repository's educational notebook.

This module is not a clinical or diagnostic tool. Dataset provenance and target
class meanings are intentionally not inferred by the code.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = (
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
)
TARGET_COLUMN = "diagnosis"
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DataSplits:
    """Disjoint splits; only the training partition should be passed to fit."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the required feature and target columns from a CSV."""
    path = Path(csv_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist or is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            raw_header = next(csv.reader(csv_file), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read dataset CSV header: {path}") from exc
    if not raw_header:
        raise ValueError("Dataset CSV must contain a header row.")
    if len(raw_header) != len(set(raw_header)):
        raise ValueError("Dataset contains duplicate column names.")

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read dataset CSV: {path}") from exc

    if not frame.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")
    required = (*FEATURE_COLUMNS, TARGET_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    try:
        features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All required feature columns must contain numeric values.") from exc
    if features.isna().any().any():
        raise ValueError("Required feature columns must not contain missing values.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Required feature columns must contain only finite values.")

    target = frame[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} must not contain missing values.")
    if target.nunique(dropna=True) < 2:
        raise ValueError(f"Target column {TARGET_COLUMN!r} must contain at least two classes.")
    if target.map(type).nunique() != 1:
        raise ValueError("Target labels must use one consistent value type.")

    return features.astype(np.float64), target.reset_index(drop=True)


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DataSplits:
    """Create deterministic, stratified and disjoint train/validation/test sets."""
    if len(features) != len(target):
        raise ValueError("Feature and target row counts must match.")
    if len(features) == 0:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must each be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size plus test_size must be less than 1.")
    class_counts = target.value_counts(dropna=False)
    if class_counts.size < 2:
        raise ValueError("At least two target classes are required for stratified splitting.")
    if (class_counts < 3).any():
        raise ValueError(
            "Each target class needs at least three rows for disjoint stratified "
            "train/validation/test splits."
        )

    indices = np.arange(len(target))
    try:
        remaining_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=target.to_numpy(),
        )
        relative_validation_size = validation_size / (1 - test_size)
        train_indices, validation_indices = train_test_split(
            remaining_indices,
            test_size=relative_validation_size,
            random_state=random_state,
            stratify=target.iloc[remaining_indices].to_numpy(),
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create stratified train/validation/test splits; provide "
            "more rows per class or adjust the requested split sizes."
        ) from exc

    def select_rows(rows: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        return (
            features.iloc[rows].reset_index(drop=True),
            target.iloc[rows].reset_index(drop=True),
        )

    X_train, y_train = select_rows(train_indices)
    X_validation, y_validation = select_rows(validation_indices)
    X_test, y_test = select_rows(test_indices)
    return DataSplits(X_train, X_validation, X_test, y_train, y_validation, y_test)


def fit_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_estimators: int = 200,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Fit the fixed educational Random Forest pipeline on training rows only."""
    if len(features) != len(target) or len(features) == 0:
        raise ValueError("Training features and labels must have the same nonzero row count.")
    if not isinstance(n_estimators, int) or isinstance(n_estimators, bool) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError(f"Training features must be ordered as {list(FEATURE_COLUMNS)}.")
    if target.isna().any() or target.nunique() < 2:
        raise ValueError("Training labels must be non-missing and contain at least two classes.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Training features must contain only finite numeric values.")

    model = Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=1,
                    class_weight=None,
                ),
            )
        ]
    )
    return model.fit(features, target)


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Return aggregate metrics for an explicitly selected held-out partition."""
    if len(features) != len(target) or len(features) == 0:
        raise ValueError("Evaluation features and labels must have the same nonzero row count.")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError(f"Evaluation features must be ordered as {list(FEATURE_COLUMNS)}.")
    if target.isna().any() or not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Evaluation inputs must be non-missing and finite.")

    predictions = model.predict(features)
    classes = model.named_steps["classifier"].classes_
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=classes).tolist(),
        "classification_report": classification_report(
            target, predictions, labels=classes, output_dict=True, zero_division=0
        ),
    }


def save_model_artifact(model: Pipeline, artifact_path: str | Path) -> Path:
    """Atomically save the estimator and its feature/target schema locally.

    The artifact contains a fitted model and must not be loaded from an
    untrusted source: joblib/pickle deserialization can execute code.
    """
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise ValueError("Expected a fitted Pipeline with a 'classifier' step.")
    try:
        classes = model.named_steps["classifier"].classes_.tolist()
    except AttributeError as exc:
        raise ValueError("The classifier must be fitted before it can be saved.") from exc

    destination = Path(artifact_path).expanduser()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise ValueError("Artifact path must name a file, not a directory.")

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "classes": classes,
        "model": model,
        "runtime_versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
        joblib.dump(artifact, temporary_path, compress=3)
        if os.name == "posix":
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise OSError(f"Could not save model artifact to {destination}") from exc
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination


def load_model_artifact(artifact_path: str | Path) -> dict[str, Any]:
    """Load and schema-check an artifact created locally by this project.

    Never call this on an artifact from an untrusted source; deserialization
    can execute arbitrary code.
    """
    path = Path(artifact_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist or is not a file: {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Could not load trusted model artifact: {path}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a mapping.")
    expected = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("Model artifact schema is incompatible with this code version.")
    model = artifact.get("model")
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise ValueError("Model artifact does not contain the expected fitted pipeline.")
    return artifact


def predict_one(artifact: Mapping[str, Any], values: Mapping[str, Any]) -> Any:
    """Predict one model class after strict feature-name and value validation."""
    if not isinstance(artifact, Mapping):
        raise ValueError("Artifact must be a mapping returned by load_model_artifact.")
    expected_artifact_schema = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
    }
    if any(artifact.get(key) != value for key, value in expected_artifact_schema.items()):
        raise ValueError("Model artifact schema is incompatible with this code version.")
    if not isinstance(values, Mapping):
        raise ValueError("Prediction input must be a mapping of feature names to values.")
    expected = set(FEATURE_COLUMNS)
    provided = set(values)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(str(key) for key in provided - expected)
        raise ValueError(f"Prediction input schema mismatch; missing={missing}, extra={extra}.")
    try:
        row_values = [[float(values[name]) for name in FEATURE_COLUMNS]]
        row = pd.DataFrame(row_values, columns=FEATURE_COLUMNS)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction feature values must be numeric.") from exc
    if not np.isfinite(row.to_numpy(dtype=np.float64)).all():
        raise ValueError("Prediction feature values must be finite.")
    model = artifact.get("model")
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        raise ValueError("Model artifact does not contain the expected fitted pipeline.")
    return model.predict(row)[0]
