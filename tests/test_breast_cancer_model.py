from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from breast_cancer_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_model,
    fit_model,
    load_dataset,
    load_model_artifact,
    predict_one,
    save_model_artifact,
    split_dataset,
)


def toy_frame(rows_per_class: int = 30) -> pd.DataFrame:
    """Generate deterministic non-clinical rows for unit tests only."""
    count = rows_per_class * 2
    values = np.arange(count, dtype=float)
    labels = np.repeat([0, 1], rows_per_class)
    frame = pd.DataFrame(
        {
            "mean_radius": values + 0.1,
            "mean_texture": values + 1.1,
            "mean_perimeter": values + 2.1,
            "mean_area": values + 3.1,
            "mean_smoothness": values + 4.1,
            "diagnosis": labels,
            "unneeded_column": "ignored",
        }
    )
    return frame


def test_load_dataset_selects_named_features_and_rejects_invalid_schema(tmp_path):
    path = tmp_path / "toy.csv"
    frame = toy_frame()
    frame.to_csv(path, index=False)

    features, target = load_dataset(path)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(target) == len(frame)
    assert target.name == TARGET_COLUMN
    assert "unneeded_column" not in features

    missing = frame.drop(columns=[FEATURE_COLUMNS[0]])
    missing.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)

    path.write_text(
        "mean_radius,mean_radius,mean_perimeter,mean_area,mean_smoothness,diagnosis\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        load_dataset(path)


def test_load_dataset_rejects_missing_non_numeric_and_non_finite_values(tmp_path):
    path = tmp_path / "toy.csv"
    frame = toy_frame()

    for value, message in [(np.nan, "must not contain missing"), ("not-a-number", "numeric")]:
        invalid = frame.copy()
        invalid[FEATURE_COLUMNS[0]] = invalid[FEATURE_COLUMNS[0]].astype(object)
        invalid.loc[0, FEATURE_COLUMNS[0]] = value
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError, match=message):
            load_dataset(path)

    invalid = frame.copy()
    invalid.loc[0, FEATURE_COLUMNS[0]] = np.inf
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite"):
        load_dataset(path)

    invalid = frame.copy()
    invalid.loc[0, TARGET_COLUMN] = np.nan
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Target column"):
        load_dataset(path)


def test_load_dataset_rejects_missing_file_and_single_class_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does-not-exist.csv")

    path = tmp_path / "one-class.csv"
    frame = toy_frame()
    frame[TARGET_COLUMN] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least two classes"):
        load_dataset(path)


def test_split_is_deterministic_stratified_and_disjoint():
    frame = toy_frame()
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]

    first = split_dataset(features, target)
    second = split_dataset(features, target)

    assert first.X_train.equals(second.X_train)
    assert first.X_validation.equals(second.X_validation)
    assert first.X_test.equals(second.X_test)
    row_ids = [
        set(part["mean_radius"].tolist())
        for part in (first.X_train, first.X_validation, first.X_test)
    ]
    assert row_ids[0].isdisjoint(row_ids[1])
    assert row_ids[0].isdisjoint(row_ids[2])
    assert row_ids[1].isdisjoint(row_ids[2])
    assert len(first.X_train) + len(first.X_validation) + len(first.X_test) == len(frame)
    for target_part in (first.y_train, first.y_validation, first.y_test):
        assert set(target_part.unique()) == {0, 1}


def test_split_rejects_misaligned_or_too_small_data():
    frame = toy_frame(rows_per_class=2)
    with pytest.raises(ValueError, match="at least three rows"):
        split_dataset(frame.loc[:, FEATURE_COLUMNS], frame[TARGET_COLUMN])
    with pytest.raises(ValueError, match="row counts must match"):
        split_dataset(frame.loc[:, FEATURE_COLUMNS], frame[TARGET_COLUMN].iloc[:-1])


def test_fit_evaluate_and_artifact_round_trip(tmp_path):
    frame = toy_frame()
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    splits = split_dataset(features, target)
    model = fit_model(splits.X_train, splits.y_train, n_estimators=15)

    metrics = evaluate_model(model, splits.X_test, splits.y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
    assert set(metrics["classification_report"]) >= {"0", "1", "accuracy"}

    artifact_path = save_model_artifact(model, tmp_path / "nested" / "model.joblib")
    assert artifact_path.exists()
    if os.name == "posix":
        assert artifact_path.stat().st_mode & 0o777 == 0o600
    artifact = load_model_artifact(artifact_path)
    assert artifact["feature_columns"] == list(FEATURE_COLUMNS)
    assert artifact["target_column"] == TARGET_COLUMN
    prediction = predict_one(
        artifact,
        {name: float(splits.X_test.iloc[0][name]) for name in FEATURE_COLUMNS},
    )
    assert prediction in {0, 1}


def test_predict_one_validates_feature_schema_and_values(tmp_path):
    frame = toy_frame()
    model = fit_model(frame.loc[:, FEATURE_COLUMNS], frame[TARGET_COLUMN], n_estimators=5)
    artifact = {
        "schema_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "model": model,
    }
    valid = {name: 1.0 for name in FEATURE_COLUMNS}

    with pytest.raises(ValueError, match="missing=.*mean_smoothness"):
        incomplete = {key: value for key, value in valid.items() if key != "mean_smoothness"}
        predict_one(artifact, incomplete)
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        predict_one(artifact, {**valid, "unexpected": 2.0})
    with pytest.raises(ValueError, match="numeric"):
        predict_one(artifact, {**valid, "mean_radius": "not numeric"})
    with pytest.raises(ValueError, match="finite"):
        predict_one(artifact, {**valid, "mean_radius": float("nan")})
    with pytest.raises(ValueError, match="schema"):
        predict_one({**artifact, "schema_version": 99}, valid)


def test_load_model_artifact_rejects_wrong_schema(tmp_path):
    frame = toy_frame()
    model = fit_model(frame.loc[:, FEATURE_COLUMNS], frame[TARGET_COLUMN], n_estimators=5)
    artifact_path = save_model_artifact(model, tmp_path / "model.joblib")

    # Keep the test artifact trusted and locally generated; this is not a
    # security test for untrusted pickle/joblib deserialization.
    import joblib

    artifact = joblib.load(artifact_path)
    artifact["schema_version"] = 999
    joblib.dump(artifact, artifact_path)
    with pytest.raises(ValueError, match="schema is incompatible"):
        load_model_artifact(artifact_path)
