import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from breast_cancer_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    load_dataset,
    save_model,
    train_and_evaluate,
    validate_dataset,
)


class ModelValidationTests(unittest.TestCase):
    @staticmethod
    def valid_dataset(rows: int = 8) -> pd.DataFrame:
        values = {column: np.arange(rows, dtype=float) for column in FEATURE_COLUMNS}
        values[TARGET_COLUMN] = [0, 1] * (rows // 2)
        return pd.DataFrame(values)

    def test_missing_dataset_fails_with_actionable_message(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Dataset not found"):
            load_dataset(Path("does-not-exist.csv"))

    def test_missing_schema_column_is_rejected(self) -> None:
        dataset = pd.DataFrame({TARGET_COLUMN: [0, 1]})
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_dataset(dataset)

    def test_pipeline_contains_scaler_before_classifier(self) -> None:
        pipeline = build_pipeline()
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "classifier"])
        self.assertEqual(len(FEATURE_COLUMNS), 5)

    def test_non_finite_features_are_rejected(self) -> None:
        dataset = self.valid_dataset().assign(mean_radius=np.inf)
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            validate_dataset(dataset)

    def test_invalid_labels_have_a_stable_diagnostic(self) -> None:
        dataset = self.valid_dataset()
        dataset[TARGET_COLUMN] = [0, "unknown"] * 4
        with self.assertRaisesRegex(ValueError, "only binary 0/1 labels"):
            validate_dataset(dataset)

    def test_split_contract_rejects_partitions_too_small_for_binary_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            train_and_evaluate(self.valid_dataset(rows=4))

    def test_training_returns_metrics_and_save_requires_fitted_model(self) -> None:
        dataset = self.valid_dataset(rows=12)
        model, metrics = train_and_evaluate(dataset)
        self.assertEqual(metrics["test_rows"], 3)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "model.joblib"
            save_model(model, output)
            self.assertTrue(output.is_file())

        with self.assertRaises(NotFittedError):
            save_model(build_pipeline(), Path("unused.joblib"))


if __name__ == "__main__":
    unittest.main()
