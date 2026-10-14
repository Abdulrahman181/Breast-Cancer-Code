import unittest
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from breast_cancer_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    load_dataset,
    validate_dataset,
)


class ModelValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
