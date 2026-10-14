# Breast Cancer Classification

This repository contains a small scikit-learn workflow for binary breast-cancer diagnosis. It is an engineering repair of the original notebook, not a clinical system.

## Data

The dataset is **not included** in this repository. No metrics or model artifacts are claimed until a user supplies a legitimate CSV. The training command stops with an actionable error when the file is absent.

Expected columns:

- `mean_radius`
- `mean_texture`
- `mean_perimeter`
- `mean_area`
- `mean_smoothness`
- `diagnosis` — binary labels encoded as `0` and `1`

Place private data outside Git (for example, `data/Breast_cancer_data.csv`; `*.csv` is ignored) and run from the repository root:

```bash
python breast_cancer_model.py \
  --data data/Breast_cancer_data.csv \
  --model-out artifacts/breast_cancer_pipeline.joblib
```

The command prints metrics for the held-out test partition and writes one fitted pipeline. It does not download data, fabricate rows, or claim reproducibility without the input data and dependency versions.

## Methodology

- The split is a **stratified** 75/25 train/test split with `random_state=0` by default.
- `StandardScaler` and the classifier are in one scikit-learn `Pipeline`; the scaler is fitted only on the training partition, preventing preprocessing leakage.
- The test partition is used once for reporting accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and a confusion matrix. These are evaluation outputs, not published benchmark claims.
- The estimator is a 200-tree `RandomForestClassifier` with a fixed seed. A fixed seed improves repeatability but is not a substitute for external validation or uncertainty analysis.

## Notebook

`Breast Cancer Code.ipynb` is intentionally clean: code cells have no stale outputs or execution counts. It uses the same validation and pipeline helpers as the CLI and therefore also stops clearly when the expected CSV is unavailable. Run it only after supplying data.

## Development checks

Install runtime and test dependencies, then run checks from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q breast_cancer_model.py tests
```

The repository does not commit CSV files, pickles, joblib files, or generated output. Do not use this project for diagnosis or treatment decisions.
