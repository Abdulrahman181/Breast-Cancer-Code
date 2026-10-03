# Breast-cancer classification (educational example)

This repository is a small, reproducible **educational** example of a Random Forest classifier. It is **not a medical device, diagnostic tool, or clinically validated model**. Do not use its code, predictions, or metrics for screening, diagnosis, treatment, triage, or decisions about an individual. No external validation, clinical utility, or performance claim is made.

## Data and provenance

The repository does **not** include a dataset. The original source does not establish dataset provenance, authorization, license, or the meaning of the `diagnosis` class values. Obtain data only from a source you are authorized to use; independently verify its source, license, and label definitions. The code deliberately does not infer that a class means “benign” or “malignant.” Do not commit patient, personal, confidential, or otherwise sensitive data.

Provide a CSV with the following required columns; extra columns are ignored:

- Features, in this order: `mean_radius`, `mean_texture`, `mean_perimeter`, `mean_area`, `mean_smoothness`
- Target: `diagnosis`, with at least two non-missing classes

Features must be numeric and finite. The loader rejects missing required values, duplicate headers, and incompatible inputs. By default the notebook looks for `Breast_cancer_data.csv` in the current working directory. Set `BREAST_CANCER_DATA` to an authorized CSV path outside the repository to use another location.

## Setup and run

Use Python 3.10, 3.11, or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab "Breast Cancer Code.ipynb"
```

The notebook validates inputs, then makes deterministic, stratified, disjoint train/validation/test partitions (approximately 65%/15%/20%; exact row counts depend on dataset size and class proportions). It fits the fixed model on training rows only, reports aggregate metrics on validation rows, then evaluates the held-out test partition once. The validation partition is available for future model selection; this example does not tune on it. If you change modeling choices after viewing test results, that test partition is no longer an unbiased final evaluation. Small or imbalanced datasets may not support all three stratified partitions and will be rejected with an actionable error.

The split is row-wise because the repository does not define patient, site, or time identifiers. If rows are repeated measurements or otherwise grouped, related samples could cross partitions; use an appropriate group/time split and independent external validation for a real research study. This repository cannot verify whether the supplied data has such structure.

The notebook's metrics depend on the local dataset, its verified labels, and the chosen split; they are not evidence of generalization or clinical effectiveness. No individual records or paired predictions are printed. The notebook is not run end-to-end in a clean checkout because the dataset is not supplied.

## Reusable code, artifacts, and tests

`breast_cancer_model.py` contains input validation, deterministic splitting, training/evaluation, and artifact helpers. The fitted estimator is a Random Forest pipeline; feature scaling is omitted because it is not needed by this estimator. Model fitting receives only the training partition. Artifacts include an explicit feature/target schema and runtime version metadata and are atomically written to `models/breast_cancer_model.joblib` by default with mode `0600` on POSIX systems; on Windows, effective protection follows the destination directory's filesystem ACLs. Set `BREAST_CANCER_ARTIFACT` to change the local output path. Dataset and model artifacts are ignored by Git.

**Only load artifacts you created or fully trust.** `joblib`/pickle deserialization can execute code. Artifacts are local educational outputs, are not a secure model-serving format, and should not be shared as if they were clinically approved.

To run tests and static checks (tests use generated toy rows only, not a supplied or clinical dataset):

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q breast_cancer_model.py tests
python -m ruff check breast_cancer_model.py tests
python -m pytest
```

GitHub Actions runs these checks on Python 3.10, 3.11, and 3.12. Dependency ranges are intentionally bounded but are not a fully resolved lockfile; exact environments may vary.
