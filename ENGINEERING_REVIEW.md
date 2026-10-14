# Engineering Review — Breast-Cancer-Code

## Scope

Audited the complete tracked repository at commit `c109305` (one notebook), including notebook code, embedded outputs, data references, dependencies, and Git state. No other repository was modified and no remote push was performed.

## Baseline findings

- The repository contained only `Breast Cancer Code.ipynb`; the referenced `Breast_cancer_data.csv` was absent.
- The notebook had stale executed outputs (including rendered tables/plots), execution counts, and an embedded 569-row snapshot. Those outputs were not treated as reproducible evidence.
- The notebook used a machine-relative data path and wrote `breastcancer.pkl`/`sc.pkl` into the working directory without an artifact policy.
- The inference cell called `classifier[0]` after `classifier.predict(...)`. `RandomForestClassifier` is not subscriptable; this would fail before the intended message.
- The original split used a fixed seed but was not stratified, so class proportions were not explicitly preserved.
- Scaling was fit after the split, so the current sequence did not directly fit the scaler on test rows; it was nevertheless separated from the estimator and easy to misuse. The repaired code keeps preprocessing and model fitting in one pipeline.
- There was no declared dependency set, README, test/validator coverage, or ignore policy.
- No Python syntax errors were found in the original code cells by AST parsing, but Ruff identified unused-expression cells, unsorted imports, and unsafe `open()` calls around pickle writes.

## Repairs

- Added `breast_cancer_model.py` with:
  - explicit required schema and binary-label validation;
  - clear missing-file/read errors;
  - stratified 75/25 split with fixed seed;
  - `StandardScaler` + 200-tree random forest in one leakage-safe pipeline;
  - held-out accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and confusion matrix calculation;
  - deterministic artifact output using `joblib` and an explicit output path.
- Replaced the notebook with a clean version using the shared module. All code-cell outputs are empty and execution counts are unset; no stale metrics are retained.
- Added `README.md`, `requirements.txt`, `.gitignore`, `pyproject.toml`, and unit tests under `tests/`.
- Tests cover missing-data failure messaging, schema validation, pipeline ordering, notebook AST parseability, no stale outputs, and no machine-specific absolute paths.

## Validation performed

- `ruff check .` — passed after repairs.
- `python3 -m compileall -q breast_cancer_model.py tests` — passed.
- `python3 -m unittest discover -s tests -v` — 5 tests passed.
- Notebook JSON parsing and integrity check — 17 cells, 11 code cells, 0 outputs, 0 executed code cells.
- The training CLI was not run end-to-end because the repository has no dataset. No metrics or model artifact are claimed in this review.

## Remaining limitation

Supply a legitimate CSV matching the documented schema before training. This review does not validate the provenance, clinical suitability, external generalization, calibration, subgroup performance, or statistical uncertainty of any future model run. The project is not a medical device or diagnostic recommendation.
