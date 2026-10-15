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

## Second Senior Review

### Scope and findings

Reviewed every tracked file, the complete notebook JSON/code state, the previous commit
(`532ee2e`), the original commit, and the current Git diff. The first repair was sound, but
the following residual engineering issues were found:

- Feature validation accepted `inf`/`-inf`, which can fail later inside scikit-learn with a
  less actionable error.
- Mixed-type invalid labels could raise a `TypeError` while formatting the error because
  Python cannot sort unlike label types.
- `train_and_evaluate` delegated invalid or too-small `test_size` values to
  `train_test_split`, producing version-dependent low-level exceptions or a one-row test
  partition where binary ROC AUC is impossible.
- `evaluate_model` did not explicitly require a fitted model, matching test lengths, or both
  classes in the test labels; `save_model` could persist an unfitted pipeline.
- Development tools used by the documented checks were not declared separately from runtime
  dependencies.

No API/UI code, secrets, tracked data, or model artifact exists in this repository. The
notebook remains intentionally unexecuted because the required private CSV is absent; no
metrics or end-to-end training claim is made.

### Repairs in this review

- Added duplicate-column, finite-feature, DataFrame-type, and robust mixed-label validation.
- Added a clear binary stratified split-size contract and explicit evaluation preconditions.
- Made artifact saving require a fitted estimator and added tests for nested artifact paths.
- Added `requirements-dev.txt` for pytest, Ruff, and notebook validation tooling, and documented
  the artifact trust boundary plus the lack of an API/UI.
- Added unit coverage for non-finite features, invalid labels, small partitions, real training,
  metrics, and artifact persistence.

### Second-review validation

- `ruff check .` — passed.
- `python3 -m compileall -q .` — passed.
- `python3 -m pytest -q` — passed (9 tests).
- Notebook JSON/AST integrity — 17 cells, 11 code cells, 0 outputs, all execution counts null.
- Direct training smoke test on an in-memory synthetic frame — passed only to validate code
  paths; it is not a project metric and is not a substitute for the missing legitimate data.

### Remaining risks

The CSV is still absent, so real-data training, artifact reload in a deployed process, clinical
validity, external generalization, calibration, subgroup performance, and uncertainty remain
unverified. `joblib` artifacts must be treated as trusted code. Dependency ranges improve
installation compatibility but do not pin a lockfile; exact environment reproduction still
requires recording the installed versions.
