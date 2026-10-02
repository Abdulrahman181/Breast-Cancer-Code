# Breast Cancer Classification (educational notebook)

This repository contains a Jupyter notebook illustrating a Random Forest classification workflow over five numeric input features. It is an educational example only, **not a medical device or diagnostic tool**. Do not use its outputs for screening, diagnosis, treatment, or decisions about an individual.

## Data required

The notebook expects `Breast_cancer_data.csv` in the repository root. The CSV is **not included**. The original repository does not document the dataset's source, license, provenance, or the meaning/mapping of values in `diagnosis`; these details cannot be verified here. Obtain the data only from a source you are authorized to use, and verify its license and class definitions yourself. Do not commit personal or otherwise sensitive data.

The notebook explicitly selects these columns, in this order:

1. `mean_radius`
2. `mean_texture`
3. `mean_perimeter`
4. `mean_area`
5. `mean_smoothness`

It also requires a `diagnosis` target column with at least two classes. The notebook validates these requirements before training. The class meanings are intentionally not inferred or labeled as “benign” or “malignant,” because the repository does not establish their mapping.

## Setup and run

Use Python 3.10 or 3.11 in a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab "Breast Cancer Code.ipynb"
```

Place your authorized `Breast_cancer_data.csv` in the repository root before running the notebook from top to bottom. `requirements.txt` constrains compatible package ranges, but is not a fully resolved lockfile; exact environments may vary within those ranges.

The notebook uses a stratified holdout split (`random_state=0`) and displays a confusion matrix, per-class report, and accuracy for that split. These are educational, dataset-specific results only. The dataset is absent here, so the notebook cannot be run end to end from a clean clone and no performance claim is validated by this repository change. Previously saved notebook outputs were cleared because they did not correspond reliably to the current source and could not be reproduced from the repository.

## Generated files and safety

Running the final cells writes `breastcancer.pkl` and `sc.pkl` locally; these artifacts are ignored by Git. Only load pickle files you created or otherwise fully trust: unpickling data from an untrusted source can execute code. The notebook does not load pickle files.
