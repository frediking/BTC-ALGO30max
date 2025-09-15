Bitcoin Analysis

Purpose

This repository contains experiments, models, and utilities for predicting Bitcoin price movement and returns. It includes a wide set of algorithms (RNN/LSTM/GRU variants, XGBoost, LightGBM, ensemble techniques), feature engineering scripts, hyperparameter tuning, and result analysis.

Project layout

- `*.py` — many scripts and experiments. Core directories and files of interest:
  - `models/` — (expected) saved model artifacts (.pkl, .h5). If missing, look for `.pkl` and `.h5` files across the repo.
  - `data/` — prepared datasets used for training and evaluation (e.g., `Bitcoin_Historical_Data_Enhanced.csv`, `all_stocks_5yr.csv`).
  - `results/` — evaluation outputs and logs (e.g., `results/result.md`, `tune.log`).
  - `rnntune.py`, `rnntuning.py`, `rnn_model_enhanced.py` — RNN/Deep learning experiments.
  - `optimized_lgbm.py`, `optimized_xgb.py`, `xgboost_tuning.py` — tree-based models and tuning.
  - `improved_ensemble.py`, `consolidated_ensemble.py` — ensemble scripts.

Quickstart (reproduce a baseline model run)

1. Create a virtual environment and install dependencies. Example using pip:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

If `requirements.txt` is not present, install these packages (versions should be pinned):
- tensorflow
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- optuna
- joblib

2. Prepare data

Place the raw or prepared CSVs in `data/` (e.g., `Bitcoin_Historical_Data_Enhanced.csv`). Run data prep if needed:

```powershell
python prepare_data.py
```

3. Run a baseline training script

A lightweight wrapper `train_baseline.py` is provided to run `rnntune.py` with safe defaults. Example:

```powershell
python train_baseline.py --data-dir data --trials 20 --output-prefix baseline
```

This expects `data/Xrn_prepared.csv` and `data/y1_prepared.csv` to exist. The wrapper exports simple environment variables that `rnntune.py` can be adapted to read for paths and trial counts.

4. Inspect results

Trained models are saved as `.h5` or `.pkl` files; evaluation metrics and logs are in `results/`.

Project notes and best next steps

- There are 36 algorithm implementation scripts and ~306 saved model artifacts in various formats. Many scripts are experimental; consider consolidating the canonical training pathway.
- Add `requirements.txt` or `environment.yml` to pin dependencies and improve reproducibility.
- Organize experiments under `experiments/` and production code under `src/`.

Contact / Author

Add author and contact information here.
