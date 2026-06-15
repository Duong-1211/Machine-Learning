# Machine-Learning — Ames Housing Price Prediction

A regression project that predicts house sale prices using the **Ames Housing dataset**. The pipeline supports 7 models, from linear regression to a PyTorch neural network, and generates Kaggle-ready submission CSVs.

## Project Structure

```
Machine-Learning/
├── main.py                          # Entry point: train, predict, generate submission
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw/                         # train.csv, test.csv, sample_submission.csv
│   ├── preprocessed/                # Cleaned & feature-engineered CSVs
│   └── submissions/                 # Output CSVs in Kaggle format
├── src/
│   ├── config.py                    # Paths, model names, default hyperparameters
│   ├── datapreprocessing.py         # Load, clean, impute missing values
│   ├── features.py                  # Feature engineering (TotalSF, HouseAge, etc.)
│   ├── training.py                  # Outlier removal, model building, training & CV
│   ├── prediction.py                # Inference + inverse log transform
│   ├── submission.py                # Save predictions as submission CSV
│   ├── evaluation.py                # RMSE, MAE, R² metrics (log & real space)
│   ├── benchmark.py                 # Cross-model comparison & report generation
│   └── models/
│       ├── regression.py            # Linear / Ridge / Lasso
│       ├── random_forest.py         # Random Forest (500 trees)
│       ├── gradient_boosting.py     # Gradient Boosting
│       ├── xgboost.py               # XGBoost (tuned hyperparameters)
│       └── deep_learning.py         # PyTorch feed-forward network (128→64→1)
└── reports/
    ├── model_benchmark.csv          # Per-model metrics table
    └── model_benchmark_report.txt   # Human-readable benchmark summary
```

## Installation

1. **Clone or download** the repository.
2. Place the competition data (`train.csv`, `test.csv`, `sample_submission.csv`) inside `data/raw/`.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Train a model and generate a submission CSV:

```bash
python main.py --model ridge
```

### Available Models

| CLI name            | Model                        | Library    |
|---------------------|------------------------------|------------|
| `linear`            | Linear Regression            | scikit-learn |
| `ridge`             | Ridge Regression (α=10)      | scikit-learn |
| `lasso`             | Lasso Regression (α=0.001)   | scikit-learn |
| `random_forest`     | Random Forest (500 trees)    | scikit-learn |
| `gradient_boosting` | Gradient Boosting            | scikit-learn |
| `xgboost`           | XGBoost (tuned)              | xgboost    |
| `deep_learning`     | 3-layer NN (128→64→1)        | PyTorch    |

Default model: `ridge`.

### Cross-Validation

```bash
python main.py --model xgboost --cv-folds 5
```

### Deep Learning Epochs

```bash
python main.py --model deep_learning --epochs 100
```

### Benchmark All Models

```bash
python -m src.benchmark
```

This runs all models with 5-fold CV and outputs a comparison table to `reports/`.

## Pipeline

1. **Preprocessing** — Strip whitespace in categoricals; fill numeric missings with median, categorical with mode (or `"None"` for columns where NA means "feature not present").
2. **Feature engineering** — Adds `TotalSF`, `TotalBathrooms`, `HouseAge`, `HasGarage`, `HasFireplace`, interaction terms (`OverallQual_TotalSF`), etc.
3. **Outlier removal** — Drops houses with `GrLivArea > 4000` and `SalePrice < $300k`, or `LotArea > 100k`.
4. **Training** — Log-transforms `SalePrice`; fits model (sklearn pipeline with OHE + scaling); optionally runs k-fold CV.
5. **Prediction** — Runs inference and exponentiates (`np.expm1`) back to dollar scale.
6. **Submission** — Saves `data/submissions/{model}_submission.csv`.

## Dependencies

- pandas, numpy
- scikit-learn
- xgboost
- torch
- joblib

## Results (Benchmark)

| Model              | Eval Log-RMSE | Eval MAE ($) | Eval R² |
|--------------------|---------------|--------------|---------|
| XGBoost            | 0.02358       | $3,008       | 0.9973  |
| Random Forest      | 0.05256       | $6,207       | 0.9856  |
| Gradient Boosting  | 0.07983       | $10,919      | 0.9639  |
| Linear Regression  | 0.09382       | $11,616      | 0.9576  |
| Ridge              | 0.10316       | $12,429      | 0.9525  |
| Lasso              | 0.11094       | $13,101      | 0.9449  |
| Deep Learning      | 0.14237       | $19,234      | 0.8951  |
