# 🏥 Patient Readmission Prediction Pipeline

> Predicting 30-day hospital readmission for diabetic patients using an end-to-end ML pipeline with SHAP explainability and fairness auditing.

## Dataset

**Kaggle Diabetic Readmission Dataset** — 101,766 encounters × 50 features, from 130 US hospitals (1999–2008).

- Download from Kaggle: `diabetes+130-us+hospitals+for+years+1999-2008`
- Place `diabetic_data.csv` and `IDs_mapping.csv` inside `dataset_diabetes/`

| Metric | Value |
|---|---|
| Rows | 101,766 |
| Columns | 50 |
| Target | `<30` (11.2%), `>30` (34.9%), `NO` (53.9%) — binarized to `<30` vs rest |

## Setup

```bash
# Clone & enter project
cd patient-readmission

# Create virtual environment and install dependencies
make setup

# Activate environment
source .venv/bin/activate
```

## Reproduce

```bash
make all          # Full pipeline: preprocess → train → evaluate
make test         # Run unit & acceptance tests
make dashboard    # Launch Streamlit explainability dashboard
```

## Project Structure

```
src/
├── data/
│   ├── loader.py       # Raw CSV loading + validation
│   ├── cleaner.py      # Cleaning & dedup & target binarization
│   └── splitter.py     # Stratified train/val/test splits
├── features/
│   ├── engineer.py     # Feature engineering
│   ├── icd_grouper.py  # ICD-9 → clinical categories
│   └── pipeline.py     # sklearn ColumnTransformer pipeline
├── models/
│   ├── baseline.py     # Logistic Regression
│   ├── xgboost_model.py  # XGBoost (primary model)
│   └── trainer.py      # HPO with Optuna + cross-validation
├── evaluation/
│   ├── metrics.py      # AUROC, AUPRC, calibration
│   ├── fairness.py     # Subgroup performance audit
│   └── shap_analysis.py  # SHAP explainability
└── utils/
    ├── config.py       # Constants, paths, feature groups
    └── logger.py       # Structured logging
dashboard/
└── app.py              # Streamlit SHAP dashboard
```

## Results

*To be updated after training runs.*

| Model | AUROC | AUPRC | Notes |
|---|---|---|---|
| Logistic Regression | — | — | Baseline |
| XGBoost (tuned) | — | — | Primary model |

## Limitations & Future Work

- No explicit date column → stratified random split used (temporal split possible with MIMIC-IV)
- ClinicalBERT on discharge notes (stretch goal, requires MIMIC-IV)
- LSTM for sequential lab/vital trajectories (stretch goal)