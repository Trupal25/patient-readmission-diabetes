# Patient Readmission Prediction Pipeline

Predicting 30-day hospital readmission risk for diabetic patients using an end-to-end machine learning workflow with explainability, subgroup auditing, and a live demo dashboard.

## Capstone Framing

This project is positioned as a healthcare decision-support system that addresses several common hospital challenges:

- rising chronic disease burden
- uncertainty in predicting patient outcomes
- delays in identifying high-risk patients before discharge
- pressure on hospital resources and care-transition planning

The core question is:

**Can structured hospital encounter data be used to identify diabetic patients at high risk of 30-day readmission and support earlier intervention?**

## Objectives Covered

This project directly supports the capstone objectives:

- perform EDA on a real clinical dataset
- identify important health indicators and readmission risk factors
- build machine learning models for patient risk classification
- develop a predictive system that supports healthcare decision-making

Deep learning is intentionally left as future scope because the current dataset is tabular EHR-style data rather than medical imaging or physiological signal data.

## Dataset

This repo uses the public diabetic readmission dataset commonly shared on Kaggle:

- `diabetes+130-us+hospitals+for+years+1999-2008`
- 101,766 encounters
- 50 source variables
- target: readmission within 30 days (`<30`) vs not readmitted within 30 days

Place these files inside `dataset_diabetes/`:

- `diabetic_data.csv`
- `IDs_mapping.csv`

## Project Components

- `notebooks/01_eda.ipynb`: exploratory data analysis and early data understanding
- `src/data/`: loading, cleaning, and train/validation/test splitting
- `src/features/`: ICD grouping, utilization features, medication aggregation, and preprocessing pipeline
- `src/models/`: logistic regression baseline, XGBoost model, and hyperparameter tuning
- `src/evaluation/`: metrics, SHAP analysis, and fairness audit
- `dashboard/app.py`: Streamlit dashboard with held-out patient exploration and manual intake scoring

## Current Workflow

1. Load and validate the public dataset
2. Clean the encounters and remove leakage-prone cases
3. Engineer readmission-relevant features
4. Train baseline and boosted-tree models
5. Evaluate on a held-out test set
6. Interpret the model with SHAP
7. Review subgroup behavior across demographic slices
8. Demonstrate inference in a dashboard

## Setup

```bash
make setup
source .venv/bin/activate
```

## Run The Project

Full pipeline:

```bash
make all
```

Run tests:

```bash
make test
```

Launch dashboard:

```bash
make dashboard
```

## Dashboard Demo

The Streamlit app supports two demo flows:

- inspect an anonymized patient from the held-out test cohort
- score a manually entered patient profile using a short intake form

The manual form is intentionally compact for capstone demos. Any fields not shown in the UI are backfilled from cohort medians or modes so you can demonstrate live scoring without re-creating the full source dataset schema.

## Modeling Summary

Current modeling work includes:

- Logistic Regression as an interpretable baseline
- XGBoost as the primary non-linear model
- SHAP-based feature attribution for global and local explanations

Representative risk drivers in this project include:

- discharge disposition and transfer patterns
- prior inpatient and emergency utilization
- diagnosis groupings
- medication burden and comorbidity load

## Evaluation Notes

The repository includes:

- AUROC, AUPRC, calibration, confusion matrix, and thresholded classification metrics
- SHAP summary plots and single-patient explanations
- subgroup audit outputs for race, gender, and age

Fairness outputs should be presented as an audit, not as a blanket claim of fairness across all groups. Small subgroup sizes and threshold choice can materially affect those results.

## Submission-Ready Talking Points

- clear healthcare motivation tied to readmission risk and resource planning
- end-to-end ML workflow from raw data to dashboard
- interpretable outputs suitable for a capstone demo
- realistic scope for a short internship submission

## Limitations And Future Work

- the dataset does not support a true temporal split with explicit event timestamps
- the current system is based on structured tabular data only
- external validation on a second hospital dataset is not included

Future extensions:

- richer hospital intake forms with database-backed persistence
- sequence modeling on longitudinal EHR datasets
- clinical notes NLP on discharge summaries
- deeper fairness and calibration review by subgroup
