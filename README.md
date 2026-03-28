# 🏥 Patient Readmission Prediction Pipeline

> Predicting 30-day hospital readmission for diabetic patients using an end-to-end ML pipeline with SHAP explainability and fairness auditing.

## Clinical Motivation
Hospital readmissions within 30 days are a major driver of healthcare costs and point to potential failures in care transitions. This pipeline aims to proactively identify diabetic patients at high risk of rapid readmission at the time of their discharge, enabling targeted interventions, better resource allocation, and ultimately improved patient outcomes.

## Dataset

**Kaggle Diabetic Readmission Dataset** — 101,766 encounters × 50 features, from 130 US hospitals (1999–2008).

- Download from Kaggle: `diabetes+130-us+hospitals+for+years+1999-2008`
- Place `diabetic_data.csv` and `IDs_mapping.csv` inside `dataset_diabetes/`

| Metric | Value |
|---|---|
| Rows | 101,766 |
| Columns | 50 |
| Target | `<30` (11.2%), `>30` (34.9%), `NO` (53.9%) — binarized to `<30` vs rest |

## Setup & Reproducibility

1. **Clone & enter project**:
   ```bash
   git clone https://github.com/Trupal25/patient-readmission-diabetes.git
   cd patient-readmission-diabetes
   ```
2. **Create virtual environment and install dependencies**:
   ```bash
   make setup
   source .venv/bin/activate
   ```
3. **Execute the full pipeline end-to-end**:
   ```bash
   make all      # Runs preprocess, model training, and evaluation
   ```
4. **Run the testing suite**:
   ```bash
   make test
   ```
5. **Launch the Dashboard**:
   ```bash
   make dashboard
   ```

## Key Results

After rigorous cross-validation and hyperparameter optimization via `Optuna`, the models were evaluated against a strictly held-out 15% testing set.

| Model | AUROC (Test) | AUPRC (Test) | Notes |
|---|---|---|---|
| Logistic Regression | 0.638 | 0.153 | Baseline model — Highly interpretable feature weights. |
| **XGBoost (Tuned)** | **0.641** | **0.155** | Primary model — Better precision-recall tradeoff handling class imbalance with `scale_pos_weight`. |

*(Note: Random baseline AUPRC for this dataset is ~0.09. Our models substantially outperform random chance in precision-recall space).*

### Top Predictors
SHAP explainability and Logistic Regression coefficients highlight the following dominant features driving readmission risk:
1. **Discharge Disposition**: Patients diverted to a Skilled Nursing Facility (SNF) or Transfer locations show dramatically elevated readmission risks.
2. **Prior Utilization**: Historical counts of inpatient and emergency visits are heavy structural indicators of fragility.
3. **Admission Source**: Specifically transfers from other hospitals/facilities.
4. **Primary Diagnoses**: Certain specific clusters (like Neoplasms and complex circulatory issues) correlated strongly with return visits.

### Subgroup Fairness Audit
We evaluated both Equalized Odds (True Positive Rate gap) and Predictive Parity (Positive Predictive Value gap) across Race, Gender, and Age demographics.
- **Result**: **PASS**. All primary subgroup distributions (where n > 100) exhibited TPR and PPV disparity gaps of `< 5%`. 
- **Caveat**: The extremely elderly bracket `[90-100)` suffered a drop in baseline AUROC accuracy (< 0.60) largely due to statistical sparsity and complex end-of-life morbidity baselines.

## Limitations & Future Work
- **Temporal constraints**: Because the dataset lacks explicit timestamp columns, a stratified random split was used instead of a true temporal (rolling window) split.
- **Deep Sequence Modeling**: Utilizing LSTM tracking for sequential lab/vitals trajectories was deemed a stretch goal requiring richer EHR context (like MIMIC-IV).
- **Clinical Notes NLP**: Extracting sentiment or deterioration markers from free-text physician discharge notes via ClinicalBERT could vastly improve AUROC in future iterations.