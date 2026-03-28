"""
Configuration settings for the patient readmission ML pipeline.

All constants, paths, and feature group definitions live here so
every other module imports from a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset_diabetes"
RAW_DATA_PATH = DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = DATA_DIR / "IDs_mapping.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Ensure output dirs exist at import time (safe, idempotent)
for _d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL = "readmitted"
TARGET_BINARY_COL = "readmitted_30day"

# Map the three-class target to binary: 1 = readmitted within 30 days
TARGET_BINARY_MAP = {"<30": 1, ">30": 0, "NO": 0}

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15   # 15 % held-out test set
VAL_SIZE = 0.15    # 15 % validation set (from remaining train pool)

# ---------------------------------------------------------------------------
# Discharge disposition IDs that indicate patient cannot be readmitted
# (expired, hospice). These rows must be removed to avoid label leakage.
# ---------------------------------------------------------------------------
DECEASED_DISPOSITION_IDS = {11, 13, 14, 19, 20, 21}

# ---------------------------------------------------------------------------
# The 23 individual medication columns present in the raw dataset
# ---------------------------------------------------------------------------
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# ---------------------------------------------------------------------------
# Feature groups (final, post-engineering — used by pipeline.py)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age_midpoint",
    "polypharmacy_score",
    "total_prior_visits",
    "num_active_medications",
    "insulin_intensity",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_category",
    "discharge_category",
    "admission_source_id",
    "medical_specialty_grouped",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
]

BINARY_FEATURES = [
    "change_binary",
    "diabetesMed_binary",
    "has_diabetes_diag",
    "a1c_tested",
    "a1c_abnormal",
    "glu_tested",
    "glu_abnormal",
    "high_utilizer_flag",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ---------------------------------------------------------------------------
# Age midpoint mapping  (used by feature engineer)
# ---------------------------------------------------------------------------
AGE_MIDPOINT_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

# ---------------------------------------------------------------------------
# Insulin intensity ordinal mapping  (used by feature engineer)
# ---------------------------------------------------------------------------
INSULIN_INTENSITY_MAP = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}