import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_ids_mapping
from src.evaluation.metrics import compute_optimal_threshold
from src.features.pipeline import get_processed_data, load_modeling_dataframe
from src.utils.config import (
    DASHBOARD_BUNDLE_PATH,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_BINARY_COL,
)

logger = logging.getLogger(__name__)

COHORT_SAMPLE_SIZE = 501
REQUIRED_DASHBOARD_BUNDLE_KEYS = {
    "admission_source_labels",
    "cohort_X",
    "cohort_y",
    "feature_names",
    "form_defaults",
    "form_options",
    "pipeline",
    "threshold",
}


def safe_mode(series: pd.Series) -> Any:
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def build_form_defaults(reference_frame: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in reference_frame.columns:
        if pd.api.types.is_numeric_dtype(reference_frame[column]):
            defaults[column] = float(reference_frame[column].median())
        else:
            defaults[column] = safe_mode(reference_frame[column])
    return defaults


def build_form_options(reference_frame: pd.DataFrame) -> dict[str, list[Any]]:
    return {
        "admission_category": sorted(reference_frame["admission_category"].dropna().unique().tolist()),
        "admission_source_id": sorted(
            reference_frame["admission_source_id"].dropna().astype(int).unique().tolist()
        ),
        "diag_1_group": sorted(reference_frame["diag_1_group"].dropna().unique().tolist()),
        "discharge_category": sorted(reference_frame["discharge_category"].dropna().unique().tolist()),
        "gender": sorted(reference_frame["gender"].dropna().unique().tolist()),
        "medical_specialty_grouped": sorted(
            reference_frame["medical_specialty_grouped"].dropna().unique().tolist()
        ),
        "race": sorted(reference_frame["race"].dropna().unique().tolist()),
    }


def build_admission_source_labels() -> dict[int, str]:
    mappings = load_ids_mapping()
    mapping_df = mappings.get("admission_source_id")
    if mapping_df is None:
        return {}

    return {
        int(row["id"]): row["description"]
        for _, row in mapping_df.iterrows()
    }


def resolve_xgb_model_path() -> Path:
    optimized_path = MODELS_DIR / "xgboost_optimized.joblib"
    if optimized_path.exists():
        return optimized_path

    initial_path = MODELS_DIR / "xgboost_initial.joblib"
    if initial_path.exists():
        return initial_path

    raise FileNotFoundError(
        "No XGBoost model artifact was found in models/. "
        "Expected `models/xgboost_optimized.joblib` or `models/xgboost_initial.joblib`."
    )


def load_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    metrics_path = METRICS_DIR / "model_comparison.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        xgb_metrics = metrics[metrics["Model"].astype(str).str.contains("XGBoost", na=False)]
        if not xgb_metrics.empty and "Opt_Threshold" in xgb_metrics.columns:
            return float(xgb_metrics["Opt_Threshold"].iloc[0])

    return compute_optimal_threshold(
        y_true.astype(np.int_),
        y_prob.astype(np.float64),
    )


def build_dashboard_bundle(cohort_size: int = COHORT_SAMPLE_SIZE) -> dict[str, Any]:
    model = joblib.load(resolve_xgb_model_path())
    reference_frame = load_modeling_dataframe().drop(columns=[TARGET_BINARY_COL]).copy()
    _, _, X_test, _, _, y_test, feature_names, pipeline = get_processed_data("xgb")

    X_test_array = np.asarray(X_test)
    y_test_array = np.asarray(y_test.to_numpy(), dtype=np.int_)
    y_prob = np.asarray(model.predict_proba(X_test_array)[:, 1], dtype=np.float64)
    threshold = load_threshold(y_test_array, y_prob)

    cohort_rows = min(cohort_size, len(X_test_array))
    bundle = {
        "admission_source_labels": build_admission_source_labels(),
        "cohort_X": X_test_array[:cohort_rows],
        "cohort_y": y_test_array[:cohort_rows],
        "feature_names": list(feature_names),
        "form_defaults": build_form_defaults(reference_frame),
        "form_options": build_form_options(reference_frame),
        "pipeline": pipeline,
        "threshold": threshold,
    }

    missing_keys = REQUIRED_DASHBOARD_BUNDLE_KEYS - bundle.keys()
    if missing_keys:
        raise ValueError(f"Dashboard bundle is missing required keys: {sorted(missing_keys)}")

    return bundle


def save_dashboard_bundle(
    output_path: Path = DASHBOARD_BUNDLE_PATH,
    cohort_size: int = COHORT_SAMPLE_SIZE,
) -> Path:
    bundle = build_dashboard_bundle(cohort_size=cohort_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    logger.info("Saved dashboard demo bundle to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    saved_path = save_dashboard_bundle()
    print(saved_path)
