import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Ensure src modules can be imported
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.evaluation.dashboard_bundle import REQUIRED_DASHBOARD_BUNDLE_KEYS
from src.utils.config import (
    AGE_MIDPOINT_MAP,
    ALL_FEATURES,
    DASHBOARD_BUNDLE_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)

AGE_LABELS = list(AGE_MIDPOINT_MAP.keys())
MIDPOINT_TO_AGE_LABEL = {value: key for key, value in AGE_MIDPOINT_MAP.items()}
INSULIN_INTENSITY_LABELS = {0: "No", 1: "Steady", 2: "Down", 3: "Up"}

st.set_page_config(
    page_title="Diabetic Readmission Risk Dashboard",
    layout="wide",
    page_icon="🏥",
)

st.markdown(
    """
<style>
    .reportview-container .main .block-container { max-width: 1200px; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🏥 Patient Readmission Prediction Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "SHAP Explanations",
        "Fairness Audit",
        "Interactive Predictor",
    ],
)
def feature_name_list(feature_names, expected_width: int) -> list[str]:
    if len(feature_names) == expected_width:
        return list(feature_names)
    return [f"Feature_{idx}" for idx in range(expected_width)]


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold * 1.5:
        return "High risk"
    if probability >= threshold:
        return "Watchlist"
    return "Lower risk"


def clipped_default(defaults: dict[str, object], column: str, minimum: int, maximum: int) -> int:
    value = int(round(float(defaults[column])))
    return min(max(value, minimum), maximum)


@st.cache_data
def load_metrics():
    metrics_path = METRICS_DIR / "model_comparison.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


@st.cache_data
def load_fairness():
    fairness_path = METRICS_DIR / "fairness_audit.csv"
    if not fairness_path.exists():
        return None
    return pd.read_csv(fairness_path)


def relative_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def render_missing_artifact_notice(path: Path, label: str):
    st.info(
        f"{label} is unavailable in this deployment because `{relative_artifact_path(path)}` is missing."
    )


def render_artifact_image(path: Path, label: str):
    if path.exists():
        st.image(str(path), width="stretch")
        return
    render_missing_artifact_notice(path, label)


@st.cache_resource
def load_dashboard_demo_bundle():
    if not DASHBOARD_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Interactive predictor assets are missing. Commit "
            f"`{relative_artifact_path(DASHBOARD_BUNDLE_PATH)}` to the repo before deploying."
        )

    bundle = joblib.load(DASHBOARD_BUNDLE_PATH)
    missing_keys = REQUIRED_DASHBOARD_BUNDLE_KEYS - bundle.keys()
    if missing_keys:
        raise ValueError(
            f"Dashboard bundle is incomplete. Missing keys: {sorted(missing_keys)}."
        )
    return bundle


@st.cache_resource
def load_model_bundle():
    xgb_path = MODELS_DIR / "xgboost_optimized.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "xgboost_initial.joblib"
    if not xgb_path.exists():
        raise FileNotFoundError(
            "No XGBoost model artifact was found. Expected "
            "`models/xgboost_optimized.joblib` or `models/xgboost_initial.joblib`."
        )

    model = joblib.load(xgb_path)
    explainer = shap.TreeExplainer(model)
    bundle = load_dashboard_demo_bundle()

    return (
        model,
        explainer,
        bundle["pipeline"],
        np.asarray(bundle["cohort_X"]),
        np.asarray(bundle["cohort_y"]),
        list(bundle["feature_names"]),
        float(bundle["threshold"]),
    )


def score_patient(model, explainer, pipeline, feature_names, patient_frame: pd.DataFrame):
    processed = pipeline.transform(patient_frame)
    probability = float(model.predict_proba(processed)[0, 1])

    processed_frame = pd.DataFrame(
        processed,
        columns=feature_name_list(feature_names, processed.shape[1]),
    )

    shap_values = explainer(processed_frame)
    if len(shap_values.shape) > 2:
        shap_values = shap_values[:, :, 1]

    impacts = np.asarray(shap_values.values)[0]
    contributions = (
        pd.DataFrame(
            {
                "Feature": processed_frame.columns,
                "Encoded Value": processed_frame.iloc[0].values,
                "SHAP Impact": impacts,
            }
        )
        .assign(AbsImpact=lambda frame: frame["SHAP Impact"].abs())
        .sort_values("AbsImpact", ascending=False)
        .head(10)
    )

    return probability, shap_values, contributions


def render_waterfall(shap_values, title: str):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(title)
    st.pyplot(fig, clear_figure=True)


def render_manual_predictor():
    dashboard_bundle = load_dashboard_demo_bundle()
    defaults = dashboard_bundle["form_defaults"]
    form_options = dashboard_bundle["form_options"]
    admission_source_labels = dashboard_bundle["admission_source_labels"]
    model, explainer, pipeline, _, _, feature_names, threshold = load_model_bundle()

    st.subheader("Quick Intake Form")
    st.caption(
        "To keep the demo short, fields not shown here are filled from cohort medians or modes."
    )

    with st.form("manual_patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            default_age = MIDPOINT_TO_AGE_LABEL.get(int(defaults["age_midpoint"]), AGE_LABELS[0])
            age_label = st.selectbox("Age Group", AGE_LABELS, index=AGE_LABELS.index(default_age))
            time_in_hospital = st.slider(
                "Time in Hospital (days)",
                1,
                14,
                clipped_default(defaults, "time_in_hospital", 1, 14),
            )
            number_inpatient = st.slider(
                "Prior Inpatient Visits",
                0,
                10,
                clipped_default(defaults, "number_inpatient", 0, 10),
            )
            number_emergency = st.slider(
                "Prior Emergency Visits",
                0,
                10,
                clipped_default(defaults, "number_emergency", 0, 10),
            )
            number_outpatient = st.slider(
                "Prior Outpatient Visits",
                0,
                20,
                clipped_default(defaults, "number_outpatient", 0, 20),
            )
            number_diagnoses = st.slider(
                "Number of Diagnoses",
                1,
                16,
                clipped_default(defaults, "number_diagnoses", 1, 16),
            )

        with col2:
            num_lab_procedures = st.slider(
                "Lab Procedures",
                1,
                120,
                clipped_default(defaults, "num_lab_procedures", 1, 120),
            )
            num_procedures = st.slider(
                "Other Procedures",
                0,
                10,
                clipped_default(defaults, "num_procedures", 0, 10),
            )
            num_medications = st.slider(
                "Number of Medications",
                1,
                50,
                clipped_default(defaults, "num_medications", 1, 50),
            )
            num_active_medications = st.slider(
                "Active Diabetes Medications",
                0,
                10,
                clipped_default(defaults, "num_active_medications", 0, 10),
            )
            polypharmacy_score = st.slider(
                "Medication Change Burden",
                0,
                10,
                clipped_default(defaults, "polypharmacy_score", 0, 10),
            )
            elixhauser_score = st.slider(
                "Comorbidity Score",
                -5,
                20,
                clipped_default(defaults, "elixhauser_score", -5, 20),
            )

        with col3:
            discharge_category = st.selectbox(
                "Discharge Category",
                form_options["discharge_category"],
                index=0,
            )
            admission_category = st.selectbox(
                "Admission Category",
                form_options["admission_category"],
                index=0,
            )
            admission_source_options = form_options["admission_source_id"]
            default_source = int(round(float(defaults["admission_source_id"])))
            admission_source_id = st.selectbox(
                "Admission Source",
                admission_source_options,
                index=admission_source_options.index(default_source)
                if default_source in admission_source_options
                else 0,
                format_func=lambda source_id: f"{source_id}: {admission_source_labels.get(source_id, 'Unknown')}",
            )
            diag_1_group = st.selectbox(
                "Primary Diagnosis Group",
                form_options["diag_1_group"],
                index=0,
            )
            medical_specialty_grouped = st.selectbox(
                "Medical Specialty",
                form_options["medical_specialty_grouped"],
                index=0,
            )
            race = st.selectbox(
                "Race",
                form_options["race"],
                index=0,
            )
            gender = st.selectbox(
                "Gender",
                form_options["gender"],
                index=0,
            )

        st.markdown("**Clinical Flags**")
        flag_col1, flag_col2, flag_col3 = st.columns(3)
        with flag_col1:
            diabetes_med = st.checkbox(
                "On Diabetes Medication",
                value=bool(int(defaults["diabetesMed_binary"])),
            )
            medication_changed = st.checkbox(
                "Medication Changed",
                value=bool(int(defaults["change_binary"])),
            )
            has_diabetes_diag = st.checkbox(
                "Diabetes Diagnosis Present",
                value=bool(int(defaults["has_diabetes_diag"])),
            )
        with flag_col2:
            a1c_tested = st.checkbox(
                "A1C Tested",
                value=bool(int(defaults["a1c_tested"])),
            )
            a1c_abnormal = st.checkbox(
                "A1C Abnormal",
                value=bool(int(defaults["a1c_abnormal"])),
                disabled=not a1c_tested,
            )
            glu_tested = st.checkbox(
                "Glucose Tested",
                value=bool(int(defaults["glu_tested"])),
            )
        with flag_col3:
            glu_abnormal = st.checkbox(
                "Glucose Abnormal",
                value=bool(int(defaults["glu_abnormal"])),
                disabled=not glu_tested,
            )
            insulin_label = st.selectbox(
                "Insulin Intensity",
                list(INSULIN_INTENSITY_LABELS.values()),
                index=int(round(float(defaults["insulin_intensity"]))),
            )

        submitted = st.form_submit_button("Score Patient")

    if not submitted:
        return

    profile = defaults.copy()
    profile.update(
        {
            "age_midpoint": AGE_MIDPOINT_MAP[age_label],
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "polypharmacy_score": polypharmacy_score,
            "total_prior_visits": number_outpatient + number_emergency + number_inpatient,
            "num_active_medications": num_active_medications,
            "insulin_intensity": next(
                key for key, value in INSULIN_INTENSITY_LABELS.items() if value == insulin_label
            ),
            "elixhauser_score": elixhauser_score,
            "race": race,
            "gender": gender,
            "admission_category": admission_category,
            "discharge_category": discharge_category,
            "admission_source_id": admission_source_id,
            "medical_specialty_grouped": medical_specialty_grouped,
            "diag_1_group": diag_1_group,
            "change_binary": int(medication_changed),
            "diabetesMed_binary": int(diabetes_med),
            "has_diabetes_diag": int(has_diabetes_diag),
            "a1c_tested": int(a1c_tested),
            "a1c_abnormal": int(a1c_tested and a1c_abnormal),
            "glu_tested": int(glu_tested),
            "glu_abnormal": int(glu_tested and glu_abnormal),
            "high_utilizer_flag": int(number_inpatient >= 2),
        }
    )

    patient_frame = pd.DataFrame([profile])[ALL_FEATURES]
    probability, shap_values, contributions = score_patient(
        model,
        explainer,
        pipeline,
        feature_names,
        patient_frame,
    )

    st.subheader("Manual Risk Assessment")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Readmission Risk", f"{probability * 100:.1f}%")
    col2.metric("Decision Threshold", f"{threshold * 100:.1f}%")
    col3.metric("Risk Band", risk_band(probability, threshold))

    if probability >= threshold:
        st.warning("This profile is above the current operating threshold and should be reviewed for intervention.")
    else:
        st.success("This profile is below the current operating threshold.")

    st.subheader("Top Contributors")
    st.dataframe(
        contributions[["Feature", "Encoded Value", "SHAP Impact"]].style.format(
            {"Encoded Value": "{:.3f}", "SHAP Impact": "{:.3f}"}
        ),
        width="stretch",
    )
    render_waterfall(shap_values, "Manual patient explanation")


def render_cohort_explorer():
    st.subheader("Test Cohort Explorer")
    st.markdown("Select an anonymized patient from the held-out test set to inspect the model output.")

    model, explainer, _, X_test, y_test, feature_names, threshold = load_model_bundle()
    max_index = max(0, min(500, len(X_test) - 1))
    patient_idx = st.slider("Select Patient Index ID", 0, max_index, min(10, max_index), 1)

    patient_data = X_test[patient_idx]
    actual_outcome = "Readmitted < 30 Days" if int(y_test[patient_idx]) == 1 else "Not Readmitted / > 30 Days"
    probability = float(model.predict_proba(patient_data.reshape(1, -1))[0, 1])

    st.subheader("Risk Assessment")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Readmission Risk", f"{probability * 100:.1f}%")
    col2.metric("Risk Band", risk_band(probability, threshold))
    col3.metric("Actual Ground Truth", actual_outcome)

    patient_frame = pd.DataFrame(
        [patient_data],
        columns=feature_name_list(feature_names, len(patient_data)),
    )
    shap_values = explainer(patient_frame)
    if len(shap_values.shape) > 2:
        shap_values = shap_values[:, :, 1]

    render_waterfall(shap_values, "Held-out patient explanation")


if page == "Overview":
    st.header("Project Overview")
    st.markdown(
        """
    This dashboard visualizes an end-to-end machine learning workflow for predicting
    **30-day hospital readmission risk** in diabetic patients.
    The goal is to support earlier intervention, better discharge planning, and more
    efficient use of hospital resources.
    """
    )

    st.subheader("Data Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Encounters", "101,766")
    col2.metric("Features", "50")
    col3.metric("Baseline Readmission Rate", "11.2%")

    st.subheader("Model Performance Summary (Held-Out Test Set)")
    metrics = load_metrics()
    if metrics is None:
        render_missing_artifact_notice(METRICS_DIR / "model_comparison.csv", "Model comparison table")
    else:
        st.dataframe(
            metrics.round(4).style.highlight_max(subset=["AUROC", "AUPRC", "F1"], color="lightgreen"),
            width="stretch",
        )

elif page == "Model Performance":
    st.header("Global Model Performance")

    st.subheader("ROC and Precision-Recall Curves")
    render_artifact_image(FIGURES_DIR / "roc_prc_comparison.png", "ROC and precision-recall chart")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration (Reliability)")
        st.markdown("*Checking if predicted probabilities match actual frequencies.*")
        render_artifact_image(FIGURES_DIR / "calibration_curve.png", "Calibration chart")
    with col2:
        st.subheader("Confusion Matrix (XGBoost)")
        st.markdown("*Threshold selected with a count-based 5:1 false-negative to false-positive cost rule.*")
        render_artifact_image(FIGURES_DIR / "xgboost_confusion_matrix.png", "Confusion matrix")

elif page == "SHAP Explanations":
    st.header("SHAP Feature Explainability")
    st.markdown(
        "XGBoost decisions explained using SHAP values. Positive impact values push readmission risk upward."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        render_artifact_image(FIGURES_DIR / "shap_summary_bar.png", "Global SHAP bar chart")
    with col2:
        st.subheader("Directional Impact (Beeswarm)")
        render_artifact_image(FIGURES_DIR / "shap_beeswarm.png", "SHAP beeswarm chart")

    st.subheader("Partial Dependence")
    st.markdown("Select a top predictor to see its isolated, non-linear effect on readmission risk:")

    deps = list(FIGURES_DIR.glob("shap_dependence_*.png"))
    dep_names = [dep.stem.replace("shap_dependence_", "") for dep in deps]
    if dep_names:
        feature = st.selectbox("Select Feature", dep_names)
        render_artifact_image(
            FIGURES_DIR / f"shap_dependence_{feature}.png",
            f"SHAP dependence chart for {feature}",
        )
    else:
        st.info("No SHAP dependence plots were bundled with this deployment.")

elif page == "Fairness Audit":
    st.header("Demographic Fairness Audit")
    df_fairness = load_fairness()
    if df_fairness is None:
        render_missing_artifact_notice(METRICS_DIR / "fairness_audit.csv", "Fairness audit table")
        st.stop()
    df_fairness = df_fairness.round(3)

    st.markdown(
        """
    **Equalized Odds Check**: Does the model recover high-risk patients at similar rates across groups?

    **Predictive Parity Check**: When the model flags a patient as high-risk, is that flag similarly reliable across groups?
    """
    )

    attributes = df_fairness["Attribute"].unique()
    tabs = st.tabs(list(attributes))

    for idx, attr in enumerate(attributes):
        with tabs[idx]:
            subset = df_fairness[df_fairness["Attribute"] == attr].copy()
            subset = subset.drop(columns=["Attribute"]).sort_values("N", ascending=False)
            st.dataframe(
                subset.style.background_gradient(subset=["TPR", "PPV"], cmap="Blues").highlight_min(
                    subset=["AUROC"], color="lightcoral"
                ),
                width="stretch",
            )

            small_samples = subset[subset["N"] < 100]
            if not small_samples.empty:
                st.warning(
                    f"Note: Subgroups {list(small_samples['Subgroup'])} have N < 100, which leads to higher variance."
                )

elif page == "Interactive Predictor":
    st.header("Interactive Patient Risk Profiler")
    st.markdown(
        "Use the quick intake form for a live demo, or inspect a real held-out patient from the test cohort."
    )

    try:
        manual_tab, cohort_tab = st.tabs(["Quick Intake", "Held-Out Cohort"])
        with manual_tab:
            render_manual_predictor()
        with cohort_tab:
            render_cohort_explorer()
    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"Interactive predictor assets are invalid: {exc}")
    except Exception as exc:
        st.error(f"Dashboard assets are not ready yet. Check backend artifacts and models: {exc}")
