import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure src modules can be imported
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR, FIGURES_DIR, METRICS_DIR

st.set_page_config(page_title="Diabetic Readmission Risk Dashboard", layout="wide", page_icon="🏥")

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container .main .block-container { max-width: 1200px; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Patient Readmission Prediction Dashboard")

# --- Navigation ---
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Model Performance", 
    "SHAP Explanations", 
    "Fairness Audit",
    "Interactive Predictor"
])

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_DIR / "model_comparison.csv")

@st.cache_data
def load_fairness():
    return pd.read_csv(METRICS_DIR / "fairness_audit.csv")

@st.cache_resource
def load_model_and_data():
    xgb_path = MODELS_DIR / "xgboost_optimized.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "xgboost_initial.joblib"
    model = joblib.load(xgb_path)
    _, _, X_test, _, _, y_test, feature_names, _ = get_processed_data("xgb")
    return model, X_test, y_test, feature_names

# --- PAGE: OVERVIEW ---
if page == "Overview":
    st.header("Project Overview")
    st.markdown("""
    This dashboard visualizes the machine learning pipeline developed to predict **30-day hospital readmission risk** for diabetic patients.
    By predicting readmission at the time of discharge, healthcare providers can proactively intervene with high-risk patients, reducing mortality and hospital costs.
    """)
    
    st.subheader("Data Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Encounters", "101,766")
    col2.metric("Features", "50")
    col3.metric("Baseline Readmission Rate", "11.2%")
    
    st.subheader("Model Performance Summary (Held-Out Test Set)")
    metrics = load_metrics().round(4)
    st.dataframe(metrics.style.highlight_max(subset=['AUROC', 'AUPRC', 'F1'], color='lightgreen'))

# --- PAGE: MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.header("Global Model Performance")
    
    st.subheader("ROC and Precision-Recall Curves")
    st.image(str(FIGURES_DIR / "roc_prc_comparison.png"), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Calibration (Reliability)")
        st.markdown("*Checking if predicted probabilities match actual frequencies.*")
        st.image(str(FIGURES_DIR / "calibration_curve.png"), use_container_width=True)
    with col2:
        st.subheader("Confusion Matrix (XGBoost)")
        st.markdown("*Threshold dynamically optimized using Youden's J statistic.*")
        st.image(str(FIGURES_DIR / "xgboost_confusion_matrix.png"), use_container_width=True)

# --- PAGE: SHAP EXPLANATIONS ---
elif page == "SHAP Explanations":
    st.header("SHAP Feature Explainability")
    st.markdown("XGBoost decisions explained using game-theoretic SHAP log-odds. Features pushing the prediction higher are shown in red.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        st.image(str(FIGURES_DIR / "shap_summary_bar.png"), use_container_width=True)
    with col2:
        st.subheader("Directional Impact (Beeswarm)")
        st.image(str(FIGURES_DIR / "shap_beeswarm.png"), use_container_width=True)
        
    st.subheader("Partial Dependence")
    st.markdown("Select a top predictor to see its isolated, non-linear effect on readmission risk:")
    
    deps = list(FIGURES_DIR.glob("shap_dependence_*.png"))
    dep_names = [d.stem.replace("shap_dependence_", "") for d in deps]
    if dep_names:
        feat = st.selectbox("Select Feature", dep_names)
        st.image(str(FIGURES_DIR / f"shap_dependence_{feat}.png"))

# --- PAGE: FAIRNESS AUDIT ---
elif page == "Fairness Audit":
    st.header("Demographic Fairness Audit")
    df_fairness = load_fairness().round(3)
    
    st.markdown("""
    **Equalized Odds Check**: Does the model detect high-risk patients (True Positive Rate) equally across groups? (Gap should be < 5%)
    
    **Predictive Parity Check**: When the model flags high-risk (Positive Predictive Value), is it equally accurate across groups? (Gap should be < 5%)
    """)
    
    attributes = df_fairness['Attribute'].unique()
    tabs = st.tabs(list(attributes))
    
    for i, attr in enumerate(attributes):
        with tabs[i]:
            sub = df_fairness[df_fairness['Attribute'] == attr].copy()
            # Drop the redundant column for cleaner display
            sub = sub.drop(columns=['Attribute']).sort_values('N', ascending=False)
            
            # Apply color maps to identify disparities
            st.dataframe(
                sub.style.background_gradient(subset=['TPR', 'PPV'], cmap='Blues')
                         .highlight_min(subset=['AUROC'], color='lightcoral')
            )
            
            # Warn if sample size is extremely small
            small_samples = sub[sub['N'] < 100]
            if not small_samples.empty:
                st.warning(f"Note: Subgroups {list(small_samples['Subgroup'])} have N < 100, which leads to high statistical variance in TPR/PPV.")

# --- PAGE: INTERACTIVE PREDICTOR ---
elif page == "Interactive Predictor":
    st.header("Interactive Patient Risk Profiler")
    st.markdown("Select an anonymized patient from the test set to analyze their real-time SHAP explanation.")
    
    try:
        model, X_test, y_test, feature_names = load_model_and_data()
        
        # We will allow selecting from the first 500 test patients
        patient_idx = st.slider("Select Patient Index ID", 0, 500, 10, 1)
        
        # Get patient data
        patient_data = X_test[patient_idx]
        actual_outcome = "Readmitted < 30 Days" if y_test.iloc[patient_idx] == 1 else "Not Readmitted / > 30 Days"
        
        # Predict
        prob = model.predict_proba(patient_data.reshape(1, -1))[0, 1]
        
        st.subheader(f"Risk Assessment")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Readmission Risk", f"{prob * 100:.1f}%", delta="HIGH RISK" if prob > 0.1 else "LOW RISK", delta_color="inverse")
        col2.metric("Actual Ground Truth", actual_outcome)
        
        st.subheader("Why did the model make this prediction?")
        
        # Calculate single patient SHAP
        explainer = shap.TreeExplainer(model)
        
        # Format df for shap
        if len(feature_names) == len(patient_data):
            patient_df = pd.DataFrame([patient_data], columns=feature_names)
        else:
            patient_df = pd.DataFrame([patient_data], columns=[f"Feature_{i}" for i in range(len(patient_data))])
            
        shap_values = explainer(patient_df)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]
            
        # Plot waterfall natively in st.pyplot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Waiting for backend models to compile. If models are trained, check error: {e}")
