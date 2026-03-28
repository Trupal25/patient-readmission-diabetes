import logging
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR, FIGURES_DIR

logger = logging.getLogger(__name__)

def run_shap_analysis():
    logger.info("Starting SHAP Explanability Analysis on XGBoost Model...")
    
    # 1. Load Data & Model
    _, _, X_test_arr, _, _, y_test, feature_names, _ = get_processed_data(model_type="xgb")
    
    xgb_path = MODELS_DIR / "xgboost_optimized.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "xgboost_initial.joblib"
    model = joblib.load(xgb_path)
    
    # Convert test set to DataFrame for better SHAP plotting labels
    # XGBoost output feature names might just be raw array, but we can use the ones extracted from pipeline
    # Actually OrdinalEncoder feature names might be the original string columns
    # Let's check if feature_names matches X_test_arr shape
    if len(feature_names) == X_test_arr.shape[1]:
        X_test = pd.DataFrame(X_test_arr, columns=feature_names)
    else:
        # Fallback
        X_test = pd.DataFrame(X_test_arr, columns=[f"Feature_{i}" for i in range(X_test_arr.shape[1])])
        
    y_test_arr = y_test.values
    y_prob = model.predict_proba(X_test_arr)[:, 1]
    # Optimal threshold approx (from metrics, let's say 0.1, but we just use basic classification)
    y_pred = (y_prob > 0.1).astype(int) 
    
    # 2. Compute SHAP Values
    logger.info("Computing SHAP values (this may take a minute)...")
    # TreeExplainer is extremely fast for XGBoost
    explainer = shap.TreeExplainer(model)
    # XGBoost output is typically margin, but SHAP plots look best with probabilities
    # We will use the raw SHAP values (log-odds) for beeswarm as customary
    shap_values = explainer(X_test)
    
    # Check if shap_values is a list (multiclass) or object. For binary, usually it's one Explainer object.
    # Handle if shap returns values for both classes
    if len(shap_values.shape) > 2:
        shap_values = shap_values[:, :, 1]
        
    # 3. Summary & Beeswarm Plots
    logger.info("Generating Global Importance & Beeswarm plots...")
    
    # Summary Bar
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance (Top Features)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm Plot (Directional Impact)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Dependence Plots for Top 5 features
    logger.info("Generating Partial Dependence Plots for Top 5...")
    # Get top 5 features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = X_test.columns[top_indices].tolist()
    
    for feature in top_features:
        try:
            plt.figure()
            shap.dependence_plot(feature, shap_values.values, X_test, show=False, interaction_index="auto")
            plt.title(f"SHAP Dependence: {feature}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_dependence_{feature.replace('/', '_')}.png", dpi=100)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate dependence plot for {feature}: {e}")
            plt.close()

    # 5. Individual Explanations (Waterfall)
    logger.info("Generating Individual Patient Waterfall Explanations...")
    # Find specific cases
    true_positives = np.where((y_pred == 1) & (y_test_arr == 1))[0]
    true_negatives = np.where((y_pred == 0) & (y_test_arr == 0))[0]
    false_positives = np.where((y_pred == 1) & (y_test_arr == 0))[0] # Model flagged high risk, but they weren't readmitted
    
    def generate_waterfall(idx, title, filename):
        if len(idx) > 0:
            i = idx[0] # Pick the first example
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[i], show=False)
            plt.title(f"{title} (Patient {i})")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
            plt.close()
    
    generate_waterfall(true_positives, "True Positive: Correctly Flagged High-Risk", "shap_waterfall_TP.png")
    generate_waterfall(true_negatives, "True Negative: Correctly Flagged Low-Risk", "shap_waterfall_TN.png")
    generate_waterfall(false_positives, "False Positive: Misclassified as High-Risk", "shap_waterfall_FP.png")

    logger.info("SHAP analysis complete. Plots saved to reports/figures/")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_shap_analysis()
