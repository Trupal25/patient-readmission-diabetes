import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix

from src.evaluation.metrics import compute_optimal_threshold
from src.features.pipeline import get_processed_data, load_engineered_dataframe
from src.utils.config import MODELS_DIR, METRICS_DIR

logger = logging.getLogger(__name__)

def evaluate_subgroup(y_true, y_prob, threshold):
    """Calculates AUROC, TPR, FPR, PPV for a subgroup."""
    if len(y_true) == 0:
        return {"AUROC": np.nan, "TPR": np.nan, "FPR": np.nan, "PPV": np.nan, "N": 0}
        
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Happens if subgroup only has one class
        auroc = np.nan
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return {
        "N": len(y_true),
        "AUROC": auroc,
        "TPR": tpr,
        "FPR": fpr,
        "PPV": ppv
    }

def run_fairness_audit():
    logger.info("Starting Fairness Audit on the XGBoost Model...")
    
    # 1. Get processed arrays and trained model
    _, _, X_test_arr, _, _, y_test_series, _, _ = get_processed_data("xgb")
    
    xgb_path = MODELS_DIR / "xgboost_optimized.joblib"
    if not xgb_path.exists():
        xgb_path = MODELS_DIR / "xgboost_initial.joblib"
    model = joblib.load(xgb_path)
    
    # 2. Get predictions
    y_prob = model.predict_proba(X_test_arr)[:, 1]
    
    # Reuse the same asymmetric operating point selected during the main evaluation.
    threshold = compute_optimal_threshold(
        y_test_series.to_numpy(dtype=np.int_),
        y_prob.astype(np.float64),
    )
    
    # 3. Retrieve original demographic features using the test split indices
    logger.info("Loading original dataset to recover demographics...")
    df = load_engineered_dataframe()
    
    # Subset to test data
    test_idx = y_test_series.index
    df_test = df.loc[test_idx].copy()
    
    # Check alignment
    assert len(df_test) == len(y_prob), "Mismatch in test set sizes between processed array and raw dataframe!"
    
    df_test['y_true'] = y_test_series.values
    df_test['y_prob'] = y_prob
    
    # Define the fairness dimensions to audit
    audit_dimensions = {
        "Race": "race",
        "Gender": "gender",
        "Age": "age" # Use the original age brackets
    }
    
    results = []
    
    for dim_name, col_name in audit_dimensions.items():
        if col_name not in df_test.columns:
            continue
            
        logger.info(f"Auditing demographic attribute: {dim_name}")
        subgroups = df_test[col_name].dropna().unique()
        
        for sg in subgroups:
            mask = df_test[col_name] == sg
            sg_data = df_test[mask]
            
            metrics = evaluate_subgroup(sg_data['y_true'], sg_data['y_prob'], threshold)
            
            results.append({
                "Attribute": dim_name,
                "Subgroup": sg,
                "N": metrics['N'],
                "AUROC": metrics['AUROC'],
                "TPR": metrics['TPR'],
                "FPR": metrics['FPR'],
                "PPV": metrics['PPV']
            })
            
    # 4. Compile and format report
    report_df = pd.DataFrame(results)
    
    # Check for equalized odds and predictive parity
    logger.info(f"\nDisparity Report (Threshold = {threshold:.4f}):")
    
    # Print groupings
    for attr in report_df['Attribute'].unique():
        sub_df = report_df[report_df['Attribute'] == attr]
        logger.info(f"\n--- {attr} ---")
        for _, row in sub_df.sort_values(by="N", ascending=False).iterrows():
            if row['N'] < 50: # Skip tiny groups for stats clarity
                continue
            logger.info(f"{row['Subgroup']:>15} (n={row['N']:>4}): AUROC={row['AUROC']:.3f} | TPR={row['TPR']:.3f} | FPR={row['FPR']:.3f} | PPV={row['PPV']:.3f}")
            
            if row['AUROC'] < 0.60:
                logger.warning(f"  -> WARNING: AUROC below 0.60 for {row['Subgroup']} ({row['AUROC']:.3f})")

    # Overall max disparities for primary subgroups (n > 100)
    valid_groups = report_df[report_df['N'] > 100]
    for attr in valid_groups['Attribute'].unique():
        sub_df = valid_groups[valid_groups['Attribute'] == attr]
        max_tpr_gap = sub_df['TPR'].max() - sub_df['TPR'].min()
        max_ppv_gap = sub_df['PPV'].max() - sub_df['PPV'].min()
        
        logger.info(f"\nDisparities for {attr}:")
        logger.info(f"  Max TPR gap (Equalized Odds violation): {max_tpr_gap:.3f}")
        logger.info(f"  Max PPV gap (Predictive Parity violation): {max_ppv_gap:.3f}")
        
        if max_tpr_gap > 0.05:
            logger.warning(f"  -> Equalized Odds check failed (>5% gap) for {attr}!")
        if max_ppv_gap > 0.05:
            logger.warning(f"  -> Predictive Parity check failed (>5% gap) for {attr}!")

    # Save to CSV
    out_path = METRICS_DIR / "fairness_audit.csv"
    report_df.to_csv(out_path, index=False)
    logger.info(f"Saved fairness report to {out_path}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_fairness_audit()
