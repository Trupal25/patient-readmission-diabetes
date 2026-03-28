import logging
import joblib
import pandas as pd
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred_proba) -> Dict[str, float]:
    """Helper to calculate AUROC and AUPRC."""
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    return {"auroc": auroc, "auprc": auprc}

def train_baseline() -> Tuple[LogisticRegression, Dict[str, float]]:
    logger.info("Initializing baseline Logistic Regression training...")
    
    # 1. Load data processed specifically for Logistic Regression (OneHot, StandardScaler)
    X_train, X_val, _, y_train, y_val, _, feature_names, _ = get_processed_data(model_type="lr")
    
    # 2. Initialize Model
    # class_weight='balanced' automatically handles our 1:8 class imbalance
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    # 3. Train
    logger.info("Fitting Logistic Regression model...")
    model.fit(X_train, y_train)
    
    # 4. Evaluate on Validation Set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(y_val, y_val_proba)
    
    logger.info(f"Baseline Validation Metrics:")
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"AUPRC: {metrics['auprc']:.4f}")
    
    # 5. Extract Feature Coefficients
    if len(feature_names) == X_train.shape[1]:
        coefs = pd.DataFrame({
            "feature": feature_names,
            "coefficient": model.coef_[0]
        }).sort_values(by="coefficient", key=abs, ascending=False)
        
        logger.info("\nTop 10 Feature Coefficients (Absolute Impact):")
        logger.info(f"\n{coefs.head(10).to_string(index=False)}")
        
        # Save coefficients for analysis
        coefs.to_csv(MODELS_DIR / "lr_coefficients.csv", index=False)
    
    # 6. Save Model
    model_path = MODELS_DIR / "logistic_baseline.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved baseline model to {model_path}")
    
    return model, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_baseline()
