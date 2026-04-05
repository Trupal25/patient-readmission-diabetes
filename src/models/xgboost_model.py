import logging
import joblib
from xgboost import XGBClassifier
from typing import Tuple, Dict

from sklearn.metrics import roc_auc_score, average_precision_score
from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred_proba) -> Dict[str, float]:
    """Helper to calculate AUROC and AUPRC."""
    auroc = float(roc_auc_score(y_true, y_pred_proba))
    auprc = float(average_precision_score(y_true, y_pred_proba))
    return {"auroc": auroc, "auprc": auprc}

def train_initial_xgboost() -> Tuple[XGBClassifier, Dict[str, float]]:
    logger.info("Initializing baseline XGBoost training...")
    
    with mlflow_utils.start_run(run_name="XGBoost_Initial"):
        # 1. Load data processed specifically for XGBoost (OrdinalEncoder, Passthrough numeric)
        X_train, X_val, _, y_train, y_val, _, feature_names, _ = get_processed_data(model_type="xgb")
        
        # 2. Setup class weight - Removed scale_pos_weight because SMOTENC balances the pipeline 1:1 natively.
        
        # 3. Initialize Model with default params from implementation plan
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr', # Use precision-recall curve for trees natively
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=20
        )
        
        # 4. Train
        logger.info("Fitting XGBoost model...")
        # New XGBoost API prefers adding early_stopping directly in the constructor, 
        # but eval_set is still required in fit():
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # 5. Evaluate on Validation Set
        y_val_proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_val_proba)
        
        logger.info(f"XGBoost Validation Metrics:")
        logger.info(f"AUROC: {metrics['auroc']:.4f}")
        logger.info(f"AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"Best Iteration: {model.best_iteration}")
        
        mlflow_utils.log_params(model.get_params())
        mlflow_utils.log_metrics({
            "val_auroc": metrics['auroc'],
            "val_auprc": metrics['auprc'],
            "best_iteration": model.best_iteration
        })
        
        # 6. Save Model
        model_path = MODELS_DIR / "xgboost_initial.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved initial XGBoost model to {model_path}")
        
        mlflow_utils.log_xgboost_model(model, "xgboost_model")
        
    return model, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_initial_xgboost()
