import logging
import json
import joblib
import optuna
import numpy as np
from xgboost import XGBClassifier
from typing import Dict, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR, METRICS_DIR

logger = logging.getLogger(__name__)

# Reduce verbosity of Optuna globally if desired (INFO is good for 100 trials though)
optuna.logging.set_verbosity(optuna.logging.INFO)

def optimize_hyperparameters(n_trials: int = 50) -> Dict[str, Any]:
    """Runs Optuna optimization for XGBoost over the validation AUPRC."""
    logger.info(f"Starting Hyperparameter Optimization ({n_trials} trials)...")
    
    X_train, X_val, _, y_train, y_val, _, _, _ = get_processed_data(model_type="xgb")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            # we'll use a fixed value to not clutter trial space, or just use pruning:
            'early_stopping_rounds': 30
        }
        
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_val_proba)
        
        return auprc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_val_auprc = study.best_value
    
    logger.info(f"Best Optuna AUPRC: {best_val_auprc:.4f}")
    logger.info(f"Best Params: {best_params}")
    
    # Save best metrics to json
    out_path = METRICS_DIR / "best_hpo_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    return best_params

def cross_validate_best_model(best_params: Dict[str, Any]):
    """Run 5-Fold Stratified CV using the best hyperparams to check for high variance."""
    logger.info("Running 5-Fold Stratified CV with optimized parameters...")
    
    X_train, _, _, y_train, _, _, _, _ = get_processed_data(model_type="xgb")
    
    # Needs a combined dataset. Actually the implementation plan asked to run CV on the 
    # initial training set.
    X_arr = X_train
    y_arr = y_train.values  # Convert from pd.Series to numpy 1D
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auroc_scores = []
    auprc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_t, y_t = X_arr[train_idx], y_arr[train_idx]
        X_v, y_v = X_arr[val_idx], y_arr[val_idx]
        
        model_params = best_params.copy()
        model_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'early_stopping_rounds': 30
        })
        
        model = XGBClassifier(**model_params)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )
        
        y_v_proba = model.predict_proba(X_v)[:, 1]
        
        auroc = roc_auc_score(y_v, y_v_proba)
        auprc = average_precision_score(y_v, y_v_proba)
        
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        
        logger.info(f"  Fold {fold+1} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    mean_auroc, std_auroc = np.mean(auroc_scores), np.std(auroc_scores)
    mean_auprc, std_auprc = np.mean(auprc_scores), np.std(auprc_scores)
    
    logger.info("CV Results across 5 Folds:")
    logger.info(f"AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    logger.info(f"AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")

    if std_auprc > 0.05:
        logger.warning(f"High variance detected in AUPRC ({std_auprc:.4f})! The model might be overfitting to specific CV splits.")

def retrain_and_save_optimized_model(best_params: Dict[str, Any]):
    """Retrain on the FULL training set using the best params and save the joblib."""
    logger.info("Retraining optimized model on full training set...")
    
    X_train, X_val, _, y_train, y_val, _, _, _ = get_processed_data(model_type="xgb")
    
    model_params = best_params.copy()
    model_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': 42,
        'early_stopping_rounds': 30
    })
    
    model = XGBClassifier(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    logger.info(f"Final Retrained Validation AUROC: {roc_auc_score(y_val, y_val_proba):.4f}")
    logger.info(f"Final Retrained Validation AUPRC: {average_precision_score(y_val, y_val_proba):.4f}")
    
    out_path = MODELS_DIR / "xgboost_optimized.joblib"
    joblib.dump(model, out_path)
    logger.info(f"Saved optimized XGBoost model to {out_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    import sys
    # For quick demonstrations we shouldn't run 100 optuna trials directly unless requested
    # Let's run a small number just to assert everything functions 
    n_trials = 10 if "--quick" in sys.argv else 100
    
    best_params = optimize_hyperparameters(n_trials=n_trials)
    cross_validate_best_model(best_params)
    retrain_and_save_optimized_model(best_params)
