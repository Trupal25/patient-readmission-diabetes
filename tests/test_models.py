import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.features.pipeline import get_processed_data
from src.utils.config import MODELS_DIR
from sklearn.metrics import roc_auc_score

@pytest.fixture(scope="module")
def xgb_model():
    """Load the trained XGBoost model if it exists."""
    path = MODELS_DIR / "xgboost_optimized.joblib"
    if not path.exists():
        path = MODELS_DIR / "xgboost_initial.joblib"
    if not path.exists():
        pytest.skip("XGBoost model file not found in models/. Skipping model tests.")
    return joblib.load(path)

@pytest.fixture(scope="module")
def xgb_data():
    """Load validation data for XGBoost testing."""
    _, X_val, _, _, y_val, _, _, _ = get_processed_data(model_type="xgb")
    return X_val, y_val

@pytest.fixture(scope="module")
def lr_model():
    """Load the trained Logistic Regression model."""
    path = MODELS_DIR / "logistic_baseline.joblib"
    if not path.exists():
        pytest.skip("Logistic regression model file not found. Skipping model tests.")
    return joblib.load(path)

@pytest.fixture(scope="module")
def lr_data():
    """Load validation data for Logistic Regression testing."""
    _, X_val, _, _, y_val, _, _, _ = get_processed_data(model_type="lr")
    return X_val, y_val


def test_xgb_model_save_load(xgb_model):
    """Test model can be saved and loaded (which fixture implicitly does, but verify it's valid)."""
    assert xgb_model is not None, "Failed to load XGBoost model."
    assert hasattr(xgb_model, "predict"), "Loaded object missing predict method."

def test_lr_model_save_load(lr_model):
    """Test model can be saved and loaded."""
    assert lr_model is not None, "Failed to load Logistic Regression model."
    assert hasattr(lr_model, "predict"), "Loaded object missing predict method."

def test_xgb_predict_shape(xgb_model, xgb_data):
    """Test model.predict returns correct shape."""
    X_val, y_val = xgb_data
    # Use a small sample to speed up tests
    X_sample = X_val[:100]
    preds = xgb_model.predict(X_sample)
    
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be binary (0 or 1)."

def test_lr_predict_shape(lr_model, lr_data):
    """Test model.predict returns correct shape."""
    X_val, y_val = lr_data
    X_sample = X_val[:100]
    preds = lr_model.predict(X_sample)
    
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be binary (0 or 1)."

def test_xgb_auroc_better_than_random(xgb_model, xgb_data):
    """Test AUROC > 0.5 (better than random performance) on validation set."""
    X_val, y_val = xgb_data
    probs = xgb_model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, probs)
    
    assert auroc > 0.5, f"XGBoost AUROC ({auroc:.3f}) is not better than random."
    assert auroc < 1.0, "AUROC cannot be 1.0 (leakage suspected if so)."

def test_lr_auroc_better_than_random(lr_model, lr_data):
    """Test AUROC > 0.5 on validation set."""
    X_val, y_val = lr_data
    probs = lr_model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, probs)
    
    assert auroc > 0.5, f"Logistic Regression AUROC ({auroc:.3f}) is not better than random."
    assert auroc < 1.0, "AUROC cannot be 1.0 (leakage suspected if so)."
