import pandas as pd
import logging
import argparse

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.data.loader import load_raw_data
from src.data.cleaner import clean
from src.features.icd_grouper import add_icd_groups
from src.features.engineer import engineer_features
from src.features.elixhauser import calculate_elixhauser_score
from src.data.splitter import split_data
from imblearn.over_sampling import SMOTENC

from src.utils.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    TARGET_BINARY_COL,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)

def build_pipeline(model_type: str = "xgb") -> Pipeline:
    """
    Builds a scikit-learn preprocessing pipeline.
    
    Args:
        model_type: "lr" for Logistic Regression or "xgb" for XGBoost.
            - LR gets StandardScaler and OneHotEncoder
            - XGB gets passthrough numeric and OrdinalEncoder
            
    Returns:
        A valid sklearn Pipeline
    """
    if model_type not in ["lr", "xgb"]:
        raise ValueError("model_type must be 'lr' or 'xgb'")
        
    logger.info(f"Building preprocessing pipeline for model_type='{model_type}'")
    
    # Define numeric transformer
    if model_type == "lr":
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"
        
    # Define categorical transformer
    if model_type == "lr":
        # drop='first' prevents multicollinearity which is useful for LR interpretation
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
    else:
        # For tree-based models, OrdinalEncoder or native categorical handling
        # We'll use ordinal encoding for simplicity as XGBoost handles splits natively.
        categorical_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        
    # Column transformer to apply transformations to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop" # Drops any columns not explicitly defined in the feature lists
    )
    
    # Wrap in a Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    
    return pipeline

def get_processed_data(model_type: str = "xgb"):
    """
    End-to-end data processing function that:
      1. Loads raw data
      2. Cleans it
      3. Adds ICD groups
      4. Engineers features
      5. Splits it
      6. Fits pipeline and transforms X
      
    Returns:
        X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    """
    # 1. Load data
    df = load_raw_data(decode_ids=False)
    
    # 2. Clean data
    df = clean(df)
    
    # 3. Add ICD groups
    df = add_icd_groups(df)
    
    # 4. Engineer features
    df = engineer_features(df)
    
    # 4.5 Add Elixhauser Score
    df = calculate_elixhauser_score(df)
    
    # Check that all required columns exist before splitting
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TARGET_BINARY_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature engineering missed required columns: {missing_cols}")
        
    # Keep only the features we need + the target
    df = df[required_cols]
    
    # 5. Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col=TARGET_BINARY_COL)
    
    # 6. Build and fit pipeline
    pipeline = build_pipeline(model_type=model_type)
    
    logger.info("Fitting pipeline on training data...")
    X_train_processed = pipeline.fit_transform(X_train)
    
    logger.info("Transforming validation and test data...")
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)
    
    logger.info(f"Pipeline created initial feature matrix with shape: {X_train_processed.shape}")
    
    # ONLY apply SMOTENC if XGBoost pipeline (LR uses sparse OHE which is hard to track without deep mapping)
    if model_type == "xgb":
        logger.info("Applying SMOTENC oversampling to synthesize minority class patients...")
        # Numeric comes first, then categorical, then binary.
        # Everything after numeric is treated as categorical by SMOTENC to avoid inventing floating point 
        # ages and maintaining discrete mappings.
        num_len = len(NUMERIC_FEATURES)
        total_len = X_train_processed.shape[1]
        cat_indices = list(range(num_len, total_len))
        
        smotenc = SMOTENC(random_state=RANDOM_STATE, categorical_features=cat_indices, sampling_strategy='auto')
        X_train_processed, y_train = smotenc.fit_resample(X_train_processed, y_train)
        logger.info(f"SMOTENC synthetically expanded training matrix to: {X_train_processed.shape}")
        
    # The output is a numpy array. We can wrap it back into a DataFrame if we want,
    # but numpy arrays are standard for sklearn/xgboost. We'll return the arrays.
    # To get column names extracted from OneHotEncoder:
    feature_names = []
    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    return (X_train_processed, X_val_processed, X_test_processed, 
            y_train, y_val, y_test, feature_names, pipeline)


if __name__ == "__main__":
    # Configure logging for script execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    logger.info("Running pipeline test...")
    get_processed_data(model_type="xgb")
    logger.info("Successfully executed end-to-end data processing for XGBoost model type.")
