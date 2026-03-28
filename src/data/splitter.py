import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.utils.config import TEST_SIZE, VAL_SIZE, RANDOM_STATE, TARGET_BINARY_COL

logger = logging.getLogger(__name__)

def split_data(
    df: pd.DataFrame, 
    target_col: str = TARGET_BINARY_COL
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataframe into Train (70%), Validation (15%), and Test (15%) sets
    using stratified sampling on the target column.
    
    Args:
        df: Feature engineered DataFrame containing the target column.
        target_col: Name of the target column.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Splitting data into train/val/test sets...")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. Split out Test set first (15%)
    # Train + Val will be 85%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # 2. Split the remaining (Train + Val) into Train (70/85) and Val (15/85)
    # 15 / (15 + 70) = 15 / 85 = 0.17647...
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    logger.info(f"Split completed.")
    logger.info(f"Train size: {len(X_train)} ({len(X_train)/len(df):.1%}) - Posatives: {y_train.mean():.1%}")
    logger.info(f"Val size:   {len(X_val)} ({len(X_val)/len(df):.1%}) - Posatives: {y_val.mean():.1%}")
    logger.info(f"Test size:  {len(X_test)} ({len(X_test)/len(df):.1%}) - Posatives: {y_test.mean():.1%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
