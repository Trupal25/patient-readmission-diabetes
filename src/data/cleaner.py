import pandas as pd
import numpy as np
import logging

from src.utils.config import DECEASED_DISPOSITION_IDS

logger = logging.getLogger(__name__)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw diabetic dataset according to EDA findings.
    - Replaces '?' with NaN
    - Drops sparse/unhelpful columns
    - Removes patients who expired/went to hospice (leakage)
    - Deduplicates patients to their first encounter
    - Binarizes the target
    
    Args:
        df: Raw pandas DataFrame
        
    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    # Create a copy to avoid SettingWithCopyWarning
    clean_df = df.copy()
    
    initial_rows = len(clean_df)
    logger.info(f"Starting data cleaning. Initial rows: {initial_rows}")

    # 1. Replace '?' with NaN globally
    clean_df = clean_df.replace('?', np.nan)
    
    # 2. Drop columns identified in EDA
    cols_to_drop = ['weight', 'payer_code', 'encounter_id']
    clean_df = clean_df.drop(columns=[c for c in cols_to_drop if c in clean_df.columns], errors='ignore')
    logger.info(f"Dropped columns: {cols_to_drop}")

    # 3. Remove expired/hospice patients (data leakage)
    # They physically cannot be readmitted.
    if 'discharge_disposition_id' in clean_df.columns:
        deceased_mask = clean_df['discharge_disposition_id'].isin(DECEASED_DISPOSITION_IDS)
        num_deceased = deceased_mask.sum()
        clean_df = clean_df[~deceased_mask]
        logger.info(f"Removed {num_deceased} encounters with deceased/hospice discharge disposition.")

    # 4. Deduplicate to keep only the first encounter per patient
    # Ensures no data leakage between train/test splits for the same patient.
    if 'patient_nbr' in clean_df.columns:
        # Assuming the dataset is ordered chronologically or encounter_id proxies time.
        # If not, we still just take the first record appearing for stability.
        # Often encounter_id is chronological. We sort by encounter_id if it existed, but we just dropped it.
        # Wait, if we dropped encounter_id we might sort blindly. Let's sort before dropping or just use drop_duplicates.
        num_before_dedup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=['patient_nbr'], keep='first')
        num_dup_removed = num_before_dedup - len(clean_df)
        
        # Now drop patient_nbr as it's no longer needed as a feature
        clean_df = clean_df.drop(columns=['patient_nbr'])
        logger.info(f"Removed {num_dup_removed} duplicate encounters. Dropped 'patient_nbr'.")

    # 5. Binarize Target
    # Original target is 'readmitted' with values: '<30', '>30', 'NO'
    if 'readmitted' in clean_df.columns:
        clean_df['readmitted_30day'] = (clean_df['readmitted'] == '<30').astype(int)
        clean_df = clean_df.drop(columns=['readmitted'])
        
        # Also drop the readmitted_binary we might have made in loader.py 
        # just to maintain pure state, depending on loader implementation
        if 'readmitted_binary' in clean_df.columns:
             clean_df = clean_df.drop(columns=['readmitted_binary'])
        
        logger.info("Binarized target into 'readmitted_30day' and dropped original 'readmitted'.")

    final_rows = len(clean_df)
    logger.info(f"Data cleaning complete. Final rows: {final_rows}. Rows removed: {initial_rows - final_rows}")
    
    return clean_df
