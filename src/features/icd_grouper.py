import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def map_icd9_code(code: str) -> str:
    """
    Maps a single ICD-9 code to a clinical category.
    Handles 'V', 'E', '?' prefixes and float/int representations.
    
    Categories:
      - 250.xx: Diabetes
      - 390-459, 785: Circulatory
      - 460-519, 786: Respiratory
      - 520-579, 787: Digestive
      - 800-999: Injury
      - 710-739: Musculoskeletal
      - 580-629, 788: Genitourinary
      - 140-239: Neoplasms
      - Other: Other
      - Missing/invalid: Missing
    """
    if pd.isna(code) or str(code) in ('?', 'Missing', 'NaN', 'nan', ''):
        return 'Missing'
        
    code_str = str(code).strip().upper()
    
    # External causes or supplementary classifications
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
        
    try:
        # Get the first part of the code before decimal point
        main_code = float(code_str)
        
        # Exact matching for diabetes
        if 250 <= main_code < 251:
            return 'Diabetes'
            
        # ICD-9 specific ranges
        if (390 <= main_code <= 459) or main_code == 785:
            return 'Circulatory'
        elif (460 <= main_code <= 519) or main_code == 786:
            return 'Respiratory'
        elif (520 <= main_code <= 579) or main_code == 787:
            return 'Digestive'
        elif 800 <= main_code <= 999:
            return 'Injury'
        elif 710 <= main_code <= 739:
            return 'Musculoskeletal'
        elif (580 <= main_code <= 629) or main_code == 788:
            return 'Genitourinary'
        elif 140 <= main_code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
            
    except ValueError:
        # Anything we can't parse as float after 'V' and 'E' check goes to 'Other'
        logger.warning(f"Unable to parse ICD-9 code: {code_str}. Mapping to 'Other'.")
        return 'Other'

def add_icd_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the ICD-9 mapping to diag_1, diag_2, and diag_3 columns.
    Adds a binary flag for any diabetes diagnosis.
    """
    df = df.copy()
    
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        if col in df.columns:
            logger.info(f"Grouping ICD-9 codes in {col}...")
            df[f'{col}_group'] = df[col].apply(map_icd9_code)
    
    # Create a binary flag indicating if ANY diagnosis was categorized as 'Diabetes'
    if all(col in df.columns for col in diag_cols):
        df['has_diabetes_diag'] = (
            (df['diag_1_group'] == 'Diabetes') | 
            (df['diag_2_group'] == 'Diabetes') | 
            (df['diag_3_group'] == 'Diabetes')
        ).astype(int)
        logger.info("Added 'has_diabetes_diag' binary flag.")
        
    return df
