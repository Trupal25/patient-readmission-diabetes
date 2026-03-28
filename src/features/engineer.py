import pandas as pd
import numpy as np
import logging

from src.utils.config import (
    MEDICATION_COLS, 
    AGE_MIDPOINT_MAP, 
    INSULIN_INTENSITY_MAP
)

logger = logging.getLogger(__name__)

def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    if 'age' in df.columns:
        df['age_midpoint'] = df['age'].map(AGE_MIDPOINT_MAP)
    return df

def create_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    # Total prior visits
    util_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if all(c in df.columns for c in util_cols):
        df['total_prior_visits'] = df[util_cols].sum(axis=1)
    
    # High utilizer flag
    if 'number_inpatient' in df.columns:
        df['high_utilizer_flag'] = (df['number_inpatient'] >= 2).astype(int)
    
    return df

def extract_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Polypharmacy score: count medication cols not in {'No', 'Steady'}
    active_change_meds = ~df[MEDICATION_COLS].isin(['No', 'Steady'])
    df['polypharmacy_score'] = active_change_meds.sum(axis=1)
    
    # 2. Number of active medications: count medication cols not in {'No'}
    active_meds = (df[MEDICATION_COLS] != 'No')
    df['num_active_medications'] = active_meds.sum(axis=1)
    
    # 3. Insulin usage intensity
    if 'insulin' in df.columns:
        df['insulin_intensity'] = df['insulin'].map(INSULIN_INTENSITY_MAP).fillna(0)
    
    # Drop all 23 individual medication columns to reduce dimensionality
    df = df.drop(columns=[c for c in MEDICATION_COLS if c in df.columns], errors='ignore')
    logger.info(f"Extracted medication features and dropped {len(MEDICATION_COLS)} individual drug columns.")
    
    return df

def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    # medication change flag
    if 'change' in df.columns:
        df['change_binary'] = (df['change'] == 'Ch').astype(int)
    
    # diabetes medication flag
    if 'diabetesMed' in df.columns:
        df['diabetesMed_binary'] = (df['diabetesMed'] == 'Yes').astype(int)
        
    return df

def encode_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    # A1C status
    if 'A1Cresult' in df.columns:
        # None is missing value (in pandas it might be 'None' string or actual NaN depending on preprocessing)
        df['a1c_tested'] = (~df['A1Cresult'].isin(['None', np.nan]) & df['A1Cresult'].notna()).astype(int)
        df['a1c_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)
        
    # Glucose status
    if 'max_glu_serum' in df.columns:
        df['glu_tested'] = (~df['max_glu_serum'].isin(['None', np.nan]) & df['max_glu_serum'].notna()).astype(int)
        df['glu_abnormal'] = df['max_glu_serum'].isin(['>200', '>300']).astype(int)
        
    return df

def group_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Discharge category
    if 'discharge_disposition_id' in df.columns:
        # 1 = Discharged to home
        # 6, 8 = Discharged to home with home health service
        # others often Transfer or SNF
        def map_discharge(val):
            if pd.isna(val): return 'Other'
            val = int(val)
            if val == 1: return 'Home'
            elif val in [6, 8]: return 'Home Health'
            elif val in [2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30]: return 'Transfer/SNF'
            return 'Other'
        df['discharge_category'] = df['discharge_disposition_id'].apply(map_discharge)
        logger.info("Grouped 'discharge_disposition_id' into 'discharge_category'")

    # Admission category
    if 'admission_type_id' in df.columns:
        def map_admission(val):
            if pd.isna(val): return 'Other'
            val = int(val)
            if val == 1: return 'Emergency'
            elif val == 2: return 'Urgent'
            elif val == 3: return 'Elective'
            return 'Other'
        df['admission_category'] = df['admission_type_id'].apply(map_admission)
        logger.info("Grouped 'admission_type_id' into 'admission_category'")

    # Medical specialty
    if 'medical_specialty' in df.columns:
        # Fill missing with Unknown
        df['medical_specialty_grouped'] = df['medical_specialty'].fillna('Unknown')
        
        # Keep top specialties, group rest to 'Other'
        top_specs = ['Unknown', 'InternalMedicine', 'Emergency/Trauma', 
                     'Family/GeneralPractice', 'Cardiology', 'Surgery-General']
        mask = ~df['medical_specialty_grouped'].isin(top_specs)
        df.loc[mask, 'medical_specialty_grouped'] = 'Other'
        logger.info("Grouped 'medical_specialty' into 'medical_specialty_grouped'")
        
    # Fill race missing
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for feature engineering pipeline.
    Assumes df has been cleaned by src.data.cleaner
    and ICD codes grouped by src.features.icd_grouper.
    """
    logger.info("Starting feature engineering...")
    df = df.copy()
    
    df = encode_age(df)
    df = create_utilization_features(df)
    df = extract_medication_features(df)
    df = encode_binary_flags(df)
    df = encode_lab_features(df)
    df = group_categories(df)
    
    logger.info("Feature engineering complete.")
    return df
