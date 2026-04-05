import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Elixhauser Mapping (simplified ICD-9 prefix matching for 31 categories)
# Using van Walraven weights which predict hospital mortality and readmission risk.
ELIXHAUSER_WEIGHTS = {
    "congestive_heart_failure": 7,
    "cardiac_arrhythmia": 5,
    "valvular_disease": -1,
    "pulmonary_circulation": 4,
    "peripheral_vascular": 2,
    "hypertension": 0,
    "paralysis": 7,
    "other_neurological": 6,
    "chronic_pulmonary": 3,
    "diabetes": 0, # They are already diabetic in this dataset
    "hypothyroidism": 0,
    "renal_failure": 5,
    "liver_disease": 11,
    "peptic_ulcer": 0,
    "aids_hiv": 0,
    "lymphoma": 9,
    "metastatic_cancer": 12,
    "solid_tumor": 4,
    "rheumatoid_arthritis": 0,
    "coagulopathy": 3,
    "obesity": -4,
    "weight_loss": 6,
    "fluid_electrolyte": 5,
    "blood_loss_anemia": -2,
    "deficiency_anemia": -2,
    "alcohol_abuse": 0,
    "drug_abuse": -7,
    "psychoses": 0,
    "depression": -3
}

# Simplified prefix/regex matching for ICD-9 groups
ELIXHAUSER_REGEX = {
    "congestive_heart_failure": r"^398\.91|^428",
    "cardiac_arrhythmia": r"^426|^427",
    "valvular_disease": r"^093\.2|^394|^395|^396|^397|^424|^V42\.2|^V43\.3",
    "pulmonary_circulation": r"^416|^417\.9",
    "peripheral_vascular": r"^440|^441|^442|^443|^444|^447\.1|^449|^V43\.4",
    "hypertension": r"^401",
    "paralysis": r"^342|^343|^344",
    "other_neurological": r"^331\.9|^332|^333|^334|^335|^336|^340|^341|^345|^348|^349|^780\.3|^784\.3",
    "chronic_pulmonary": r"^490|^491|^492|^493|^494|^495|^496|^500|^501|^502|^503|^504|^505",
    "renal_failure": r"^584|^585|^586|^V42\.0|^V45\.1|^V56",
    "liver_disease": r"^070\.22|^070\.23|^070\.32|^070\.33|^070\.44|^070\.54|^456\.0|^456\.1|^456\.2|^570|^571|^572|^573\.3|^573\.4|^573\.8|^573\.9|^V42\.7",
    "metastatic_cancer": r"^196|^197|^198|^199",
    "solid_tumor": r"^14|^15|^16|^17|^18|^190|^191|^192|^193|^194|^195",
    "fluid_electrolyte": r"^276",
    "weight_loss": r"^260|^261|^262|^263|^783\.2|^799\.4",
    "coagulopathy": r"^286|^287\.1|^287\.3|^287\.4|^287\.5",
    "lymphoma": r"^200|^201|^202|^203|^204|^205|^206|^207|^208",
    "obesity": r"^278\.0"
}

def clean_icd(code: str) -> str:
    """Removes V/E, handles floats masquerading as strings."""
    if pd.isna(code) or str(code) == '?' or str(code).lower() == 'nan':
        return ""
    code_str = str(code).strip().upper()
    return code_str

def calculate_elixhauser_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a composite Elixhauser Comorbidity Score (Van Walraven weighted).
    Uses diag_1, diag_2, diag_3.
    """
    logger.info("Computing Elixhauser Comorbidity Index scores...")
    df_copy = df.copy()
    
    # Initialize score column
    # Use float32 to save memory
    df_copy['elixhauser_score'] = np.zeros(len(df_copy), dtype=np.float32)
    
    # Pre-clean codes for regex
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        if col in df_copy.columns:
            df_copy[f"{col}_clean"] = df_copy[col].apply(clean_icd)
            
    # Iterate through categories and assign weights if matched
    for category, pattern in ELIXHAUSER_REGEX.items():
        weight = ELIXHAUSER_WEIGHTS.get(category, 0)
        
        if weight == 0:
            continue
            
        compiled_regex = re.compile(pattern)
        
        # Check if patient has this comorbidity across any of the 3 diags
        has_comorbidity = np.zeros(len(df_copy), dtype=bool)
        for col in [f"{c}_clean" for c in diag_cols if c in df_copy.columns]:
            # Regex match
            has_comorbidity = has_comorbidity | df_copy[col].str.match(compiled_regex).fillna(False)
            
        # Add weights
        df_copy.loc[has_comorbidity, 'elixhauser_score'] += weight
        
    # Clean up intermediate columns
    columns_to_drop = [f"{c}_clean" for c in diag_cols if f"{c}_clean" in df_copy.columns]
    df_copy = df_copy.drop(columns=columns_to_drop)
    
    score_mean = df_copy['elixhauser_score'].mean()
    score_positive = (df_copy['elixhauser_score'] > 0).mean()
    
    logger.info(f"Elixhauser computation complete. Mean score: {score_mean:.2f}, Positively scored patients: {score_positive:.1%}")
    return df_copy
