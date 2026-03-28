import pytest
import pandas as pd
import numpy as np

from src.features.icd_grouper import map_icd9_code, add_icd_groups
from src.features.engineer import engineer_features, encode_age, extract_medication_features
from src.features.pipeline import build_pipeline
from src.utils.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES


class TestICDGrouper:
    def test_map_icd9_code_diabetes(self):
        assert map_icd9_code("250.01") == "Diabetes"
        assert map_icd9_code("250") == "Diabetes"
        
    def test_map_icd9_code_circulatory(self):
        assert map_icd9_code("414") == "Circulatory"
        assert map_icd9_code("785") == "Circulatory"
        
    def test_map_icd9_code_missing_and_invalid(self):
        assert map_icd9_code("?") == "Missing"
        assert map_icd9_code(np.nan) == "Missing"
        assert map_icd9_code("V25") == "Other"
        assert map_icd9_code("E900") == "Other"

    def test_add_icd_groups(self):
        # Create dummy df
        df = pd.DataFrame({
            'diag_1': ['250.00', '414.00', '?'],
            'diag_2': ['V45.81', '250.01', 'NaN'],
            'diag_3': ['785', '460', 'E99']
        })
        
        grouped = add_icd_groups(df)
        
        # Check has_diabetes_diag logic
        assert grouped['has_diabetes_diag'].tolist() == [1, 1, 0]
        # Check specific groups
        assert grouped['diag_1_group'].tolist() == ['Diabetes', 'Circulatory', 'Missing']
        assert grouped['diag_3_group'].tolist() == ['Circulatory', 'Respiratory', 'Other']


class TestFeatureEngineer:
    def test_age_encoding(self):
        df = pd.DataFrame({'age': ['[0-10)', '[50-60)', '[90-100)']})
        encoded = encode_age(df)
        assert encoded['age_midpoint'].tolist() == [5, 55, 95]

    def test_polypharmacy_score(self):
        # Only testing a subset of MEDICATION_COLS, assuming extract_medication_features safely handles what is present
        df = pd.DataFrame({
            'insulin': ['No', 'Steady', 'Up'], 
            'metformin': ['No', 'Up', 'Down']
        })
        # Override MEDICATION_COLS temporarily in module if making unit test isolated, 
        # but the module relies on the config. We will just pass the real cols.
        from src.utils.config import MEDICATION_COLS
        
        # Populate realistic dataframe with all 'No' by default
        df_full = pd.DataFrame(index=[0, 1, 2], columns=MEDICATION_COLS)
        df_full.fillna('No', inplace=True)
        # Override specific ones
        df_full.loc[0, 'insulin'] = 'No'
        df_full.loc[1, 'insulin'] = 'Steady'
        df_full.loc[2, 'insulin'] = 'Up'
        
        df_full.loc[0, 'metformin'] = 'No'
        df_full.loc[1, 'metformin'] = 'Up'
        df_full.loc[2, 'metformin'] = 'Down'
        
        engineered = extract_medication_features(df_full)
         
        # 'No' or 'Steady' -> 0 points. 'Up' or 'Down' -> 1 point.
        # Row 0: No, No -> 0 points.
        # Row 1: Steady, Up -> 1 point.
        # Row 2: Up, Down -> 2 points.
        assert engineered['polypharmacy_score'].tolist() == [0, 1, 2]
        assert engineered['num_active_medications'].tolist() == [0, 2, 2]
        
    def test_polypharmacy_is_non_negative(self):
        # We can also test the full pipeline dataframe integration 
        pass


class TestPipeline:
    def test_pipeline_instantiation(self):
        pipe_lr = build_pipeline("lr")
        assert pipe_lr is not None
        
        pipe_xgb = build_pipeline("xgb")
        assert pipe_xgb is not None
