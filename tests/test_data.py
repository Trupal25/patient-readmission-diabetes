"""
Phase 1 acceptance tests — data loading, cleaning, and splitting.

Run with:
    pytest tests/test_data.py -v
"""
import pytest
import pandas as pd

from src.data.loader import load_raw_data, load_ids_mapping
from src.data.cleaner import clean
from src.data.splitter import split_data
from src.utils.config import (
    DECEASED_DISPOSITION_IDS,
    TARGET_BINARY_COL,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw_data()


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_shape(self, raw_df: pd.DataFrame):
        """Dataset must have exactly 101,766 rows and 50 columns."""
        assert raw_df.shape == (101_766, 50), (
            f"Unexpected shape: {raw_df.shape}"
        )

    def test_target_values(self, raw_df: pd.DataFrame):
        """Target column must only contain the three expected labels."""
        valid = {"NO", ">30", "<30"}
        actual = set(raw_df["readmitted"].unique())
        assert actual.issubset(valid), f"Unexpected target values: {actual - valid}"

    def test_missing_sentinel_replaced(self, raw_df: pd.DataFrame):
        """'?' should have been converted to NaN, not left as a string."""
        for col in raw_df.columns:
            if raw_df[col].dtype == object:
                assert "?" not in raw_df[col].values, (
                    f"Column '{col}' still contains '?' after loading."
                )

    def test_diag_cols_are_strings(self, raw_df: pd.DataFrame):
        """Diagnosis columns must stay as strings (not cast to float)."""
        for col in ("diag_1", "diag_2", "diag_3"):
            assert raw_df[col].dtype == object, (
                f"Column '{col}' should be string dtype, got {raw_df[col].dtype}"
            )

    def test_ids_mapping_keys(self):
        """IDs mapping must return all three expected keys."""
        mappings = load_ids_mapping()
        expected_keys = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
        assert expected_keys == set(mappings.keys()), (
            f"Mapping keys mismatch: {set(mappings.keys())}"
        )

    def test_ids_mapping_non_empty(self):
        """Each mapping table must have at least one row."""
        for name, df in load_ids_mapping().items():
            assert len(df) > 0, f"Mapping '{name}' is empty."


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestCleaner:
    def test_no_deceased_patients(self, clean_df: pd.DataFrame):
        """Expired/hospice discharge dispositions must be removed."""
        remaining = clean_df["discharge_disposition_id"].isin(DECEASED_DISPOSITION_IDS)
        assert not remaining.any(), (
            f"Found {remaining.sum()} expired/hospice rows after cleaning."
        )

    def test_no_duplicate_patients(self, clean_df: pd.DataFrame):
        """patient_nbr column must be dropped after deduplication."""
        assert "patient_nbr" not in clean_df.columns, (
            "patient_nbr should be dropped after dedup."
        )

    def test_weight_dropped(self, clean_df: pd.DataFrame):
        assert "weight" not in clean_df.columns

    def test_encounter_id_dropped(self, clean_df: pd.DataFrame):
        assert "encounter_id" not in clean_df.columns

    def test_target_is_binary(self, clean_df: pd.DataFrame):
        """Target must be 0/1 integer after binarization."""
        assert TARGET_BINARY_COL in clean_df.columns, (
            f"'{TARGET_BINARY_COL}' column missing."
        )
        unique_vals = set(clean_df[TARGET_BINARY_COL].unique())
        assert unique_vals.issubset({0, 1}), (
            f"Non-binary values in target: {unique_vals}"
        )

    def test_original_target_dropped(self, clean_df: pd.DataFrame):
        assert "readmitted" not in clean_df.columns

    def test_row_count_reduced(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame):
        """Cleaning should reduce the row count (dedup + deceased removal)."""
        assert len(clean_df) < len(raw_df), (
            "Expected fewer rows after cleaning."
        )

    def test_positive_rate_reasonable(self, clean_df: pd.DataFrame):
        """30-day readmission rate must be between 5% and 25%."""
        rate = clean_df[TARGET_BINARY_COL].mean()
        assert 0.05 < rate < 0.25, f"Unexpected positive rate: {rate:.3f}"


# ---------------------------------------------------------------------------
# Splitter tests
# ---------------------------------------------------------------------------

class TestSplitter:
    def test_split_sizes(self, clean_df: pd.DataFrame):
        """Splits must sum to total rows with correct proportions."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            clean_df, save_indices=False
        )
        total = len(clean_df)
        assert abs(len(X_test) / total - 0.15) < 0.02, "Test split ~15%"
        assert abs(len(X_val) / total - 0.15) < 0.02, "Val split ~15%"
        assert abs(len(X_train) / total - 0.70) < 0.02, "Train split ~70%"
        assert len(X_train) + len(X_val) + len(X_test) == total

    def test_stratification(self, clean_df: pd.DataFrame):
        """Positive-class rate must be consistent across all splits (± 2%)."""
        _, _, _, y_train, y_val, y_test = split_data(
            clean_df, save_indices=False
        )
        rate_train = y_train.mean()
        rate_val = y_val.mean()
        rate_test = y_test.mean()
        assert abs(rate_train - rate_val) < 0.02, "Train/val positive rate diverges"
        assert abs(rate_train - rate_test) < 0.02, "Train/test positive rate diverges"

    def test_no_overlap(self, clean_df: pd.DataFrame):
        """Train, val, and test indices must be disjoint."""
        X_train, X_val, X_test, *_ = split_data(clean_df, save_indices=False)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx), "Train and val share indices."
        assert train_idx.isdisjoint(test_idx), "Train and test share indices."
        assert val_idx.isdisjoint(test_idx), "Val and test share indices."
