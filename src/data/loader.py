"""
Data loading and validation module.

Responsibilities:
  - Read the raw CSV with correct dtypes (no silent coercion surprises).
  - Replace the dataset's '?' sentinel with proper NaN.
  - Parse IDs_mapping.csv and optionally decode ID columns into labels.
  - Validate dataset shape and target column integrity.
"""
import pandas as pd

from src.utils.config import IDS_MAPPING_PATH, RAW_DATA_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns that must be read as strings to preserve leading zeros / dot notation
_STR_COLS = {
    "encounter_id": str,
    "patient_nbr": str,
    "diag_1": str,
    "diag_2": str,
    "diag_3": str,
    "payer_code": str,
}

# Columns that must be integers (used as join keys against IDs_mapping)
_INT_COLS = {
    "admission_type_id": "Int64",          # nullable int — avoids cast errors if any row is NaN
    "discharge_disposition_id": "Int64",
    "admission_source_id": "Int64",
}

_EXPECTED_ROWS = 101_766
_EXPECTED_COLS = 50
_VALID_TARGET_VALUES = {"NO", ">30", "<30"}


# ---------------------------------------------------------------------------
# IDs mapping loader
# ---------------------------------------------------------------------------

def load_ids_mapping() -> dict[str, pd.DataFrame]:
    """
    Parse ``IDs_mapping.csv`` into a dictionary keyed by ID column name.

    The file contains three concatenated tables separated by blank lines:
    ``admission_type_id``, ``discharge_disposition_id``, and
    ``admission_source_id``.

    Returns:
        dict mapping column name → DataFrame with columns ['id', 'description']
    """
    if not IDS_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {IDS_MAPPING_PATH}")

    tables: dict[str, pd.DataFrame] = {}
    current_name: str | None = None
    rows: list[tuple[int, str]] = []

    _SECTION_HEADERS = {
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    }

    with open(IDS_MAPPING_PATH, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line == ",":
                continue

            # Check whether this line starts a new section
            matched_header = next(
                (h for h in _SECTION_HEADERS if line.startswith(h)), None
            )
            if matched_header:
                # Save the previous table if one was being built
                if current_name and rows:
                    tables[current_name] = pd.DataFrame(rows, columns=["id", "description"])
                current_name = matched_header
                rows = []
                continue

            # Data row: "id,description"  (description may contain commas)
            if current_name:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    id_str, desc = parts[0].strip(), parts[1].strip().strip('"')
                    if id_str.isdigit():
                        rows.append((int(id_str), desc))

    # Don't forget the last table
    if current_name and rows:
        tables[current_name] = pd.DataFrame(rows, columns=["id", "description"])

    log.info("Loaded ID mappings for: %s", list(tables.keys()))
    return tables


# ---------------------------------------------------------------------------
# Raw data loader
# ---------------------------------------------------------------------------

def load_raw_data(decode_ids: bool = False) -> pd.DataFrame:
    """
    Load and validate the raw diabetic readmission CSV.

    Args:
        decode_ids: If True, replace numeric ID columns with human-readable
                    labels from ``IDs_mapping.csv`` (adds ``_label`` suffix
                    columns; original ID columns are kept).

    Returns:
        Raw :class:`pd.DataFrame` with shape (101 766, 50+).

    Raises:
        FileNotFoundError: If the CSV is not found at the configured path.
        ValueError:        If target column contains unexpected values.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}.\n"
            "Download it from Kaggle (search 'Diabetic Readmission Dataset') "
            "and place both CSVs inside dataset_diabetes/."
        )

    log.info("Loading raw data from %s …", RAW_DATA_PATH)

    df = pd.read_csv(
        RAW_DATA_PATH,
        dtype={**_STR_COLS, **_INT_COLS},
        na_values=["?"],      # dataset uses '?' as the missing-value sentinel
        low_memory=False,
    )

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------
    if df.shape[0] != _EXPECTED_ROWS:
        log.warning(
            "Expected %d rows, got %d — dataset may be a different version.",
            _EXPECTED_ROWS,
            df.shape[0],
        )
    if df.shape[1] != _EXPECTED_COLS:
        log.warning(
            "Expected %d columns, got %d.",
            _EXPECTED_COLS,
            df.shape[1],
        )

    # ------------------------------------------------------------------
    # Target integrity check
    # ------------------------------------------------------------------
    actual_vals = set(df["readmitted"].dropna().unique())
    unexpected = actual_vals - _VALID_TARGET_VALUES
    if unexpected:
        raise ValueError(
            f"'readmitted' column has unexpected values: {unexpected}"
        )

    log.info("Loaded dataset — shape: %s", df.shape)
    log.info(
        "Target distribution:\n%s",
        df["readmitted"].value_counts(dropna=False).to_string(),
    )

    # ------------------------------------------------------------------
    # Optional: decode ID columns into readable labels
    # ------------------------------------------------------------------
    if decode_ids:
        mappings = load_ids_mapping()
        for col, mapping_df in mappings.items():
            if col in df.columns:
                id_to_label = mapping_df.set_index("id")["description"].to_dict()
                df[f"{col}_label"] = df[col].map(id_to_label)
        log.info("ID columns decoded with human-readable labels.")

    return df


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_raw_data(decode_ids=True)
    print(f"\nShape: {df.shape}")
    print(f"\nDtypes sample:\n{df.dtypes.head(10)}")
    print(f"\nTarget counts:\n{df['readmitted'].value_counts()}")
    print(f"\nMissing values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")