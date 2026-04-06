import pandas as pd

from src.evaluation.dashboard_bundle import build_form_defaults, build_form_options


def test_build_form_defaults_uses_numeric_medians_and_categorical_modes():
    frame = pd.DataFrame(
        {
            "time_in_hospital": [1, 5, 9],
            "race": ["Caucasian", "AfricanAmerican", "Caucasian"],
            "gender": ["Male", "Female", "Female"],
        }
    )

    defaults = build_form_defaults(frame)

    assert defaults["time_in_hospital"] == 5.0
    assert defaults["race"] == "Caucasian"
    assert defaults["gender"] == "Female"


def test_build_form_options_returns_sorted_unique_values():
    frame = pd.DataFrame(
        {
            "admission_category": ["Emergency", "Urgent", "Emergency"],
            "admission_source_id": [7, 1, 7],
            "diag_1_group": ["Circulatory", "Respiratory", "Circulatory"],
            "discharge_category": ["Home", "SNF", "Home"],
            "gender": ["Female", "Male", "Female"],
            "medical_specialty_grouped": ["InternalMedicine", "Cardiology", "Cardiology"],
            "race": ["Caucasian", "AfricanAmerican", "Caucasian"],
        }
    )

    options = build_form_options(frame)

    assert options["admission_category"] == ["Emergency", "Urgent"]
    assert options["admission_source_id"] == [1, 7]
    assert options["diag_1_group"] == ["Circulatory", "Respiratory"]
    assert options["discharge_category"] == ["Home", "SNF"]
    assert options["gender"] == ["Female", "Male"]
    assert options["medical_specialty_grouped"] == ["Cardiology", "InternalMedicine"]
    assert options["race"] == ["AfricanAmerican", "Caucasian"]
