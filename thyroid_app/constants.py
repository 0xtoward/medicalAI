"""Shared constants for the Streamlit deployment layer."""

from utils.config import STATIC_NAMES, TIME_STAMPS

STATE_TO_CODE = {"Hyper": 0, "Normal": 1, "Hypo": 2}
CODE_TO_STATE = {v: k for k, v in STATE_TO_CODE.items()}

RELAPSE_CURRENT_LANDMARKS = ["1M", "3M", "6M", "12M", "18M"]
FIXED_LANDMARKS = {"3M": 3, "6M": 4}

RELAPSE_FEATURE_COLUMNS = STATIC_NAMES + [
    "FT3_Current",
    "FT4_Current",
    "logTSH_Current",
    "Ever_Hyper_Before",
    "Ever_Hypo_Before",
    "Time_In_Normal",
    "Delta_FT4_k0",
    "Delta_TSH_k0",
    "Delta_FT4_1step",
    "Delta_TSH_1step",
]

SERIES_NAMES = ["FT3", "FT4", "TSH"]

STATIC_FIELD_HELP = {
    "Sex": "0 = Female, 1 = Male",
    "Age": "Years",
    "Height": "cm",
    "Weight": "kg",
    "BMI": "kg/m^2",
    "Exophthalmos": "0 = No, 1 = Yes",
    "ThyroidW": "Thyroid weight / volume field from the source dataset",
    "RAI3d": "3-day RAI-related field from the source dataset",
    "TreatCount": "Prior treatment count",
    "TGAb": "Thyroglobulin antibody",
    "TPOAb": "Thyroid peroxidase antibody",
    "TRAb": "Thyrotropin receptor antibody",
    "Uptake24h": "24-hour uptake",
    "MaxUptake": "Maximum uptake",
    "HalfLife": "Effective half-life",
    "Dose": "Administered RAI dose",
}

STATIC_FIELD_RANGES = {
    "Sex": (0.0, 1.0),
    "Age": (0.0, 120.0),
    "Height": (0.5, 2.5),
    "Weight": (20.0, 300.0),
    "BMI": (10.0, 80.0),
    "Exophthalmos": (0.0, 1.0),
    "ThyroidW": (0.0, 500.0),
    "RAI3d": (0.0, 100.0),
    "TreatCount": (0.0, 20.0),
    "TGAb": (0.0, 5000.0),
    "TPOAb": (0.0, 5000.0),
    "TRAb": (0.0, 1000.0),
    "Uptake24h": (0.0, 100.0),
    "MaxUptake": (0.0, 100.0),
    "HalfLife": (0.0, 30.0),
    "Dose": (0.0, 1000.0),
}

LAB_FIELD_RANGES = {
    "FT3": (0.0, 100.0),
    "FT4": (0.0, 200.0),
    "TSH": (0.0, 500.0),
}


def timestamp_to_index(timestamp: str) -> int:
    return TIME_STAMPS.index(timestamp)


def required_timestamps_for_landmark(landmark: str) -> list[str]:
    end_idx = timestamp_to_index(landmark)
    return TIME_STAMPS[: end_idx + 1]


def required_state_timestamps_for_landmark(landmark: str) -> list[str]:
    end_idx = timestamp_to_index(landmark)
    return TIME_STAMPS[1 : end_idx + 1]
