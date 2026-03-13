"""Shared constants for all prediction scripts."""

FILE_PATH = "1003.xlsx"
SEED = 42

COL_IDX = {
    'ID': 0, 'Outcome': 14,
    'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
    'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
    'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
    'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    'Eval_Cols': [35, 44, 53, 62, 71, 80],
}

STATIC_NAMES = [
    "Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos",
    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb",
    "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose",
]

TIME_STAMPS = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
STATE_NAMES = ["Hyper", "Normal", "Hypo"]
