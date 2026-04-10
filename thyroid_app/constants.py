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
    "Sex": "0 = 女性, 1 = 男性",
    "Age": "单位：岁",
    "Height": "单位：米",
    "Weight": "单位：千克",
    "BMI": "单位：kg/m^2",
    "Exophthalmos": "0 = 无, 1 = 有",
    "ThyroidW": "原始数据中的甲状腺重量/体积字段",
    "RAI3d": "原始数据中的 3 日 RAI 相关字段",
    "TreatCount": "既往治疗次数",
    "TGAb": "甲状腺球蛋白抗体",
    "TPOAb": "甲状腺过氧化物酶抗体",
    "TRAb": "促甲状腺激素受体抗体",
    "Uptake24h": "24 小时摄取率",
    "MaxUptake": "最大摄取率",
    "HalfLife": "有效半衰期",
    "Dose": "放射性碘剂量",
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
