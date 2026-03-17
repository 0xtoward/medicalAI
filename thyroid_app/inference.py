"""Runtime inference helpers for exported Streamlit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from thyroid_app.constants import (
    FIXED_LANDMARKS,
    LAB_FIELD_RANGES,
    STATIC_FIELD_RANGES,
    STATE_TO_CODE,
    required_state_timestamps_for_landmark,
    required_timestamps_for_landmark,
    timestamp_to_index,
)
from thyroid_app.features import make_relapse_features
from utils.config import STATIC_NAMES, TIME_STAMPS
from utils.data import extract_flat_features


def load_manifest(base_dir: str | Path = "artifacts") -> dict:
    manifest_path = Path(base_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle(task: str, base_dir: str | Path = "artifacts") -> dict:
    manifest = load_manifest(base_dir)
    bundle_path = Path(manifest["tasks"][task]["bundle_path"])
    if not bundle_path.is_absolute():
        bundle_path = Path(base_dir).parent / bundle_path
    return joblib.load(bundle_path)


def _check_range(field_name: str, value: float, lower: float, upper: float, messages: list[str], label: str) -> None:
    if value < lower or value > upper:
        messages.append(f"{label} `{field_name}` is outside the expected range [{lower}, {upper}].")


def validate_case_payload(case_payload: dict, landmark: str, task: str) -> list[str]:
    messages = []
    static = case_payload.get("static", {})
    series = case_payload.get("series", {})
    states = case_payload.get("states", {})

    for field in STATIC_NAMES:
        if field not in static or static[field] in ("", None):
            messages.append(f"Missing required static field: `{field}`.")
            continue
        _check_range(field, float(static[field]), *STATIC_FIELD_RANGES[field], messages, "Static field")

    required_timestamps = required_timestamps_for_landmark(landmark)
    for lab_name in ["FT3", "FT4", "TSH"]:
        lab_series = series.get(lab_name, {})
        for timestamp in required_timestamps:
            if timestamp not in lab_series or lab_series[timestamp] in ("", None):
                messages.append(f"Missing `{lab_name}` value at `{timestamp}`.")
                continue
            _check_range(lab_name, float(lab_series[timestamp]), *LAB_FIELD_RANGES[lab_name], messages, "Lab value")

    if task == "relapse":
        required_states = required_state_timestamps_for_landmark(landmark)
        for timestamp in required_states:
            state = states.get(timestamp)
            if state is None:
                messages.append(f"Missing clinician state at `{timestamp}`.")
                continue
            if state not in STATE_TO_CODE:
                messages.append(f"Invalid clinician state at `{timestamp}`: `{state}`.")
        if states.get(landmark) != "Normal":
            messages.append(
                "Relapse prediction requires the current state to be `Normal`. Use the Fixed Landmark Calculator otherwise."
            )
    return messages


def _series_to_dynamic_tensor(series: dict, required_timestamps: list[str]) -> np.ndarray:
    return np.stack(
        [
            np.array([float(series["FT3"][timestamp]) for timestamp in required_timestamps], dtype=float),
            np.array([float(series["FT4"][timestamp]) for timestamp in required_timestamps], dtype=float),
            np.array([float(series["TSH"][timestamp]) for timestamp in required_timestamps], dtype=float),
        ],
        axis=-1,
    )


def _build_relapse_feature_row(case_payload: dict, landmark: str, bundle: dict) -> pd.DataFrame:
    required_timestamps = required_timestamps_for_landmark(landmark)
    current_index = timestamp_to_index(landmark)
    lab_matrix = _series_to_dynamic_tensor(case_payload["series"], required_timestamps)
    state_codes = [STATE_TO_CODE["Hyper"]]
    for timestamp in required_state_timestamps_for_landmark(landmark):
        state_codes.append(STATE_TO_CODE[case_payload["states"][timestamp]])
    hist_states = np.asarray(state_codes[:current_index], dtype=int)
    prior_relapse_count = int(
        sum(int(hist_states[idx] == 1 and hist_states[idx + 1] == 0) for idx in range(len(hist_states) - 1))
    )
    prev_state = int(state_codes[current_index - 1]) if current_index > 0 else -1
    labs_k = lab_matrix[current_index]
    labs_0 = lab_matrix[0]
    if current_index > 0:
        labs_prev = lab_matrix[current_index - 1]
        delta_ft4_1step = float(labs_k[1] - labs_prev[1])
        delta_tsh_1step = float(np.log1p(max(labs_k[2], 0.0)) - np.log1p(max(labs_prev[2], 0.0)))
    else:
        delta_ft4_1step = 0.0
        delta_tsh_1step = 0.0
    row = pd.DataFrame(
        [
            {
                **{field: float(case_payload["static"][field]) for field in STATIC_NAMES},
                "Interval_Name": f"{landmark}->{TIME_STAMPS[current_index + 1]}",
                "Prev_State": str(prev_state),
                "FT3_Current": float(labs_k[0]),
                "FT4_Current": float(labs_k[1]),
                "logTSH_Current": float(np.log1p(max(labs_k[2], 0.0))),
                "Ever_Hyper_Before": int(prior_relapse_count > 0),
                "Ever_Hypo_Before": int(2 in hist_states),
                "Time_In_Normal": int(np.sum(hist_states == 1)),
                "Delta_FT4_k0": float(labs_k[1] - labs_0[1]),
                "Delta_TSH_k0": float(np.log1p(max(labs_k[2], 0.0)) - np.log1p(max(labs_0[2], 0.0))),
                "Delta_FT4_1step": delta_ft4_1step,
                "Delta_TSH_1step": delta_tsh_1step,
            }
        ]
    )
    return make_relapse_features(row, bundle["interval_categories"], bundle["prev_state_categories"])


def _build_fixed_feature_row(case_payload: dict, landmark: str, bundle: dict) -> pd.DataFrame:
    seq_len = FIXED_LANDMARKS[landmark]
    required_timestamps = TIME_STAMPS[:seq_len]
    static_vector = np.array([[float(case_payload["static"][field]) for field in STATIC_NAMES]], dtype=float)
    dynamic = _series_to_dynamic_tensor(case_payload["series"], required_timestamps)[None, :, :]
    feature_row, _ = extract_flat_features(static_vector, dynamic, seq_len)
    scaled = bundle["outer_scaler"].transform(feature_row)
    return pd.DataFrame(scaled, columns=bundle["feature_names"])


def _linear_top_drivers(bundle: dict, feature_frame: pd.DataFrame, limit: int = 5) -> list[dict]:
    model = bundle["model"]
    if "lr" not in getattr(model, "named_steps", {}):
        return []
    scaler = model.named_steps["scaler"]
    lr = model.named_steps["lr"]
    transformed = scaler.transform(feature_frame)[0]
    contributions = lr.coef_.ravel() * transformed
    ranked = sorted(
        zip(bundle["feature_names"], contributions),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [{"feature": name, "contribution": float(value)} for name, value in ranked[:limit]]


def _tree_top_drivers(bundle: dict, limit: int = 5) -> list[dict]:
    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        ranked = sorted(
            zip(bundle["feature_names"], model.feature_importances_),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return [{"feature": name, "importance": float(value)} for name, value in ranked[:limit]]
    return []


def predict_relapse(case_payload: dict, base_dir: str | Path = "artifacts") -> dict:
    landmark = case_payload["landmark"]
    bundle = load_bundle("relapse", base_dir)
    validation_messages = validate_case_payload(case_payload, landmark, "relapse")
    if validation_messages:
        return {
            "model_name": bundle["model_name"],
            "landmark": landmark,
            "predicted_probability": None,
            "decision_threshold": bundle["decision_threshold"],
            "decision_label": None,
            "top_drivers": [],
            "validation_messages": validation_messages,
        }

    feature_row = _build_relapse_feature_row(case_payload, landmark, bundle)
    probability = float(bundle["model"].predict_proba(feature_row)[:, 1][0])
    remaining_visits = max(1, len(TIME_STAMPS) - 1 - timestamp_to_index(landmark))
    projected_any_relapse = float(1 - (1 - probability) ** remaining_visits)
    top_drivers = _linear_top_drivers(bundle, feature_row, limit=5)
    quartiles = bundle["patient_quartiles"]
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    risk_group = quartile_labels[min(sum(projected_any_relapse > q for q in quartiles[1:-1]), 3)]
    decision_label = "High risk" if probability >= bundle["decision_threshold"] else "Lower risk"
    return {
        "model_name": bundle["model_name"],
        "landmark": landmark,
        "predicted_probability": probability,
        "decision_threshold": bundle["decision_threshold"],
        "decision_label": decision_label,
        "top_drivers": top_drivers,
        "validation_messages": [],
        "projected_any_relapse_risk": projected_any_relapse,
        "projected_risk_group": risk_group,
        "projection_note": "Projected follow-up risk assumes the current next-window risk persists across remaining visits.",
    }


def predict_fixed_landmark(case_payload: dict, landmark: str, base_dir: str | Path = "artifacts") -> dict:
    task = "fixed_3m" if landmark == "3M" else "fixed_6m"
    bundle = load_bundle(task, base_dir)
    validation_messages = validate_case_payload(case_payload, landmark, "fixed")
    if validation_messages:
        return {
            "model_name": bundle["model_name"],
            "landmark": landmark,
            "predicted_probability": None,
            "decision_threshold": bundle["decision_threshold"],
            "decision_label": None,
            "top_drivers": [],
            "validation_messages": validation_messages,
        }
    feature_row = _build_fixed_feature_row(case_payload, landmark, bundle)
    probability = float(bundle["model"].predict_proba(feature_row)[:, 1][0])
    top_drivers = _linear_top_drivers(bundle, feature_row, limit=5) or _tree_top_drivers(bundle, limit=5)
    decision_label = "Likely Hyper" if probability >= bundle["decision_threshold"] else "Likely Non-Hyper"
    return {
        "model_name": bundle["model_name"],
        "landmark": landmark,
        "predicted_probability": probability,
        "decision_threshold": bundle["decision_threshold"],
        "decision_label": decision_label,
        "top_drivers": top_drivers,
        "validation_messages": [],
    }
