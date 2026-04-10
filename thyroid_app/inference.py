"""Runtime inference helpers for exported Streamlit artifacts."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from thyroid_app.constants import (
    FIXED_LANDMARKS,
    LAB_FIELD_RANGES,
    SERIES_NAMES,
    STATIC_FIELD_RANGES,
    STATE_TO_CODE,
    required_state_timestamps_for_landmark,
    required_timestamps_for_landmark,
    timestamp_to_index,
)
from thyroid_app.features import make_relapse_features
from utils.physio_forecast import CURRENT_CONTEXT_COLS, add_stage1_prediction_family, build_physio_history_features
from utils.config import STATIC_NAMES, TIME_STAMPS
from utils.data import extract_flat_features, split_imputed
from utils.recurrence import build_interval_risk_data


HIGH_RISK_WINDOWS = ["1M->3M", "3M->6M", "6M->12M"]
GLOBAL_CORE_FEATURES = [
    "Time_In_Normal",
    "FT3_Current",
    "Delta_TSH_k0",
    "FT4_Current",
    "logTSH_Current",
    "Prior_Relapse_Count",
    "Ever_Hyper_Before",
    "Ever_Hypo_Before",
    "Start_Time",
    "Interval_Width",
]
GLOBAL_CORE_INTERACTION_FEATURES = [
    "Time_In_Normal",
    "FT3_Current",
    "Delta_TSH_k0",
]
THREEHEAD_DROPOUT = 0.15


def load_manifest(base_dir: str | Path = "artifacts") -> dict:
    manifest_path = Path(base_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=16)
def load_bundle(task: str, base_dir: str | Path = "artifacts") -> dict:
    manifest = load_manifest(base_dir)
    bundle_path = Path(manifest["tasks"][task]["bundle_path"])
    if not bundle_path.is_absolute():
        bundle_path = Path(base_dir).parent / bundle_path
    return joblib.load(bundle_path)


def _build_landmark_features_local(X_s, X_d, seq_len):
    X_base, feat_names = extract_flat_features(X_s, X_d, seq_len)
    labs = X_d[:, :seq_len, :]
    final = labs[:, -1, :]
    prev = labs[:, -2, :] if seq_len > 1 else labs[:, -1, :]
    first = labs[:, 0, :]
    eps = 1e-6

    thyroid_idx = STATIC_NAMES.index("ThyroidW")
    trab_idx = STATIC_NAMES.index("TRAb")
    extra_blocks = [
        np.column_stack(
            [
                final[:, 0] / (first[:, 0] + eps),
                final[:, 1] / (first[:, 1] + eps),
                (final[:, 2] + 1.0) / (first[:, 2] + 1.0),
            ]
        ),
        np.column_stack(
            [
                final[:, 0] / (final[:, 1] + eps),
                final[:, 1] / (final[:, 2] + 1.0),
            ]
        ),
        labs.mean(axis=1),
        labs.std(axis=1),
        final - prev,
        np.column_stack(
            [
                X_s[:, thyroid_idx] * final[:, 0],
                X_s[:, thyroid_idx] * final[:, 1],
                X_s[:, trab_idx] * final[:, 0],
                X_s[:, trab_idx] * final[:, 1],
            ]
        ),
    ]
    extra_names = [
        "FT3_last_over_0M",
        "FT4_last_over_0M",
        "TSH_last_over_0M_plus1",
        "FT3_last_over_FT4_last",
        "FT4_last_over_TSH_last_plus1",
        *[f"{lab}_mean" for lab in ["FT3", "FT4", "TSH"][: labs.shape[-1]]],
        *[f"{lab}_std" for lab in ["FT3", "FT4", "TSH"][: labs.shape[-1]]],
        *[f"{lab}_delta_last_prev" for lab in ["FT3", "FT4", "TSH"][: labs.shape[-1]]],
        "ThyroidW_x_FT3_last",
        "ThyroidW_x_FT4_last",
        "TRAb_x_FT3_last",
        "TRAb_x_FT4_last",
    ]
    X_full = np.hstack([X_base] + extra_blocks)
    return X_full, feat_names + extra_names


def _build_threehead_feature_blocks(df, interval_cats, prev_state_cats):
    static_df = df[STATIC_NAMES].copy().reset_index(drop=True)

    local_cols = [
        "FT3_Current",
        "FT4_Current",
        "logTSH_Current",
        "Delta_FT4_1step",
        "Delta_TSH_1step",
    ]
    local_df = df[local_cols].copy().reset_index(drop=True)
    for cat in interval_cats:
        local_df[f"Window_{cat}"] = (df["Interval_Name"].values == cat).astype(float)
    for cat in prev_state_cats:
        local_df[f"PrevState_{cat}"] = (df["Prev_State"].values == cat).astype(float)

    global_df = df[GLOBAL_CORE_FEATURES].copy().reset_index(drop=True)
    for cat in HIGH_RISK_WINDOWS:
        dummy = f"CoreWindow_{cat}"
        global_df[dummy] = (df["Interval_Name"].values == cat).astype(float)
        for feature in GLOBAL_CORE_INTERACTION_FEATURES:
            global_df[f"{feature}_x_{dummy}"] = global_df[feature].values * global_df[dummy].values

    return (
        static_df.apply(pd.to_numeric, errors="coerce").astype(float),
        local_df.apply(pd.to_numeric, errors="coerce").astype(float),
        global_df.apply(pd.to_numeric, errors="coerce").astype(float),
    )


def _transform_feature_blocks(scalers, static_df, local_df, global_df):
    return {
        "static": scalers["static"].transform(static_df).astype(np.float32),
        "local": scalers["local"].transform(local_df).astype(np.float32),
        "global": scalers["global"].transform(global_df).astype(np.float32),
    }


class _MLPBranch(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super().__init__()
        hidden_dim = max(embed_dim, min(128, max(32, input_dim * 2)))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _LinearBranch(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class _TeacherPretrainNet(nn.Module):
    def __init__(
        self,
        static_dim,
        local_dim,
        global_dim,
        embed_dim=48,
        dropout=THREEHEAD_DROPOUT,
        gate_mode="side_cap",
        side_gate_cap=None,
        global_floor=None,
        temperature=1.0,
    ):
        super().__init__()
        self.static_branch = _MLPBranch(static_dim, embed_dim, dropout)
        self.local_branch = _MLPBranch(local_dim, embed_dim, dropout)
        self.global_branch = _LinearBranch(global_dim, embed_dim)
        self.landmark_adapter = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.landmark_head = nn.Linear(embed_dim, 1)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
        )
        self.shared_norm = nn.LayerNorm(embed_dim)
        self.teacher_head = nn.Linear(embed_dim + 1, 1)
        self.next_hyper_head = nn.Linear(embed_dim + 1, 1)
        self.side_gate_cap = side_gate_cap
        self.global_floor = global_floor
        self.temperature = temperature

    def _apply_gate_constraints(self, gate_logits):
        gate = torch.softmax(gate_logits / self.temperature, dim=1)
        if self.side_gate_cap is not None:
            side_gate = torch.clamp(gate[:, :2], max=self.side_gate_cap)
            side_overflow = (gate[:, :2] - side_gate).clamp_min(0.0).sum(dim=1, keepdim=True)
            global_gate = gate[:, 2:3] + side_overflow
            gate = torch.cat([side_gate, global_gate], dim=1)
        if self.global_floor is not None:
            global_gate = gate[:, 2:3]
            side_gate = gate[:, :2]
            needs_floor = global_gate < self.global_floor
            if needs_floor.any():
                remaining = 1.0 - self.global_floor
                side_sum = side_gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
                rescaled_side = side_gate * (remaining / side_sum)
                side_gate = torch.where(needs_floor.expand_as(side_gate), rescaled_side, side_gate)
                floored_global = torch.full_like(global_gate, self.global_floor)
                global_gate = torch.where(needs_floor, floored_global, global_gate)
                gate = torch.cat([side_gate, global_gate], dim=1)
        return gate

    def forward(self, static_x, local_x, global_x, landmark_obs_mask=None, landmark_obs_value=None):
        h_static = self.static_branch(static_x)
        h_local = self.local_branch(local_x)
        h_global = self.global_branch(global_x)
        joint = torch.cat([h_static, h_local, h_global], dim=1)

        landmark_hidden = self.landmark_adapter(joint)
        landmark_logit = self.landmark_head(landmark_hidden).squeeze(1)
        landmark_prob = torch.sigmoid(landmark_logit)
        if landmark_obs_mask is None or landmark_obs_value is None:
            landmark_signal = landmark_prob
        else:
            landmark_signal = landmark_obs_mask * landmark_obs_value + (1.0 - landmark_obs_mask) * landmark_prob

        gate_in = torch.cat([joint, landmark_signal.unsqueeze(1)], dim=1)
        gate = self._apply_gate_constraints(self.gate_net(gate_in))
        fused = gate[:, 0:1] * h_static + gate[:, 1:2] * h_local + gate[:, 2:3] * h_global
        fused = self.shared_norm(fused)
        fused_with_landmark = torch.cat([fused, landmark_signal.unsqueeze(1)], dim=1)
        teacher_logit = self.teacher_head(fused_with_landmark).squeeze(1)
        next_hyper_logit = self.next_hyper_head(fused_with_landmark).squeeze(1)
        return {
            "teacher_logit": teacher_logit,
            "landmark_logit": landmark_logit,
            "next_hyper_logit": next_hyper_logit,
            "teacher_prob": torch.sigmoid(teacher_logit),
            "landmark_prob": landmark_prob,
            "landmark_signal": landmark_signal,
            "next_hyper_prob": torch.sigmoid(next_hyper_logit),
            "embedding": fused,
            "gate": gate,
        }


def _check_range(field_name: str, value: float, lower: float, upper: float, messages: list[str], label: str) -> None:
    if value < lower or value > upper:
        messages.append(f"{label} `{field_name}` 超出预期范围 [{lower}, {upper}]。")


def validate_case_payload(case_payload: dict, landmark: str, task: str) -> list[str]:
    messages = []
    static = case_payload.get("static", {})
    series = case_payload.get("series", {})
    states = case_payload.get("states", {})

    for field in STATIC_NAMES:
        if field not in static or static[field] in ("", None):
            messages.append(f"缺少必填静态变量：`{field}`。")
            continue
        _check_range(field, float(static[field]), *STATIC_FIELD_RANGES[field], messages, "静态变量")

    required_timestamps = required_timestamps_for_landmark(landmark)
    for lab_name in ["FT3", "FT4", "TSH"]:
        lab_series = series.get(lab_name, {})
        for timestamp in required_timestamps:
            if timestamp not in lab_series or lab_series[timestamp] in ("", None):
                messages.append(f"缺少 `{timestamp}` 时点的 `{lab_name}` 数值。")
                continue
            _check_range(lab_name, float(lab_series[timestamp]), *LAB_FIELD_RANGES[lab_name], messages, "实验室指标")

    if task == "relapse":
        required_states = required_state_timestamps_for_landmark(landmark)
        for timestamp in required_states:
            state = states.get(timestamp)
            if state is None:
                messages.append(f"缺少 `{timestamp}` 时点的临床状态。")
                continue
            if state not in STATE_TO_CODE:
                messages.append(f"`{timestamp}` 时点的临床状态无效：`{state}`。")
        if states.get(landmark) != "Normal":
            messages.append("动态复发预测要求当前时点状态为 `Normal`；若当前仍为甲亢，请先使用 `3M` 早期反应评估。")
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


def _build_relapse_base_row(case_payload: dict, landmark: str) -> pd.DataFrame:
    required_timestamps = required_timestamps_for_landmark(landmark)
    current_index = timestamp_to_index(landmark)
    static_vector = np.array([[float(case_payload["static"][field]) for field in STATIC_NAMES]], dtype=float)
    lab_matrix = _series_to_dynamic_tensor(case_payload["series"], required_timestamps)
    dyn_tensor = np.concatenate([lab_matrix, lab_matrix[-1:, :]], axis=0)[None, :, :]
    state_codes = [STATE_TO_CODE["Hyper"]]
    for timestamp in required_state_timestamps_for_landmark(landmark):
        state_codes.append(STATE_TO_CODE[case_payload["states"][timestamp]])
    state_codes.append(state_codes[-1])
    state_matrix = np.asarray(state_codes, dtype=int)[None, :]

    risk_df = build_interval_risk_data(
        static_vector,
        dyn_tensor,
        state_matrix,
        np.array(["case"], dtype=object),
        target_k=current_index,
        row_ids=np.array([0], dtype=int),
    )
    history_df = build_physio_history_features(
        dyn_tensor,
        state_matrix,
        np.array(["case"], dtype=object),
        target_k=current_index,
        row_ids=np.array([0], dtype=int),
    )
    return risk_df.merge(history_df, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")


def _build_relapse_feature_row(case_payload: dict, landmark: str, bundle: dict) -> pd.DataFrame:
    row = _build_relapse_base_row(case_payload, landmark)
    return make_relapse_features(row, bundle["interval_categories"], bundle["prev_state_categories"])


def _align_frame_to_bundle(frame: pd.DataFrame, feature_names: list[str], medians: dict | None = None) -> pd.DataFrame:
    out = frame.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    if medians:
        out = out.fillna(pd.Series(medians, dtype=float))
    out = out.reindex(columns=feature_names, fill_value=0.0)
    if medians:
        out = out.fillna(pd.Series(medians, dtype=float))
    return out.astype(float)


def _truncate_case_payload(case_payload: dict, landmark: str) -> dict:
    required = required_timestamps_for_landmark(landmark)
    return {
        "landmark": landmark,
        "static": case_payload["static"],
        "series": {
            lab: {ts: case_payload["series"][lab][ts] for ts in required}
            for lab in SERIES_NAMES
        },
        "states": {},
    }


def _pipeline_linear_top_drivers(model, feature_frame: pd.DataFrame, feature_names: list[str], limit: int = 5) -> list[dict]:
    if not hasattr(model, "named_steps"):
        return []
    scaler = model.named_steps.get("scaler")
    linear = model.named_steps.get("lr") or model.named_steps.get("clf")
    if scaler is None or linear is None or not hasattr(linear, "coef_"):
        return []
    transformed = scaler.transform(feature_frame)[0]
    contributions = linear.coef_.ravel() * transformed
    ranked = sorted(zip(feature_names, contributions), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "contribution": float(value)} for name, value in ranked[:limit]]


def _tree_top_drivers_from_model(model, feature_names: list[str], limit: int = 5) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda item: abs(item[1]), reverse=True)
        return [{"feature": name, "importance": float(value)} for name, value in ranked[:limit]]
    return []


def _build_stage1_feature_row(base_row: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    frame = base_row[[col for col in CURRENT_CONTEXT_COLS if col in base_row.columns]].copy().reset_index(drop=True)
    interval_name = str(base_row.iloc[0]["Interval_Name"])
    prev_state = str(base_row.iloc[0]["Prev_State"])
    for category in bundle["interval_categories"]:
        frame[f"Interval_Name_{category}"] = float(interval_name == category)
    for category in bundle["prev_state_categories"]:
        frame[f"Prev_State_{category}"] = float(prev_state == category)
    return _align_frame_to_bundle(frame, bundle["stage1_feature_names"], bundle["stage1_feature_medians"])


def _predict_stage1_delta_model(model, x_frame: pd.DataFrame, interval_name: str) -> np.ndarray:
    if isinstance(model, dict) and "__global__" in model:
        fitted = model.get(str(interval_name), model["__global__"])
        return np.asarray(fitted.predict(x_frame), dtype=float)
    return np.asarray(model.predict(x_frame), dtype=float)


def _predict_stage1_state_model(model, x_frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x_frame), dtype=float)


def _build_stage1_payload_for_case(base_row: pd.DataFrame, bundle: dict) -> dict:
    x_stage1 = _build_stage1_feature_row(base_row, bundle)
    interval_name = str(base_row.iloc[0]["Interval_Name"])

    delta_results = {}
    delta_base = {}
    for name, payload in bundle["stage1_delta_models"].items():
        pred = _predict_stage1_delta_model(payload["model"], x_stage1, interval_name)
        delta_base[name] = pred
        delta_results[name] = {"test_delta": pred}
    if delta_base:
        delta_results["EnsembleMean"] = {"test_delta": np.mean(np.stack(list(delta_base.values()), axis=0), axis=0)}

    state_results = {}
    state_base = {}
    for name, model in bundle["stage1_state_models"].items():
        proba = _predict_stage1_state_model(model, x_stage1)
        state_base[name] = proba
        state_results[name] = {"test_proba": proba}
    if state_base:
        state_results["EnsembleMean"] = {"test_proba": np.mean(np.stack(list(state_base.values()), axis=0), axis=0)}

    return {
        "delta_results": delta_results,
        "state_results": state_results,
        "hyper_margin_reference": bundle["hyper_margin_reference"],
    }


def _build_sidecar_feature_row(enriched_row: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    frame = enriched_row[list(bundle["sidecar_feature_names"])].copy().reset_index(drop=True)
    return _align_frame_to_bundle(frame, bundle["sidecar_feature_names"], bundle["sidecar_feature_medians"])


def _build_sidecar_calibration_row(base_row: pd.DataFrame, sidecar_logit: float, bundle: dict) -> pd.DataFrame:
    frame = pd.DataFrame({"Score_Logit": [float(sidecar_logit)]})
    phase = "HighRisk" if str(base_row.iloc[0]["Interval_Name"]) in set(bundle["high_risk_windows"]) else "Late"
    if bundle.get("sidecar_calibration_strategy") == "phase":
        frame["Phase_HighRisk"] = float(phase == "HighRisk")
        frame["Phase_Late"] = float(phase == "Late")
    return _align_frame_to_bundle(
        frame,
        bundle["sidecar_calibration_feature_names"],
        bundle["sidecar_calibration_feature_medians"],
    )


def _build_direct_windowed_feature_row(base_row: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    frame = base_row[[col for col in CURRENT_CONTEXT_COLS if col in base_row.columns]].copy().reset_index(drop=True)
    interval_name = str(base_row.iloc[0]["Interval_Name"])
    prev_state = str(base_row.iloc[0]["Prev_State"])
    for category in bundle["interval_categories"]:
        frame[f"Interval_Name_{category}"] = float(interval_name == category)
    for category in bundle["prev_state_categories"]:
        frame[f"Prev_State_{category}"] = float(prev_state == category)

    for dummy in [f"Interval_Name_{name}" for name in bundle["high_risk_windows"]]:
        if dummy not in frame.columns:
            frame[dummy] = float(interval_name == dummy.replace("Interval_Name_", ""))
    for feature in bundle["windowed_core_interaction_features"]:
        if feature not in frame.columns:
            continue
        for dummy in [f"Interval_Name_{name}" for name in bundle["high_risk_windows"]]:
            frame[f"{feature}_x_{dummy}"] = frame[feature].values * frame[dummy].values
    return _align_frame_to_bundle(frame, bundle["direct_feature_names"], bundle["direct_feature_medians"])


def _build_fusion_feature_row(base_row: pd.DataFrame, direct_proba: float, sidecar_proba: float, bundle: dict) -> pd.DataFrame:
    interval_name = str(base_row.iloc[0]["Interval_Name"])
    frame = pd.DataFrame(
        {
            "Direct_Logit": [float(np.log(np.clip(direct_proba, 1e-6, 1 - 1e-6) / np.clip(1 - direct_proba, 1e-6, 1 - 1e-6)))],
            "Physio_Main_Logit": [float(np.log(np.clip(sidecar_proba, 1e-6, 1 - 1e-6) / np.clip(1 - sidecar_proba, 1e-6, 1 - 1e-6)))],
        }
    )
    frame["Direct_x_Physio_Main"] = frame["Direct_Logit"].values * frame["Physio_Main_Logit"].values
    for category in bundle["interval_categories"]:
        dummy = f"Interval_Name_{category}"
        frame[dummy] = float(interval_name == category)
        frame[f"{dummy}_x_Direct_Logit"] = frame[dummy].values * frame["Direct_Logit"].values
        frame[f"{dummy}_x_Physio_Main_Logit"] = frame[dummy].values * frame["Physio_Main_Logit"].values
    return _align_frame_to_bundle(frame, bundle["feature_names"], bundle["fusion_feature_medians"])


def _predict_relapse_two_stage(case_payload: dict, bundle: dict) -> dict:
    landmark = case_payload["landmark"]
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

    base_row = _build_relapse_base_row(case_payload, landmark)
    case_stage1_payload = _build_stage1_payload_for_case(base_row, bundle)
    enriched_row = add_stage1_prediction_family(base_row, case_stage1_payload, "test")

    sidecar_x = _build_sidecar_feature_row(enriched_row, bundle)
    sidecar_ref_proba = float(bundle["sidecar_model"].predict_proba(sidecar_x)[:, 1][0])
    cal_x = _build_sidecar_calibration_row(base_row, np.log(np.clip(sidecar_ref_proba, 1e-6, 1 - 1e-6) / np.clip(1 - sidecar_ref_proba, 1e-6, 1 - 1e-6)), bundle)
    sidecar_proba = float(bundle["sidecar_calibration_model"].predict_proba(cal_x)[:, 1][0])

    direct_x = _build_direct_windowed_feature_row(enriched_row, bundle)
    direct_proba = float(bundle["direct_model"].predict_proba(direct_x)[:, 1][0])

    fusion_x = _build_fusion_feature_row(base_row, direct_proba, sidecar_proba, bundle)
    probability = float(bundle["model"].predict_proba(fusion_x)[:, 1][0])
    remaining_visits = max(1, len(TIME_STAMPS) - 1 - timestamp_to_index(landmark))
    projected_any_relapse = float(1 - (1 - probability) ** remaining_visits)
    quartiles = bundle["patient_quartiles"]
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    risk_group = quartile_labels[min(sum(projected_any_relapse > q for q in quartiles[1:-1]), 3)]
    decision_label = "高风险" if probability >= bundle["decision_threshold"] else "较低风险"
    top_drivers = _pipeline_linear_top_drivers(bundle["model"], fusion_x, bundle["feature_names"], limit=5)
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
        "projection_note": "累计随访风险基于“当前下一窗口风险在后续时点近似延续”的简化假设，仅用于临床沟通而非严格事件率估计。",
    }


def _build_fixed_feature_row(case_payload: dict, landmark: str, bundle: dict) -> pd.DataFrame:
    seq_len = FIXED_LANDMARKS[landmark]
    required_timestamps = TIME_STAMPS[:seq_len]
    static_vector = np.array([[float(case_payload["static"][field]) for field in STATIC_NAMES]], dtype=float)
    dynamic = _series_to_dynamic_tensor(case_payload["series"], required_timestamps)[None, :, :]
    feature_row, _ = extract_flat_features(static_vector, dynamic, seq_len)
    scaled = bundle["outer_scaler"].transform(feature_row)
    return pd.DataFrame(scaled, columns=bundle["feature_names"])


def _build_fixed_routed_feature_row(case_payload: dict, landmark: str, bundle: dict) -> pd.DataFrame:
    seq_len = bundle["seq_len"]
    required_timestamps = TIME_STAMPS[:seq_len]
    static_vector = np.array([[float(case_payload["static"][field]) for field in STATIC_NAMES]], dtype=float)
    dynamic = _series_to_dynamic_tensor(case_payload["series"], required_timestamps)[None, :, :]
    raw = np.hstack(
        [
            static_vector,
            dynamic[:, :, 0],
            dynamic[:, :, 1],
            dynamic[:, :, 2],
        ]
    )
    filled = bundle["imputer"].transform(raw)
    xs, ft3, ft4, tsh = split_imputed(filled, len(STATIC_NAMES), seq_len)
    x_feat, _ = _build_landmark_features_local(xs, np.stack([ft3, ft4, tsh], axis=-1), seq_len)
    frame = pd.DataFrame(x_feat, columns=bundle["feature_names"])
    frame = _align_frame_to_bundle(frame, bundle["feature_names"], bundle.get("feature_medians"))
    return frame


def _make_teacher_pretrain_model(bundle: dict):
    model = _TeacherPretrainNet(
        static_dim=len(bundle["static_features"]),
        local_dim=len(bundle["local_features"]),
        global_dim=len(bundle["global_features"]),
        embed_dim=bundle["embed_dim"],
        dropout=THREEHEAD_DROPOUT,
        gate_mode=bundle["gate_variant"]["gate_mode"],
        side_gate_cap=bundle["gate_variant"]["side_cap"],
        global_floor=bundle["gate_variant"]["global_floor"],
        temperature=bundle["gate_variant"]["temperature"],
    ).to("cpu")
    model.load_state_dict(bundle["backbone_state_dict"])
    model.eval()
    return model


def _predict_relapse_teacher_frozen(case_payload: dict, bundle: dict, base_dir: str | Path) -> dict:
    landmark = case_payload["landmark"]
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

    base_row = _build_relapse_base_row(case_payload, landmark)
    static_df, local_df, global_df = _build_threehead_feature_blocks(
        base_row,
        bundle["interval_categories"],
        bundle["prev_state_categories"],
    )
    tensors = _transform_feature_blocks(bundle["block_scalers"], static_df, local_df, global_df)

    if timestamp_to_index(landmark) >= timestamp_to_index("3M"):
        landmark_case = _truncate_case_payload(case_payload, "3M")
        landmark_obs = predict_fixed_landmark(landmark_case, "3M", base_dir)
        landmark_obs_value = float(landmark_obs["predicted_probability"])
        landmark_obs_mask = 1.0
    else:
        landmark_obs_value = 0.0
        landmark_obs_mask = 0.0

    model = _make_teacher_pretrain_model(bundle)
    with torch.no_grad():
        out = model(
            torch.tensor(tensors["static"], dtype=torch.float32),
            torch.tensor(tensors["local"], dtype=torch.float32),
            torch.tensor(tensors["global"], dtype=torch.float32),
            torch.tensor([landmark_obs_mask], dtype=torch.float32),
            torch.tensor([landmark_obs_value], dtype=torch.float32),
        )

    teacher_logit = float(out["teacher_logit"].cpu().numpy()[0])
    teacher_prob = float(out["teacher_prob"].cpu().numpy()[0])
    landmark_prob = float(out["landmark_prob"].cpu().numpy()[0])
    landmark_signal = float(out["landmark_signal"].cpu().numpy()[0])
    next_hyper_prob = float(out["next_hyper_prob"].cpu().numpy()[0])
    gate = out["gate"].cpu().numpy()[0]

    fuse_x = pd.DataFrame(
        {
            "Teacher_Logit": [teacher_logit],
            "Landmark_Signal": [landmark_signal],
            "Gate_Static": [float(gate[0])],
            "Gate_Local": [float(gate[1])],
        }
    )
    fuse_x = _align_frame_to_bundle(fuse_x, bundle["fuse_feature_names"])
    probability = float(bundle["fuse_model"].predict_proba(fuse_x)[:, 1][0])
    remaining_visits = max(1, len(TIME_STAMPS) - 1 - timestamp_to_index(landmark))
    projected_any_relapse = float(1 - (1 - probability) ** remaining_visits)
    quartiles = bundle["patient_quartiles"]
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    risk_group = quartile_labels[min(sum(projected_any_relapse > q for q in quartiles[1:-1]), 3)]
    decision_label = "高风险预警" if probability >= bundle["decision_threshold"] else "低于预警阈值"
    top_drivers = _pipeline_linear_top_drivers(bundle["fuse_model"], fuse_x, bundle["fuse_feature_names"], limit=5)

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
        "projection_note": "累计随访风险基于“当前下一窗口风险在后续时点近似延续”的简化假设，仅用于风险沟通。",
        "teacher_prob": teacher_prob,
        "landmark_prob": landmark_prob,
        "landmark_signal": landmark_signal,
        "next_hyper_prob": next_hyper_prob,
        "gate_static": float(gate[0]),
        "gate_local": float(gate[1]),
        "gate_global": float(gate[2]),
        "landmark_source": "observed_3m_model" if landmark_obs_mask > 0 else "predicted_before_3m",
    }


def _predict_fixed_landmark_routed(case_payload: dict, landmark: str, bundle: dict) -> dict:
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

    feature_row = _build_fixed_routed_feature_row(case_payload, landmark, bundle)
    elastic_proba = float(bundle["anchor_elastic_model"].predict_proba(feature_row)[:, 1][0])
    lgbm_proba = float(bundle["anchor_lgbm_model"].predict_proba(feature_row)[:, 1][0])
    w_elastic, w_lgbm = bundle["anchor_blend_weights"]
    blend_proba = float(w_elastic * elastic_proba + w_lgbm * lgbm_proba)

    route_value = float(feature_row.iloc[0][bundle["route_feature"]])
    routed = route_value >= bundle["route_cutoff"]
    if routed:
        probability = float(bundle["specialist_model"].predict_proba(feature_row)[:, 1][0])
        specialist_drivers = (
            _tree_top_drivers_from_model(bundle["specialist_model"], bundle["feature_names"], limit=4)
            or _pipeline_linear_top_drivers(bundle["specialist_model"], feature_row, bundle["feature_names"], limit=4)
        )
        top_drivers = [
            {
                "feature": f"Route: {bundle['route_feature']}",
                "importance": float(route_value - bundle["route_cutoff"]),
            }
        ] + specialist_drivers
    else:
        probability = blend_proba
        top_drivers = _pipeline_linear_top_drivers(bundle["anchor_elastic_model"], feature_row, bundle["feature_names"], limit=5)
        if not top_drivers:
            top_drivers = _tree_top_drivers_from_model(bundle["anchor_lgbm_model"], bundle["feature_names"], limit=5)

    decision_label = "倾向甲亢持续" if probability >= bundle["decision_threshold"] else "倾向进入非甲亢状态"
    return {
        "model_name": bundle["model_name"],
        "landmark": landmark,
        "predicted_probability": probability,
        "decision_threshold": bundle["decision_threshold"],
        "decision_label": decision_label,
        "top_drivers": top_drivers[:5],
        "validation_messages": [],
        "routed": routed,
        "route_feature": bundle["route_feature"],
        "route_cutoff": bundle["route_cutoff"],
        "route_value": route_value,
        "specialist_name": bundle["specialist_name"] if routed else "anchor_blend",
        "elastic_probability": elastic_proba,
        "lgbm_probability": lgbm_proba,
        "blend_probability": blend_proba,
    }


def predict_relapse(case_payload: dict, base_dir: str | Path = "artifacts") -> dict:
    landmark = case_payload["landmark"]
    bundle = load_bundle("relapse", base_dir)
    if bundle.get("task") == "two_stage_relapse":
        return _predict_relapse_two_stage(case_payload, bundle)
    if bundle.get("task") == "teacher_frozen_relapse":
        return _predict_relapse_teacher_frozen(case_payload, bundle, base_dir)
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
    top_drivers = _pipeline_linear_top_drivers(bundle["model"], feature_row, bundle["feature_names"], limit=5)
    quartiles = bundle["patient_quartiles"]
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    risk_group = quartile_labels[min(sum(projected_any_relapse > q for q in quartiles[1:-1]), 3)]
    decision_label = "高风险" if probability >= bundle["decision_threshold"] else "较低风险"
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
        "projection_note": "累计随访风险基于“当前下一窗口风险在后续时点近似延续”的简化假设，仅用于风险沟通。",
    }


def predict_fixed_landmark(case_payload: dict, landmark: str, base_dir: str | Path = "artifacts") -> dict:
    task = "fixed_3m" if landmark == "3M" else "fixed_6m"
    bundle = load_bundle(task, base_dir)
    if bundle.get("task") == "fixed_landmark_routed":
        return _predict_fixed_landmark_routed(case_payload, landmark, bundle)
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
    top_drivers = (
        _pipeline_linear_top_drivers(bundle["model"], feature_row, bundle["feature_names"], limit=5)
        or _tree_top_drivers_from_model(bundle["model"], bundle["feature_names"], limit=5)
    )
    decision_label = "倾向甲亢持续" if probability >= bundle["decision_threshold"] else "倾向进入非甲亢状态"
    return {
        "model_name": bundle["model_name"],
        "landmark": landmark,
        "predicted_probability": probability,
        "decision_threshold": bundle["decision_threshold"],
        "decision_label": decision_label,
        "top_drivers": top_drivers,
        "validation_messages": [],
    }
