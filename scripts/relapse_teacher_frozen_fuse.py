import argparse
import copy
import os
import pickle
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import scipy.integrate

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from scripts import relapse_threehead_landmark as base
from scripts import relapse_two_stage_transition as teacher_exp
from utils.config import SEED
from utils.evaluation import (
    compute_binary_metrics,
    save_calibration_figure,
    save_dca_figure,
    save_threshold_sensitivity_figure,
)


class Config:
    OUT_DIR = Path("./results/relapse_teacher_frozen_fuse/")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    MAX_EPOCHS = 80
    PATIENCE = 12
    BASE_LR = 8e-4
    DROPOUT = 0.15
    EMBED_DIM = 48
    GATE_COLLAPSE_STD = 0.005
    SELECTION_TOL = 0.003
    WIDE_TOP_BACKBONES = 3
    CURRENT_BASELINE_VAL_PRAUC = 0.344562
    CURRENT_BASELINE_TEST_PRAUC = 0.279256


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
KEY_COLS = ["Patient_ID", "Source_Row", "Interval_ID"]
MATCH_COLS = [
    "Patient_ID",
    "Interval_ID",
    "Start_Time",
    "Stop_Time",
    "Y_Relapse",
    "FT3_Current",
    "FT4_Current",
    "logTSH_Current",
]
BACKBONE_DIR = Config.OUT_DIR / "backbones"
BACKBONE_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_logit(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def get_gate_variants():
    return [
        {
            "run_name": "hard_cap_025",
            "display_name": "hard_cap_025",
            "gate_mode": "side_cap",
            "side_cap": 0.25,
            "global_floor": None,
            "temperature": 1.0,
        },
        {
            "run_name": "softmax_free",
            "display_name": "softmax_free",
            "gate_mode": "softmax_free",
            "side_cap": None,
            "global_floor": None,
            "temperature": 1.0,
        },
        {
            "run_name": "side_cap_035",
            "display_name": "side_cap_035",
            "gate_mode": "side_cap",
            "side_cap": 0.35,
            "global_floor": None,
            "temperature": 1.0,
        },
        {
            "run_name": "global_floor_040",
            "display_name": "global_floor_040",
            "gate_mode": "global_floor",
            "side_cap": None,
            "global_floor": 0.40,
            "temperature": 1.0,
        },
        {
            "run_name": "global_floor_050",
            "display_name": "global_floor_050",
            "gate_mode": "global_floor",
            "side_cap": None,
            "global_floor": 0.50,
            "temperature": 1.0,
        },
        {
            "run_name": "softmax_temp_070",
            "display_name": "softmax_temp_070",
            "gate_mode": "softmax_free",
            "side_cap": None,
            "global_floor": None,
            "temperature": 0.7,
        },
        {
            "run_name": "softmax_temp_130",
            "display_name": "softmax_temp_130",
            "gate_mode": "softmax_free",
            "side_cap": None,
            "global_floor": None,
            "temperature": 1.3,
        },
    ]


def normalized_match_frame(df):
    out = df[MATCH_COLS].copy()
    for col in ["Start_Time", "Stop_Time", "FT3_Current", "FT4_Current", "logTSH_Current"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    out["Y_Relapse"] = out["Y_Relapse"].astype(int)
    return out


def get_model_specs():
    specs = OrderedDict()
    specs["lr"] = {
        "complexity": 1,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, penalty="l2", solver="lbfgs", random_state=SEED)),
                ]
            ),
            {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
            4,
        ),
    }
    specs["elastic_lr"] = {
        "complexity": 2,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, penalty="elasticnet", solver="saga", random_state=SEED)),
                ]
            ),
            {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            6,
        ),
    }
    specs["elastic_lr_sparse_fixed"] = {
        "complexity": 2,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="elasticnet",
                            solver="saga",
                            random_state=SEED,
                            C=0.05,
                            l1_ratio=0.7,
                        ),
                    ),
                ]
            ),
            {},
            0,
        ),
    }
    specs["elastic_lr_dense_fixed"] = {
        "complexity": 2,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="elasticnet",
                            solver="saga",
                            random_state=SEED,
                            C=0.5,
                            l1_ratio=0.3,
                        ),
                    ),
                ]
            ),
            {},
            0,
        ),
    }
    specs["lr_balanced"] = {
        "complexity": 1,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="l2",
                            solver="lbfgs",
                            class_weight="balanced",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
            4,
        ),
    }
    if LGBMClassifier is not None:
        specs["lightgbm"] = {
            "complexity": 3,
            "builder": lambda: (
                LGBMClassifier(
                    objective="binary",
                    random_state=SEED,
                    n_jobs=-1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
                {
                    "num_leaves": [7, 15, 31],
                    "max_depth": [2, 3, 4, -1],
                    "learning_rate": [0.03, 0.05, 0.08],
                    "n_estimators": [80, 120, 180],
                    "min_child_samples": [10, 20, 40],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [0.0, 0.5, 1.0],
                },
                6,
            ),
        }
        specs["lightgbm_small_fixed"] = {
            "complexity": 3,
            "builder": lambda: (
                LGBMClassifier(
                    objective="binary",
                    random_state=SEED,
                    n_jobs=-1,
                    verbosity=-1,
                    force_col_wise=True,
                    num_leaves=15,
                    max_depth=3,
                    learning_rate=0.05,
                    n_estimators=120,
                    min_child_samples=20,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    reg_lambda=0.5,
                ),
                {},
                0,
            ),
        }
    else:  # pragma: no cover
        specs["hist_gbm"] = {
            "complexity": 3,
            "builder": lambda: (
                HistGradientBoostingClassifier(random_state=SEED),
                {
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [2, 3, 4],
                    "max_leaf_nodes": [15, 31],
                    "min_samples_leaf": [10, 20, 30],
                    "l2_regularization": [0.0, 0.1, 0.5],
                },
                6,
            ),
        }
    specs["extra_trees"] = {
        "complexity": 3,
        "builder": lambda: (
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [2, 3, 4, 5, None],
                "min_samples_leaf": [1, 3, 5, 10],
                "max_features": ["sqrt", 0.5, 1.0],
                "class_weight": [None, "balanced"],
            },
            6,
        ),
    }
    specs["tiny_mlp"] = {
        "complexity": 4,
        "builder": lambda: (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            max_iter=600,
                            random_state=SEED,
                            early_stopping=False,
                        ),
                    ),
                ]
            ),
            {
                "clf__hidden_layer_sizes": [(4,), (8,), (16,)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 3e-3],
            },
            6,
        ),
    }
    return specs


def fit_model_pack(model_name, x_fit, y_fit, groups_fit, x_eval_map):
    model_specs = get_model_specs()
    base_est, param_grid, n_iter = model_specs[model_name]["builder"]()
    n_splits = min(3, len(pd.Index(groups_fit).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)
    best_est, cv_score, best_params = base.relapse_base.fit_candidate_model(
        base_est,
        param_grid,
        n_iter,
        -1,
        gkf,
        x_fit,
        y_fit,
        groups_fit,
    )

    oof = np.zeros(len(y_fit), dtype=float)
    for tr_idx, val_idx in gkf.split(x_fit, y_fit, groups_fit):
        fitted = clone(best_est)
        fitted.fit(x_fit.iloc[tr_idx], y_fit[tr_idx])
        oof[val_idx] = fitted.predict_proba(x_fit.iloc[val_idx])[:, 1]

    fitted = clone(best_est)
    fitted.fit(x_fit, y_fit)
    out = {
        "model": fitted,
        "cv_prauc": float(cv_score),
        "best_params": best_params,
        "oof_proba": oof,
        "train_fit_proba": fitted.predict_proba(x_fit)[:, 1],
    }
    for split_name, x_df in x_eval_map.items():
        out[f"{split_name}_proba"] = fitted.predict_proba(x_df)[:, 1]
    return out


def run_teacher035_pipeline(train_df, eval_df, tag):
    cache_path = Config.OUT_DIR / f"Teacher035_{tag}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(MATCH_COLS).issubset(cached.columns):
            return cached

    teacher_cache_dir = Config.OUT_DIR / "teacher035_cache"
    teacher_cache_dir.mkdir(parents=True, exist_ok=True)
    original_out_dir = teacher_exp.Config.OUT_DIR
    teacher_exp.Config.OUT_DIR = teacher_cache_dir
    teacher_exp.Config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        x1_tr, x1_ev, _ = teacher_exp.make_stage1_feature_frames(train_df, eval_df)
        groups = train_df["Patient_ID"].values
        train_intervals = train_df["Interval_Name"].values
        eval_intervals = eval_df["Interval_Name"].values
        stage1_payload = teacher_exp.run_stage1_oof_forecast(
            x1_tr,
            train_df,
            x1_ev,
            eval_df,
            groups,
            train_intervals,
            eval_intervals,
            teacher_cache_dir / f"stage1_{tag}",
        )
        train_pred = teacher_exp.add_stage1_prediction_family(train_df, stage1_payload, "oof")
        eval_pred = teacher_exp.add_stage1_prediction_family(eval_df, stage1_payload, "test")

        _, direct_name, _, direct_results = teacher_exp.fit_stage2_mode(
            f"teacher_direct_anchor_{tag}",
            train_pred,
            eval_pred,
            mode="direct",
            use_feature_selection=True,
            allowed_models=None,
        )
        _, direct_payload = teacher_exp.select_transparent_payload(direct_results)

        _, side_name, _, side_results = teacher_exp.fit_stage2_mode(
            f"teacher_sidecar_ref_{tag}",
            train_pred,
            eval_pred,
            mode="predicted_only_state_rule",
            use_feature_selection=False,
            allowed_models=["Logistic Reg.", "Elastic LR"],
        )
        _, side_payload = teacher_exp.select_transparent_payload(side_results)
        champion_sidecar_ref = {
            **side_payload,
            "variant_name": f"predicted_only_state_rule:{side_name}",
        }
        champion_calibrated = teacher_exp.build_sidecar_calibrated_pack(
            f"teacher_champion_cal_{tag}",
            train_pred,
            eval_pred,
            champion_sidecar_ref,
            strategy_key="phase",
        )
        _, _, direct_window_payload, _ = teacher_exp.fit_direct_windowed_core(
            train_pred,
            eval_pred,
            direct_payload,
        )
        _, _, final_payload, _ = teacher_exp.fit_fixed_fusion(
            f"teacher035_{tag}",
            train_pred,
            eval_pred,
            direct_window_payload,
            champion_calibrated,
            note=f"teacher035 direct_windowed_core_plus_champion_calibrated ({tag})",
        )
    finally:
        teacher_exp.Config.OUT_DIR = original_out_dir

    fit_df = normalized_match_frame(train_df)
    fit_df["Teacher035_Prob"] = final_payload["oof_proba"]
    fit_df["Split"] = "Fit"
    eval_out = normalized_match_frame(eval_df)
    eval_out["Teacher035_Prob"] = final_payload["proba"]
    eval_out["Split"] = "Eval"
    out_df = pd.concat([fit_df, eval_out], ignore_index=True)
    out_df.to_csv(cache_path, index=False)
    return out_df


def build_teacher035_targets(train_patient_order):
    cache_path = Config.OUT_DIR / "Teacher035_Targets.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(MATCH_COLS).issubset(cached.columns):
            return cached

    old_train, old_test = teacher_exp.build_two_stage_long_data()
    split_idx = int(len(train_patient_order) * (1 - base.Config.VAL_FRACTION))
    fit_pids = set(train_patient_order[:split_idx])
    val_pids = set(train_patient_order[split_idx:])
    old_fit = old_train[old_train["Patient_ID"].isin(fit_pids)].reset_index(drop=True)
    old_val = old_train[old_train["Patient_ID"].isin(val_pids)].reset_index(drop=True)
    old_test = old_test.reset_index(drop=True)

    fit_val = run_teacher035_pipeline(old_fit, old_val, "fit_val")
    fit_test = run_teacher035_pipeline(old_fit, old_test, "fit_test")

    fit_df = fit_val[fit_val["Split"] == "Fit"][MATCH_COLS + ["Teacher035_Prob"]].copy()
    val_df = fit_val[fit_val["Split"] == "Eval"][MATCH_COLS + ["Teacher035_Prob"]].copy()
    te_df = fit_test[fit_test["Split"] == "Eval"][MATCH_COLS + ["Teacher035_Prob"]].copy()
    fit_df["TargetSplit"] = "Fit"
    val_df["TargetSplit"] = "Validation"
    te_df["TargetSplit"] = "TemporalTest"
    out_df = pd.concat([fit_df, val_df, te_df], ignore_index=True)
    out_df.to_csv(cache_path, index=False)
    return out_df


def attach_teacher_targets(df_ref, teacher_df, split_name):
    merged = normalized_match_frame(df_ref).merge(
        teacher_df[teacher_df["TargetSplit"] == split_name][MATCH_COLS + ["Teacher035_Prob"]],
        on=MATCH_COLS,
        how="left",
        validate="one_to_one",
    )
    df_out = df_ref.copy().reset_index(drop=True)
    df_out["Teacher035_Prob"] = merged["Teacher035_Prob"].values
    if df_out["Teacher035_Prob"].isna().any():
        missing = int(df_out["Teacher035_Prob"].isna().sum())
        raise ValueError(f"Teacher035 target missing for {split_name}: {missing} rows")
    return df_out


class PretrainDataset(Dataset):
    def __init__(
        self,
        tensors,
        y_hazard,
        y_teacher,
        y_landmark,
        y_next_hyper,
        obs_mask,
        obs_value,
        patient_ids,
        interval_names,
    ):
        self.static = torch.tensor(tensors["static"], dtype=torch.float32)
        self.local = torch.tensor(tensors["local"], dtype=torch.float32)
        self.global_x = torch.tensor(tensors["global"], dtype=torch.float32)
        self.y_hazard = torch.tensor(y_hazard, dtype=torch.float32)
        self.y_teacher = torch.tensor(y_teacher, dtype=torch.float32)
        self.y_landmark = torch.tensor(y_landmark, dtype=torch.float32)
        self.y_next_hyper = torch.tensor(y_next_hyper, dtype=torch.float32)
        self.landmark_obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
        self.landmark_obs_value = torch.tensor(obs_value, dtype=torch.float32)
        self.patient_ids = np.asarray(patient_ids)
        self.interval_names = np.asarray(interval_names)

    def __len__(self):
        return len(self.y_hazard)

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "local": self.local[idx],
            "global": self.global_x[idx],
            "y_hazard": self.y_hazard[idx],
            "y_teacher": self.y_teacher[idx],
            "y_landmark": self.y_landmark[idx],
            "y_next_hyper": self.y_next_hyper[idx],
            "landmark_obs_mask": self.landmark_obs_mask[idx],
            "landmark_obs_value": self.landmark_obs_value[idx],
        }


class TeacherPretrainNet(nn.Module):
    def __init__(
        self,
        static_dim,
        local_dim,
        global_dim,
        embed_dim=48,
        dropout=0.15,
        gate_mode="side_cap",
        side_gate_cap=None,
        global_floor=None,
        temperature=1.0,
    ):
        super().__init__()
        self.static_branch = base.MLPBranch(static_dim, embed_dim, dropout)
        self.local_branch = base.MLPBranch(local_dim, embed_dim, dropout)
        self.global_branch = base.LinearBranch(global_dim, embed_dim)
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
        self.gate_mode = gate_mode
        self.side_gate_cap = side_gate_cap
        self.global_floor = global_floor
        self.temperature = temperature

    def apply_gate_constraints(self, gate_logits):
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
                remaining = (1.0 - self.global_floor)
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
        gate = self.apply_gate_constraints(self.gate_net(gate_in))
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


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def build_datasets(df_fit, df_val, df_te, landmark_train_map, landmark_test_map):
    interval_cats = sorted(df_fit["Interval_Name"].unique())
    prev_state_cats = sorted(df_fit["Prev_State"].unique())
    fit_static, fit_local, fit_global = base.build_feature_blocks(df_fit, interval_cats, prev_state_cats)
    val_static, val_local, val_global = base.build_feature_blocks(df_val, interval_cats, prev_state_cats)
    te_static, te_local, te_global = base.build_feature_blocks(df_te, interval_cats, prev_state_cats)
    scalers = base.fit_block_scalers(fit_static, fit_local, fit_global)
    fit_tensors = base.transform_blocks(scalers, fit_static, fit_local, fit_global)
    val_tensors = base.transform_blocks(scalers, val_static, val_local, val_global)
    te_tensors = base.transform_blocks(scalers, te_static, te_local, te_global)

    def pack(df, landmark_map):
        y_hazard = df["Y_Relapse"].values.astype(np.float32)
        y_teacher = df["Teacher035_Prob"].values.astype(np.float32)
        y_landmark = df["Patient_ID"].map(landmark_map).fillna(0).values.astype(np.float32)
        y_next_hyper = (df["Next_State"].astype(str).values == "Hyper").astype(np.float32)
        obs_mask = (df["Start_Time"].values >= 3.0).astype(np.float32)
        obs_value = y_landmark.copy()
        return y_hazard, y_teacher, y_landmark, y_next_hyper, obs_mask, obs_value

    fit_targets = pack(df_fit, landmark_train_map)
    val_targets = pack(df_val, landmark_train_map)
    te_targets = pack(df_te, landmark_test_map)
    fit_ds = PretrainDataset(fit_tensors, *fit_targets, df_fit["Patient_ID"].values, df_fit["Interval_Name"].values)
    val_ds = PretrainDataset(val_tensors, *val_targets, df_val["Patient_ID"].values, df_val["Interval_Name"].values)
    te_ds = PretrainDataset(te_tensors, *te_targets, df_te["Patient_ID"].values, df_te["Interval_Name"].values)
    return fit_ds, val_ds, te_ds, scalers


def collect_predictions(model, dataset):
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    model.eval()
    store = {
        "teacher_logit": [],
        "teacher_prob": [],
        "landmark_logit": [],
        "landmark_prob": [],
        "landmark_signal": [],
        "next_hyper_logit": [],
        "next_hyper_prob": [],
        "embedding": [],
        "gate": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, Config.DEVICE)
            out = model(
                batch["static"],
                batch["local"],
                batch["global"],
                batch["landmark_obs_mask"],
                batch["landmark_obs_value"],
            )
            for key in store:
                store[key].append(out[key].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in store.items()}


def fit_pretrain(model, fit_ds, val_ds):
    fit_loader = DataLoader(fit_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=1e-4)
    land_pos = float(fit_ds.y_landmark.numpy().sum())
    land_neg = float(len(fit_ds) - land_pos)
    next_pos = float(fit_ds.y_next_hyper.numpy().sum())
    next_neg = float(len(fit_ds) - next_pos)
    landmark_pos_weight = torch.tensor([land_neg / max(land_pos, 1.0)], dtype=torch.float32, device=Config.DEVICE)
    next_pos_weight = torch.tensor([next_neg / max(next_pos, 1.0)], dtype=torch.float32, device=Config.DEVICE)

    history = []
    best = None
    stale = 0

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in fit_loader:
            batch = move_batch(batch, Config.DEVICE)
            out = model(
                batch["static"],
                batch["local"],
                batch["global"],
                batch["landmark_obs_mask"],
                batch["landmark_obs_value"],
            )
            teacher_loss = nn.functional.binary_cross_entropy_with_logits(out["teacher_logit"], batch["y_teacher"])
            landmark_loss = nn.functional.binary_cross_entropy_with_logits(
                out["landmark_logit"], batch["y_landmark"], pos_weight=landmark_pos_weight
            )
            next_hyper_loss = nn.functional.binary_cross_entropy_with_logits(
                out["next_hyper_logit"], batch["y_next_hyper"], pos_weight=next_pos_weight
            )
            loss = teacher_loss + 0.7 * landmark_loss + 0.5 * next_hyper_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        fit_preds = collect_predictions(model, fit_ds)
        val_preds = collect_predictions(model, val_ds)
        fit_thr = base.select_threshold(fit_ds.y_hazard.numpy(), fit_preds["teacher_prob"], objective="f1")
        val_metrics = compute_binary_metrics(val_ds.y_hazard.numpy(), val_preds["teacher_prob"], fit_thr)
        val_land_auc = roc_auc_score(val_ds.y_landmark.numpy(), val_preds["landmark_prob"])
        val_next_auc = roc_auc_score(val_ds.y_next_hyper.numpy(), val_preds["next_hyper_prob"])
        score = float(val_metrics["prauc"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_batches, 1),
                "fit_threshold": fit_thr,
                "val_auc": val_metrics["auc"],
                "val_prauc": val_metrics["prauc"],
                "val_f1": val_metrics["f1"],
                "val_landmark_auc": val_land_auc,
                "val_next_hyper_auc": val_next_auc,
            }
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "threshold": fit_thr,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= Config.PATIENCE:
                break

    model.load_state_dict(best["state_dict"])
    return model, pd.DataFrame(history), best


def save_checkpoint(model, scalers, history_df, best_payload, gate_variant):
    ckpt_path = BACKBONE_DIR / f"Pretrained_Backbone_{gate_variant['run_name']}.pkl"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scalers": scalers,
            "history": history_df.to_dict(orient="records"),
            "best_epoch": best_payload["epoch"],
            "best_score": best_payload["score"],
            "teacher_threshold": best_payload["threshold"],
            "embed_dim": Config.EMBED_DIM,
            "gate_variant": gate_variant,
        },
        ckpt_path,
    )
    return ckpt_path


def instantiate_model(fit_ds, gate_variant):
    return TeacherPretrainNet(
        static_dim=fit_ds.static.shape[1],
        local_dim=fit_ds.local.shape[1],
        global_dim=fit_ds.global_x.shape[1],
        embed_dim=Config.EMBED_DIM,
        dropout=Config.DROPOUT,
        gate_mode=gate_variant["gate_mode"],
        side_gate_cap=gate_variant["side_cap"],
        global_floor=gate_variant["global_floor"],
        temperature=gate_variant["temperature"],
    ).to(Config.DEVICE)


def summarize_gate_stats(preds_by_split, gate_variant):
    rows = []
    for split_name, preds in preds_by_split.items():
        gate = preds["gate"]
        row = {
            "Split": split_name,
            "Gate_Mode": gate_variant["gate_mode"],
            "Temperature": gate_variant["temperature"],
            "Side_Cap": gate_variant["side_cap"],
            "Global_Floor": gate_variant["global_floor"],
        }
        gate_names = ["Static", "Local", "Global"]
        for idx, gate_name in enumerate(gate_names):
            values = gate[:, idx]
            row[f"Gate_{gate_name}_Mean"] = float(np.mean(values))
            row[f"Gate_{gate_name}_Std"] = float(np.std(values))
            row[f"Gate_{gate_name}_Min"] = float(np.min(values))
            row[f"Gate_{gate_name}_Max"] = float(np.max(values))
            if gate_name != "Global" and gate_variant["side_cap"] is not None:
                row[f"Gate_{gate_name}_CapHitRatio"] = float(np.mean(np.isclose(values, gate_variant["side_cap"], atol=1e-6)))
            else:
                row[f"Gate_{gate_name}_CapHitRatio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def detect_gate_collapse(gate_stats_df, gate_variant):
    val_row = gate_stats_df.loc[gate_stats_df["Split"] == "Validation"].iloc[0]
    stds = [
        float(val_row["Gate_Static_Std"]),
        float(val_row["Gate_Local_Std"]),
        float(val_row["Gate_Global_Std"]),
    ]
    max_std = max(stds)
    cap_saturated = False
    if gate_variant["side_cap"] is not None:
        cap_saturated = (
            min(float(val_row["Gate_Static_CapHitRatio"]), float(val_row["Gate_Local_CapHitRatio"])) > 0.80
            and max_std < Config.GATE_COLLAPSE_STD
        )
    hard_pattern = (
        abs(float(val_row["Gate_Static_Mean"]) - 0.25) < 0.02
        and abs(float(val_row["Gate_Local_Mean"]) - 0.25) < 0.02
        and abs(float(val_row["Gate_Global_Mean"]) - 0.50) < 0.02
        and max_std < Config.GATE_COLLAPSE_STD
    )
    free_flat = max_std < (Config.GATE_COLLAPSE_STD / 2.0)
    return bool(cap_saturated or hard_pattern or free_flat)


def evaluate_teacher_head(y_true, proba, threshold):
    return compute_binary_metrics(np.asarray(y_true).astype(int), np.asarray(proba), threshold)


def build_raw_feature_frame(preds):
    gate = preds["gate"]
    return pd.DataFrame(
        {
            "Teacher_Logit": preds["teacher_logit"],
            "Teacher_Prob": preds["teacher_prob"],
            "Landmark_Logit": preds["landmark_logit"],
            "Landmark_Prob": preds["landmark_prob"],
            "Landmark_Signal": preds["landmark_signal"],
            "NextHyper_Logit": preds["next_hyper_logit"],
            "NextHyper_Prob": preds["next_hyper_prob"],
            "Gate_Static": gate[:, 0],
            "Gate_Local": gate[:, 1],
            "Gate_Global": gate[:, 2],
        }
    )


def build_signal_frame(values):
    values = np.asarray(values, dtype=float)
    if np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0:
        col = safe_logit(values)
    else:
        col = values
    return pd.DataFrame({"Signal": col})


def sanitize_frames(fit_df, eval_df_map):
    medians = fit_df.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).fillna(medians)
    out = {"fit": fit_df}
    for split_name, frame in eval_df_map.items():
        out[split_name] = frame.replace([np.inf, -np.inf], np.nan).fillna(medians)
    keep_cols = [c for c in fit_df.columns if fit_df[c].nunique(dropna=False) > 1]
    if not keep_cols:
        keep_cols = list(fit_df.columns)
    return out["fit"][keep_cols], {k: v[keep_cols] for k, v in out.items() if k != "fit"}, keep_cols


def build_screen_feature_frames(df_fit, df_val, df_te, fit_preds, val_preds, te_preds):
    frames = {
        "fit": build_raw_feature_frame(fit_preds),
        "val": build_raw_feature_frame(val_preds),
        "test": build_raw_feature_frame(te_preds),
    }

    interval_cats = sorted(df_fit["Interval_Name"].astype(str).unique())
    dummy_cols = []
    for cat in interval_cats:
        dummy_col = f"Interval_Name_{cat}"
        dummy_cols.append(dummy_col)
        fit_dummy = (df_fit["Interval_Name"].astype(str).values == cat).astype(float)
        val_dummy = (df_val["Interval_Name"].astype(str).values == cat).astype(float)
        test_dummy = (df_te["Interval_Name"].astype(str).values == cat).astype(float)
        frames["fit"][dummy_col] = fit_dummy
        frames["val"][dummy_col] = val_dummy
        frames["test"][dummy_col] = test_dummy

        for split_name, src in [("fit", fit_dummy), ("val", val_dummy), ("test", test_dummy)]:
            frames[split_name][f"{dummy_col}_x_Teacher_Logit"] = src * frames[split_name]["Teacher_Logit"].values
            frames[split_name][f"{dummy_col}_x_Landmark_Signal"] = src * frames[split_name]["Landmark_Signal"].values

    for split_name in frames:
        frame = frames[split_name]
        frame["Teacher_x_LandmarkSignal"] = frame["Teacher_Logit"].values * frame["Landmark_Signal"].values
        frame["Teacher_x_NextHyper"] = frame["Teacher_Logit"].values * frame["NextHyper_Prob"].values
        frame["Teacher_x_GateGlobal"] = frame["Teacher_Logit"].values * frame["Gate_Global"].values

    fit_frame, other_frames, keep_cols = sanitize_frames(
        frames["fit"],
        {"val": frames["val"], "test": frames["test"]},
    )
    frames["fit"] = fit_frame
    frames["val"] = other_frames["val"]
    frames["test"] = other_frames["test"]
    return frames, interval_cats, keep_cols


def build_master_feature_frames(df_fit, df_val, df_te, fit_preds, val_preds, te_preds, y_fit, groups_fit):
    frames, interval_cats, _ = build_screen_feature_frames(df_fit, df_val, df_te, fit_preds, val_preds, te_preds)

    calib_rows = []
    for cal_name, src_col in [
        ("Teacher_Cal", "Teacher_Prob"),
        ("LandmarkSignal_Cal", "Landmark_Signal"),
        ("LandmarkProb_Cal", "Landmark_Prob"),
        ("NextHyper_Cal", "NextHyper_Prob"),
    ]:
        x_fit = build_signal_frame(frames["fit"][src_col].values)
        x_val = build_signal_frame(frames["val"][src_col].values)
        x_test = build_signal_frame(frames["test"][src_col].values)
        pack = fit_model_pack("elastic_lr", x_fit, y_fit, groups_fit, {"val": x_val, "test": x_test})
        frames["fit"][cal_name] = pack["oof_proba"]
        frames["val"][cal_name] = pack["val_proba"]
        frames["test"][cal_name] = pack["test_proba"]
        calib_rows.append(
            {
                "Feature": cal_name,
                "Source": src_col,
                "CV_PR_AUC": pack["cv_prauc"],
                "Best_Params": pack["best_params"],
            }
        )

    pca = PCA(n_components=min(4, fit_preds["embedding"].shape[1]))
    fit_pca = pca.fit_transform(fit_preds["embedding"])
    val_pca = pca.transform(val_preds["embedding"])
    te_pca = pca.transform(te_preds["embedding"])
    pca_cols = [f"Emb_PCA{i + 1}" for i in range(fit_pca.shape[1])]
    for idx, col in enumerate(pca_cols):
        frames["fit"][col] = fit_pca[:, idx]
        frames["val"][col] = val_pca[:, idx]
        frames["test"][col] = te_pca[:, idx]

    dummy_cols = [f"Interval_Name_{cat}" for cat in interval_cats]
    for dummy_col in dummy_cols:
        for split_name in frames:
            src = frames[split_name][dummy_col].values
            frames[split_name][f"{dummy_col}_x_Teacher_Cal"] = src * frames[split_name]["Teacher_Cal"].values
            frames[split_name][f"{dummy_col}_x_LandmarkSignal_Cal"] = src * frames[split_name]["LandmarkSignal_Cal"].values
            frames[split_name]["TeacherCal_x_LandmarkSignalCal"] = (
                frames[split_name]["Teacher_Cal"].values * frames[split_name]["LandmarkSignal_Cal"].values
            )
            frames[split_name]["TeacherCal_x_NextHyperCal"] = (
                frames[split_name]["Teacher_Cal"].values * frames[split_name]["NextHyper_Cal"].values
            )

    fit_frame, other_frames, keep_cols = sanitize_frames(
        frames["fit"],
        {"val": frames["val"], "test": frames["test"]},
    )
    frames["fit"] = fit_frame
    frames["val"] = other_frames["val"]
    frames["test"] = other_frames["test"]
    return frames, pd.DataFrame(calib_rows), interval_cats, pca_cols, keep_cols


def build_screen_pack_defs(interval_cats):
    dummy_cols = [f"Interval_Name_{cat}" for cat in interval_cats]
    return OrderedDict(
        {
            "teacher_only_raw": ["Teacher_Logit"],
            "teacher_landmark_raw": ["Teacher_Logit", "Landmark_Signal"],
            "teacher_landmark_gate_raw": ["Teacher_Logit", "Landmark_Signal", "Gate_Static", "Gate_Local", "Gate_Global"],
            "teacher_landmark_windowed_raw": ["Teacher_Logit", "Landmark_Signal"]
            + dummy_cols
            + [f"{col}_x_Teacher_Logit" for col in dummy_cols]
            + [f"{col}_x_Landmark_Signal" for col in dummy_cols],
        }
    )


def build_feature_pack_defs(interval_cats, pca_cols):
    dummy_cols = [f"Interval_Name_{cat}" for cat in interval_cats]
    window_teacher_raw = [f"{col}_x_Teacher_Logit" for col in dummy_cols]
    window_land_raw = [f"{col}_x_Landmark_Signal" for col in dummy_cols]
    window_teacher_cal = [f"{col}_x_Teacher_Cal" for col in dummy_cols]
    window_land_cal = [f"{col}_x_LandmarkSignal_Cal" for col in dummy_cols]

    pack_defs = OrderedDict()
    pack_defs["teacher_only_raw"] = ["Teacher_Logit"]
    pack_defs["teacher_only_cal"] = ["Teacher_Cal"]
    pack_defs["teacher_only_raw_cal"] = ["Teacher_Logit", "Teacher_Cal"]
    pack_defs["teacher_landmark_raw"] = ["Teacher_Logit", "Landmark_Signal"]
    pack_defs["teacher_landmark_cal"] = ["Teacher_Cal", "LandmarkSignal_Cal"]
    pack_defs["teacher_landmark_raw_cal"] = ["Teacher_Logit", "Landmark_Signal", "Teacher_Cal", "LandmarkSignal_Cal"]
    pack_defs["teacher_next_raw"] = ["Teacher_Logit", "NextHyper_Prob"]
    pack_defs["teacher_landmark_next_raw"] = ["Teacher_Logit", "Landmark_Signal", "NextHyper_Prob"]
    pack_defs["teacher_gate_raw"] = ["Teacher_Logit", "Gate_Static", "Gate_Local", "Gate_Global"]
    pack_defs["teacher_landmark_gate_raw"] = ["Teacher_Logit", "Landmark_Signal", "Gate_Static", "Gate_Local", "Gate_Global"]
    pack_defs["teacher_interact_raw"] = [
        "Teacher_Logit",
        "Landmark_Signal",
        "NextHyper_Prob",
        "Gate_Global",
        "Teacher_x_LandmarkSignal",
        "Teacher_x_NextHyper",
        "Teacher_x_GateGlobal",
    ]
    pack_defs["teacher_landmark_windowed_raw"] = ["Teacher_Logit", "Landmark_Signal"] + dummy_cols + window_teacher_raw + window_land_raw
    pack_defs["teacher_landmark_windowed_cal"] = ["Teacher_Cal", "LandmarkSignal_Cal"] + dummy_cols + window_teacher_cal + window_land_cal
    pack_defs["teacher_landmark_gate_pca4"] = [
        "Teacher_Logit",
        "Landmark_Signal",
        "Gate_Static",
        "Gate_Local",
        "Gate_Global",
    ] + pca_cols
    pack_defs["teacher_full_lowdim"] = [
        "Teacher_Logit",
        "Teacher_Cal",
        "Landmark_Signal",
        "LandmarkSignal_Cal",
        "NextHyper_Prob",
        "NextHyper_Cal",
        "Gate_Static",
        "Gate_Local",
        "Gate_Global",
        "Teacher_x_LandmarkSignal",
        "Teacher_x_NextHyper",
        "Teacher_x_GateGlobal",
        "TeacherCal_x_LandmarkSignalCal",
        "TeacherCal_x_NextHyperCal",
    ] + pca_cols
    return pack_defs


def choose_best_rows(df, top_n=3):
    if df.empty:
        return df
    ranked = df.sort_values(
        ["Val_PR_AUC", "ComplexityScore", "Features", "CV_PR_AUC", "Val_AUC"],
        ascending=[False, True, True, False, False],
    )
    return ranked.head(top_n)


def evaluate_probability_row(y_val, y_te, val_proba, test_proba):
    thr = base.select_threshold(y_val, val_proba, objective="f1")
    val_metrics = compute_binary_metrics(y_val, val_proba, thr)
    te_metrics = compute_binary_metrics(y_te, test_proba, thr)
    return thr, val_metrics, te_metrics


def resolve_available_columns(frame, cols):
    return [col for col in cols if col in frame.columns]


def run_single_fuse_experiment(
    backbone_meta,
    df_te,
    pack_name,
    model_name,
    x_fit,
    x_val,
    x_te,
    y_fit,
    y_val,
    y_te,
    groups_fit,
):
    payload = fit_model_pack(model_name, x_fit, y_fit, groups_fit, {"val": x_val, "test": x_te})
    thr, val_metrics, te_metrics = evaluate_probability_row(y_val, y_te, payload["val_proba"], payload["test_proba"])
    spec = get_model_specs()[model_name]
    experiment_id = f"{backbone_meta['Backbone_Run']}::{pack_name}::{model_name}"
    row = {
        **backbone_meta,
        "SearchStage": "wide",
        "Experiment_ID": experiment_id,
        "Experiment": f"{pack_name}__{model_name}",
        "PackName": pack_name,
        "ModelFamily": model_name,
        "ComplexityScore": spec["complexity"],
        "Features": x_fit.shape[1],
        "CV_PR_AUC": payload["cv_prauc"],
        "Val_AUC": val_metrics["auc"],
        "Val_PR_AUC": val_metrics["prauc"],
        "Val_F1": val_metrics["f1"],
        "Val_Threshold": thr,
        "Test_AUC": te_metrics["auc"],
        "Test_PR_AUC": te_metrics["prauc"],
        "Test_F1": te_metrics["f1"],
        "Test_Recall": te_metrics["recall"],
        "Test_Specificity": te_metrics["specificity"],
        "Test_C_Index": base.relapse_base.compute_recurrent_c_index_from_intervals(df_te, payload["test_proba"]),
        "Best_Params": payload["best_params"],
        "Status": "ok",
        "ComponentModels": "",
    }
    payload.update(
        {
            "threshold": thr,
            "val_metrics": val_metrics,
            "test_metrics": te_metrics,
            "feature_names": list(x_fit.columns),
            "model_family": model_name,
            "pack_name": pack_name,
        }
    )
    return row, payload


def run_screen_experiments(backbone_meta, df_te, master_frames, pack_defs, y_fit, y_val, y_te, groups_fit):
    rows = []
    payloads = {}
    for pack_name in pack_defs:
        cols = resolve_available_columns(master_frames["fit"], pack_defs[pack_name])
        if not cols:
            continue
        for model_name in ["lr", "elastic_lr", "lr_balanced"]:
            row, payload = run_single_fuse_experiment(
                backbone_meta,
                df_te,
                pack_name,
                model_name,
                master_frames["fit"][cols],
                master_frames["val"][cols],
                master_frames["test"][cols],
                y_fit,
                y_val,
                y_te,
                groups_fit,
            )
            row["SearchStage"] = "screen"
            payloads[row["Experiment_ID"]] = payload
            rows.append(row)
    return rows, payloads


def run_top3_ensembles(backbone_meta, df_te, single_rows_df, payloads, y_fit, y_val, y_te, groups_fit):
    rows = []
    extra_payloads = {}
    top_df = choose_best_rows(single_rows_df, top_n=3)
    if len(top_df) < 2:
        return rows, extra_payloads

    component_ids = top_df["Experiment_ID"].tolist()
    component_names = top_df["Experiment"].tolist()
    fit_mat = np.column_stack([payloads[exp_id]["oof_proba"] for exp_id in component_ids])
    val_mat = np.column_stack([payloads[exp_id]["val_proba"] for exp_id in component_ids])
    te_mat = np.column_stack([payloads[exp_id]["test_proba"] for exp_id in component_ids])

    def rank_avg(mat):
        cols = []
        for idx in range(mat.shape[1]):
            cols.append(pd.Series(mat[:, idx]).rank(method="average", pct=True).values)
        return np.mean(np.column_stack(cols), axis=1)

    rank_fit = rank_avg(fit_mat)
    rank_val = rank_avg(val_mat)
    rank_te = rank_avg(te_mat)
    thr, val_metrics, te_metrics = evaluate_probability_row(y_val, y_te, rank_val, rank_te)
    experiment_id = f"{backbone_meta['Backbone_Run']}::rank_average_top3"
    rows.append(
        {
            **backbone_meta,
            "SearchStage": "ensemble",
            "Experiment_ID": experiment_id,
            "Experiment": "rank_average_top3",
            "PackName": "top3_rank_average",
            "ModelFamily": "rank_average",
            "ComplexityScore": 5,
            "Features": fit_mat.shape[1],
            "CV_PR_AUC": average_precision_score(y_fit, rank_fit),
            "Val_AUC": val_metrics["auc"],
            "Val_PR_AUC": val_metrics["prauc"],
            "Val_F1": val_metrics["f1"],
            "Val_Threshold": thr,
            "Test_AUC": te_metrics["auc"],
            "Test_PR_AUC": te_metrics["prauc"],
            "Test_F1": te_metrics["f1"],
            "Test_Recall": te_metrics["recall"],
            "Test_Specificity": te_metrics["specificity"],
            "Test_C_Index": base.relapse_base.compute_recurrent_c_index_from_intervals(df_te, rank_te),
            "Best_Params": None,
            "Status": "ok",
            "ComponentModels": "|".join(component_names),
        }
    )
    extra_payloads[experiment_id] = {
        "threshold": thr,
        "val_proba": rank_val,
        "test_proba": rank_te,
        "oof_proba": rank_fit,
        "val_metrics": val_metrics,
        "test_metrics": te_metrics,
        "feature_names": component_names,
        "model": None,
        "model_family": "rank_average",
        "pack_name": "top3_rank_average",
        "component_ids": component_ids,
    }

    stack_cols = [f"Comp_{i + 1}" for i in range(fit_mat.shape[1])]
    x_fit = pd.DataFrame(fit_mat, columns=stack_cols)
    x_val = pd.DataFrame(val_mat, columns=stack_cols)
    x_te = pd.DataFrame(te_mat, columns=stack_cols)
    stack_payload = fit_model_pack("elastic_lr", x_fit, y_fit, groups_fit, {"val": x_val, "test": x_te})
    thr, val_metrics, te_metrics = evaluate_probability_row(y_val, y_te, stack_payload["val_proba"], stack_payload["test_proba"])
    experiment_id = f"{backbone_meta['Backbone_Run']}::stack_elastic_top3"
    rows.append(
        {
            **backbone_meta,
            "SearchStage": "ensemble",
            "Experiment_ID": experiment_id,
            "Experiment": "stack_elastic_top3",
            "PackName": "top3_stack",
            "ModelFamily": "stack_elastic",
            "ComplexityScore": 4,
            "Features": x_fit.shape[1],
            "CV_PR_AUC": stack_payload["cv_prauc"],
            "Val_AUC": val_metrics["auc"],
            "Val_PR_AUC": val_metrics["prauc"],
            "Val_F1": val_metrics["f1"],
            "Val_Threshold": thr,
            "Test_AUC": te_metrics["auc"],
            "Test_PR_AUC": te_metrics["prauc"],
            "Test_F1": te_metrics["f1"],
            "Test_Recall": te_metrics["recall"],
            "Test_Specificity": te_metrics["specificity"],
            "Test_C_Index": base.relapse_base.compute_recurrent_c_index_from_intervals(df_te, stack_payload["test_proba"]),
            "Best_Params": stack_payload["best_params"],
            "Status": "ok",
            "ComponentModels": "|".join(component_names),
        }
    )
    stack_payload.update(
        {
            "threshold": thr,
            "val_metrics": val_metrics,
            "test_metrics": te_metrics,
            "feature_names": stack_cols,
            "model_family": "stack_elastic",
            "pack_name": "top3_stack",
            "component_ids": component_ids,
        }
    )
    extra_payloads[experiment_id] = stack_payload
    return rows, extra_payloads


def run_wide_fuse_sweep(backbone_meta, df_te, master_frames, pack_defs, y_fit, y_val, y_te, groups_fit):
    rows = []
    payloads = {}
    linear_models = ["elastic_lr_sparse_fixed", "elastic_lr_dense_fixed"]
    nonlinear_models = ["lightgbm_small_fixed"] if "lightgbm_small_fixed" in get_model_specs() else ["hist_gbm"]
    linear_packs = {
        "teacher_only_cal",
        "teacher_only_raw_cal",
        "teacher_landmark_raw_cal",
        "teacher_landmark_next_raw",
        "teacher_interact_raw",
        "teacher_full_lowdim",
    }
    nonlinear_packs = {
        "teacher_landmark_raw_cal",
        "teacher_full_lowdim",
    }
    for pack_name, cols in pack_defs.items():
        if pack_name not in linear_packs and pack_name not in nonlinear_packs:
            continue
        cols = resolve_available_columns(master_frames["fit"], cols)
        if not cols:
            continue
        x_fit = master_frames["fit"][cols]
        x_val = master_frames["val"][cols]
        x_te = master_frames["test"][cols]
        model_names = list(linear_models) if pack_name in linear_packs else []
        if pack_name in nonlinear_packs:
            model_names += nonlinear_models
        for model_name in model_names:
            try:
                row, payload = run_single_fuse_experiment(
                    backbone_meta,
                    df_te,
                    pack_name,
                    model_name,
                    x_fit,
                    x_val,
                    x_te,
                    y_fit,
                    y_val,
                    y_te,
                    groups_fit,
                )
                rows.append(row)
                payloads[row["Experiment_ID"]] = payload
            except Exception as exc:
                rows.append(
                    {
                        **backbone_meta,
                        "SearchStage": "wide",
                        "Experiment_ID": f"{backbone_meta['Backbone_Run']}::{pack_name}::{model_name}",
                        "Experiment": f"{pack_name}__{model_name}",
                        "PackName": pack_name,
                        "ModelFamily": model_name,
                        "ComplexityScore": get_model_specs()[model_name]["complexity"],
                        "Features": x_fit.shape[1],
                        "CV_PR_AUC": np.nan,
                        "Val_AUC": np.nan,
                        "Val_PR_AUC": np.nan,
                        "Val_F1": np.nan,
                        "Val_Threshold": np.nan,
                        "Test_AUC": np.nan,
                        "Test_PR_AUC": np.nan,
                        "Test_F1": np.nan,
                        "Test_Recall": np.nan,
                        "Test_Specificity": np.nan,
                        "Test_C_Index": np.nan,
                        "Best_Params": str(exc),
                        "Status": "failed",
                        "ComponentModels": "",
                    }
                )
    return rows, payloads


def pick_best_experiment(results_df):
    ok_df = results_df.loc[results_df["Status"] == "ok"].copy()
    best_val = ok_df["Val_PR_AUC"].max()
    keep = ok_df.loc[ok_df["Val_PR_AUC"] >= (best_val - Config.SELECTION_TOL)].copy()
    keep = keep.sort_values(
        ["ComplexityScore", "Features", "CV_PR_AUC", "Val_AUC", "Test_PR_AUC"],
        ascending=[True, True, False, False, False],
    )
    return keep.iloc[0]


def process_backbone(run_name, source_name, gate_variant, fit_ds, val_ds, te_ds, scalers, reuse_checkpoint=None):
    if reuse_checkpoint is None:
        model = instantiate_model(fit_ds, gate_variant)
        model, history_df, best_payload = fit_pretrain(model, fit_ds, val_ds)
        ckpt_path = save_checkpoint(model, scalers, history_df, best_payload, gate_variant)
    else:
        model = instantiate_model(fit_ds, gate_variant)
        try:
            state = torch.load(reuse_checkpoint, map_location=Config.DEVICE, weights_only=False)
        except TypeError:  # pragma: no cover
            state = torch.load(reuse_checkpoint, map_location=Config.DEVICE)
        model.load_state_dict(state["state_dict"])
        history_df = pd.DataFrame(state.get("history", []))
        best_payload = {
            "epoch": state.get("best_epoch"),
            "score": state.get("best_score"),
            "threshold": state.get("teacher_threshold"),
        }
        ckpt_path = Path(reuse_checkpoint)

    history_path = BACKBONE_DIR / f"Pretrain_History_{run_name}.csv"
    history_df.to_csv(history_path, index=False)

    for param in model.parameters():
        param.requires_grad = False

    fit_preds = collect_predictions(model, fit_ds)
    val_preds = collect_predictions(model, val_ds)
    te_preds = collect_predictions(model, te_ds)
    gate_stats_df = summarize_gate_stats(
        {"Fit": fit_preds, "Validation": val_preds, "TemporalTest": te_preds},
        gate_variant,
    )
    gate_stats_df.to_csv(BACKBONE_DIR / f"Gate_Stats_{run_name}.csv", index=False)

    collapsed = detect_gate_collapse(gate_stats_df, gate_variant)
    raw_thr = base.select_threshold(val_ds.y_hazard.numpy(), val_preds["teacher_prob"], objective="f1")
    raw_val_metrics = evaluate_teacher_head(val_ds.y_hazard.numpy(), val_preds["teacher_prob"], raw_thr)
    raw_te_metrics = evaluate_teacher_head(te_ds.y_hazard.numpy(), te_preds["teacher_prob"], raw_thr)
    val_row = gate_stats_df.loc[gate_stats_df["Split"] == "Validation"].iloc[0]

    backbone_row = {
        "Backbone_Run": run_name,
        "Source": source_name,
        "Gate_Mode": gate_variant["gate_mode"],
        "Gate_Temperature": gate_variant["temperature"],
        "Gate_Side_Cap": gate_variant["side_cap"],
        "Gate_Global_Floor": gate_variant["global_floor"],
        "Checkpoint": str(ckpt_path),
        "Best_Epoch": best_payload["epoch"],
        "Best_Pretrain_Val_PR_AUC": best_payload["score"],
        "TeacherHead_Val_AUC": raw_val_metrics["auc"],
        "TeacherHead_Val_PR_AUC": raw_val_metrics["prauc"],
        "TeacherHead_Test_AUC": raw_te_metrics["auc"],
        "TeacherHead_Test_PR_AUC": raw_te_metrics["prauc"],
        "TeacherHead_Threshold": raw_thr,
        "Gate_Static_Mean_Val": val_row["Gate_Static_Mean"],
        "Gate_Local_Mean_Val": val_row["Gate_Local_Mean"],
        "Gate_Global_Mean_Val": val_row["Gate_Global_Mean"],
        "Gate_Static_Std_Val": val_row["Gate_Static_Std"],
        "Gate_Local_Std_Val": val_row["Gate_Local_Std"],
        "Gate_Global_Std_Val": val_row["Gate_Global_Std"],
        "Gate_Static_CapHit_Val": val_row["Gate_Static_CapHitRatio"],
        "Gate_Local_CapHit_Val": val_row["Gate_Local_CapHitRatio"],
        "Gate_Global_Min_Val": val_row["Gate_Global_Min"],
        "GateCollapsed": collapsed,
        "EligibleWideSearch": not collapsed,
    }
    return {
        "model": model,
        "history_df": history_df,
        "fit_preds": fit_preds,
        "val_preds": val_preds,
        "te_preds": te_preds,
        "backbone_row": backbone_row,
        "checkpoint": ckpt_path,
        "gate_stats_df": gate_stats_df,
    }


def save_best_fuse(best_row, best_payload, best_backbone):
    out = {
        "best_row": best_row.to_dict(),
        "threshold": best_payload["threshold"],
        "feature_names": best_payload["feature_names"],
        "model_family": best_payload["model_family"],
        "pack_name": best_payload["pack_name"],
        "component_ids": best_payload.get("component_ids"),
        "component_models": best_row.get("ComponentModels", ""),
        "backbone_checkpoint": str(best_backbone["checkpoint"]),
        "backbone_run": best_backbone["backbone_row"]["Backbone_Run"],
        "model": best_payload.get("model"),
    }
    with open(Config.OUT_DIR / "Best_Frozen_Fuse.pkl", "wb") as f:
        pickle.dump(out, f)
    joblib.dump(out, Config.OUT_DIR / "Best_Frozen_Fuse.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-current-only", action="store_true")
    parser.add_argument("--retrain-only", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    print("=" * 88)
    print("  Teacher-Pretrained Frozen-Backbone Fuse Optimization")
    print("=" * 88)
    print(f"  Device: {Config.DEVICE}")

    df_tr, df_te, landmark_train_map, landmark_test_map, unique_pids = base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = base.split_train_validation(df_tr, train_patient_order)
    teacher_targets = build_teacher035_targets(train_patient_order)
    df_fit = attach_teacher_targets(df_fit, teacher_targets, "Fit")
    df_val = attach_teacher_targets(df_val, teacher_targets, "Validation")
    df_te = attach_teacher_targets(df_te, teacher_targets, "TemporalTest")

    fit_ds, val_ds, te_ds, scalers = build_datasets(df_fit, df_val, df_te, landmark_train_map, landmark_test_map)
    y_fit = df_fit["Y_Relapse"].values.astype(int)
    y_val = df_val["Y_Relapse"].values.astype(int)
    y_te = df_te["Y_Relapse"].values.astype(int)
    groups_fit = df_fit["Patient_ID"].values

    do_reuse = not args.retrain_only
    do_retrain = not args.reuse_current_only
    gate_variants = get_gate_variants()

    backbone_runs = []
    if do_reuse:
        current_ckpt = Config.OUT_DIR / "Pretrained_Backbone.pkl"
        if not current_ckpt.exists():
            raise FileNotFoundError(f"Missing current checkpoint: {current_ckpt}")
        print("\n--- Backbone source: reuse current pkl ---")
        reused = process_backbone(
            run_name="reuse_current_hard_cap_025",
            source_name="reuse_current",
            gate_variant=gate_variants[0],
            fit_ds=fit_ds,
            val_ds=val_ds,
            te_ds=te_ds,
            scalers=scalers,
            reuse_checkpoint=current_ckpt,
        )
        backbone_runs.append(reused)
        print(
            f"  reuse_current_hard_cap_025  val teacher PR-AUC={reused['backbone_row']['TeacherHead_Val_PR_AUC']:.3f}  "
            f"collapsed={reused['backbone_row']['GateCollapsed']}"
        )

    if do_retrain:
        print("\n--- Backbone source: retrain gate variants ---")
        for gate_variant in gate_variants:
            run_name = gate_variant["run_name"]
            cached_ckpt = BACKBONE_DIR / f"Pretrained_Backbone_{run_name}.pkl"
            if cached_ckpt.exists():
                print(f"  loading cached backbone {run_name} ...")
                result = process_backbone(
                    run_name=run_name,
                    source_name="retrained",
                    gate_variant=gate_variant,
                    fit_ds=fit_ds,
                    val_ds=val_ds,
                    te_ds=te_ds,
                    scalers=scalers,
                    reuse_checkpoint=cached_ckpt,
                )
            else:
                print(f"  pretraining {run_name} ...")
                result = process_backbone(
                    run_name=run_name,
                    source_name="retrained",
                    gate_variant=gate_variant,
                    fit_ds=fit_ds,
                    val_ds=val_ds,
                    te_ds=te_ds,
                    scalers=scalers,
                    reuse_checkpoint=None,
                )
            backbone_runs.append(result)
            print(
                f"    best epoch={result['backbone_row']['Best_Epoch']}  "
                f"val teacher PR-AUC={result['backbone_row']['TeacherHead_Val_PR_AUC']:.3f}  "
                f"collapsed={result['backbone_row']['GateCollapsed']}"
            )

    backbone_rows = []
    all_rows = []
    all_payloads = {}
    screening_artifacts = {}

    for result in backbone_runs:
        row = result["backbone_row"]
        backbone_rows.append(row)
        print(f"\n--- Screen fuse search on {row['Backbone_Run']} ---")
        screen_frames, interval_cats, _ = build_screen_feature_frames(
            df_fit,
            df_val,
            df_te,
            result["fit_preds"],
            result["val_preds"],
            result["te_preds"],
        )
        screen_pack_defs = build_screen_pack_defs(interval_cats)
        backbone_meta = {
            "Backbone_Run": row["Backbone_Run"],
            "Backbone_Source": row["Source"],
            "Gate_Mode": row["Gate_Mode"],
            "Gate_Temperature": row["Gate_Temperature"],
            "Gate_Side_Cap": row["Gate_Side_Cap"],
            "Gate_Global_Floor": row["Gate_Global_Floor"],
        }

        screen_rows, screen_payloads = run_screen_experiments(
            backbone_meta,
            df_te,
            screen_frames,
            screen_pack_defs,
            y_fit,
            y_val,
            y_te,
            groups_fit,
        )
        all_rows.extend(screen_rows)
        all_payloads.update(screen_payloads)
        row["Screen_Best_Val_PR_AUC"] = max(r["Val_PR_AUC"] for r in screen_rows)
        row["Wide_Best_Val_PR_AUC"] = np.nan
        screening_artifacts[row["Backbone_Run"]] = {
            "backbone_meta": backbone_meta,
            "result": result,
        }
        pd.DataFrame(backbone_rows).to_csv(Config.OUT_DIR / "Backbone_Experiment_Summary.csv", index=False)
        pd.DataFrame(all_rows).to_csv(Config.OUT_DIR / "FrozenFuse_Experiment_Summary.csv", index=False)

    eligible_names = (
        pd.DataFrame(backbone_rows)
        .loc[lambda x: x["EligibleWideSearch"]]
        .sort_values(["Screen_Best_Val_PR_AUC", "TeacherHead_Val_PR_AUC"], ascending=[False, False])
        .head(Config.WIDE_TOP_BACKBONES)["Backbone_Run"]
        .tolist()
    )

    for row in backbone_rows:
        row["SelectedForWide"] = row["Backbone_Run"] in eligible_names

    for backbone_name in eligible_names:
        artifact = screening_artifacts[backbone_name]
        row = next(item for item in backbone_rows if item["Backbone_Run"] == backbone_name)
        result = artifact["result"]
        backbone_meta = artifact["backbone_meta"]
        print(f"\n--- Wide fuse search on {backbone_name} ---")
        master_frames, calibrator_df, interval_cats, pca_cols, _ = build_master_feature_frames(
            df_fit,
            df_val,
            df_te,
            result["fit_preds"],
            result["val_preds"],
            result["te_preds"],
            y_fit,
            groups_fit,
        )
        calibrator_df.to_csv(BACKBONE_DIR / f"Calibrator_Summary_{backbone_name}.csv", index=False)
        pack_defs = build_feature_pack_defs(interval_cats, pca_cols)
        wide_rows, wide_payloads = run_wide_fuse_sweep(
            backbone_meta,
            df_te,
            master_frames,
            pack_defs,
            y_fit,
            y_val,
            y_te,
            groups_fit,
        )
        all_rows.extend(wide_rows)
        all_payloads.update(wide_payloads)

        wide_ok_df = pd.DataFrame([r for r in wide_rows if r["Status"] == "ok"])
        ensemble_rows, ensemble_payloads = run_top3_ensembles(
            backbone_meta,
            df_te,
            wide_ok_df,
            wide_payloads,
            y_fit,
            y_val,
            y_te,
            groups_fit,
        )
        all_rows.extend(ensemble_rows)
        all_payloads.update(ensemble_payloads)
        if not wide_ok_df.empty:
            row["Wide_Best_Val_PR_AUC"] = float(wide_ok_df["Val_PR_AUC"].max())

        pd.DataFrame(backbone_rows).to_csv(Config.OUT_DIR / "Backbone_Experiment_Summary.csv", index=False)
        pd.DataFrame(all_rows).to_csv(Config.OUT_DIR / "FrozenFuse_Experiment_Summary.csv", index=False)

    backbone_df = pd.DataFrame(backbone_rows).sort_values(
        ["EligibleWideSearch", "TeacherHead_Val_PR_AUC", "Best_Pretrain_Val_PR_AUC"],
        ascending=[False, False, False],
    )
    backbone_df.to_csv(Config.OUT_DIR / "Backbone_Experiment_Summary.csv", index=False)

    fuse_df = pd.DataFrame(all_rows)
    fuse_df = fuse_df.sort_values(
        ["Val_PR_AUC", "ComplexityScore", "Features", "CV_PR_AUC", "Val_AUC"],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    fuse_df.to_csv(Config.OUT_DIR / "FrozenFuse_Experiment_Summary.csv", index=False)

    best_row = pick_best_experiment(fuse_df)
    best_payload = all_payloads[best_row["Experiment_ID"]]
    best_backbone = next(item for item in backbone_runs if item["backbone_row"]["Backbone_Run"] == best_row["Backbone_Run"])
    save_best_fuse(best_row, best_payload, best_backbone)

    best_summary = pd.DataFrame(
        [
            {
                "Best_Backbone_Run": best_row["Backbone_Run"],
                "Best_Backbone_Source": best_row["Backbone_Source"],
                "Best_Gate_Mode": best_row["Gate_Mode"],
                "Best_Gate_Temperature": best_row["Gate_Temperature"],
                "Best_Gate_Side_Cap": best_row["Gate_Side_Cap"],
                "Best_Gate_Global_Floor": best_row["Gate_Global_Floor"],
                "Best_Experiment": best_row["Experiment"],
                "Best_ModelFamily": best_row["ModelFamily"],
                "Best_PackName": best_row["PackName"],
                "Best_Features": best_row["Features"],
                "Best_Val_PR_AUC": best_row["Val_PR_AUC"],
                "Best_Test_PR_AUC": best_row["Test_PR_AUC"],
                "Best_Val_AUC": best_row["Val_AUC"],
                "Best_Test_AUC": best_row["Test_AUC"],
                "Best_Val_F1": best_row["Val_F1"],
                "Best_Test_F1": best_row["Test_F1"],
                "Best_Threshold": best_row["Val_Threshold"],
                "CurrentBaseline_Val_PR_AUC": Config.CURRENT_BASELINE_VAL_PRAUC,
                "CurrentBaseline_Test_PR_AUC": Config.CURRENT_BASELINE_TEST_PRAUC,
                "Backbone_Checkpoint": str(best_backbone["checkpoint"]),
            }
        ]
    )
    best_summary.to_csv(Config.OUT_DIR / "Best_Pipeline_Summary.csv", index=False)

    summary_rows = [
        {"Split": "Validation", "Task": "FuseBest", "Metric": "AUC", "Value": best_row["Val_AUC"]},
        {"Split": "Validation", "Task": "FuseBest", "Metric": "PR-AUC", "Value": best_row["Val_PR_AUC"]},
        {"Split": "Validation", "Task": "FuseBest", "Metric": "F1", "Value": best_row["Val_F1"]},
        {"Split": "Validation", "Task": "FuseBest", "Metric": "Threshold", "Value": best_row["Val_Threshold"]},
        {"Split": "Validation", "Task": "FuseBest", "Metric": "BestExperiment", "Value": best_row["Experiment"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "AUC", "Value": best_row["Test_AUC"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "PR-AUC", "Value": best_row["Test_PR_AUC"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "F1", "Value": best_row["Test_F1"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "Recall", "Value": best_row["Test_Recall"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "Specificity", "Value": best_row["Test_Specificity"]},
        {"Split": "TemporalTest", "Task": "FuseBest", "Metric": "BestExperiment", "Value": best_row["Experiment"]},
    ]
    pd.DataFrame(summary_rows).to_csv(Config.OUT_DIR / "TeacherFrozenFuse_Summary.csv", index=False)

    best_test_pred = (best_payload["test_proba"] >= best_payload["threshold"]).astype(int)
    cm = confusion_matrix(y_te, best_test_pred, labels=[0, 1])
    base.plot_confusion(
        cm,
        f"Teacher Frozen Fuse CM ({best_row['Experiment']}, thr={best_payload['threshold']:.2f})",
        Config.OUT_DIR / "Confusion_Temporal.png",
    )

    te_preds = best_backbone["te_preds"]
    pred_out = df_te[["Patient_ID", "Source_Row", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse"]].copy()
    pred_out["Teacher035_Target"] = df_te["Teacher035_Prob"].values
    pred_out["TeacherHead_Prob"] = te_preds["teacher_prob"]
    pred_out["Landmark_Prob"] = te_preds["landmark_prob"]
    pred_out["Landmark_Signal"] = te_preds["landmark_signal"]
    pred_out["NextHyper_Prob"] = te_preds["next_hyper_prob"]
    pred_out["Gate_Static"] = te_preds["gate"][:, 0]
    pred_out["Gate_Local"] = te_preds["gate"][:, 1]
    pred_out["Gate_Global"] = te_preds["gate"][:, 2]
    pred_out["FuseBest_Prob"] = best_payload["test_proba"]
    pred_out["FuseBest_Pred"] = best_test_pred
    pred_out["Best_Backbone_Run"] = best_row["Backbone_Run"]
    pred_out["Best_Experiment"] = best_row["Experiment"]
    pred_out.to_csv(Config.OUT_DIR / "TemporalTest_Predictions.csv", index=False)

    save_calibration_figure(
        y_te,
        best_payload["test_proba"],
        f"Teacher Frozen Fuse Calibration ({best_row['Experiment']})",
        Config.OUT_DIR / "Calibration_Temporal.png",
    )
    save_dca_figure(
        y_te,
        best_payload["test_proba"],
        f"Teacher Frozen Fuse DCA ({best_row['Experiment']})",
        Config.OUT_DIR / "DCA_Temporal.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        best_payload["test_proba"],
        best_payload["threshold"],
        f"Teacher Frozen Fuse Threshold Sensitivity ({best_row['Experiment']})",
        Config.OUT_DIR / "Threshold_Sensitivity_Temporal.png",
    )

    print("\n--- Best pipeline ---")
    print(
        f"  {best_row['Backbone_Run']} / {best_row['Experiment']}  "
        f"val PR-AUC={best_row['Val_PR_AUC']:.3f}  test PR-AUC={best_row['Test_PR_AUC']:.3f}"
    )
    print("\n--- Top 10 fuse experiments by val PR-AUC ---")
    print(
        fuse_df[
            [
                "Backbone_Run",
                "Experiment",
                "ModelFamily",
                "Features",
                "Val_PR_AUC",
                "Test_PR_AUC",
                "Val_AUC",
                "Test_AUC",
                "Status",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\n  Saved results to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
