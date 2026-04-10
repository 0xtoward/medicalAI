import os
import sys
import copy
import math
import random
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from scripts import relapse as relapse_base
from utils.config import SEED, STATIC_NAMES, TIME_STAMPS, STATE_NAMES
from utils.data import (
    load_data as _load_data,
    apply_missforest,
    split_imputed,
    build_states_from_labels,
)
from utils.evaluation import (
    compute_binary_metrics,
    compute_calibration_stats,
    save_calibration_figure,
    save_dca_figure,
    save_threshold_sensitivity_figure,
)
from utils.plot_style import apply_publication_style

apply_publication_style()


class Config:
    OUT_DIR = Path("./results/relapse_threehead_landmark/")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_FRACTION = 0.15
    BATCH_SIZE = 128
    BASE_LR = 1e-3
    FINETUNE_LR = 3e-4
    MAX_EPOCHS = 140
    HARD_EPOCHS = 50
    PATIENCE = 20
    HARD_PATIENCE = 12
    DROPOUT = 0.15
    EMBED_DIM = 48
    SIDE_GATE_CAP = 0.25


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


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_threshold(y_true, proba, objective="f1", low=0.05, high=0.80, step=0.005):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    thresholds = np.arange(low, high + 1e-9, step)
    best_thr = 0.5
    best_key = (-np.inf, -np.inf, -np.inf, -np.inf)
    for thr in thresholds:
        m = compute_binary_metrics(y_true, proba, thr)
        if objective == "acc":
            key = (m["acc"], m["bacc"], m["f1"], m["auc"])
        elif objective == "bacc":
            key = (m["bacc"], m["f1"], m["acc"], m["auc"])
        else:
            key = (m["f1"], m["bacc"], m["acc"], m["auc"])
        if key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr


def binary_focal_loss_with_logits(logits, targets, pos_weight, gamma=1.5):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal_factor = (1.0 - pt).pow(gamma)
    return focal_factor * bce


def encode_next_state(series):
    mapping = {name: idx for idx, name in enumerate(STATE_NAMES)}
    return series.map(mapping).astype(int).values


def build_temporal_masks(pids):
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    return unique_pids, tr_mask, te_mask


def build_three_month_landmark_maps(X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, pids, tr_mask):
    """Derive the patient-level 3M hyper label using only data available by 3M."""
    n_static = X_s_raw.shape[1]
    depth = 3
    raw = np.hstack(
        [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
    )
    imp_path = relapse_base.Config.OUT_DIR / f"missforest_depth{depth}.pkl"
    legacy_path = relapse_base.Config.LEGACY_OUT_DIR / f"missforest_depth{depth}.pkl"
    imputer = relapse_base.load_or_fit_depth_imputer(
        raw[tr_mask], imp_path, fallback_cache_path=legacy_path, label="depth-3 landmark"
    )

    filled_tr = apply_missforest(raw[tr_mask], imputer, depth)
    filled_te = apply_missforest(raw[~tr_mask], imputer, depth)

    xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
    xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)

    s_tr = build_states_from_labels(ev_tr)
    s_te = build_states_from_labels(ev_te)

    train_map = {pid: int(s_tr[i, 2] == 0) for i, pid in enumerate(np.asarray(pids)[tr_mask])}
    test_map = {pid: int(s_te[i, 2] == 0) for i, pid in enumerate(np.asarray(pids)[~tr_mask])}
    return train_map, test_map


def build_longitudinal_tables():
    """Reuse relapse.py temporal-safe pooled interval construction."""
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    unique_pids, tr_mask, te_mask = build_temporal_masks(pids)

    landmark_train_map, landmark_test_map = build_three_month_landmark_maps(
        X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, pids, tr_mask
    )

    df_tr_parts, df_te_parts = [], []
    for depth in range(1, 7):
        k = depth - 1
        n_lab = depth
        n_ev = depth
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"
        raw = np.hstack(
            [
                X_s_raw,
                ft3_raw[:, :n_lab],
                ft4_raw[:, :n_lab],
                tsh_raw[:, :n_lab],
                eval_raw[:, :n_ev],
            ]
        )

        imp_path = relapse_base.Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        legacy_path = relapse_base.Config.LEGACY_OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer = relapse_base.load_or_fit_depth_imputer(
            raw[tr_mask], imp_path, fallback_cache_path=legacy_path, label=f"depth-{depth} ({interval_name})"
        )

        filled_tr = apply_missforest(raw[tr_mask], imputer, n_ev)
        filled_te = apply_missforest(raw[te_mask], imputer, n_ev)

        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, n_lab, n_ev)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, n_lab, n_ev)

        s_tr = build_states_from_labels(ev_tr)
        s_te = build_states_from_labels(ev_te)

        x_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        x_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_tr_parts.append(relapse_base.build_long_format_data(xs_tr, x_d_tr, s_tr, pids[tr_mask], target_k=k))
        df_te_parts.append(relapse_base.build_long_format_data(xs_te, x_d_te, s_te, pids[te_mask], target_k=k))

    df_tr = pd.concat(df_tr_parts, ignore_index=True)
    df_te = pd.concat(df_te_parts, ignore_index=True)
    return df_tr, df_te, landmark_train_map, landmark_test_map, unique_pids


def split_train_validation(df_tr, train_patient_order):
    split_idx = int(len(train_patient_order) * (1 - Config.VAL_FRACTION))
    fit_pids = set(train_patient_order[:split_idx])
    val_pids = set(train_patient_order[split_idx:])
    df_fit = df_tr[df_tr["Patient_ID"].isin(fit_pids)].reset_index(drop=True)
    df_val = df_tr[df_tr["Patient_ID"].isin(val_pids)].reset_index(drop=True)
    return df_fit, df_val


def build_feature_blocks(df, interval_cats, prev_state_cats):
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


def fit_block_scalers(static_fit, local_fit, global_fit):
    scalers = {
        "static": StandardScaler().fit(static_fit),
        "local": StandardScaler().fit(local_fit),
        "global": StandardScaler().fit(global_fit),
    }
    return scalers


def transform_blocks(scalers, static_df, local_df, global_df):
    return {
        "static": scalers["static"].transform(static_df).astype(np.float32),
        "local": scalers["local"].transform(local_df).astype(np.float32),
        "global": scalers["global"].transform(global_df).astype(np.float32),
    }


class IntervalDataset(Dataset):
    def __init__(
        self,
        tensors,
        y_hazard,
        y_landmark,
        y_next_state,
        landmark_obs_mask,
        landmark_obs_value,
        sample_weight=None,
        patient_ids=None,
        interval_ids=None,
        interval_names=None,
    ):
        self.static = torch.tensor(tensors["static"], dtype=torch.float32)
        self.local = torch.tensor(tensors["local"], dtype=torch.float32)
        self.global_x = torch.tensor(tensors["global"], dtype=torch.float32)
        self.y_hazard = torch.tensor(y_hazard, dtype=torch.float32)
        self.y_landmark = torch.tensor(y_landmark, dtype=torch.float32)
        self.y_next_state = torch.tensor(y_next_state, dtype=torch.long)
        self.landmark_obs_mask = torch.tensor(landmark_obs_mask, dtype=torch.float32)
        self.landmark_obs_value = torch.tensor(landmark_obs_value, dtype=torch.float32)
        if sample_weight is None:
            sample_weight = np.ones(len(y_hazard), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
        self.patient_ids = np.asarray(patient_ids) if patient_ids is not None else None
        self.interval_ids = np.asarray(interval_ids) if interval_ids is not None else None
        self.interval_names = np.asarray(interval_names) if interval_names is not None else None

    def __len__(self):
        return len(self.y_hazard)

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "local": self.local[idx],
            "global": self.global_x[idx],
            "y_hazard": self.y_hazard[idx],
            "y_landmark": self.y_landmark[idx],
            "y_next_state": self.y_next_state[idx],
            "landmark_obs_mask": self.landmark_obs_mask[idx],
            "landmark_obs_value": self.landmark_obs_value[idx],
            "sample_weight": self.sample_weight[idx],
        }


class MLPBranch(nn.Module):
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


class LinearBranch(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class LandmarkLocalGlobalHazardNet(nn.Module):
    def __init__(self, static_dim, local_dim, global_dim, embed_dim=48, dropout=0.15):
        super().__init__()
        self.static_branch = MLPBranch(static_dim, embed_dim, dropout)
        self.local_branch = MLPBranch(local_dim, embed_dim, dropout)
        self.global_branch = LinearBranch(global_dim, embed_dim)

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
        self.hazard_head = nn.Linear(embed_dim + 1, 1)
        self.next_state_head = nn.Linear(embed_dim + 1, len(STATE_NAMES))
        self.task_log_vars = nn.Parameter(torch.zeros(3))

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
        gate = torch.softmax(self.gate_net(gate_in), dim=1)
        side_gate = torch.clamp(gate[:, :2], max=Config.SIDE_GATE_CAP)
        side_overflow = (gate[:, :2] - side_gate).clamp_min(0.0).sum(dim=1, keepdim=True)
        global_gate = gate[:, 2:3] + side_overflow
        gate = torch.cat([side_gate, global_gate], dim=1)
        fused = (
            gate[:, 0:1] * h_static
            + gate[:, 1:2] * h_local
            + gate[:, 2:3] * h_global
        )
        fused = self.shared_norm(fused)
        fused_with_landmark = torch.cat([fused, landmark_signal.unsqueeze(1)], dim=1)

        hazard_logit = self.hazard_head(fused_with_landmark).squeeze(1)
        next_state_logits = self.next_state_head(fused_with_landmark)
        return {
            "hazard_logit": hazard_logit,
            "landmark_logit": landmark_logit,
            "next_state_logits": next_state_logits,
            "landmark_prob": landmark_prob,
            "landmark_signal": landmark_signal,
            "embedding": fused,
            "gate": gate,
            "task_log_vars": self.task_log_vars,
        }


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device)
    return out


def collect_predictions(model, dataset, device):
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    model.eval()
    store = {
        "hazard_prob": [],
        "landmark_prob": [],
        "landmark_signal": [],
        "next_state_prob": [],
        "embedding": [],
        "gate": [],
    }
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(
                batch["static"],
                batch["local"],
                batch["global"],
                batch["landmark_obs_mask"],
                batch["landmark_obs_value"],
            )
            store["hazard_prob"].append(torch.sigmoid(out["hazard_logit"]).cpu().numpy())
            store["landmark_prob"].append(torch.sigmoid(out["landmark_logit"]).cpu().numpy())
            store["landmark_signal"].append(out["landmark_signal"].cpu().numpy())
            store["next_state_prob"].append(torch.softmax(out["next_state_logits"], dim=1).cpu().numpy())
            store["embedding"].append(out["embedding"].cpu().numpy())
            store["gate"].append(out["gate"].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in store.items()}


def summarize_auxiliary_targets(dataset, pred_dict):
    landmark_df = pd.DataFrame(
        {
            "Patient_ID": dataset.patient_ids,
            "Landmark_Y": dataset.y_landmark.numpy(),
            "Landmark_Prob": pred_dict["landmark_prob"],
        }
    )
    landmark_patient = (
        landmark_df.groupby("Patient_ID", sort=False)
        .agg(Landmark_Y=("Landmark_Y", "max"), Landmark_Prob=("Landmark_Prob", "mean"))
        .reset_index(drop=True)
    )
    y_land = landmark_patient["Landmark_Y"].values.astype(int)
    p_land = landmark_patient["Landmark_Prob"].values.astype(float)
    if len(np.unique(y_land)) > 1:
        landmark_auc = roc_auc_score(y_land, p_land)
        landmark_prauc = average_precision_score(y_land, p_land)
    else:
        landmark_auc = np.nan
        landmark_prauc = np.nan
    landmark_acc = accuracy_score(y_land, (p_land >= 0.5).astype(int))

    next_true = dataset.y_next_state.numpy()
    next_pred = pred_dict["next_state_prob"].argmax(axis=1)
    next_acc = accuracy_score(next_true, next_pred)
    next_bacc = balanced_accuracy_score(next_true, next_pred)
    next_macro_f1 = f1_score(next_true, next_pred, average="macro", zero_division=0)

    return {
        "landmark_auc": landmark_auc,
        "landmark_prauc": landmark_prauc,
        "landmark_acc": landmark_acc,
        "next_acc": next_acc,
        "next_bacc": next_bacc,
        "next_macro_f1": next_macro_f1,
    }


def evaluate_hazard(dataset, pred_dict, threshold=None):
    y_true = dataset.y_hazard.numpy().astype(int)
    proba = pred_dict["hazard_prob"]
    if threshold is None:
        threshold = select_threshold(y_true, proba, objective="f1")
    metrics = compute_binary_metrics(y_true, proba, threshold)
    return metrics, threshold


def fit_model(model, train_dataset, val_dataset, phase_name, base_lr, max_epochs, patience):
    device = Config.DEVICE
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    y_train = train_dataset.y_hazard.numpy()
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    hazard_pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    y_land = train_dataset.y_landmark.numpy()
    land_pos = float(y_land.sum())
    land_neg = float(len(y_land) - land_pos)
    landmark_pos_weight = torch.tensor([land_neg / max(land_pos, 1.0)], dtype=torch.float32, device=device)

    next_counts = np.bincount(train_dataset.y_next_state.numpy(), minlength=len(STATE_NAMES)).astype(float)
    next_weights = torch.tensor(next_counts.sum() / np.clip(next_counts, 1.0, None), dtype=torch.float32, device=device)
    next_weights = next_weights / next_weights.mean()

    best_payload = None
    history = []
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            out = model(
                batch["static"],
                batch["local"],
                batch["global"],
                batch["landmark_obs_mask"],
                batch["landmark_obs_value"],
            )

            hazard_loss = binary_focal_loss_with_logits(
                out["hazard_logit"],
                batch["y_hazard"],
                pos_weight=hazard_pos_weight,
            )
            hazard_loss = (hazard_loss * batch["sample_weight"]).mean()

            landmark_loss = nn.functional.binary_cross_entropy_with_logits(
                out["landmark_logit"],
                batch["y_landmark"],
                pos_weight=landmark_pos_weight,
            )
            next_loss = nn.functional.cross_entropy(
                out["next_state_logits"],
                batch["y_next_state"],
                weight=next_weights,
            )
            loss_terms = [hazard_loss, landmark_loss, next_loss]
            loss = 0.0
            for idx, task_loss in enumerate(loss_terms):
                log_var = model.task_log_vars[idx]
                loss = loss + torch.exp(-log_var) * task_loss + log_var

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        train_preds = collect_predictions(model, train_dataset, device)
        train_metrics, train_thr = evaluate_hazard(train_dataset, train_preds)
        val_preds = collect_predictions(model, val_dataset, device)
        val_metrics, val_thr = evaluate_hazard(val_dataset, val_preds)
        val_aux = summarize_auxiliary_targets(val_dataset, val_preds)
        score = float(val_metrics["prauc"])

        history.append(
            {
                "phase": phase_name,
                "epoch": epoch,
                "train_loss": total_loss / max(total_batches, 1),
                "train_auc": train_metrics["auc"],
                "train_prauc": train_metrics["prauc"],
                "train_acc": train_metrics["acc"],
                "train_thr": train_thr,
                "val_auc": val_metrics["auc"],
                "val_prauc": val_metrics["prauc"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "val_thr": val_thr,
                "val_landmark_auc": val_aux["landmark_auc"],
                "val_next_bacc": val_aux["next_bacc"],
                "hazard_log_var": float(model.task_log_vars[0].detach().cpu().item()),
                "landmark_log_var": float(model.task_log_vars[1].detach().cpu().item()),
                "next_state_log_var": float(model.task_log_vars[2].detach().cpu().item()),
                "score": score,
            }
        )

        if best_payload is None or score > best_payload["score"]:
            best_payload = {
                "score": score,
                "epoch": epoch,
                "threshold": val_thr,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_payload["state_dict"])
    return model, pd.DataFrame(history), best_payload["threshold"], best_payload["epoch"]


def build_hard_weights(model, dataset, threshold):
    preds = collect_predictions(model, dataset, Config.DEVICE)
    y = dataset.y_hazard.numpy().astype(int)
    proba = preds["hazard_prob"]
    pred = (proba >= threshold).astype(int)
    fp_mask = (pred == 1) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)
    near_mask = np.abs(proba - threshold) <= 0.08

    weights = np.ones(len(y), dtype=np.float32)
    weights[fp_mask] += 1.0
    weights[fn_mask] += 1.5
    weights[near_mask] += 0.4
    return weights


def export_encoder_csv(model, dataset, df_ref, split_name):
    preds = collect_predictions(model, dataset, Config.DEVICE)
    emb = preds["embedding"]
    gate = preds["gate"]
    out = df_ref[
        ["Patient_ID", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse", "Next_State"]
    ].copy().reset_index(drop=True)
    out["Hazard_Prob"] = preds["hazard_prob"]
    out["Landmark_Prob"] = preds["landmark_prob"]
    out["Landmark_Signal"] = preds["landmark_signal"]
    out["Gate_Static"] = gate[:, 0]
    out["Gate_Local"] = gate[:, 1]
    out["Gate_Global"] = gate[:, 2]
    for j in range(emb.shape[1]):
        out[f"Enc_{j:02d}"] = emb[:, j]
    out.to_csv(Config.OUT_DIR / f"Encoder_{split_name}.csv", index=False)
    return preds


def plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Relapse", "Relapse"],
        yticklabels=["No Relapse", "Relapse"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_datasets(df_fit, df_val, df_te, landmark_train_map, landmark_test_map):
    interval_cats = sorted(df_fit["Interval_Name"].unique())
    prev_state_cats = sorted(df_fit["Prev_State"].unique())

    fit_static, fit_local, fit_global = build_feature_blocks(df_fit, interval_cats, prev_state_cats)
    val_static, val_local, val_global = build_feature_blocks(df_val, interval_cats, prev_state_cats)
    te_static, te_local, te_global = build_feature_blocks(df_te, interval_cats, prev_state_cats)

    scalers = fit_block_scalers(fit_static, fit_local, fit_global)
    fit_tensors = transform_blocks(scalers, fit_static, fit_local, fit_global)
    val_tensors = transform_blocks(scalers, val_static, val_local, val_global)
    te_tensors = transform_blocks(scalers, te_static, te_local, te_global)

    def attach_targets(df, landmark_map):
        y_hazard = df["Y_Relapse"].values.astype(np.float32)
        y_landmark = df["Patient_ID"].map(landmark_map).fillna(0).values.astype(np.float32)
        y_next = encode_next_state(df["Next_State"])
        obs_mask = (df["Start_Time"].values >= 3.0).astype(np.float32)
        obs_value = y_landmark.copy()
        return y_hazard, y_landmark, y_next, obs_mask, obs_value

    fit_targets = attach_targets(df_fit, landmark_train_map)
    val_targets = attach_targets(df_val, landmark_train_map)
    te_targets = attach_targets(df_te, landmark_test_map)

    fit_ds = IntervalDataset(
        fit_tensors,
        *fit_targets,
        patient_ids=df_fit["Patient_ID"].values,
        interval_ids=df_fit["Interval_ID"].values,
        interval_names=df_fit["Interval_Name"].values,
    )
    val_ds = IntervalDataset(
        val_tensors,
        *val_targets,
        patient_ids=df_val["Patient_ID"].values,
        interval_ids=df_val["Interval_ID"].values,
        interval_names=df_val["Interval_Name"].values,
    )
    te_ds = IntervalDataset(
        te_tensors,
        *te_targets,
        patient_ids=df_te["Patient_ID"].values,
        interval_ids=df_te["Interval_ID"].values,
        interval_names=df_te["Interval_Name"].values,
    )
    return fit_ds, val_ds, te_ds


def safe_logit(proba):
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(proba / (1.0 - proba))


def get_stage2_fusion_spec():
    return (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=4000, penalty="elasticnet", solver="saga", random_state=SEED)),
            ]
        ),
        {"lr__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        8,
    )


def fit_prob_pack(x_fit, y_fit, groups_fit, x_eval_map):
    base_est, param_grid, n_iter = get_stage2_fusion_spec()
    n_splits = min(3, len(pd.Index(groups_fit).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)
    best_est, cv_score, best_params = relapse_base.fit_candidate_model(
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
    for tr_idx, val_idx in gkf.split(x_fit, y_fit, groups=groups_fit):
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


def build_stage2_feature_frames(
    fit_df,
    other_df,
    direct_fit_proba,
    direct_other_proba,
    side_fit_proba=None,
    side_other_proba=None,
    include_window_terms=True,
):
    cats = sorted(fit_df["Interval_Name"].astype(str).unique())
    x_fit = pd.DataFrame({"Direct_Logit": safe_logit(direct_fit_proba)})
    x_other = pd.DataFrame({"Direct_Logit": safe_logit(direct_other_proba)})

    if include_window_terms:
        for cat in cats:
            dummy = f"Interval_Name_{cat}"
            fit_dummy = (fit_df["Interval_Name"].astype(str).values == cat).astype(float)
            other_dummy = (other_df["Interval_Name"].astype(str).values == cat).astype(float)
            x_fit[dummy] = fit_dummy
            x_other[dummy] = other_dummy
            x_fit[f"{dummy}_x_Direct_Logit"] = fit_dummy * x_fit["Direct_Logit"].values
            x_other[f"{dummy}_x_Direct_Logit"] = other_dummy * x_other["Direct_Logit"].values

    if side_fit_proba is not None and side_other_proba is not None:
        x_fit["Physio_Main_Logit"] = safe_logit(side_fit_proba)
        x_other["Physio_Main_Logit"] = safe_logit(side_other_proba)
        x_fit["Direct_x_Physio_Main"] = x_fit["Direct_Logit"].values * x_fit["Physio_Main_Logit"].values
        x_other["Direct_x_Physio_Main"] = x_other["Direct_Logit"].values * x_other["Physio_Main_Logit"].values
        if include_window_terms:
            for cat in cats:
                dummy = f"Interval_Name_{cat}"
                x_fit[f"{dummy}_x_Physio_Main_Logit"] = x_fit[dummy].values * x_fit["Physio_Main_Logit"].values
                x_other[f"{dummy}_x_Physio_Main_Logit"] = x_other[dummy].values * x_other["Physio_Main_Logit"].values

    medians = x_fit.median(numeric_only=True)
    x_fit = x_fit.replace([np.inf, -np.inf], np.nan).fillna(medians)
    x_other = x_other.replace([np.inf, -np.inf], np.nan).fillna(medians)
    keep_cols = [c for c in x_fit.columns if x_fit[c].nunique(dropna=False) > 1]
    return x_fit[keep_cols], x_other[keep_cols], keep_cols


def fit_sidecar_expert(name, fit_matrix, val_matrix, test_matrix, df_fit, df_val, df_te):
    fit_df = pd.DataFrame(fit_matrix)
    val_df = pd.DataFrame(val_matrix, columns=fit_df.columns)
    test_df = pd.DataFrame(test_matrix, columns=fit_df.columns)
    y_fit = df_fit["Y_Relapse"].values.astype(int)
    groups_fit = df_fit["Patient_ID"].values
    pack = fit_prob_pack(
        fit_df,
        y_fit,
        groups_fit,
        {"val": val_df, "test": test_df},
    )
    return {
        "name": name,
        "fit_proba": pack["train_fit_proba"],
        "fit_oof_proba": pack["oof_proba"],
        "val_proba": pack["val_proba"],
        "test_proba": pack["test_proba"],
        "cv_prauc": pack["cv_prauc"],
        "best_params": pack["best_params"],
    }


def run_stage2_fusion_experiments(df_fit, df_val, df_te, fit_preds, val_preds, test_preds):
    y_fit = df_fit["Y_Relapse"].values.astype(int)
    y_val = df_val["Y_Relapse"].values.astype(int)
    y_te = df_te["Y_Relapse"].values.astype(int)
    groups_fit = df_fit["Patient_ID"].values
    hyper_idx = STATE_NAMES.index("Hyper")

    landmark_signal_sidecar = {
        "name": "landmark_signal_raw",
        "fit_proba": fit_preds["landmark_signal"],
        "fit_oof_proba": fit_preds["landmark_signal"],
        "val_proba": val_preds["landmark_signal"],
        "test_proba": test_preds["landmark_signal"],
        "cv_prauc": np.nan,
        "best_params": None,
    }
    landmark_prob_sidecar = {
        "name": "landmark_prob_raw",
        "fit_proba": fit_preds["landmark_prob"],
        "fit_oof_proba": fit_preds["landmark_prob"],
        "val_proba": val_preds["landmark_prob"],
        "test_proba": test_preds["landmark_prob"],
        "cv_prauc": np.nan,
        "best_params": None,
    }
    landmark_signal_cal_sidecar = fit_sidecar_expert(
        "landmark_signal_calibrated",
        fit_preds["landmark_signal"][:, None],
        val_preds["landmark_signal"][:, None],
        test_preds["landmark_signal"][:, None],
        df_fit,
        df_val,
        df_te,
    )
    landmark_prob_cal_sidecar = fit_sidecar_expert(
        "landmark_prob_calibrated",
        fit_preds["landmark_prob"][:, None],
        val_preds["landmark_prob"][:, None],
        test_preds["landmark_prob"][:, None],
        df_fit,
        df_val,
        df_te,
    )
    next_hyper_sidecar = fit_sidecar_expert(
        "next_hyper_calibrated",
        fit_preds["next_state_prob"][:, [hyper_idx]],
        val_preds["next_state_prob"][:, [hyper_idx]],
        test_preds["next_state_prob"][:, [hyper_idx]],
        df_fit,
        df_val,
        df_te,
    )
    landmark_next_sidecar = fit_sidecar_expert(
        "landmark_next_combo_calibrated",
        np.column_stack([fit_preds["landmark_signal"], fit_preds["next_state_prob"][:, hyper_idx]]),
        np.column_stack([val_preds["landmark_signal"], val_preds["next_state_prob"][:, hyper_idx]]),
        np.column_stack([test_preds["landmark_signal"], test_preds["next_state_prob"][:, hyper_idx]]),
        df_fit,
        df_val,
        df_te,
    )
    minimal_combo_sidecar = fit_sidecar_expert(
        "minimal_combo_calibrated",
        np.column_stack([fit_preds["landmark_prob"], fit_preds["landmark_signal"], fit_preds["next_state_prob"][:, hyper_idx]]),
        np.column_stack([val_preds["landmark_prob"], val_preds["landmark_signal"], val_preds["next_state_prob"][:, hyper_idx]]),
        np.column_stack([test_preds["landmark_prob"], test_preds["landmark_signal"], test_preds["next_state_prob"][:, hyper_idx]]),
        df_fit,
        df_val,
        df_te,
    )

    experiments = [
        ("direct_plain_only", None, False),
        ("direct_windowed_only", None, True),
        ("direct_plus_landmark_signal_raw", landmark_signal_sidecar, True),
        ("direct_plus_landmark_prob_raw", landmark_prob_sidecar, True),
        ("direct_plus_landmark_signal_cal", landmark_signal_cal_sidecar, True),
        ("direct_plus_landmark_prob_cal", landmark_prob_cal_sidecar, True),
        ("direct_plus_next_hyper_cal", next_hyper_sidecar, True),
        ("direct_plus_landmark_next_combo_cal", landmark_next_sidecar, True),
        ("direct_plus_minimal_combo_cal", minimal_combo_sidecar, True),
    ]

    rows = []
    payloads = {}
    base_est, param_grid, n_iter = get_stage2_fusion_spec()
    n_splits = min(3, len(pd.Index(groups_fit).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)

    for exp_name, sidecar, include_window_terms in experiments:
        fit_side = None if sidecar is None else sidecar["fit_proba"]
        val_side = None if sidecar is None else sidecar["val_proba"]
        test_side = None if sidecar is None else sidecar["test_proba"]

        x_fit, x_val, feat_names = build_stage2_feature_frames(
            df_fit,
            df_val,
            fit_preds["hazard_prob"],
            val_preds["hazard_prob"],
            fit_side,
            val_side,
            include_window_terms=include_window_terms,
        )
        _, x_test, _ = build_stage2_feature_frames(
            df_fit,
            df_te,
            fit_preds["hazard_prob"],
            test_preds["hazard_prob"],
            fit_side,
            test_side,
            include_window_terms=include_window_terms,
        )

        best_est, cv_score, best_params = relapse_base.fit_candidate_model(
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
        for tr_idx, hold_idx in gkf.split(x_fit, y_fit, groups_fit):
            fitted = clone(best_est)
            fitted.fit(x_fit.iloc[tr_idx], y_fit[tr_idx])
            oof[hold_idx] = fitted.predict_proba(x_fit.iloc[hold_idx])[:, 1]

        fitted = clone(best_est)
        fitted.fit(x_fit, y_fit)
        val_proba = fitted.predict_proba(x_val)[:, 1]
        test_proba = fitted.predict_proba(x_test)[:, 1]
        thr = select_threshold(y_val, val_proba, objective="f1")
        val_metrics = compute_binary_metrics(y_val, val_proba, thr)
        test_metrics = compute_binary_metrics(y_te, test_proba, thr)

        row = {
            "Experiment": exp_name,
            "Features": len(feat_names),
            "Sidecar": "none" if sidecar is None else sidecar["name"],
            "Windowed": include_window_terms,
            "Fusion_CV_PR_AUC": float(cv_score),
            "Val_AUC": val_metrics["auc"],
            "Val_PR_AUC": val_metrics["prauc"],
            "Val_F1": val_metrics["f1"],
            "Val_Threshold": thr,
            "Test_AUC": test_metrics["auc"],
            "Test_PR_AUC": test_metrics["prauc"],
            "Test_F1": test_metrics["f1"],
            "Test_Recall": test_metrics["recall"],
            "Test_Specificity": test_metrics["specificity"],
            "Test_C_Index": relapse_base.compute_recurrent_c_index_from_intervals(df_te, test_proba),
            "Fusion_Params": best_params,
            "Sidecar_CV_PR_AUC": np.nan if sidecar is None else sidecar["cv_prauc"],
            "Sidecar_Params": None if sidecar is None else sidecar["best_params"],
        }
        rows.append(row)
        payloads[exp_name] = {
            "feature_names": feat_names,
            "threshold": thr,
            "val_proba": val_proba,
            "test_proba": test_proba,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "sidecar_name": row["Sidecar"],
            "model": fitted,
        }

    result_df = pd.DataFrame(rows).sort_values(["Val_PR_AUC", "Val_AUC", "Test_PR_AUC"], ascending=[False, False, False]).reset_index(drop=True)
    best_name = result_df.iloc[0]["Experiment"]
    return result_df, best_name, payloads[best_name]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached .pkl files")
    parser.add_argument("--with-hard-finetune", action="store_true", help="Run the hard-example fine-tune ablation")
    args = parser.parse_args()
    if args.clear_cache:
        from utils.data import clear_pkl_cache
        clear_pkl_cache(relapse_base.Config.OUT_DIR)

    seed_everything(SEED)

    print("=" * 88)
    print("  Landmark-Guided Local-Global Dynamic Hazard Network")
    print("=" * 88)
    print(f"  Device: {Config.DEVICE}")

    df_tr, df_te, landmark_train_map, landmark_test_map, unique_pids = build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = split_train_validation(df_tr, train_patient_order)

    print(f"  Train intervals: {len(df_tr)}")
    print(f"    Fit: {len(df_fit)}  Validation: {len(df_val)}")
    print(f"  Test intervals:  {len(df_te)}")
    print(f"  3M landmark prevalence (train patients): {np.mean(list(landmark_train_map.values())):.3f}")
    print(f"  Hazard prevalence: fit={df_fit['Y_Relapse'].mean():.3f}  val={df_val['Y_Relapse'].mean():.3f}  test={df_te['Y_Relapse'].mean():.3f}")

    fit_ds, val_ds, te_ds = build_datasets(df_fit, df_val, df_te, landmark_train_map, landmark_test_map)
    model = LandmarkLocalGlobalHazardNet(
        static_dim=fit_ds.static.shape[1],
        local_dim=fit_ds.local.shape[1],
        global_dim=fit_ds.global_x.shape[1],
        embed_dim=Config.EMBED_DIM,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    print("\n--- Phase 1: main multi-task fitting ---")
    model, hist_main, val_thr, best_epoch = fit_model(
        model,
        fit_ds,
        val_ds,
        phase_name="main",
        base_lr=Config.BASE_LR,
        max_epochs=Config.MAX_EPOCHS,
        patience=Config.PATIENCE,
    )
    print(f"  Best main epoch: {best_epoch}  val threshold={val_thr:.3f}")

    history_df = hist_main.copy()
    fit_export_ds = fit_ds
    if args.with_hard_finetune:
        print("\n--- Phase 2: hard-example fine-tuning ---")
        hard_weights = build_hard_weights(model, fit_ds, val_thr)
        fit_ds_hard = IntervalDataset(
            {"static": fit_ds.static.numpy(), "local": fit_ds.local.numpy(), "global": fit_ds.global_x.numpy()},
            fit_ds.y_hazard.numpy(),
            fit_ds.y_landmark.numpy(),
            fit_ds.y_next_state.numpy(),
            fit_ds.landmark_obs_mask.numpy(),
            fit_ds.landmark_obs_value.numpy(),
            sample_weight=hard_weights,
            patient_ids=fit_ds.patient_ids,
            interval_ids=fit_ds.interval_ids,
            interval_names=fit_ds.interval_names,
        )
        model, hist_hard, val_thr, hard_epoch = fit_model(
            model,
            fit_ds_hard,
            val_ds,
            phase_name="hard_finetune",
            base_lr=Config.FINETUNE_LR,
            max_epochs=Config.HARD_EPOCHS,
            patience=Config.HARD_PATIENCE,
        )
        print(f"  Best hard-finetune epoch: {hard_epoch}  val threshold={val_thr:.3f}")
        history_df = pd.concat([hist_main, hist_hard], ignore_index=True)
        fit_export_ds = fit_ds_hard
    history_df.to_csv(Config.OUT_DIR / "Training_History.csv", index=False)

    val_preds = export_encoder_csv(model, val_ds, df_val, "Validation")
    test_preds = export_encoder_csv(model, te_ds, df_te, "TemporalTest")
    train_preds = export_encoder_csv(model, fit_export_ds, df_fit, "TrainFit")

    raw_val_metrics, raw_val_thr = evaluate_hazard(val_ds, val_preds, threshold=val_thr)
    raw_test_metrics, _ = evaluate_hazard(te_ds, test_preds, threshold=raw_val_thr)
    val_aux = summarize_auxiliary_targets(val_ds, val_preds)
    test_aux = summarize_auxiliary_targets(te_ds, test_preds)
    stage2_df, stage2_best_name, stage2_best = run_stage2_fusion_experiments(
        df_fit,
        df_val,
        df_te,
        train_preds,
        val_preds,
        test_preds,
    )
    stage2_df.to_csv(Config.OUT_DIR / "Stage2_Experiment_Summary.csv", index=False)
    final_val_metrics = stage2_best["val_metrics"]
    final_test_metrics = stage2_best["test_metrics"]
    final_thr = stage2_best["threshold"]
    final_val_proba = stage2_best["val_proba"]
    final_test_proba = stage2_best["test_proba"]

    print("\n--- Validation ---")
    print(f"  Hazard raw      AUC={raw_val_metrics['auc']:.3f}  PR-AUC={raw_val_metrics['prauc']:.3f}  Acc={raw_val_metrics['acc']:.3f}  F1={raw_val_metrics['f1']:.3f}")
    print(f"  Hazard stage2   AUC={final_val_metrics['auc']:.3f}  PR-AUC={final_val_metrics['prauc']:.3f}  Acc={final_val_metrics['acc']:.3f}  F1={final_val_metrics['f1']:.3f}  [{stage2_best_name}]")
    print(f"  Landmark AUC={val_aux['landmark_auc']:.3f}  PR-AUC={val_aux['landmark_prauc']:.3f}  Acc={val_aux['landmark_acc']:.3f}")
    print(f"  NextState Acc={val_aux['next_acc']:.3f}  BalAcc={val_aux['next_bacc']:.3f}  MacroF1={val_aux['next_macro_f1']:.3f}")

    print("\n--- Temporal Test ---")
    print(f"  Hazard raw      AUC={raw_test_metrics['auc']:.3f}  PR-AUC={raw_test_metrics['prauc']:.3f}  Acc={raw_test_metrics['acc']:.3f}  F1={raw_test_metrics['f1']:.3f}")
    print(f"  Hazard stage2   AUC={final_test_metrics['auc']:.3f}  PR-AUC={final_test_metrics['prauc']:.3f}  Acc={final_test_metrics['acc']:.3f}  F1={final_test_metrics['f1']:.3f}  [{stage2_best_name}]")
    print(f"  Landmark AUC={test_aux['landmark_auc']:.3f}  PR-AUC={test_aux['landmark_prauc']:.3f}  Acc={test_aux['landmark_acc']:.3f}")
    print(f"  NextState Acc={test_aux['next_acc']:.3f}  BalAcc={test_aux['next_bacc']:.3f}  MacroF1={test_aux['next_macro_f1']:.3f}")
    print("\n--- Stage2 sweep (top 5 by val PR-AUC) ---")
    print(stage2_df.head(5).to_string(index=False))

    test_pred = (final_test_proba >= final_thr).astype(int)
    cm = confusion_matrix(te_ds.y_hazard.numpy().astype(int), test_pred, labels=[0, 1])
    plot_confusion(
        cm,
        title=f"Three-Head Stage2 Temporal Test CM (thr={final_thr:.2f})",
        out_path=Config.OUT_DIR / "Confusion_Temporal_Stage2Best.png",
    )

    summary_rows = [
        {"Split": "Validation", "Task": "HazardRaw", "Metric": "AUC", "Value": raw_val_metrics["auc"]},
        {"Split": "Validation", "Task": "HazardRaw", "Metric": "PR-AUC", "Value": raw_val_metrics["prauc"]},
        {"Split": "Validation", "Task": "HazardRaw", "Metric": "Accuracy", "Value": raw_val_metrics["acc"]},
        {"Split": "Validation", "Task": "HazardRaw", "Metric": "F1", "Value": raw_val_metrics["f1"]},
        {"Split": "Validation", "Task": "HazardRaw", "Metric": "Threshold", "Value": raw_val_thr},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "AUC", "Value": final_val_metrics["auc"]},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "PR-AUC", "Value": final_val_metrics["prauc"]},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "Accuracy", "Value": final_val_metrics["acc"]},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "F1", "Value": final_val_metrics["f1"]},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "Threshold", "Value": final_thr},
        {"Split": "Validation", "Task": "HazardStage2Best", "Metric": "BestExperiment", "Value": stage2_best_name},
        {"Split": "Validation", "Task": "Landmark3M", "Metric": "AUC", "Value": val_aux["landmark_auc"]},
        {"Split": "Validation", "Task": "Landmark3M", "Metric": "PR-AUC", "Value": val_aux["landmark_prauc"]},
        {"Split": "Validation", "Task": "Landmark3M", "Metric": "Accuracy", "Value": val_aux["landmark_acc"]},
        {"Split": "Validation", "Task": "NextState", "Metric": "Accuracy", "Value": val_aux["next_acc"]},
        {"Split": "Validation", "Task": "NextState", "Metric": "Balanced Accuracy", "Value": val_aux["next_bacc"]},
        {"Split": "Validation", "Task": "NextState", "Metric": "Macro-F1", "Value": val_aux["next_macro_f1"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "AUC", "Value": raw_test_metrics["auc"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "PR-AUC", "Value": raw_test_metrics["prauc"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "Accuracy", "Value": raw_test_metrics["acc"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "F1", "Value": raw_test_metrics["f1"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "Recall", "Value": raw_test_metrics["recall"]},
        {"Split": "TemporalTest", "Task": "HazardRaw", "Metric": "Specificity", "Value": raw_test_metrics["specificity"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "AUC", "Value": final_test_metrics["auc"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "PR-AUC", "Value": final_test_metrics["prauc"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "Accuracy", "Value": final_test_metrics["acc"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "F1", "Value": final_test_metrics["f1"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "Recall", "Value": final_test_metrics["recall"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "Specificity", "Value": final_test_metrics["specificity"]},
        {"Split": "TemporalTest", "Task": "HazardStage2Best", "Metric": "BestExperiment", "Value": stage2_best_name},
        {"Split": "TemporalTest", "Task": "Landmark3M", "Metric": "AUC", "Value": test_aux["landmark_auc"]},
        {"Split": "TemporalTest", "Task": "Landmark3M", "Metric": "PR-AUC", "Value": test_aux["landmark_prauc"]},
        {"Split": "TemporalTest", "Task": "Landmark3M", "Metric": "Accuracy", "Value": test_aux["landmark_acc"]},
        {"Split": "TemporalTest", "Task": "NextState", "Metric": "Accuracy", "Value": test_aux["next_acc"]},
        {"Split": "TemporalTest", "Task": "NextState", "Metric": "Balanced Accuracy", "Value": test_aux["next_bacc"]},
        {"Split": "TemporalTest", "Task": "NextState", "Metric": "Macro-F1", "Value": test_aux["next_macro_f1"]},
    ]
    pd.DataFrame(summary_rows).to_csv(Config.OUT_DIR / "ThreeHead_Summary.csv", index=False)

    pred_out = df_te[
        ["Patient_ID", "Interval_ID", "Interval_Name", "Start_Time", "Stop_Time", "Y_Relapse", "Next_State"]
    ].copy().reset_index(drop=True)
    pred_out["HazardRaw_Prob"] = test_preds["hazard_prob"]
    pred_out["HazardRaw_Pred"] = (test_preds["hazard_prob"] >= raw_val_thr).astype(int)
    pred_out["HazardStage2_Prob"] = final_test_proba
    pred_out["HazardStage2_Pred"] = test_pred
    pred_out["HazardStage2_BestExperiment"] = stage2_best_name
    pred_out["Landmark3M_Prob"] = test_preds["landmark_prob"]
    pred_out["Landmark3M_Signal"] = test_preds["landmark_signal"]
    pred_out["NextState_Pred"] = [STATE_NAMES[idx] for idx in test_preds["next_state_prob"].argmax(axis=1)]
    pred_out.to_csv(Config.OUT_DIR / "TemporalTest_Predictions.csv", index=False)

    save_calibration_figure(
        te_ds.y_hazard.numpy(),
        final_test_proba,
        f"Three-Head Stage2 Hazard Calibration (Temporal Test, {stage2_best_name})",
        Config.OUT_DIR / "Calibration_Temporal_Stage2Best.png",
    )
    save_dca_figure(
        te_ds.y_hazard.numpy(),
        final_test_proba,
        f"Three-Head Stage2 Hazard DCA (Temporal Test, {stage2_best_name})",
        Config.OUT_DIR / "DCA_Temporal_Stage2Best.png",
    )
    save_threshold_sensitivity_figure(
        te_ds.y_hazard.numpy(),
        final_test_proba,
        final_thr,
        f"Three-Head Stage2 Threshold Sensitivity (Temporal Test, {stage2_best_name})",
        Config.OUT_DIR / "Threshold_Sensitivity_Temporal_Stage2Best.png",
    )

    print(f"\n  Saved results to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
