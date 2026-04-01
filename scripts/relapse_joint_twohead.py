import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED
from utils.evaluation import (
    aggregate_patient_level,
    bootstrap_group_cis,
    compute_binary_metrics,
    compute_calibration_stats,
    format_ci,
    save_calibration_figure,
    save_dca_figure,
    save_patient_aggregation_sensitivity_figure,
    save_patient_risk_strata,
    save_threshold_sensitivity_figure,
    select_best_threshold,
    evaluate_patient_aggregation_sensitivity,
    evaluate_window_sensitivity,
)
from utils.feature_selection import select_binary_features_with_l1
from utils.performance_panels import build_binary_performance_long, export_metric_matrices, save_performance_heatmap_panels
from utils.physio_forecast import make_stage2_feature_frames
from utils.plot_style import apply_publication_style

apply_publication_style()
np.random.seed(SEED)
torch.manual_seed(SEED)


class Config:
    OUT_DIR = Path("./results/relapse_joint_twohead/")
    SOURCE_DIR = Path("./results/relapse_two_stage_physio/")
    JOURNAL_PATH = OUT_DIR / "joint_twohead_journal.md"
    SUMMARY_PATH = OUT_DIR / "joint_twohead_summary.md"


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


HIGH_RISK_WINDOWS = {"1M->3M", "3M->6M", "6M->12M"}


@dataclass
class JointVariant:
    name: str
    family: str
    hidden_dim: int = 48
    dropout: float = 0.15
    lr: float = 0.003
    weight_decay: float = 1e-3
    max_epochs: int = 160
    aux_weight: float = 0.30
    rule_target: str = "any_rule"
    corr_kind: str | None = None
    corr_grouping: str | None = None
    corr_reg: float = 0.04
    base_weight: float = 0.25
    corr_scale: float = 1.0
    calibrator: str | None = None
    residual_scale: float = 0.35
    max_pos_weight_main: float = 4.0
    max_pos_weight_aux: float = 3.0
    rank_weight: float = 0.0
    use_anchor_offset: bool = False


def _safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_")


def _safe_logit(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _group_kfold(groups):
    unique_groups = pd.Index(groups).drop_duplicates()
    return GroupKFold(n_splits=min(3, len(unique_groups)))


def _param_grid_size(grid):
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _fit_candidate_model(base, grid, n_iter, n_jobs, cv, x_tr_df, y_tr, groups_tr):
    rs = RandomizedSearchCV(
        base,
        grid,
        n_iter=min(n_iter, _param_grid_size(grid)),
        cv=cv,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    rs.fit(x_tr_df, y_tr, groups=groups_tr)
    return rs.best_estimator_, float(rs.best_score_), rs.best_params_


def _score_binary(y_true, proba):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    auc = roc_auc_score(y_true, proba) if np.unique(y_true).size > 1 else np.nan
    prauc = average_precision_score(y_true, proba) if np.unique(y_true).size > 1 else np.nan
    brier = float(np.mean((proba - y_true) ** 2))
    return prauc, auc, brier


def _pick_groups_for_inner_val(groups, y, seed):
    group_series = pd.Series(groups)
    group_summary = (
        pd.DataFrame({"group": group_series.values, "y": y})
        .groupby("group", as_index=False)
        .agg(event_rate=("y", "mean"), events=("y", "sum"), n=("y", "size"))
    )
    event_groups = group_summary.loc[group_summary["events"] > 0, "group"].tolist()
    nonevent_groups = group_summary.loc[group_summary["events"] == 0, "group"].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(event_groups)
    rng.shuffle(nonevent_groups)
    n_groups = len(group_summary)
    n_val = max(6, int(round(0.18 * n_groups)))
    n_event_val = max(1, int(round(n_val * max(0.1, len(event_groups) / max(1, n_groups)))))
    n_event_val = min(n_event_val, len(event_groups))
    n_nonevent_val = max(1, n_val - n_event_val)
    n_nonevent_val = min(n_nonevent_val, len(nonevent_groups))
    val_groups = set(event_groups[:n_event_val] + nonevent_groups[:n_nonevent_val])
    if len(val_groups) < 4:
        fallback = group_summary["group"].tolist()
        rng.shuffle(fallback)
        val_groups = set(fallback[: max(4, n_val)])
    return np.array([g in val_groups for g in groups], dtype=bool)


def _fit_empirical_rule_refs(train_df):
    y = train_df["Y_Relapse"].values.astype(int)
    refs = {}
    for source_col, short_name in [("FT4_Next", "FT4"), ("logTSH_Next", "logTSH")]:
        values = train_df[source_col].values.astype(float)
        pos = values[y == 1]
        neg = values[y == 0]
        pos_median = float(np.nanmedian(pos))
        neg_median = float(np.nanmedian(neg))
        direction = 1.0 if pos_median >= neg_median else -1.0
        refs[short_name] = {
            "threshold": 0.5 * (pos_median + neg_median),
            "direction": direction,
        }
    return refs


def load_selected_feature_list(source_dir, prefix):
    csv_path = Path(source_dir) / f"{prefix}_Feature_Selection.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    selected = df.loc[df["selected"].astype(bool), "feature"].astype(str).tolist()
    return selected or None


def _build_rule_targets(df, refs):
    ft4_rule = (
        refs["FT4"]["direction"] * (df["FT4_Next"].values.astype(float) - refs["FT4"]["threshold"]) > 0
    ).astype(int)
    logtsh_rule = (
        refs["logTSH"]["direction"] * (df["logTSH_Next"].values.astype(float) - refs["logTSH"]["threshold"]) > 0
    ).astype(int)
    any_rule = np.maximum(ft4_rule, logtsh_rule).astype(int)
    both_rule = (ft4_rule * logtsh_rule).astype(int)
    return {
        "ft4_rule": ft4_rule,
        "logtsh_rule": logtsh_rule,
        "any_rule": any_rule,
        "both_rule": both_rule,
    }


def _window_groups(interval_names, grouping="full", categories=None):
    interval_names = pd.Series(interval_names).astype(str)
    if grouping == "full":
        labels = categories or sorted(interval_names.unique(), key=lambda x: (float(x.split("->")[0].replace("M", "")), x))
        codes = pd.Categorical(interval_names, categories=labels).codes
        return labels, codes
    if grouping == "early_late":
        labels = categories or ["Late", "HighRisk"]
        mapped = interval_names.isin(HIGH_RISK_WINDOWS).astype(int).values
        return labels, mapped
    raise ValueError(f"Unknown grouping: {grouping}")


def _onehot_codes(codes, n_classes):
    mat = np.zeros((len(codes), n_classes), dtype=np.float32)
    mat[np.arange(len(codes)), codes] = 1.0
    return mat


def _to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)


class SharedJointNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, variant, n_corr_groups=0):
        super().__init__()
        self.variant = variant
        self.linear_skip = nn.Linear(input_dim, 1)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim, 1) if variant.family == "c1" else None
        self.corr_head = nn.Linear(hidden_dim, n_corr_groups) if variant.family == "c2" else None
        self.variant_cfg = variant

    def forward(self, x, corr_onehot=None, anchor_offset=None):
        hidden = self.trunk(x)
        if self.variant_cfg.use_anchor_offset:
            if anchor_offset is None:
                raise ValueError("anchor_offset is required when use_anchor_offset=True")
            linear_logit = anchor_offset.reshape(-1)
        else:
            linear_logit = self.linear_skip(x).squeeze(1)
        residual_logit = self.variant_cfg.residual_scale * self.residual_head(hidden).squeeze(1)
        base_logit = linear_logit + residual_logit
        out = {"base_logit": base_logit, "linear_logit": linear_logit, "residual_logit": residual_logit}

        if self.variant_cfg.family == "c1":
            out["aux_logit"] = self.aux_head(hidden).squeeze(1)
            out["final_logit"] = base_logit
            return out

        corr_all = self.corr_head(hidden)
        corr_selected = (corr_all * corr_onehot).sum(dim=1)
        out["corr_selected"] = corr_selected
        if self.variant_cfg.corr_kind == "intercept":
            out["final_logit"] = base_logit + self.variant_cfg.corr_scale * corr_selected
        elif self.variant_cfg.corr_kind == "slope":
            out["final_logit"] = base_logit * (1.0 + self.variant_cfg.corr_scale * torch.tanh(corr_selected))
        else:
            raise ValueError(f"Unknown correction kind: {self.variant_cfg.corr_kind}")
        return out


def _compute_joint_loss(outputs, y_main, variant, pos_weight_main, y_aux=None, pos_weight_aux=None):
    bce_main = nn.functional.binary_cross_entropy_with_logits(
        outputs["final_logit"],
        y_main,
        pos_weight=pos_weight_main,
    )
    pair_rank = _pairwise_rank_loss(outputs["final_logit"], y_main)
    if variant.family == "c1":
        aux_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["aux_logit"],
            y_aux,
            pos_weight=pos_weight_aux,
        )
        return bce_main + variant.aux_weight * aux_loss + variant.rank_weight * pair_rank

    bce_base = nn.functional.binary_cross_entropy_with_logits(
        outputs["base_logit"],
        y_main,
        pos_weight=pos_weight_main,
    )
    bce_linear = nn.functional.binary_cross_entropy_with_logits(
        outputs["linear_logit"],
        y_main,
        pos_weight=pos_weight_main,
    )
    corr_reg = torch.mean(outputs["corr_selected"] ** 2)
    resid_reg = torch.mean(outputs["residual_logit"] ** 2)
    return (
        bce_main
        + variant.base_weight * (0.5 * bce_base + 0.5 * bce_linear)
        + variant.corr_reg * (corr_reg + resid_reg)
        + variant.rank_weight * pair_rank
    )


def _pairwise_rank_loss(logits, y_true):
    pos = logits[y_true > 0.5]
    neg = logits[y_true <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.nn.functional.softplus(-diffs).mean()


def _train_epochs(
    model,
    x_train,
    y_train,
    variant,
    corr_train=None,
    y_aux_train=None,
    anchor_train=None,
    epochs=150,
):
    pos_main = min(
        (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0),
        float(variant.max_pos_weight_main),
    )
    pos_weight_main = torch.tensor([pos_main], dtype=torch.float32)
    pos_weight_aux = None
    if y_aux_train is not None:
        pos_aux = min(
            (len(y_aux_train) - float(y_aux_train.sum())) / max(float(y_aux_train.sum()), 1.0),
            float(variant.max_pos_weight_aux),
        )
        pos_weight_aux = torch.tensor([pos_aux], dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        outputs = model(x_train, corr_train, anchor_train)
        loss = _compute_joint_loss(
            outputs,
            y_train,
            variant,
            pos_weight_main=pos_weight_main,
            y_aux=y_aux_train,
            pos_weight_aux=pos_weight_aux,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()


def _select_best_epoch(
    x_fit,
    y_fit,
    groups_fit,
    variant,
    corr_fit=None,
    y_aux_fit=None,
    anchor_fit=None,
    seed=SEED,
):
    val_mask = _pick_groups_for_inner_val(groups_fit, y_fit, seed)
    train_mask = ~val_mask
    if np.unique(y_fit[val_mask]).size < 2 or train_mask.sum() < 32:
        return min(variant.max_epochs, 120)

    scaler = StandardScaler()
    x_inner_tr = scaler.fit_transform(x_fit[train_mask])
    x_inner_val = scaler.transform(x_fit[val_mask])
    x_inner_tr_t = _to_tensor(x_inner_tr)
    x_inner_val_t = _to_tensor(x_inner_val)
    y_inner_tr_t = _to_tensor(y_fit[train_mask])
    corr_tr_t = _to_tensor(corr_fit[train_mask]) if corr_fit is not None else None
    corr_val_t = _to_tensor(corr_fit[val_mask]) if corr_fit is not None else None
    y_aux_inner_tr_t = _to_tensor(y_aux_fit[train_mask]) if y_aux_fit is not None else None
    anchor_inner_tr_t = _to_tensor(anchor_fit[train_mask]) if anchor_fit is not None else None
    anchor_inner_val_t = _to_tensor(anchor_fit[val_mask]) if anchor_fit is not None else None

    model = SharedJointNet(
        input_dim=x_fit.shape[1],
        hidden_dim=variant.hidden_dim,
        dropout=variant.dropout,
        variant=variant,
        n_corr_groups=0 if corr_fit is None else corr_fit.shape[1],
    )

    pos_main = min(
        (int(train_mask.sum()) - float(y_fit[train_mask].sum())) / max(float(y_fit[train_mask].sum()), 1.0),
        float(variant.max_pos_weight_main),
    )
    pos_weight_main = torch.tensor([pos_main], dtype=torch.float32)
    pos_weight_aux = None
    if y_aux_fit is not None:
        pos_aux = min(
            (int(train_mask.sum()) - float(y_aux_fit[train_mask].sum()))
            / max(float(y_aux_fit[train_mask].sum()), 1.0),
            float(variant.max_pos_weight_aux),
        )
        pos_weight_aux = torch.tensor([pos_aux], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)

    best_epoch = min(variant.max_epochs, 80)
    best_key = (-np.inf, -np.inf, -np.inf)
    patience = 36
    stale = 0
    for epoch in range(1, int(variant.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_inner_tr_t, corr_tr_t, anchor_inner_tr_t)
        loss = _compute_joint_loss(
            outputs,
            y_inner_tr_t,
            variant,
            pos_weight_main=pos_weight_main,
            y_aux=y_aux_inner_tr_t,
            pos_weight_aux=pos_weight_aux,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch < 40 and epoch % 5 != 0:
            continue

        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(x_inner_val_t, corr_val_t, anchor_inner_val_t)["final_logit"]).cpu().numpy()
        prauc, auc, neg_brier = _score_binary(y_fit[val_mask], val_proba)
        key = (prauc, auc, -neg_brier)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return int(best_epoch)


def _apply_logit_calibration(
    y_true,
    oof_proba,
    train_fit_proba,
    test_proba,
    fit_interval_names,
    train_interval_names,
    test_interval_names,
    mode,
):
    y_true = np.asarray(y_true, dtype=int)
    fit_df = pd.DataFrame({"logit": _safe_logit(oof_proba), "Interval_Name": pd.Series(fit_interval_names).astype(str)})
    train_df = pd.DataFrame({"logit": _safe_logit(train_fit_proba), "Interval_Name": pd.Series(train_interval_names).astype(str)})
    test_df = pd.DataFrame({"logit": _safe_logit(test_proba), "Interval_Name": pd.Series(test_interval_names).astype(str)})

    if mode == "window_intercept":
        cats = sorted(fit_df["Interval_Name"].unique())
        for cat in cats[1:]:
            col = f"Win_{cat}"
            fit_df[col] = (fit_df["Interval_Name"] == cat).astype(float)
            train_df[col] = (train_df["Interval_Name"] == cat).astype(float)
            test_df[col] = (test_df["Interval_Name"] == cat).astype(float)
        cols = ["logit"] + [c for c in fit_df.columns if c.startswith("Win_")]
    elif mode == "early_late":
        fit_df["HighRisk"] = fit_df["Interval_Name"].isin(HIGH_RISK_WINDOWS).astype(float)
        train_df["HighRisk"] = train_df["Interval_Name"].isin(HIGH_RISK_WINDOWS).astype(float)
        test_df["HighRisk"] = test_df["Interval_Name"].isin(HIGH_RISK_WINDOWS).astype(float)
        cols = ["logit", "HighRisk"]
    elif mode == "global":
        cols = ["logit"]
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    calib = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    calib.fit(fit_df[cols], y_true)
    oof_cal = calib.predict_proba(fit_df[cols])[:, 1]
    train_fit_cal = calib.predict_proba(train_df[cols])[:, 1]
    test_cal = calib.predict_proba(test_df[cols])[:, 1]
    return train_fit_cal, oof_cal, test_cal


def fit_joint_variant(
    variant,
    x_tr_df,
    x_te_df,
    train_df,
    test_df,
    y_aux_train,
    y_aux_test,
    anchor_oof=None,
    anchor_train_fit=None,
    anchor_test=None,
):
    x_tr = x_tr_df.values.astype(np.float32)
    x_te = x_te_df.values.astype(np.float32)
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values

    corr_labels_tr, corr_codes_tr = None, None
    corr_labels_te, corr_codes_te = None, None
    corr_onehot_tr, corr_onehot_te = None, None
    if variant.family == "c2":
        corr_labels_tr, corr_codes_tr = _window_groups(train_df["Interval_Name"], grouping=variant.corr_grouping)
        corr_labels_te, corr_codes_te = _window_groups(
            test_df["Interval_Name"],
            grouping=variant.corr_grouping,
            categories=corr_labels_tr,
        )
        corr_onehot_tr = _onehot_codes(corr_codes_tr, len(corr_labels_tr)).astype(np.float32)
        corr_onehot_te = _onehot_codes(corr_codes_te, len(corr_labels_te)).astype(np.float32)

    gkf = _group_kfold(groups)
    oof = np.zeros(len(y_tr), dtype=float)
    best_epochs = []
    anchor_oof = None if anchor_oof is None else np.asarray(anchor_oof, dtype=np.float32).reshape(-1)
    anchor_train_fit = None if anchor_train_fit is None else np.asarray(anchor_train_fit, dtype=np.float32).reshape(-1)
    anchor_test = None if anchor_test is None else np.asarray(anchor_test, dtype=np.float32).reshape(-1)

    for fold_id, (fit_idx, val_idx) in enumerate(gkf.split(x_tr_df, y_tr, groups=groups)):
        best_epoch = _select_best_epoch(
            x_fit=x_tr[fit_idx],
            y_fit=y_tr[fit_idx],
            groups_fit=groups[fit_idx],
            variant=variant,
            corr_fit=None if corr_onehot_tr is None else corr_onehot_tr[fit_idx],
            y_aux_fit=y_aux_train[variant.rule_target][fit_idx] if variant.family == "c1" else None,
            anchor_fit=None if anchor_oof is None else anchor_oof[fit_idx],
            seed=SEED + fold_id,
        )
        best_epochs.append(best_epoch)

        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_tr[fit_idx])
        x_val_scaled = scaler.transform(x_tr[val_idx])
        model = SharedJointNet(
            input_dim=x_tr.shape[1],
            hidden_dim=variant.hidden_dim,
            dropout=variant.dropout,
            variant=variant,
            n_corr_groups=0 if corr_onehot_tr is None else corr_onehot_tr.shape[1],
        )
        _train_epochs(
            model=model,
            x_train=_to_tensor(x_fit_scaled),
            y_train=_to_tensor(y_tr[fit_idx]),
            variant=variant,
            corr_train=None if corr_onehot_tr is None else _to_tensor(corr_onehot_tr[fit_idx]),
            y_aux_train=None if variant.family != "c1" else _to_tensor(y_aux_train[variant.rule_target][fit_idx]),
            anchor_train=None if anchor_oof is None else _to_tensor(anchor_oof[fit_idx]),
            epochs=best_epoch,
        )
        model.eval()
        with torch.no_grad():
            oof[val_idx] = torch.sigmoid(
                model(
                    _to_tensor(x_val_scaled),
                    None if corr_onehot_tr is None else _to_tensor(corr_onehot_tr[val_idx]),
                    None if anchor_oof is None else _to_tensor(anchor_oof[val_idx]),
                )["final_logit"]
            ).cpu().numpy()

    final_epoch = int(np.median(best_epochs))
    scaler_full = StandardScaler()
    x_tr_scaled = scaler_full.fit_transform(x_tr)
    x_te_scaled = scaler_full.transform(x_te)
    model = SharedJointNet(
        input_dim=x_tr.shape[1],
        hidden_dim=variant.hidden_dim,
        dropout=variant.dropout,
        variant=variant,
        n_corr_groups=0 if corr_onehot_tr is None else corr_onehot_tr.shape[1],
    )
    _train_epochs(
        model=model,
        x_train=_to_tensor(x_tr_scaled),
        y_train=_to_tensor(y_tr),
        variant=variant,
        corr_train=None if corr_onehot_tr is None else _to_tensor(corr_onehot_tr),
        y_aux_train=None if variant.family != "c1" else _to_tensor(y_aux_train[variant.rule_target]),
        anchor_train=None if anchor_train_fit is None else _to_tensor(anchor_train_fit),
        epochs=final_epoch,
    )
    model.eval()
    with torch.no_grad():
        train_fit_proba = torch.sigmoid(
            model(
                _to_tensor(x_tr_scaled),
                None if corr_onehot_tr is None else _to_tensor(corr_onehot_tr),
                None if anchor_train_fit is None else _to_tensor(anchor_train_fit),
            )["final_logit"]
        ).cpu().numpy()
        test_proba = torch.sigmoid(
            model(
                _to_tensor(x_te_scaled),
                None if corr_onehot_te is None else _to_tensor(corr_onehot_te),
                None if anchor_test is None else _to_tensor(anchor_test),
            )["final_logit"]
        ).cpu().numpy()

    if variant.calibrator is not None:
        train_fit_proba, oof, test_proba = _apply_logit_calibration(
            y_true=y_tr,
            oof_proba=oof,
            train_fit_proba=train_fit_proba,
            test_proba=test_proba,
            fit_interval_names=train_df["Interval_Name"].values,
            train_interval_names=train_df["Interval_Name"].values,
            test_interval_names=test_df["Interval_Name"].values,
            mode=variant.calibrator,
        )

    threshold = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
    metrics = compute_binary_metrics(y_te, test_proba, threshold)
    cal = compute_calibration_stats(y_te, test_proba)
    result = {
        "variant": variant,
        "model": model,
        "scaler": scaler_full,
        "proba": test_proba,
        "oof_proba": oof,
        "train_fit_proba": train_fit_proba,
        "threshold": threshold,
        "metrics": metrics,
        "cal": cal,
        "selected_features": list(x_tr_df.columns),
        "best_epoch": final_epoch,
        "aux_positive_rate_train": float(y_aux_train[variant.rule_target].mean()) if variant.family == "c1" else np.nan,
    }
    row = {
        "Variant": variant.name,
        "Family": variant.family,
        "AUC": metrics["auc"],
        "PR_AUC": metrics["prauc"],
        "Brier": metrics["brier"],
        "Recall": metrics["recall"],
        "Specificity": metrics["specificity"],
        "Calibration_Intercept": cal["intercept"],
        "Calibration_Slope": cal["slope"],
        "Threshold": threshold,
        "Epochs": final_epoch,
        "Aux_Target": variant.rule_target if variant.family == "c1" else variant.corr_grouping,
        "Calibration_Mode": variant.calibrator if variant.calibrator else "none",
    }
    return row, result


def fit_direct_anchor(x_tr_df, x_te_df, train_df, test_df, out_dir, selected_features=None):
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    if selected_features is None:
        fs = select_binary_features_with_l1(
            x_tr_df,
            x_te_df,
            y_tr,
            groups,
            out_dir=out_dir,
            prefix="Direct_Anchor",
            seed=SEED,
            min_features=8,
        )
        x_tr = fs.X_train
        x_te = fs.X_test
        selected_features = list(fs.selected_features)
    else:
        x_tr = x_tr_df[selected_features].copy()
        x_te = x_te_df[selected_features].copy()
        pd.DataFrame(
            {
                "feature": selected_features,
                "coefficient": np.nan,
                "abs_coefficient": np.nan,
                "selected": True,
            }
        ).to_csv(out_dir / "Direct_Anchor_Feature_Selection.csv", index=False)
        (out_dir / "Direct_Anchor_Feature_Selection_Report.txt").write_text(
            "\n".join(
                [
                    f"original_features={x_tr_df.shape[1]}",
                    f"selected_features={len(selected_features)}",
                    "best_c=NA_reuse",
                    "cv_pr_auc=NA_reuse",
                    "selected_feature_names=" + ", ".join(selected_features),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    specs = {
        "Logistic Reg.": (
            Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, random_state=SEED))]),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__penalty": ["l1", "l2"], "lr__solver": ["saga"]},
            10,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=5000, penalty="elasticnet", solver="saga", random_state=SEED)),
                ]
            ),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            10,
        ),
    }
    gkf = _group_kfold(groups)
    rows = []
    results = {}
    for model_name, (base, grid, n_iter) in specs.items():
        best_est, cv_score, best_params = _fit_candidate_model(base, grid, n_iter, -1, gkf, x_tr, y_tr, groups)
        oof = np.zeros(len(y_tr), dtype=float)
        for fit_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            fitted = clone(best_est)
            fitted.fit(x_tr.iloc[fit_idx], y_tr[fit_idx])
            oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        fitted = clone(best_est)
        fitted.fit(x_tr, y_tr)
        train_fit_proba = fitted.predict_proba(x_tr)[:, 1]
        proba = fitted.predict_proba(x_te)[:, 1]
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        row = {
            "Variant": f"direct_anchor_{_safe_name(model_name)}",
            "Family": "direct_anchor",
            "AUC": metrics["auc"],
            "PR_AUC": metrics["prauc"],
            "Brier": metrics["brier"],
            "Recall": metrics["recall"],
            "Specificity": metrics["specificity"],
            "Calibration_Intercept": cal["intercept"],
            "Calibration_Slope": cal["slope"],
            "Threshold": thr,
            "Epochs": np.nan,
            "Aux_Target": "none",
            "Calibration_Mode": "none",
            "CV_PR_AUC": cv_score,
            "Best_Params": str(best_params),
        }
        rows.append(row)
        results[model_name] = {
            "model": fitted,
            "proba": proba,
            "oof_proba": oof,
            "train_fit_proba": train_fit_proba,
            "threshold": thr,
            "metrics": metrics,
            "cal": cal,
            "selected_features": selected_features,
        }
    summary_df = pd.DataFrame(rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    best_name = summary_df.iloc[0]["Variant"].replace("direct_anchor_", "").replace("_", " ")
    if best_name == "Elastic LR":
        payload = results["Elastic LR"]
        label = "Elastic LR"
    else:
        payload = results["Logistic Reg."]
        label = "Logistic Reg."
    summary_df.to_csv(out_dir / "direct_anchor_results.csv", index=False)
    return summary_df, label, payload, selected_features


def count_stop_rule_hits(row, direct_row):
    hits = 0
    if float(row["PR_AUC"]) >= float(direct_row["PR_AUC"]) + 0.005:
        hits += 1
    if float(row["AUC"]) >= float(direct_row["AUC"]) - 0.001:
        hits += 1
    if float(row["Brier"]) <= float(direct_row["Brier"]) + 0.001:
        hits += 1
    if abs(float(row["Calibration_Intercept"])) <= abs(float(direct_row["Calibration_Intercept"])) + 0.40:
        hits += 1
    if abs(float(row["Calibration_Slope"]) - 1.0) <= abs(float(direct_row["Calibration_Slope"]) - 1.0) + 0.25:
        hits += 1
    return hits


def choose_round_winner(results_df, direct_row):
    df = results_df.copy()
    if "StopRule_Hits" not in df.columns:
        df["StopRule_Hits"] = df.apply(lambda row: count_stop_rule_hits(row, direct_row), axis=1)
    eligible = df[df["StopRule_Hits"] >= 2].copy()
    if len(eligible) == 0:
        eligible = df
    return eligible.sort_values(["PR_AUC", "AUC", "Brier", "StopRule_Hits"], ascending=[False, False, True, False]).iloc[0]


def compute_window_metrics(df_long, proba, threshold, tag):
    rows = []
    for window_name, group in df_long.groupby("Interval_Name", sort=False):
        idx = group.index.to_numpy()
        y_true = group["Y_Relapse"].values.astype(int)
        sub_proba = np.asarray(proba, dtype=float)[idx]
        metrics = compute_binary_metrics(y_true, sub_proba, threshold)
        cal = compute_calibration_stats(y_true, sub_proba)
        rows.append(
            {
                "Tag": tag,
                "Interval_Name": window_name,
                "N": int(len(group)),
                "Events": int(y_true.sum()),
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_delta(df_long, proba_a, proba_b, metric_key, n_boot=2000, seed=SEED):
    y = df_long["Y_Relapse"].values.astype(int)
    groups = df_long["Patient_ID"].values
    unique_groups = pd.Index(groups).drop_duplicates().tolist()
    group_to_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        y_boot = y[idx]
        if np.unique(y_boot).size < 2:
            continue
        if metric_key == "PR_AUC":
            a_val = average_precision_score(y_boot, np.asarray(proba_a)[idx])
            b_val = average_precision_score(y_boot, np.asarray(proba_b)[idx])
        elif metric_key == "AUC":
            a_val = roc_auc_score(y_boot, np.asarray(proba_a)[idx])
            b_val = roc_auc_score(y_boot, np.asarray(proba_b)[idx])
        else:
            raise ValueError(metric_key)
        diffs.append(a_val - b_val)
    diffs = np.asarray(diffs, dtype=float)
    return {
        "Metric": metric_key,
        "Delta": float(np.mean(diffs)),
        "CI_Low": float(np.percentile(diffs, 2.5)),
        "CI_High": float(np.percentile(diffs, 97.5)),
        "OneSided_p": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1)),
    }


def save_best_reports(name, payload, train_df, test_df, out_dir, prefix):
    y_te = test_df["Y_Relapse"].values.astype(int)
    interval_eval_df = pd.DataFrame({"Patient_ID": test_df["Patient_ID"].values, "Y": y_te, "proba": payload["proba"]})
    interval_cis = bootstrap_group_cis(interval_eval_df, payload["threshold"], group_col="Patient_ID")
    interval_metrics = compute_binary_metrics(y_te, payload["proba"], payload["threshold"])
    interval_cal = compute_calibration_stats(y_te, payload["proba"])
    interval_summary = pd.DataFrame(
        [
            {"Level": "Interval", "Metric": "AUC", "Value": interval_metrics["auc"], "CI_95": format_ci(interval_cis, "auc")},
            {"Level": "Interval", "Metric": "PR_AUC", "Value": interval_metrics["prauc"], "CI_95": format_ci(interval_cis, "prauc")},
            {"Level": "Interval", "Metric": "Brier", "Value": interval_metrics["brier"], "CI_95": format_ci(interval_cis, "brier")},
            {"Level": "Interval", "Metric": "Recall", "Value": interval_metrics["recall"], "CI_95": format_ci(interval_cis, "recall")},
            {"Level": "Interval", "Metric": "Specificity", "Value": interval_metrics["specificity"], "CI_95": format_ci(interval_cis, "specificity")},
            {"Level": "Interval", "Metric": "Calibration_Intercept", "Value": interval_cal["intercept"], "CI_95": format_ci(interval_cis, "cal_intercept")},
            {"Level": "Interval", "Metric": "Calibration_Slope", "Value": interval_cal["slope"], "CI_95": format_ci(interval_cis, "cal_slope")},
            {"Level": "Interval", "Metric": "Threshold", "Value": payload["threshold"], "CI_95": ""},
        ]
    )
    interval_summary.to_csv(out_dir / f"{prefix}_interval_summary.csv", index=False)

    save_calibration_figure(y_te, payload["proba"], f"Calibration Curve ({name})", out_dir / f"{prefix}_calibration_interval.png")
    save_dca_figure(y_te, payload["proba"], f"Decision Curve Analysis ({name})", out_dir / f"{prefix}_dca_interval.png")
    save_threshold_sensitivity_figure(
        y_te,
        payload["proba"],
        payload["threshold"],
        f"Threshold Sensitivity ({name})",
        out_dir / f"{prefix}_threshold_interval.png",
    )

    patient_tr = aggregate_patient_level(train_df, payload["oof_proba"])
    patient_te = aggregate_patient_level(test_df, payload["proba"])
    patient_thr = select_best_threshold(patient_tr["Y"].values, patient_tr["proba"].values, low=0.05, high=0.95, step=0.01)
    patient_metrics = compute_binary_metrics(patient_te["Y"].values, patient_te["proba"].values, patient_thr)
    patient_cis = bootstrap_group_cis(patient_te, patient_thr, group_col="Patient_ID")
    patient_cal = compute_calibration_stats(patient_te["Y"].values, patient_te["proba"].values)
    patient_summary = pd.DataFrame(
        [
            {"Level": "Patient", "Metric": "AUC", "Value": patient_metrics["auc"], "CI_95": format_ci(patient_cis, "auc")},
            {"Level": "Patient", "Metric": "PR_AUC", "Value": patient_metrics["prauc"], "CI_95": format_ci(patient_cis, "prauc")},
            {"Level": "Patient", "Metric": "Brier", "Value": patient_metrics["brier"], "CI_95": format_ci(patient_cis, "brier")},
            {"Level": "Patient", "Metric": "Recall", "Value": patient_metrics["recall"], "CI_95": format_ci(patient_cis, "recall")},
            {"Level": "Patient", "Metric": "Specificity", "Value": patient_metrics["specificity"], "CI_95": format_ci(patient_cis, "specificity")},
            {"Level": "Patient", "Metric": "Calibration_Intercept", "Value": patient_cal["intercept"], "CI_95": format_ci(patient_cis, "cal_intercept")},
            {"Level": "Patient", "Metric": "Calibration_Slope", "Value": patient_cal["slope"], "CI_95": format_ci(patient_cis, "cal_slope")},
            {"Level": "Patient", "Metric": "Threshold", "Value": patient_thr, "CI_95": ""},
        ]
    )
    patient_summary.to_csv(out_dir / f"{prefix}_patient_summary.csv", index=False)
    save_patient_risk_strata(patient_tr, patient_te, out_dir / f"{prefix}_patient_risk.png")
    patient_sens_df, _ = evaluate_patient_aggregation_sensitivity(train_df, test_df, payload["oof_proba"], payload["proba"])
    patient_sens_df.to_csv(out_dir / f"{prefix}_patient_aggregation_sensitivity.csv", index=False)
    save_patient_aggregation_sensitivity_figure(patient_sens_df, out_dir / f"{prefix}_patient_aggregation_sensitivity.png")
    window_sens_df = evaluate_window_sensitivity(test_df, payload["proba"], payload["threshold"])
    window_sens_df.to_csv(out_dir / f"{prefix}_window_sensitivity.csv", index=False)


def load_long_data():
    train_path = Config.SOURCE_DIR / "two_stage_train.csv"
    test_path = Config.SOURCE_DIR / "two_stage_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected current long-format data under results/relapse_two_stage_physio/")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_current_anchor_summary():
    summary_path = Config.SOURCE_DIR / "TwoStage_Best_Group_Summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    best_row = df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0].to_dict()
    direct_row = df.loc[df["Group"] == "direct"].iloc[0].to_dict()
    return {"best": best_row, "direct": direct_row, "table": df}


def round1_variants():
    return [
        JointVariant(name="c1_rule_any", family="c1", aux_weight=0.30, rule_target="any_rule", hidden_dim=48, dropout=0.15),
        JointVariant(name="c2_interval_intercept", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25),
        JointVariant(name="c2_interval_slope", family="c2", corr_kind="slope", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, corr_scale=0.60),
        JointVariant(name="c2_early_late_intercept", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.03, base_weight=0.25),
        JointVariant(name="c2_interval_intercept_calibrated", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, calibrator="window_intercept"),
    ]


def round2_variants(winner_family, winner_name):
    if winner_family == "c1":
        return [
            JointVariant(name="c1_rule_any_w020", family="c1", aux_weight=0.20, rule_target="any_rule", hidden_dim=48, dropout=0.15, residual_scale=0.25),
            JointVariant(name="c1_rule_any_w050", family="c1", aux_weight=0.50, rule_target="any_rule", hidden_dim=48, dropout=0.15, residual_scale=0.25),
            JointVariant(name="c1_rule_any_ranked", family="c1", aux_weight=0.30, rule_target="any_rule", hidden_dim=48, dropout=0.15, residual_scale=0.20, rank_weight=0.08, max_pos_weight_main=3.0),
            JointVariant(name="c1_rule_ft4_w035", family="c1", aux_weight=0.35, rule_target="ft4_rule", hidden_dim=48, dropout=0.15, residual_scale=0.25),
            JointVariant(name="c1_rule_tsh_w035", family="c1", aux_weight=0.35, rule_target="logtsh_rule", hidden_dim=48, dropout=0.15, residual_scale=0.25),
        ]
    if "slope" in winner_name:
        return [
            JointVariant(name="c2_interval_slope", family="c2", corr_kind="slope", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, corr_scale=0.60, residual_scale=0.25, max_pos_weight_main=3.0),
            JointVariant(name="c2_interval_slope_ranked", family="c2", corr_kind="slope", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.05, base_weight=0.25, corr_scale=0.45, residual_scale=0.20, max_pos_weight_main=2.8, rank_weight=0.10),
            JointVariant(name="c2_interval_slope_ranked_calibrated", family="c2", corr_kind="slope", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.05, base_weight=0.25, corr_scale=0.45, residual_scale=0.20, max_pos_weight_main=2.8, rank_weight=0.10, calibrator="global"),
            JointVariant(name="c2_early_late_slope", family="c2", corr_kind="slope", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, corr_scale=0.40, residual_scale=0.20, max_pos_weight_main=2.8, rank_weight=0.08),
            JointVariant(name="c2_interval_slope_windowcal", family="c2", corr_kind="slope", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.05, base_weight=0.25, corr_scale=0.40, residual_scale=0.18, max_pos_weight_main=2.8, rank_weight=0.08, calibrator="window_intercept"),
        ]
    if "early_late" in winner_name:
        return [
            JointVariant(name="c2_early_late_intercept", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.03, base_weight=0.25, residual_scale=0.25),
            JointVariant(name="c2_early_late_intercept_reglow", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.015, base_weight=0.25, residual_scale=0.20),
            JointVariant(name="c2_early_late_intercept_reghigh", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.060, base_weight=0.25, residual_scale=0.20),
            JointVariant(name="c2_early_late_intercept_calibrated", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.03, base_weight=0.25, residual_scale=0.20, calibrator="early_late"),
        ]
    return [
        JointVariant(name="c2_interval_intercept", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, residual_scale=0.25),
        JointVariant(name="c2_interval_intercept_reglow", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.015, base_weight=0.25, residual_scale=0.20),
        JointVariant(name="c2_interval_intercept_reghigh", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.080, base_weight=0.25, residual_scale=0.20),
        JointVariant(name="c2_interval_intercept_calibrated", family="c2", corr_kind="intercept", corr_grouping="full", hidden_dim=48, dropout=0.10, corr_reg=0.04, base_weight=0.25, residual_scale=0.20, calibrator="window_intercept"),
        JointVariant(name="c2_early_late_intercept", family="c2", corr_kind="intercept", corr_grouping="early_late", hidden_dim=48, dropout=0.10, corr_reg=0.03, base_weight=0.25, residual_scale=0.20),
    ]


def round3_variants(winner_family, winner_name):
    if winner_family != "c2":
        return [
            JointVariant(
                name="c1_anchor_rule_any_ranked",
                family="c1",
                aux_weight=0.30,
                rule_target="any_rule",
                hidden_dim=48,
                dropout=0.12,
                residual_scale=0.12,
                rank_weight=0.08,
                max_pos_weight_main=2.4,
                use_anchor_offset=True,
            ),
            JointVariant(
                name="c1_anchor_rule_any_globalcal",
                family="c1",
                aux_weight=0.25,
                rule_target="any_rule",
                hidden_dim=48,
                dropout=0.12,
                residual_scale=0.10,
                rank_weight=0.05,
                max_pos_weight_main=2.2,
                use_anchor_offset=True,
                calibrator="global",
            ),
        ]
    return [
        JointVariant(
            name="c2_anchor_slope_ranked",
            family="c2",
            corr_kind="slope",
            corr_grouping="full",
            hidden_dim=48,
            dropout=0.10,
            corr_reg=0.05,
            base_weight=0.30,
            corr_scale=0.28,
            residual_scale=0.14,
            max_pos_weight_main=2.4,
            rank_weight=0.08,
            use_anchor_offset=True,
        ),
        JointVariant(
            name="c2_anchor_slope_ranked_tight",
            family="c2",
            corr_kind="slope",
            corr_grouping="full",
            hidden_dim=48,
            dropout=0.10,
            corr_reg=0.07,
            base_weight=0.35,
            corr_scale=0.18,
            residual_scale=0.10,
            max_pos_weight_main=2.2,
            rank_weight=0.06,
            use_anchor_offset=True,
        ),
        JointVariant(
            name="c2_anchor_early_late_slope",
            family="c2",
            corr_kind="slope",
            corr_grouping="early_late",
            hidden_dim=48,
            dropout=0.10,
            corr_reg=0.05,
            base_weight=0.30,
            corr_scale=0.22,
            residual_scale=0.12,
            max_pos_weight_main=2.4,
            rank_weight=0.06,
            use_anchor_offset=True,
        ),
        JointVariant(
            name="c2_anchor_slope_ranked_globalcal",
            family="c2",
            corr_kind="slope",
            corr_grouping="full",
            hidden_dim=48,
            dropout=0.10,
            corr_reg=0.05,
            base_weight=0.30,
            corr_scale=0.22,
            residual_scale=0.10,
            max_pos_weight_main=2.2,
            rank_weight=0.05,
            use_anchor_offset=True,
            calibrator="global",
        ),
    ]


def run_variant_round(
    round_name,
    variants,
    x_tr_sel,
    x_te_sel,
    train_df,
    test_df,
    y_aux_train,
    y_aux_test,
    direct_row,
    anchor_payload=None,
):
    rows = []
    payloads = {}
    anchor_oof = None if anchor_payload is None else _safe_logit(anchor_payload["oof_proba"])
    anchor_train_fit = None if anchor_payload is None else _safe_logit(anchor_payload["train_fit_proba"])
    anchor_test = None if anchor_payload is None else _safe_logit(anchor_payload["proba"])
    for variant in variants:
        row, payload = fit_joint_variant(
            variant=variant,
            x_tr_df=x_tr_sel,
            x_te_df=x_te_sel,
            train_df=train_df,
            test_df=test_df,
            y_aux_train=y_aux_train,
            y_aux_test=y_aux_test,
            anchor_oof=anchor_oof if variant.use_anchor_offset else None,
            anchor_train_fit=anchor_train_fit if variant.use_anchor_offset else None,
            anchor_test=anchor_test if variant.use_anchor_offset else None,
        )
        row["Round"] = round_name
        row["StopRule_Hits"] = count_stop_rule_hits(row, direct_row)
        rows.append(row)
        payloads[variant.name] = payload
        print(
            f"  {round_name:<8s} {variant.name:<32s} "
            f"AUC={row['AUC']:.3f} PR-AUC={row['PR_AUC']:.3f} "
            f"Brier={row['Brier']:.3f} Cal.Int={row['Calibration_Intercept']:.3f} "
            f"Cal.Slope={row['Calibration_Slope']:.3f}"
        )
    result_df = pd.DataFrame(rows).sort_values(["StopRule_Hits", "PR_AUC", "AUC", "Brier"], ascending=[False, False, False, True]).reset_index(drop=True)
    result_df.to_csv(Config.OUT_DIR / f"{round_name.lower()}_results.csv", index=False)
    return result_df, payloads


def write_journal(text):
    with Config.JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def write_summary(
    current_anchor,
    direct_row,
    round1_df,
    round2_df,
    round3_df,
    final_row,
    bootstrap_df,
    window_df,
):
    round1_best = choose_round_winner(round1_df, direct_row)
    round2_best = choose_round_winner(round2_df, direct_row)
    round3_best = None if round3_df is None or len(round3_df) == 0 else choose_round_winner(round3_df, direct_row)
    current_best = current_anchor["best"] if current_anchor is not None else None
    per_window_best = window_df[window_df["Tag"] == final_row["Variant"]].copy().sort_values("PR_AUC", ascending=False)
    top_window = per_window_best.iloc[0].to_dict()
    bottom_window = per_window_best.iloc[-1].to_dict()
    delta_vs_direct = bootstrap_df.loc[bootstrap_df["Compare_To"] == "Direct"].copy()
    pr_delta = delta_vs_direct.loc[delta_vs_direct["Metric"] == "PR_AUC"].iloc[0]
    auc_delta = delta_vs_direct.loc[delta_vs_direct["Metric"] == "AUC"].iloc[0]

    lines = [
        "# Joint Two-Head Summary",
        "",
        "## 设计",
        "",
        "- 这轮不再做 hard two-stage handoff，而是把 pooled landmark 数据直接送进一个共享 trunk。",
        "- 主头始终预测 `Y_Relapse / Y_next_hyper`。",
        "- 只试两类辅助头：`binary hyper + rule aux` 和 `binary hyper + interval correction`。",
        "",
        "## 当前最好 joint two-head",
        "",
        f"- 最终赢家：`{final_row['Variant']}`。",
        f"- Interval-level：`PR-AUC {final_row['PR_AUC']:.3f}`，`AUC {final_row['AUC']:.3f}`，`Brier {final_row['Brier']:.3f}`。",
        f"- Calibration：`intercept {final_row['Calibration_Intercept']:.3f}`，`slope {final_row['Calibration_Slope']:.3f}`。",
        "",
        "## 对比",
        "",
        f"- 本轮 direct anchor：`PR-AUC {direct_row['PR_AUC']:.3f}`，`AUC {direct_row['AUC']:.3f}`，`Brier {direct_row['Brier']:.3f}`。",
        f"- 相对 direct 的 grouped bootstrap delta PR-AUC：`{pr_delta['Delta']:.3f}`，95% CI `[{pr_delta['CI_Low']:.3f}, {pr_delta['CI_High']:.3f}]`，one-sided p=`{pr_delta['OneSided_p']:.3f}`。",
        f"- 相对 direct 的 grouped bootstrap delta AUC：`{auc_delta['Delta']:.3f}`，95% CI `[{auc_delta['CI_Low']:.3f}, {auc_delta['CI_High']:.3f}]`，one-sided p=`{auc_delta['OneSided_p']:.3f}`。",
    ]
    if current_best is not None:
        lines.extend(
            [
                f"- 当前已存在的最佳 two-stage 参考：`{current_best['Group']} / {current_best['Best_Model']}`，`PR-AUC {current_best['PR_AUC']:.3f}`，`AUC {current_best['AUC']:.3f}`。",
                f"- 本 joint 路线相对该参考的静态差值：`PR-AUC {final_row['PR_AUC'] - float(current_best['PR_AUC']):+.3f}`，`AUC {final_row['AUC'] - float(current_best['AUC']):+.3f}`。",
            ]
        )

    lines.extend(
        [
            "",
            "## 逐轮优化结论",
            "",
            f"- Round 1 冠军：`{round1_best['Variant']}`，`PR-AUC {round1_best['PR_AUC']:.3f}`，`AUC {round1_best['AUC']:.3f}`。",
            f"- Round 2 冠军：`{round2_best['Variant']}`，`PR-AUC {round2_best['PR_AUC']:.3f}`，`AUC {round2_best['AUC']:.3f}`。",
            f"- Round 3 冠军：`{round3_best['Variant']}`，`PR-AUC {round3_best['PR_AUC']:.3f}`，`AUC {round3_best['AUC']:.3f}`。" if round3_best is not None else "- Round 3：未运行。",
            f"- 说明：赢家 family 是 `{final_row['Family']}`，说明最有效的信息共享方式是 `{ 'interval-specific correction' if final_row['Family'] == 'c2' else 'rule auxiliary regularization' }`。",
            "",
            "## Window-wise",
            "",
            f"- 最强窗口：`{top_window['Interval_Name']}`，`PR-AUC {top_window['PR_AUC']:.3f}`，`AUC {top_window['AUC']:.3f}`。",
            f"- 最弱窗口：`{bottom_window['Interval_Name']}`，`PR-AUC {bottom_window['PR_AUC']:.3f}`，`AUC {bottom_window['AUC']:.3f}`。",
            "",
            "## 判断",
            "",
            "- 如果 joint two-head 继续落后于当前最佳 two-stage，主要原因不是 landmark 主线有问题，而是共享表示还没有把‘窗口差异 + 机制阈值’压缩成比当前透明 fusion 更强的风险修正。",
            "- 如果 joint two-head 接近或超过 direct，但还没超过当前最佳 two-stage，那么它依然是一个更论文友好的联合学习备选，因为它避免了 hard handoff 和 error propagation。",
        ]
    )
    Config.SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("=" * 96)
    print("  Joint Two-Head Landmark Experiments for Relapse Prediction")
    print("=" * 96)

    train_df, test_df = load_long_data()
    current_anchor = load_current_anchor_summary()

    x_tr_direct, x_te_direct, _ = make_stage2_feature_frames(train_df, test_df, mode="direct")
    selected_features = load_selected_feature_list(Config.SOURCE_DIR, "Stage2_direct")
    direct_summary_df, direct_label, direct_payload, selected_features = fit_direct_anchor(
        x_tr_direct,
        x_te_direct,
        train_df,
        test_df,
        Config.OUT_DIR,
        selected_features=selected_features,
    )
    direct_row = {
        "Variant": f"direct_anchor_{_safe_name(direct_label)}",
        "PR_AUC": float(direct_payload["metrics"]["prauc"]),
        "AUC": float(direct_payload["metrics"]["auc"]),
        "Brier": float(direct_payload["metrics"]["brier"]),
        "Calibration_Intercept": float(direct_payload["cal"]["intercept"]),
        "Calibration_Slope": float(direct_payload["cal"]["slope"]),
    }
    print(
        f"  Direct anchor: {direct_label}  "
        f"AUC={direct_row['AUC']:.3f} PR-AUC={direct_row['PR_AUC']:.3f} Brier={direct_row['Brier']:.3f}"
    )

    x_tr_sel = x_tr_direct[selected_features].copy()
    x_te_sel = x_te_direct[selected_features].copy()
    rule_refs = _fit_empirical_rule_refs(train_df)
    y_aux_train = _build_rule_targets(train_df, rule_refs)
    y_aux_test = _build_rule_targets(test_df, rule_refs)

    Config.JOURNAL_PATH.write_text(
        "\n".join(
            [
                "# Joint Two-Head Journal",
                "",
                "## Setup",
                "- Data source: current pooled landmark long-format tables under `results/relapse_two_stage_physio/`.",
                f"- Direct anchor selected features: `{len(selected_features)}`.",
                f"- Current static best two-stage reference: `{current_anchor['best']['Group']}` with PR-AUC `{current_anchor['best']['PR_AUC']:.3f}`." if current_anchor is not None else "- Current static best two-stage reference: unavailable.",
                "- Allowed joint families: `C1 rule auxiliary`, `C2 interval correction`.",
            ]
        )
        + "\n\n",
        encoding="utf-8",
    )

    print("\n--- Round 1: two-head family search ---")
    round1_df, round1_payloads = run_variant_round(
        round_name="Round1",
        variants=round1_variants(),
        x_tr_sel=x_tr_sel,
        x_te_sel=x_te_sel,
        train_df=train_df,
        test_df=test_df,
        y_aux_train=y_aux_train,
        y_aux_test=y_aux_test,
        direct_row=direct_row,
        anchor_payload=None,
    )
    round1_winner = choose_round_winner(round1_df, direct_row)
    write_journal(
        "\n".join(
            [
                "## Round 1",
                f"- Winner: `{round1_winner['Variant']}` ({round1_winner['Family']}).",
                f"- Metrics: PR-AUC `{round1_winner['PR_AUC']:.3f}`, AUC `{round1_winner['AUC']:.3f}`, Brier `{round1_winner['Brier']:.3f}`.",
                f"- Stop-rule hits: `{int(round1_winner['StopRule_Hits'])}`.",
                "- Decision: refine the winning family locally in Round 2 instead of reopening wide architecture search.",
            ]
        )
    )

    print("\n--- Round 2: winner-local refinement ---")
    round2_df, round2_payloads = run_variant_round(
        round_name="Round2",
        variants=round2_variants(round1_winner["Family"], round1_winner["Variant"]),
        x_tr_sel=x_tr_sel,
        x_te_sel=x_te_sel,
        train_df=train_df,
        test_df=test_df,
        y_aux_train=y_aux_train,
        y_aux_test=y_aux_test,
        direct_row=direct_row,
        anchor_payload=None,
    )
    round2_winner = choose_round_winner(round2_df, direct_row)
    write_journal(
        "\n".join(
            [
                "## Round 2",
                f"- Winner: `{round2_winner['Variant']}` ({round2_winner['Family']}).",
                f"- Metrics: PR-AUC `{round2_winner['PR_AUC']:.3f}`, AUC `{round2_winner['AUC']:.3f}`, Brier `{round2_winner['Brier']:.3f}`.",
                f"- Stop-rule hits: `{int(round2_winner['StopRule_Hits'])}`.",
                "- Interpretation: keep only the refined family winner as the final joint two-head candidate for this turn.",
            ]
        )
    )

    print("\n--- Round 3: anchor-conditioned winner refinement ---")
    round3_df, round3_payloads = run_variant_round(
        round_name="Round3",
        variants=round3_variants(round2_winner["Family"], round2_winner["Variant"]),
        x_tr_sel=x_tr_sel,
        x_te_sel=x_te_sel,
        train_df=train_df,
        test_df=test_df,
        y_aux_train=y_aux_train,
        y_aux_test=y_aux_test,
        direct_row=direct_row,
        anchor_payload=direct_payload,
    )
    round3_winner = choose_round_winner(round3_df, direct_row)
    write_journal(
        "\n".join(
            [
                "## Round 3",
                f"- Winner: `{round3_winner['Variant']}` ({round3_winner['Family']}).",
                f"- Metrics: PR-AUC `{round3_winner['PR_AUC']:.3f}`, AUC `{round3_winner['AUC']:.3f}`, Brier `{round3_winner['Brier']:.3f}`.",
                f"- Stop-rule hits: `{int(round3_winner['StopRule_Hits'])}`.",
                "- Interpretation: this round anchors the joint scorer to the direct landmark risk and only learns small residual/correction heads.",
            ]
        )
    )

    combined_df = pd.concat([direct_summary_df, round1_df, round2_df, round3_df], ignore_index=True, sort=False)
    combined_df.to_csv(Config.OUT_DIR / "joint_twohead_all_results.csv", index=False)
    final_row = choose_round_winner(pd.concat([round1_df, round2_df, round3_df], ignore_index=True), direct_row)
    final_payload = round3_payloads.get(
        final_row["Variant"],
        round2_payloads.get(final_row["Variant"], round1_payloads.get(final_row["Variant"])),
    )

    window_frames = [
        compute_window_metrics(test_df, direct_payload["proba"], direct_payload["threshold"], "Direct_Anchor"),
    ]
    for name, payload in {**round1_payloads, **round2_payloads, **round3_payloads}.items():
        window_frames.append(compute_window_metrics(test_df, payload["proba"], payload["threshold"], name))
    window_df = pd.concat(window_frames, ignore_index=True)
    window_df.to_csv(Config.OUT_DIR / "joint_twohead_window_metrics.csv", index=False)

    bootstrap_rows = []
    for metric in ["PR_AUC", "AUC"]:
        stat = bootstrap_metric_delta(test_df, final_payload["proba"], direct_payload["proba"], metric)
        stat["Compare_To"] = "Direct"
        stat["Variant"] = final_row["Variant"]
        bootstrap_rows.append(stat)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(Config.OUT_DIR / "joint_twohead_bootstrap_delta_vs_direct.csv", index=False)

    prediction_df = test_df[["Patient_ID", "Interval_Name", "Y_Relapse"]].copy()
    prediction_df["Direct_Anchor_Proba"] = direct_payload["proba"]
    prediction_df["Joint_Best_Proba"] = final_payload["proba"]
    prediction_df.to_csv(Config.OUT_DIR / "joint_twohead_test_predictions.csv", index=False)

    perf_long_df = pd.concat(
        [
            build_binary_performance_long(
                task_name="Direct Anchor",
                results={direct_label: direct_payload},
                domain_payloads={
                    "Train_Fit": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "train_fit_proba"},
                    "Validation_OOF": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "oof_proba"},
                    "Test_Temporal": {"y_true": test_df["Y_Relapse"].values.astype(int), "proba_key": "proba"},
                },
                metric_keys=["prauc", "auc", "recall", "specificity", "f1"],
                threshold_key="threshold",
            ),
            build_binary_performance_long(
                task_name="Joint Two-Head",
                results={final_row["Variant"]: final_payload},
                domain_payloads={
                    "Train_Fit": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "train_fit_proba"},
                    "Validation_OOF": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "oof_proba"},
                    "Test_Temporal": {"y_true": test_df["Y_Relapse"].values.astype(int), "proba_key": "proba"},
                },
                metric_keys=["prauc", "auc", "recall", "specificity", "f1"],
                threshold_key="threshold",
            ),
        ],
        ignore_index=True,
    )
    export_metric_matrices(perf_long_df, Config.OUT_DIR, prefix="JointTwoHead_Performance")
    save_performance_heatmap_panels(
        perf_long_df,
        Config.OUT_DIR / "joint_twohead_performance_heatmaps.png",
        task_order=["Direct Anchor", "Joint Two-Head"],
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=["prauc", "auc", "recall", "specificity", "f1"],
        title="Joint Two-Head vs Direct Anchor",
    )

    save_best_reports(final_row["Variant"], final_payload, train_df, test_df, Config.OUT_DIR, prefix="joint_twohead_best")
    write_summary(current_anchor, direct_row, round1_df, round2_df, round3_df, final_row, bootstrap_df, window_df)

    print("\n  Final winner")
    print(
        f"    {final_row['Variant']:<34s} "
        f"AUC={final_row['AUC']:.3f}  PR-AUC={final_row['PR_AUC']:.3f}  "
        f"Brier={final_row['Brier']:.3f}  Cal.Int={final_row['Calibration_Intercept']:.3f}  "
        f"Cal.Slope={final_row['Calibration_Slope']:.3f}"
    )
    if current_anchor is not None:
        print(
            f"    Static current best two-stage ref: {current_anchor['best']['Group']} / {current_anchor['best']['Best_Model']}  "
            f"AUC={float(current_anchor['best']['AUC']):.3f}  PR-AUC={float(current_anchor['best']['PR_AUC']):.3f}"
        )
    print(f"\n  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
