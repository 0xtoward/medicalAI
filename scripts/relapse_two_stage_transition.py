import os
import sys
import warnings
from abc import ABC, abstractmethod
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
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED, STATIC_NAMES, STATE_NAMES, TIME_STAMPS
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    clear_pkl_cache,
    load_data as _load_data,
    load_or_fit_depth_imputer,
    split_imputed,
)
from utils.evaluation import (
    aggregate_patient_level,
    bootstrap_group_cis,
    compute_binary_metrics,
    compute_calibration_stats,
    concordance_index_simple,
    evaluate_patient_aggregation_sensitivity,
    evaluate_window_sensitivity,
    format_ci,
    save_calibration_figure,
    save_dca_figure,
    save_patient_aggregation_sensitivity_figure,
    save_patient_risk_strata,
    save_threshold_sensitivity_figure,
    select_best_threshold,
)
from utils.feature_selection import select_binary_features_with_l1
from utils.model_viz import save_logistic_regression_visuals
from utils.performance_panels import (
    build_binary_performance_long,
    export_metric_matrices,
    save_performance_heatmap_panels,
)
from utils.physio_forecast import (
    TARGET_COLS,
    CURRENT_CONTEXT_COLS,
    add_stage1_prediction_family,
    evaluate_next_state_predictions,
    evaluate_physio_predictions,
    get_next_state_models,
    make_stage1_feature_frames,
    make_stage2_feature_frames,
    run_stage1_oof_forecast,
    save_physio_scatter,
    save_stage1_metric_bar,
)
from utils.plot_style import (
    PRIMARY_BLUE,
    PRIMARY_TEAL,
    TEXT_MID,
    apply_publication_style,
)
from utils.recurrence import derive_recurrent_survival_data

apply_publication_style()
np.random.seed(SEED)


TIME_MONTHS = [0, 1, 3, 6, 12, 18, 24]
HIGH_RISK_WINDOWS = {"1M->3M", "3M->6M", "6M->12M"}
WINDOWED_CORE_CANDIDATE_FEATURES = [
    "Time_In_Normal",
    "FT3_Current",
    "Delta_TSH_k0",
    "FT4_Current",
    "FT4_Mean_ToDate",
    "logTSH_Mean_ToDate",
]
WINDOWED_CORE_INTERACTION_FEATURES = [
    "Time_In_Normal",
    "FT3_Current",
    "Delta_TSH_k0",
]
MONOTONIC_DIRECTIONS = {
    "Time_In_Normal": -1,
    "FT3_Current": 1,
    "logTSH_Current": -1,
    "Delta_TSH_k0": -1,
    "Prior_Relapse_Count": 1,
    "Ever_Hyper_Before": 1,
}
PREV_BEST_NEW_CI_LOW = -0.013
RESULT_PREFIX = "TwoStageTransition"
RESULT_REPORT_NAME = "TwoStage_Transition_Round.md"
RESULT_TITLE = "Two-Stage Transition Sidecar Optimization"
RESULT_GROUP_PLOT = "TwoStageTransition_Group_Comparison.png"
RESULT_WINDOW_PLOT = "TwoStageTransition_PerWindow_PR_AUC_Comparison.png"
RESULT_BOOTSTRAP_PLOT = "TwoStageTransition_PR_AUC_Delta_Forest.png"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    direct_variant: str
    sidecar_variant: str
    stage1_scope: str = "normal"
    use_rule_sidecar: bool = False
    phase: str = "A"


class ScoreCalibrationStrategy(ABC):
    """Reusable calibration design contract for champion sidecars."""

    key = "base"
    description = "base"

    @abstractmethod
    def build_frames(self, train_df, test_df, raw_oof_logit, raw_test_logit):
        """Return train/test calibration design matrices."""


class GlobalCalibrationStrategy(ScoreCalibrationStrategy):
    key = "global"
    description = "shared slope only"

    def build_frames(self, train_df, test_df, raw_oof_logit, raw_test_logit):
        tr = pd.DataFrame({"Score_Logit": np.asarray(raw_oof_logit, dtype=float)})
        te = pd.DataFrame({"Score_Logit": np.asarray(raw_test_logit, dtype=float)})
        return tr, te


class PhaseInterceptCalibrationStrategy(ScoreCalibrationStrategy):
    key = "phase"
    description = "shared slope + early/late intercepts"

    def build_frames(self, train_df, test_df, raw_oof_logit, raw_test_logit):
        tr = pd.DataFrame({"Score_Logit": np.asarray(raw_oof_logit, dtype=float)})
        te = pd.DataFrame({"Score_Logit": np.asarray(raw_test_logit, dtype=float)})
        train_phase = np.where(train_df["Interval_Name"].astype(str).isin(HIGH_RISK_WINDOWS), "HighRisk", "Late")
        test_phase = np.where(test_df["Interval_Name"].astype(str).isin(HIGH_RISK_WINDOWS), "HighRisk", "Late")
        for label in ["HighRisk", "Late"]:
            col = f"Phase_{label}"
            tr[col] = (train_phase == label).astype(float)
            te[col] = (test_phase == label).astype(float)
        return tr, te


class IntervalInterceptCalibrationStrategy(ScoreCalibrationStrategy):
    key = "interval"
    description = "shared slope + interval intercepts"

    def build_frames(self, train_df, test_df, raw_oof_logit, raw_test_logit):
        tr = pd.DataFrame({"Score_Logit": np.asarray(raw_oof_logit, dtype=float)})
        te = pd.DataFrame({"Score_Logit": np.asarray(raw_test_logit, dtype=float)})
        for label in sorted(train_df["Interval_Name"].astype(str).unique()):
            col = f"Interval_{label}"
            tr[col] = (train_df["Interval_Name"].astype(str).values == label).astype(float)
            te[col] = (test_df["Interval_Name"].astype(str).values == label).astype(float)
        return tr, te


CALIBRATION_STRATEGIES = {
    GlobalCalibrationStrategy.key: GlobalCalibrationStrategy(),
    PhaseInterceptCalibrationStrategy.key: PhaseInterceptCalibrationStrategy(),
    IntervalInterceptCalibrationStrategy.key: IntervalInterceptCalibrationStrategy(),
}


class Config:
    OUT_DIR = Path("./results/relapse_two_stage_transition/")
    BASE_TWO_STAGE_DIR = Path("./results/relapse_two_stage_physio/")
    SHARED_RELAPSE_DIR = Path("./results/relapse/")
    LEGACY_RELAPSE_DIR = Path("./multistate_result/")


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_fit_with_message(raw_train, cache_path, fallback_cache_path=None, label=""):
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None
    if primary.exists():
        print(f"  MissForest {label}: loaded from {primary}")
    elif fallback is not None and fallback.exists():
        print(f"  MissForest {label}: reused shared cache {fallback}")
    else:
        print(f"  MissForest {label}: fitting on train ({len(raw_train)} records)...")
    return load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=fallback_cache_path)


def _safe_logit(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _safe_model_tag(name):
    return str(name).replace(" ", "_").replace("/", "_")


def build_two_stage_long_data():
    local_train = Config.OUT_DIR / "two_stage_train.csv"
    local_test = Config.OUT_DIR / "two_stage_test.csv"
    if local_train.exists() and local_test.exists():
        print("  Reusing local two-stage long data cache")
        return pd.read_csv(local_train), pd.read_csv(local_test)

    shared_train = Config.BASE_TWO_STAGE_DIR / "two_stage_train.csv"
    shared_test = Config.BASE_TWO_STAGE_DIR / "two_stage_test.csv"
    if shared_train.exists() and shared_test.exists():
        print("  Reusing baseline two-stage long data cache")
        df_train = pd.read_csv(shared_train)
        df_test = pd.read_csv(shared_test)
        df_train.to_csv(local_train, index=False)
        df_test.to_csv(local_test, index=False)
        return df_train, df_test

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    row_ids = np.arange(len(pids))
    print(f"  Records: {len(pids)}")

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    print(f"  Train: {int(tr_mask.sum())} records")
    print(f"  Test:  {int(te_mask.sum())} records")

    print("\n--- Phase 1: Build two-stage long-format data ---")
    df_tr_parts = []
    df_te_parts = []

    from utils.recurrence import build_interval_risk_data
    from utils.physio_forecast import build_next_physio_targets, build_physio_history_features

    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"

        raw_relapse = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        relapse_cache = Config.SHARED_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        if not relapse_cache.exists():
            relapse_cache = Config.LEGACY_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        local_relapse_cache = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer_cur = load_or_fit_with_message(
            raw_relapse[tr_mask],
            local_relapse_cache,
            fallback_cache_path=relapse_cache,
            label=f"current-depth-{depth} ({interval_name})",
        )
        filled_tr = apply_missforest(raw_relapse[tr_mask], imputer_cur, depth)
        filled_te = apply_missforest(raw_relapse[te_mask], imputer_cur, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        s_tr = build_states_from_labels(ev_tr)
        s_te = build_states_from_labels(ev_te)
        x_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        x_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_cur_tr = build_interval_risk_data(
            xs_tr, x_d_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_cur_te = build_interval_risk_data(
            xs_te, x_d_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )
        df_hist_tr = build_physio_history_features(
            x_d_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_hist_te = build_physio_history_features(
            x_d_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )

        n_lab_next = depth + 1
        raw_phys = np.hstack([X_s_raw, ft3_raw[:, :n_lab_next], ft4_raw[:, :n_lab_next], tsh_raw[:, :n_lab_next]])
        shared_phys_cache = Config.BASE_TWO_STAGE_DIR / f"missforest_physio_depth{depth}.pkl"
        phys_cache = Config.OUT_DIR / f"missforest_physio_depth{depth}.pkl"
        imputer_phys = load_or_fit_with_message(
            raw_phys[tr_mask],
            phys_cache,
            fallback_cache_path=shared_phys_cache if shared_phys_cache.exists() else None,
            label=f"physio-depth-{depth} ({interval_name})",
        )
        filled_phys_tr = apply_missforest(raw_phys[tr_mask], imputer_phys, 0)
        filled_phys_te = apply_missforest(raw_phys[te_mask], imputer_phys, 0)
        _, ft3n_tr, ft4n_tr, tshn_tr = split_imputed(filled_phys_tr, n_static, n_lab_next, 0)
        _, ft3n_te, ft4n_te, tshn_te = split_imputed(filled_phys_te, n_static, n_lab_next, 0)
        x_d_next_tr = np.stack([ft3n_tr, ft4n_tr, tshn_tr], axis=-1)
        x_d_next_te = np.stack([ft3n_te, ft4n_te, tshn_te], axis=-1)
        df_next_tr = build_next_physio_targets(
            x_d_next_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_next_te = build_next_physio_targets(
            x_d_next_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )

        df_tr = df_cur_tr.merge(df_hist_tr, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_tr = df_tr.merge(df_next_tr, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_te = df_cur_te.merge(df_hist_te, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_te = df_te.merge(df_next_te, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")

        df_tr_parts.append(df_tr)
        df_te_parts.append(df_te)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr)}  test {len(df_te)} rows")

    df_train = pd.concat(df_tr_parts, ignore_index=True)
    df_test = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_train)}  test {len(df_test)} rows")
    df_train.to_csv(local_train, index=False)
    df_test.to_csv(local_test, index=False)
    return df_train, df_test


def _state_in_scope(state_code, current_scope):
    if current_scope == "normal":
        return int(state_code) == 1
    if current_scope == "nonhyper":
        return int(state_code) != 0
    raise ValueError(f"Unknown current_scope: {current_scope}")


def build_transition_scope_rows(X_s, X_d, X_d_next, S_matrix, pids, current_scope="nonhyper", target_k=None, row_ids=None):
    rows = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)
    if row_ids is None:
        row_ids = np.arange(n_samples)

    for i in range(n_samples):
        pid = pids[i]
        row_id = int(row_ids[i])
        xs = X_s[i]
        for k in k_range:
            cur_state = int(S_matrix[i, k])
            if not _state_in_scope(cur_state, current_scope):
                continue

            next_state = int(S_matrix[i, k + 1])
            labs_k = X_d[i, k, :]
            labs_0 = X_d[i, 0, :]
            labs_next = X_d_next[i, k + 1, :]
            hist = X_d[i, : k + 1, :]
            prev_labs = X_d[i, k - 1, :] if k > 0 else X_d[i, k, :]
            hist_log_tsh = np.log1p(np.clip(hist[:, 2], 0, None))
            hist_states = S_matrix[i, :k]
            prev_state = int(S_matrix[i, k - 1]) if k > 0 else -1
            prior_relapse_count = int(sum(int(hist_states[m] == 1 and hist_states[m + 1] == 0) for m in range(len(hist_states) - 1)))
            ever_hyper = int(prior_relapse_count > 0)
            ever_hypo = int(2 in hist_states)
            time_in_normal = int(np.sum(hist_states == 1))

            delta_ft4_k0 = labs_k[1] - labs_0[1]
            delta_tsh_k0 = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(np.clip(labs_0[2], 0, None))
            if k > 0:
                labs_prev = X_d[i, k - 1, :]
                delta_ft4_1step = labs_k[1] - labs_prev[1]
                delta_tsh_1step = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(np.clip(labs_prev[2], 0, None))
            else:
                delta_ft4_1step = 0.0
                delta_tsh_1step = 0.0

            rows.append(
                {
                    "Patient_ID": pid,
                    "Source_Row": row_id,
                    "Interval_ID": k,
                    "Interval_Name": f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}",
                    "Start_Time": TIME_MONTHS[k],
                    "Stop_Time": TIME_MONTHS[k + 1],
                    "Interval_Width": TIME_MONTHS[k + 1] - TIME_MONTHS[k],
                    "Current_State": STATE_NAMES[cur_state],
                    "Next_State": STATE_NAMES[next_state],
                    "Prior_Relapse_Count": prior_relapse_count,
                    "Event_Order": prior_relapse_count + 1,
                    **dict(zip(STATIC_NAMES, xs)),
                    "FT3_Current": labs_k[0],
                    "FT4_Current": labs_k[1],
                    "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                    "FT3_Baseline": float(labs_0[0]),
                    "FT4_Baseline": float(labs_0[1]),
                    "logTSH_Baseline": float(np.log1p(np.clip(labs_0[2], 0, None))),
                    "FT3_Prev": float(prev_labs[0]),
                    "FT4_Prev": float(prev_labs[1]),
                    "logTSH_Prev": float(np.log1p(np.clip(prev_labs[2], 0, None))),
                    "FT3_Mean_ToDate": float(np.mean(hist[:, 0])),
                    "FT4_Mean_ToDate": float(np.mean(hist[:, 1])),
                    "logTSH_Mean_ToDate": float(np.mean(hist_log_tsh)),
                    "FT3_STD_ToDate": float(np.std(hist[:, 0])),
                    "FT4_STD_ToDate": float(np.std(hist[:, 1])),
                    "logTSH_STD_ToDate": float(np.std(hist_log_tsh)),
                    "Prev_State": str(prev_state),
                    "Ever_Hyper_Before": ever_hyper,
                    "Ever_Hypo_Before": ever_hypo,
                    "Time_In_Normal": time_in_normal,
                    "Delta_FT4_k0": delta_ft4_k0,
                    "Delta_TSH_k0": delta_tsh_k0,
                    "Delta_FT4_1step": delta_ft4_1step,
                    "Delta_TSH_1step": delta_tsh_1step,
                    "FT3_Next": float(labs_next[0]),
                    "FT4_Next": float(labs_next[1]),
                    "logTSH_Next": float(np.log1p(np.clip(labs_next[2], 0, None))),
                }
            )

    return pd.DataFrame(rows)


def build_stage1_transition_scope_data(current_scope="nonhyper"):
    local_train = Config.OUT_DIR / f"stage1_scope_{current_scope}_train.csv"
    local_test = Config.OUT_DIR / f"stage1_scope_{current_scope}_test.csv"
    local_counts = Config.OUT_DIR / "Stage1_Scope_Counts.csv"
    if local_train.exists() and local_test.exists() and local_counts.exists():
        print(f"  Reusing local stage1 scope cache: {current_scope}")
        return pd.read_csv(local_train), pd.read_csv(local_test), pd.read_csv(local_counts)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    row_ids = np.arange(len(pids))
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask

    df_tr_parts = []
    df_te_parts = []
    count_rows = []

    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"

        raw_relapse = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        relapse_cache = Config.SHARED_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        if not relapse_cache.exists():
            relapse_cache = Config.LEGACY_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        local_relapse_cache = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer_cur = load_or_fit_with_message(
            raw_relapse[tr_mask],
            local_relapse_cache,
            fallback_cache_path=relapse_cache,
            label=f"scope-current-depth-{depth} ({interval_name})",
        )
        filled_tr = apply_missforest(raw_relapse[tr_mask], imputer_cur, depth)
        filled_te = apply_missforest(raw_relapse[te_mask], imputer_cur, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        s_tr = build_states_from_labels(ev_tr)
        s_te = build_states_from_labels(ev_te)
        x_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        x_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        n_lab_next = depth + 1
        raw_phys = np.hstack([X_s_raw, ft3_raw[:, :n_lab_next], ft4_raw[:, :n_lab_next], tsh_raw[:, :n_lab_next]])
        shared_phys_cache = Config.BASE_TWO_STAGE_DIR / f"missforest_physio_depth{depth}.pkl"
        phys_cache = Config.OUT_DIR / f"missforest_physio_depth{depth}.pkl"
        imputer_phys = load_or_fit_with_message(
            raw_phys[tr_mask],
            phys_cache,
            fallback_cache_path=shared_phys_cache if shared_phys_cache.exists() else None,
            label=f"scope-physio-depth-{depth} ({interval_name})",
        )
        filled_phys_tr = apply_missforest(raw_phys[tr_mask], imputer_phys, 0)
        filled_phys_te = apply_missforest(raw_phys[te_mask], imputer_phys, 0)
        _, ft3n_tr, ft4n_tr, tshn_tr = split_imputed(filled_phys_tr, n_static, n_lab_next, 0)
        _, ft3n_te, ft4n_te, tshn_te = split_imputed(filled_phys_te, n_static, n_lab_next, 0)
        x_d_next_tr = np.stack([ft3n_tr, ft4n_tr, tshn_tr], axis=-1)
        x_d_next_te = np.stack([ft3n_te, ft4n_te, tshn_te], axis=-1)

        df_tr = build_transition_scope_rows(
            xs_tr, x_d_tr, x_d_next_tr, s_tr, pids[tr_mask], current_scope=current_scope, target_k=k, row_ids=row_ids[tr_mask]
        )
        df_te = build_transition_scope_rows(
            xs_te, x_d_te, x_d_next_te, s_te, pids[te_mask], current_scope=current_scope, target_k=k, row_ids=row_ids[te_mask]
        )
        df_tr_parts.append(df_tr)
        df_te_parts.append(df_te)

        for split_name, S in [("Train", s_tr), ("Test", s_te)]:
            cur = S[:, k]
            nxt = S[:, k + 1]
            count_rows.append(
                {
                    "Split": split_name,
                    "Interval_Name": interval_name,
                    "current_normal": int(np.sum(cur == 1)),
                    "current_nonhyper": int(np.sum(cur != 0)),
                    "next_hyper_from_normal": int(np.sum((cur == 1) & (nxt == 0))),
                    "next_hyper_from_nonhyper": int(np.sum((cur != 0) & (nxt == 0))),
                }
            )

        print(f"    scope-{current_scope} depth-{depth} ({interval_name}): train {len(df_tr)}  test {len(df_te)} rows")

    df_train = pd.concat(df_tr_parts, ignore_index=True)
    df_test = pd.concat(df_te_parts, ignore_index=True)
    count_df = pd.DataFrame(count_rows)
    df_train.to_csv(local_train, index=False)
    df_test.to_csv(local_test, index=False)
    count_df.to_csv(local_counts, index=False)
    return df_train, df_test, count_df


def make_stage1_scope_feature_frames(train_df, test_df):
    feat_cols = [col for col in CURRENT_CONTEXT_COLS if col in train_df.columns]
    cat_cols = [col for col in ["Interval_Name", "Prev_State", "Current_State"] if col in train_df.columns]
    tr = train_df[feat_cols + cat_cols].copy().reset_index(drop=True)
    te = test_df[feat_cols + cat_cols].copy().reset_index(drop=True)

    for col in cat_cols:
        cats = sorted(train_df[col].astype(str).unique())
        for cat in cats:
            name = f"{col}_{cat}"
            tr[name] = (train_df[col].astype(str).values == cat).astype(float)
            te[name] = (test_df[col].astype(str).values == cat).astype(float)

    tr = tr.drop(columns=cat_cols).replace([np.inf, -np.inf], np.nan)
    te = te.drop(columns=cat_cols).replace([np.inf, -np.inf], np.nan)
    medians = tr.median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)
    keep_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols], keep_cols


def _binary_proba_2col(proba):
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.shape[1] == 1:
        arr = np.column_stack([1.0 - arr[:, 0], arr[:, 0]])
    return np.clip(arr, 1e-8, 1 - 1e-8)


def run_binary_suite_oof(x_train, train_df, x_test, test_df, groups, y_train=None, y_test=None):
    if y_train is None:
        y_train = (train_df["Next_State"].astype(str).values == "Hyper").astype(int)
    if y_test is None:
        y_test = (test_df["Next_State"].astype(str).values == "Hyper").astype(int)

    n_splits = min(3, len(pd.Index(groups).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)
    metric_frames = []
    results = {}

    for name, base_model in get_next_state_models().items():
        oof_proba = np.zeros((len(y_train), 2), dtype=float)
        for tr_idx, val_idx in gkf.split(x_train, y_train, groups=groups):
            model = clone(base_model)
            model.fit(x_train.iloc[tr_idx], y_train[tr_idx])
            oof_proba[val_idx] = _binary_proba_2col(model.predict_proba(x_train.iloc[val_idx]))

        fitted = clone(base_model)
        fitted.fit(x_train, y_train)
        train_fit = _binary_proba_2col(fitted.predict_proba(x_train))
        test_proba = _binary_proba_2col(fitted.predict_proba(x_test))
        metric_frames.append(evaluate_next_state_predictions(y_train, oof_proba, "Train_OOF", name))
        test_metrics = evaluate_next_state_predictions(y_test, test_proba, "Test", name)
        results[name] = {
            "model": fitted,
            "train_fit_proba": train_fit[:, 1],
            "oof_proba": oof_proba,
            "test_proba": test_proba,
            "metrics": pd.concat([metric_frames[-1], test_metrics], ignore_index=True),
        }

    base_names = list(results.keys())
    if base_names:
        oof_stack = np.stack([results[name]["oof_proba"] for name in base_names], axis=0)
        test_stack = np.stack([results[name]["test_proba"] for name in base_names], axis=0)
        train_fit_stack = np.stack([_binary_proba_2col(results[name]["train_fit_proba"]) for name in base_names], axis=0)
        ensemble_oof = np.mean(oof_stack, axis=0)
        ensemble_test = np.mean(test_stack, axis=0)
        ensemble_train_fit = np.mean(train_fit_stack, axis=0)
        metric_frames.append(evaluate_next_state_predictions(y_train, ensemble_oof, "Train_OOF", "EnsembleMean"))
        test_metrics = evaluate_next_state_predictions(y_test, ensemble_test, "Test", "EnsembleMean")
        results["EnsembleMean"] = {
            "model": {"members": base_names, "type": "mean_ensemble"},
            "train_fit_proba": ensemble_train_fit[:, 1],
            "oof_proba": ensemble_oof,
            "test_proba": ensemble_test,
            "metrics": pd.concat([metric_frames[-1], test_metrics], ignore_index=True),
        }

    metrics_df = pd.concat([payload["metrics"] for payload in results.values()], ignore_index=True)
    score_df = metrics_df[metrics_df["Split"] == "Train_OOF"].pivot_table(index="Model", columns="Metric", values="Value")
    best_name = score_df.sort_values(["Hyper_PR_AUC", "AUC"], ascending=[False, False]).index[0]
    return best_name, results, metrics_df


def _align_prediction_array(reference_df, scope_df, values):
    keys = ["Patient_ID", "Source_Row", "Interval_ID"]
    arr = np.asarray(values)
    if arr.ndim == 1:
        tmp = scope_df[keys].copy()
        tmp["pred"] = arr
        merged = reference_df[keys].merge(tmp, on=keys, how="left")
        return merged["pred"].values.astype(float)

    cols = [f"pred_{idx}" for idx in range(arr.shape[1])]
    tmp = scope_df[keys].copy()
    for idx, col in enumerate(cols):
        tmp[col] = arr[:, idx]
    merged = reference_df[keys].merge(tmp, on=keys, how="left")
    return merged[cols].values.astype(float)


def align_scope_state_results_to_reference(reference_df, scope_df, scope_results):
    aligned = {}
    for name, pack in scope_results.items():
        aligned[name] = {
            "model": pack["model"],
            "train_fit_proba": _align_prediction_array(reference_df, scope_df, pack["train_fit_proba"]),
            "oof_proba": _align_prediction_array(reference_df, scope_df, pack["oof_proba"]),
            "test_proba": None,
        }
    return aligned


def align_scope_state_results_to_test(reference_df, scope_df, aligned_train_results, scope_results):
    for name, pack in scope_results.items():
        aligned_train_results[name]["test_proba"] = _align_prediction_array(reference_df, scope_df, pack["test_proba"])
    return aligned_train_results


def get_stage2_tune_specs():
    s = SEED
    return {
        "Logistic Reg.": (
            Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=s))]),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__penalty": ["l1", "l2"], "lr__solver": ["saga"]},
            "#1f77b4",
            "-.",
            10,
            -1,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=4000, penalty="elasticnet", solver="saga", random_state=s)),
                ]
            ),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8",
            "-.",
            10,
            -1,
        ),
        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                n_estimators=240,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=s,
                n_jobs=1,
                verbose=-1,
            ),
            {
                "num_leaves": [15, 31, 63],
                "min_child_samples": [10, 20, 40],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [0.0, 0.1, 0.5, 1.0],
            },
            "#f58518",
            ":",
            10,
            -1,
        ),
        "Balanced RF": (
            BalancedRandomForestClassifier(random_state=s, n_jobs=1),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "min_samples_leaf": [1, 3, 5, 10],
                "sampling_strategy": ["all", "not minority"],
            },
            "#72b7b2",
            "--",
            8,
            -1,
        ),
    }


def get_linear_tune_specs():
    specs = get_stage2_tune_specs()
    return {name: specs[name] for name in ["Logistic Reg.", "Elastic LR"]}


def get_fixed_fusion_spec():
    s = SEED
    return (
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=4000, penalty="elasticnet", solver="saga", random_state=s)),
            ]
        ),
        {"lr__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "#4c78a8",
        "-.",
        8,
        -1,
    )


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, x_tr_df, y_tr, groups_tr):
    if grid:
        rs = RandomizedSearchCV(
            base,
            grid,
            n_iter=n_iter,
            cv=cv,
            scoring="average_precision",
            random_state=SEED,
            n_jobs=n_jobs,
        )
        rs.fit(x_tr_df, y_tr, groups=groups_tr)
        return rs.best_estimator_, rs.best_score_, rs.best_params_

    scores = []
    for f_tr, f_val in cv.split(x_tr_df, y_tr, groups=groups_tr):
        model = clone(base)
        model.fit(x_tr_df.iloc[f_tr], y_tr[f_tr])
        proba = model.predict_proba(x_tr_df.iloc[f_val])[:, 1]
        scores.append(average_precision_score(y_tr[f_val], proba))
    return clone(base), float(np.mean(scores)), None


def write_fixed_feature_manifest(prefix, feat_names, note=""):
    feat_names = list(feat_names)
    summary_df = pd.DataFrame(
        {
            "feature": feat_names,
            "coefficient": np.nan,
            "abs_coefficient": np.nan,
            "selected": True,
        }
    )
    summary_df.to_csv(Config.OUT_DIR / f"{prefix}_Feature_Selection.csv", index=False)
    lines = [
        f"original_features={len(feat_names)}",
        f"selected_features={len(feat_names)}",
        "best_c=NA_fixed",
        "cv_pr_auc=NA_fixed",
        "selected_feature_names=" + ", ".join(feat_names),
    ]
    if note:
        lines.append(f"note={note}")
    (Config.OUT_DIR / f"{prefix}_Feature_Selection_Report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_recurrent_c_index_from_intervals(df_long, proba):
    tmp = df_long.copy().reset_index(drop=True)
    tmp["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    recurrent_df = derive_recurrent_survival_data(tmp)
    if len(recurrent_df) == 0:
        return np.nan
    return concordance_index_simple(
        recurrent_df["Gap_Time"].values,
        recurrent_df["proba"].values,
        recurrent_df["Event"].values,
    )


def fit_from_frames(group_name, x_tr, x_te, y_tr, y_te, groups, tune_specs, note=""):
    feat_names = list(x_tr.columns)
    write_fixed_feature_manifest(prefix=f"Stage2_{group_name}", feat_names=feat_names, note=note)
    gkf = GroupKFold(n_splits=3)
    rows = []
    results = {}

    for model_name, (base_est, param_grid, color, ls, n_iter, n_jobs) in tune_specs.items():
        best_est, cv_score, best_params = fit_candidate_model(base_est, param_grid, n_iter, n_jobs, gkf, x_tr, y_tr, groups)
        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            fitted = clone(best_est)
            fitted.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]

        fitted = clone(best_est)
        fitted.fit(x_tr, y_tr)
        train_fit = fitted.predict_proba(x_tr)[:, 1]
        proba = fitted.predict_proba(x_te)[:, 1]
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        rows.append(
            {
                "Group": group_name,
                "Model": model_name,
                "CV_PR_AUC": cv_score,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "F1": metrics["f1"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": thr,
                "Best_Params": "" if best_params is None else str(best_params),
            }
        )
        results[model_name] = {
            "model": fitted,
            "oof_proba": oof,
            "train_fit_proba": train_fit,
            "proba": proba,
            "threshold": thr,
            "metrics": metrics,
            "cal": cal,
            "feat_names": feat_names,
            "auc": metrics["auc"],
            "prauc": metrics["prauc"],
            "color": color,
            "ls": ls,
        }

    best_name = max(results, key=lambda name: (results[name]["prauc"], results[name]["auc"]))
    return pd.DataFrame(rows), best_name, results[best_name], results


def fit_stage2_mode(group_name, train_df, test_df, mode, base_current_features=None, use_feature_selection=False, allowed_models=None):
    x_tr_full, x_te_full, _ = make_stage2_feature_frames(
        train_df, test_df, mode=mode, base_current_features=base_current_features
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values

    if use_feature_selection:
        fs = select_binary_features_with_l1(
            x_tr_full,
            x_te_full,
            y_tr,
            groups,
            out_dir=Config.OUT_DIR,
            prefix=f"Stage2_{group_name}",
            seed=SEED,
            min_features=8,
        )
        x_tr = fs.X_train
        x_te = fs.X_test
        feat_names = fs.selected_features
        print(
            f"    {group_name:<40s} feature selection: {fs.original_feature_count} -> {len(feat_names)} "
            f"(best C={fs.best_c}, CV PR-AUC={fs.cv_score:.3f})"
        )
    else:
        x_tr = x_tr_full.copy()
        x_te = x_te_full.copy()
        feat_names = list(x_tr.columns)
        write_fixed_feature_manifest(prefix=f"Stage2_{group_name}", feat_names=feat_names, note=f"mode={mode}")

    specs = get_stage2_tune_specs()
    if allowed_models is not None:
        specs = {name: specs[name] for name in allowed_models}
    group_df, best_name, best_payload, results = fit_from_frames(
        group_name, x_tr, x_te, y_tr, y_te, groups, specs, note=f"mode={mode}"
    )
    for payload in results.values():
        payload["feat_names"] = feat_names
        payload["c_index"] = compute_recurrent_c_index_from_intervals(test_df, payload["proba"])
    group_df["C_Index"] = group_df["Model"].map({k: v["c_index"] for k, v in results.items()})
    best_payload["c_index"] = results[best_name]["c_index"]
    return group_df, best_name, best_payload, results


def select_transparent_payload(results):
    linear_names = [name for name in ["Logistic Reg.", "Elastic LR"] if name in results]
    if not linear_names:
        best_name = max(results, key=lambda name: (results[name]["prauc"], results[name]["auc"]))
        return best_name, results[best_name]
    best_name = max(linear_names, key=lambda name: (results[name]["prauc"], results[name]["auc"]))
    return best_name, results[best_name]


def build_direct_windowed_core_frames(train_df, test_df, selected_features):
    x_tr_full, x_te_full, _ = make_stage2_feature_frames(train_df, test_df, mode="direct", base_current_features=None)
    early_dummies = [f"Interval_Name_{name}" for name in sorted(HIGH_RISK_WINDOWS)]
    keep_cols = list(
        dict.fromkeys(list(selected_features) + WINDOWED_CORE_CANDIDATE_FEATURES + WINDOWED_CORE_INTERACTION_FEATURES + early_dummies)
    )
    keep_cols = [col for col in keep_cols if col in x_tr_full.columns]
    tr = x_tr_full[keep_cols].copy()
    te = x_te_full[keep_cols].copy()

    for dummy in early_dummies:
        if dummy not in tr.columns:
            interval_name = dummy.replace("Interval_Name_", "")
            tr[dummy] = (train_df["Interval_Name"].astype(str).values == interval_name).astype(float)
            te[dummy] = (test_df["Interval_Name"].astype(str).values == interval_name).astype(float)

    for feature in WINDOWED_CORE_INTERACTION_FEATURES:
        if feature not in x_tr_full.columns:
            continue
        if feature not in tr.columns:
            tr[feature] = x_tr_full[feature].values
            te[feature] = x_te_full[feature].values
        for dummy in early_dummies:
            inter_name = f"{feature}_x_{dummy}"
            tr[inter_name] = tr[feature].values * tr[dummy].values
            te[inter_name] = te[feature].values * te[dummy].values

    keep_final = [col for col in tr.columns if tr[col].nunique(dropna=False) > 1]
    return tr[keep_final], te[keep_final], keep_final


def fit_direct_windowed_core(train_df, test_df, direct_ref_payload):
    x_tr, x_te, feat_names = build_direct_windowed_core_frames(train_df, test_df, direct_ref_payload["feat_names"])
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    write_fixed_feature_manifest(
        prefix="Stage2_direct_windowed_core",
        feat_names=feat_names,
        note=(
            "selected direct feature set + narrowed evidence-driven early-window interactions; "
            f"candidate_core={','.join(WINDOWED_CORE_CANDIDATE_FEATURES)}; "
            f"interaction_core={','.join(WINDOWED_CORE_INTERACTION_FEATURES)}"
        ),
    )
    group_df, best_name, best_payload, results = fit_from_frames(
        "direct_windowed_core_anchor",
        x_tr,
        x_te,
        y_tr,
        y_te,
        groups,
        get_linear_tune_specs(),
        note=(
            "direct selected features + narrowed early-window core interactions "
            f"for {','.join(WINDOWED_CORE_INTERACTION_FEATURES)}"
        ),
    )
    for payload in results.values():
        payload["feat_names"] = feat_names
        payload["c_index"] = compute_recurrent_c_index_from_intervals(test_df, payload["proba"])
    group_df["C_Index"] = group_df["Model"].map({k: v["c_index"] for k, v in results.items()})
    best_payload["c_index"] = results[best_name]["c_index"]
    return group_df, best_name, best_payload, results


def fit_monotonic_tree_direct(train_df, test_df, direct_ref_payload):
    x_tr_full, x_te_full, _ = make_stage2_feature_frames(
        train_df, test_df, mode="direct", base_current_features=direct_ref_payload["feat_names"]
    )
    feat_names = list(x_tr_full.columns)
    constraints = [MONOTONIC_DIRECTIONS.get(name, 0) for name in feat_names]
    base = LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
        n_jobs=1,
    )
    specs = {
        "Monotonic LightGBM": (
            base,
            {},
            "#f58518",
            ":",
            0,
            1,
        )
    }
    write_fixed_feature_manifest(
        prefix="Stage2_monotonic_tree_direct",
        feat_names=feat_names,
        note="selected direct features + partial monotone constraints",
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    group_df, best_name, best_payload, results = fit_from_frames(
        "monotonic_tree_direct_anchor",
        x_tr_full,
        x_te_full,
        y_tr,
        y_te,
        groups,
        specs,
        note="partial monotone constraints on selected direct features",
    )
    for payload in results.values():
        payload["feat_names"] = feat_names
        payload["c_index"] = compute_recurrent_c_index_from_intervals(test_df, payload["proba"])
    group_df["C_Index"] = group_df["Model"].map({k: v["c_index"] for k, v in results.items()})
    best_payload["c_index"] = results[best_name]["c_index"]
    return group_df, best_name, best_payload, results


def fit_intercept_calibrator(train_df, test_df, raw_oof_logit, raw_test_logit, y_train, groups, strategy_key="phase"):
    if strategy_key not in CALIBRATION_STRATEGIES:
        raise ValueError(f"Unknown calibration strategy: {strategy_key}")
    strategy = CALIBRATION_STRATEGIES[strategy_key]
    tr, te = strategy.build_frames(train_df, test_df, raw_oof_logit, raw_test_logit)
    keep_cols = [col for col in tr.columns if tr[col].nunique(dropna=False) > 1]
    tr = tr[keep_cols]
    te = te[keep_cols]

    gkf = GroupKFold(n_splits=min(3, len(pd.Index(groups).drop_duplicates())))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in gkf.split(tr, y_train, groups=groups):
        fitted = clone(model)
        fitted.fit(tr.iloc[tr_idx], y_train[tr_idx])
        oof[val_idx] = fitted.predict_proba(tr.iloc[val_idx])[:, 1]

    fitted = clone(model)
    fitted.fit(tr, y_train)
    train_fit = fitted.predict_proba(tr)[:, 1]
    test_proba = fitted.predict_proba(te)[:, 1]
    return fitted, train_fit, oof, test_proba, strategy.description


def build_sidecar_calibrated_pack(group_name, train_df, test_df, sidecar_payload, strategy_key="phase"):
    y_tr = train_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    raw_oof = _safe_logit(sidecar_payload["oof_proba"])
    raw_test = _safe_logit(sidecar_payload["proba"])
    model, train_fit, oof, test_proba, strategy_desc = fit_intercept_calibrator(
        train_df, test_df, raw_oof, raw_test, y_tr, groups, strategy_key=strategy_key
    )
    return {
        "model": model,
        "train_fit_proba": train_fit,
        "oof_proba": oof,
        "proba": test_proba,
        "raw_logit_oof": raw_oof,
        "raw_logit_test": raw_test,
        "variant_name": group_name,
        "calibration_strategy": strategy_key,
        "calibration_note": strategy_desc,
    }


def _score_pack_from_state_results(state_results):
    base_names = [name for name in state_results if name != "EnsembleMean"]
    if not base_names:
        base_names = list(state_results.keys())
    oof_stack = np.stack([_binary_proba_2col(state_results[name]["oof_proba"])[:, 1] for name in base_names], axis=0)
    test_stack = np.stack([_binary_proba_2col(state_results[name]["test_proba"])[:, 1] for name in base_names], axis=0)
    train_fit_stack = np.stack([_binary_proba_2col(state_results[name]["train_fit_proba"])[:, 1] for name in base_names], axis=0)
    return {
        "model": {"type": "state_score_mean", "members": base_names},
        "train_fit_proba": np.mean(train_fit_stack, axis=0),
        "oof_proba": np.mean(oof_stack, axis=0),
        "proba": np.mean(test_stack, axis=0),
    }


def build_fusion_feature_frames(train_df, test_df, direct_payload, physio_payload, aux_rule_payload=None):
    tr = pd.DataFrame(
        {
            "Direct_Logit": _safe_logit(direct_payload["oof_proba"]),
            "Physio_Main_Logit": _safe_logit(physio_payload["oof_proba"]),
        }
    )
    te = pd.DataFrame(
        {
            "Direct_Logit": _safe_logit(direct_payload["proba"]),
            "Physio_Main_Logit": _safe_logit(physio_payload.get("proba", physio_payload.get("test_proba"))),
        }
    )
    tr["Direct_x_Physio_Main"] = tr["Direct_Logit"].values * tr["Physio_Main_Logit"].values
    te["Direct_x_Physio_Main"] = te["Direct_Logit"].values * te["Physio_Main_Logit"].values

    cats = sorted(train_df["Interval_Name"].astype(str).unique())
    for cat in cats:
        dummy = f"Interval_Name_{cat}"
        tr[dummy] = (train_df["Interval_Name"].astype(str).values == cat).astype(float)
        te[dummy] = (test_df["Interval_Name"].astype(str).values == cat).astype(float)
        tr[f"{dummy}_x_Direct_Logit"] = tr[dummy].values * tr["Direct_Logit"].values
        te[f"{dummy}_x_Direct_Logit"] = te[dummy].values * te["Direct_Logit"].values
        tr[f"{dummy}_x_Physio_Main_Logit"] = tr[dummy].values * tr["Physio_Main_Logit"].values
        te[f"{dummy}_x_Physio_Main_Logit"] = te[dummy].values * te["Physio_Main_Logit"].values

    if aux_rule_payload is not None:
        tr["Physio_Rule_Logit"] = _safe_logit(aux_rule_payload["oof_proba"])
        te["Physio_Rule_Logit"] = _safe_logit(aux_rule_payload.get("proba", aux_rule_payload.get("test_proba")))

    medians = tr.median(numeric_only=True)
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(medians)
    te = te.replace([np.inf, -np.inf], np.nan).fillna(medians)
    keep_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols], keep_cols


def fit_fixed_fusion(group_name, train_df, test_df, direct_payload, sidecar_payload, aux_rule_payload=None, note=""):
    x_tr, x_te, feat_names = build_fusion_feature_frames(
        train_df,
        test_df,
        direct_payload,
        sidecar_payload,
        aux_rule_payload=aux_rule_payload,
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    fusion_note = note or "fixed both_windowed + Elastic LR fusion"
    if sidecar_payload.get("calibration_note"):
        fusion_note = f"{fusion_note}; sidecar_calibration={sidecar_payload['calibration_note']}"
    if aux_rule_payload is not None:
        fusion_note = f"{fusion_note}; aux_rule_sidecar=yes"
    write_fixed_feature_manifest(
        prefix=f"Stage2_{group_name}",
        feat_names=feat_names,
        note=fusion_note,
    )
    specs = {"Elastic LR": get_fixed_fusion_spec()}
    group_df, best_name, best_payload, results = fit_from_frames(
        group_name,
        x_tr,
        x_te,
        y_tr,
        y_te,
        groups,
        specs,
        note=fusion_note,
    )
    for payload in results.values():
        payload["feat_names"] = feat_names
        payload["c_index"] = compute_recurrent_c_index_from_intervals(test_df, payload["proba"])
        payload["direct_name"] = group_name
    group_df["C_Index"] = group_df["Model"].map({k: v["c_index"] for k, v in results.items()})
    best_payload["c_index"] = results[best_name]["c_index"]
    return group_df, best_name, best_payload, results


def record_group_result(group_name, group_df, best_name, payload, group_results, train_df, test_df, all_rows, perf_frames, best_groups, all_group_results):
    all_rows.append(group_df)
    best_groups[group_name] = (best_name, payload)
    all_group_results[group_name] = group_results
    print(
        f"  Best {group_name:<52s}: {best_name:<18s} "
        f"AUC={payload['metrics']['auc']:.3f} PR-AUC={payload['metrics']['prauc']:.3f} "
        f"Brier={payload['metrics']['brier']:.3f}"
    )
    perf_frames.append(
        build_binary_performance_long(
            task_name=f"{RESULT_PREFIX} {group_name}",
            results=group_results,
            domain_payloads={
                "Train_Fit": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "train_fit_proba"},
                "Validation_OOF": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "oof_proba"},
                "Test_Temporal": {"y_true": test_df["Y_Relapse"].values.astype(int), "proba_key": "proba"},
            },
            metric_keys=["prauc", "auc", "recall", "specificity", "f1"],
            threshold_key="threshold",
        )
    )


def compute_window_metric_table(group_name, test_df, payload):
    rows = []
    threshold = float(payload["threshold"])
    for interval_name, subset in test_df.groupby("Interval_Name", sort=False):
        idx = subset.index.to_numpy()
        y_true = subset["Y_Relapse"].values.astype(int)
        proba = np.asarray(payload["proba"], dtype=float)[idx]
        metrics = compute_binary_metrics(y_true, proba, threshold)
        cal = compute_calibration_stats(y_true, proba)
        rows.append(
            {
                "Group": group_name,
                "Interval_Name": str(interval_name),
                "n": int(len(y_true)),
                "positives": int(np.sum(y_true)),
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "Brier": metrics["brier"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_group_delta(y_true, proba_a, proba_b, groups, metric="prauc", n_boot=2000, seed=SEED):
    unique_groups = pd.Index(groups).drop_duplicates().to_list()
    group_to_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([group_to_idx[group] for group in sampled])
        y_boot = y_true[boot_idx]
        if np.unique(y_boot).size < 2:
            continue
        if metric == "prauc":
            diff = average_precision_score(y_boot, proba_a[boot_idx]) - average_precision_score(y_boot, proba_b[boot_idx])
        else:
            diff = roc_auc_score(y_boot, proba_a[boot_idx]) - roc_auc_score(y_boot, proba_b[boot_idx])
        diffs.append(float(diff))

    point = (
        average_precision_score(y_true, proba_a) - average_precision_score(y_true, proba_b)
        if metric == "prauc"
        else roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)
    )
    if not diffs:
        return {"Metric": metric, "Delta": point, "CI_Low": np.nan, "CI_High": np.nan, "OneSided_P": np.nan}
    diffs = np.asarray(diffs, dtype=float)
    return {
        "Metric": metric,
        "Delta": point,
        "CI_Low": float(np.percentile(diffs, 2.5)),
        "CI_High": float(np.percentile(diffs, 97.5)),
        "OneSided_P": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1)),
    }


def build_bootstrap_delta_rows(group_name, test_df, candidate_payload, baseline_name, baseline_payload):
    y_true = test_df["Y_Relapse"].values.astype(int)
    groups = test_df["Patient_ID"].values
    proba_a = np.asarray(candidate_payload["proba"], dtype=float)
    proba_b = np.asarray(baseline_payload["proba"], dtype=float)
    rows = []
    for metric in ["prauc", "auc"]:
        out = bootstrap_group_delta(y_true, proba_a, proba_b, groups, metric=metric)
        out["Group"] = group_name
        out["Baseline"] = baseline_name
        rows.append(out)
    return pd.DataFrame(rows)


def calibration_distance(intercept, slope):
    return abs(float(intercept)) + abs(float(slope) - 1.0)


def build_anchor_champion_stop_rule(summary_df, per_window_df, delta_boot_df, reference_group):
    ref_row = summary_df.loc[summary_df["Group"] == reference_group].iloc[0]
    rows = []
    candidate_groups = [
        group
        for group in summary_df["Group"].tolist()
        if group not in {"direct_anchor_ref", reference_group}
    ]
    for group in candidate_groups:
        row = summary_df.loc[summary_df["Group"] == group].iloc[0]
        delta_pr = float(row["PR_AUC"] - ref_row["PR_AUC"])
        auc_not_down = float(row["AUC"]) >= float(ref_row["AUC"]) - 0.001
        brier_not_worse = float(row["Brier"]) <= float(ref_row["Brier"]) + 0.002
        cal_not_worse = calibration_distance(row["Calibration_Intercept"], row["Calibration_Slope"]) <= calibration_distance(
            ref_row["Calibration_Intercept"], ref_row["Calibration_Slope"]
        ) + 0.15

        group_window = per_window_df[per_window_df["Group"] == group].set_index("Interval_Name")
        ref_window = per_window_df[per_window_df["Group"] == reference_group].set_index("Interval_Name")
        shared_intervals = [name for name in group_window.index if name in ref_window.index]
        improved_windows = 0
        for interval_name in shared_intervals:
            if float(group_window.loc[interval_name, "PR_AUC"] - ref_window.loc[interval_name, "PR_AUC"]) >= 0.003:
                improved_windows += 1

        pr_boot = delta_boot_df[
            (delta_boot_df["Group"] == group)
            & (delta_boot_df["Baseline"] == reference_group)
            & (delta_boot_df["Metric"] == "prauc")
        ]
        ci_right_shift = False
        ci_low = np.nan
        if len(pr_boot):
            ci_low = float(pr_boot.iloc[0]["CI_Low"])
            ci_right_shift = ci_low >= PREV_BEST_NEW_CI_LOW

        pass_count = (
            int(delta_pr >= 0.005)
            + int(auc_not_down)
            + int(brier_not_worse)
            + int(cal_not_worse)
            + int(improved_windows >= 2)
            + int(ci_right_shift)
        )
        rows.append(
            {
                "Group": group,
                "PR_AUC_Gain_vs_Champion": delta_pr,
                "AUC_Not_Down": auc_not_down,
                "Brier_Not_Worse": brier_not_worse,
                "Calibration_Not_Worse": cal_not_worse,
                "Improved_Window_Count": improved_windows,
                "Bootstrap_CI_Low_vs_Champion": ci_low,
                "Bootstrap_CI_RightShift": ci_right_shift,
                "Pass_Count": pass_count,
                "Pass_StopRule": pass_count >= 3,
            }
        )
    return pd.DataFrame(rows)


def _annotate_bars(ax, bars):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=TEXT_MID,
        )


def save_anchor_champion_comparison(summary_df):
    label_map = {
        "direct_anchor_ref": "Direct",
        "current_champion_ref": "Champion ref",
        "champion_sidecar_calibrated": "Champion phase-cal",
        "direct_windowed_core": "Direct+window",
        "direct_windowed_core_plus_champion_calibrated": "Direct+window+cal",
        "stage1_nonhyper_scope_plus_champion_calibrated": "Nonhyper+cal",
        "monotonic_tree_direct_plus_champion_calibrated": "Mono tree+cal",
        "direct_windowed_core_plus_champion_calibrated_plus_rule_sidecar": "Direct+cal+rule",
    }
    plot_df = summary_df.copy()
    plot_df["Label"] = plot_df["Group"].map(label_map).fillna(plot_df["Group"])
    fig_width = max(12.5, 1.45 * len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    auc_bars = axes[0].bar(plot_df["Label"], plot_df["AUC"], color=PRIMARY_BLUE, alpha=0.88, width=0.62)
    pr_bars = axes[1].bar(plot_df["Label"], plot_df["PR_AUC"], color=PRIMARY_TEAL, alpha=0.88, width=0.62)
    axes[0].set_title("Two-stage transition comparison: AUC", fontsize=9)
    axes[1].set_title("Two-stage transition comparison: PR-AUC", fontsize=9)
    axes[0].set_ylim(0, max(0.9, float(plot_df["AUC"].max()) + 0.08))
    axes[1].set_ylim(0, max(0.40, float(plot_df["PR_AUC"].max()) + 0.10))
    for ax, bars in [(axes[0], auc_bars), (axes[1], pr_bars)]:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=6.5, rotation=24)
        ax.tick_params(axis="y", labelsize=7)
        _annotate_bars(ax, bars)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(Config.OUT_DIR / RESULT_GROUP_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_window_pr_auc_comparison(per_window_df, reference_group, best_group):
    order = [f"{TIME_STAMPS[idx]}->{TIME_STAMPS[idx + 1]}" for idx in range(len(TIME_STAMPS) - 1)]
    ref = (
        per_window_df[per_window_df["Group"] == reference_group][["Interval_Name", "PR_AUC"]]
        .rename(columns={"PR_AUC": "Reference_PR_AUC"})
        .set_index("Interval_Name")
    )
    best = (
        per_window_df[per_window_df["Group"] == best_group][["Interval_Name", "PR_AUC"]]
        .rename(columns={"PR_AUC": "Best_PR_AUC"})
        .set_index("Interval_Name")
    )
    plot_df = ref.join(best, how="outer").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(plot_df["Interval_Name"], plot_df["Reference_PR_AUC"], marker="o", lw=2.0, color=PRIMARY_BLUE, label="Current champion")
    ax.plot(plot_df["Interval_Name"], plot_df["Best_PR_AUC"], marker="o", lw=2.0, color=PRIMARY_TEAL, label="Optimized two-stage")
    ax.set_title("Per-window PR-AUC: current champion vs optimized two-stage", fontsize=9)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=22, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / RESULT_WINDOW_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pr_auc_delta_forest(delta_boot_df, baseline_name="current_champion_ref"):
    df = delta_boot_df[(delta_boot_df["Metric"] == "prauc") & (delta_boot_df["Baseline"] == baseline_name)].copy()
    if len(df) == 0:
        return
    df = df.sort_values("Delta", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig_h = max(3.6, 0.55 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ax.axvline(0, color=TEXT_MID, lw=1.0, ls="--", alpha=0.6)
    ax.hlines(y, df["CI_Low"], df["CI_High"], color=PRIMARY_BLUE, lw=2.0)
    ax.scatter(df["Delta"], y, color=PRIMARY_TEAL, s=36, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Group"], fontsize=7)
    ax.set_xlabel("Delta PR-AUC vs current champion")
    ax.set_title("Grouped bootstrap delta PR-AUC", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / RESULT_BOOTSTRAP_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_best_mode_reports_prefixed(prefix, mode, payload, train_df, test_df):
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
            {"Level": "Interval", "Metric": "Calibration_Intercept", "Value": interval_cal["intercept"], "CI_95": format_ci(interval_cis, "cal_intercept")},
            {"Level": "Interval", "Metric": "Calibration_Slope", "Value": interval_cal["slope"], "CI_95": format_ci(interval_cis, "cal_slope")},
            {"Level": "Interval", "Metric": "Threshold", "Value": payload["threshold"], "CI_95": ""},
        ]
    )
    interval_summary.to_csv(Config.OUT_DIR / f"{prefix}_Interval_Summary.csv", index=False)

    save_calibration_figure(
        y_te,
        payload["proba"],
        f"Calibration Curve ({mode}, interval-level)",
        Config.OUT_DIR / f"{prefix}_Calibration_Interval.png",
    )
    save_dca_figure(
        y_te,
        payload["proba"],
        f"Decision Curve Analysis ({mode}, interval-level)",
        Config.OUT_DIR / f"{prefix}_DCA_Interval.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        payload["proba"],
        payload["threshold"],
        f"Threshold Sensitivity ({mode}, interval-level)",
        Config.OUT_DIR / f"{prefix}_Threshold_Sensitivity_Interval.png",
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
            {"Level": "Patient", "Metric": "Calibration_Intercept", "Value": patient_cal["intercept"], "CI_95": format_ci(patient_cis, "cal_intercept")},
            {"Level": "Patient", "Metric": "Calibration_Slope", "Value": patient_cal["slope"], "CI_95": format_ci(patient_cis, "cal_slope")},
            {"Level": "Patient", "Metric": "Threshold", "Value": patient_thr, "CI_95": ""},
        ]
    )
    patient_summary.to_csv(Config.OUT_DIR / f"{prefix}_Patient_Summary.csv", index=False)
    save_calibration_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Calibration Curve ({mode}, patient-level)",
        Config.OUT_DIR / f"{prefix}_Calibration_Patient.png",
    )
    save_dca_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Decision Curve Analysis ({mode}, patient-level)",
        Config.OUT_DIR / f"{prefix}_DCA_Patient.png",
    )
    save_threshold_sensitivity_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        patient_thr,
        f"Threshold Sensitivity ({mode}, patient-level)",
        Config.OUT_DIR / f"{prefix}_Threshold_Sensitivity_Patient.png",
    )
    save_patient_risk_strata(patient_tr, patient_te, Config.OUT_DIR / f"{prefix}_Patient_Risk_Q1Q4.png")
    patient_sens_df, _ = evaluate_patient_aggregation_sensitivity(train_df, test_df, payload["oof_proba"], payload["proba"])
    patient_sens_df.to_csv(Config.OUT_DIR / f"{prefix}_Patient_Aggregation_Sensitivity.csv", index=False)
    save_patient_aggregation_sensitivity_figure(
        patient_sens_df,
        Config.OUT_DIR / f"{prefix}_Patient_Aggregation_Sensitivity.png",
    )
    window_sens_df = evaluate_window_sensitivity(test_df, payload["proba"], payload["threshold"])
    window_sens_df.to_csv(Config.OUT_DIR / f"{prefix}_Window_Sensitivity.csv", index=False)


def write_anchor_champion_round_summary(summary_df, stop_rule_df, delta_boot_df, per_window_df, stage1_payload, scope_count_df, prior_best_row):
    direct_row = summary_df.loc[summary_df["Group"] == "direct_anchor_ref"].iloc[0]
    champion_row = summary_df.loc[summary_df["Group"] == "current_champion_ref"].iloc[0]
    experiment_df = summary_df[~summary_df["Group"].isin({"direct_anchor_ref", "current_champion_ref"})].sort_values(
        ["PR_AUC", "AUC"], ascending=[False, False]
    )
    best_row = experiment_df.iloc[0]
    best_group = str(best_row["Group"])

    window_delta_lines = []
    exp_window = per_window_df[per_window_df["Group"] == best_group].set_index("Interval_Name")
    ref_window = per_window_df[per_window_df["Group"] == "current_champion_ref"].set_index("Interval_Name")
    for interval_name in exp_window.index:
        if interval_name not in ref_window.index:
            continue
        delta_pr = float(exp_window.loc[interval_name, "PR_AUC"] - ref_window.loc[interval_name, "PR_AUC"])
        if delta_pr >= -0.05:
            window_delta_lines.append((interval_name, delta_pr))
    window_delta_lines = sorted(window_delta_lines, key=lambda item: item[1], reverse=True)[:5]

    delta_vs_direct = delta_boot_df[
        (delta_boot_df["Group"] == best_group) & (delta_boot_df["Baseline"] == "direct_anchor_ref") & (delta_boot_df["Metric"] == "prauc")
    ]
    delta_vs_champion = delta_boot_df[
        (delta_boot_df["Group"] == best_group) & (delta_boot_df["Baseline"] == "current_champion_ref") & (delta_boot_df["Metric"] == "prauc")
    ]

    lines = [
        "# Two-Stage Transition Round",
        "",
        "## Frozen Baselines",
        "",
        f"- Direct anchor reference: `PR-AUC {direct_row['PR_AUC']:.3f}`, `AUC {direct_row['AUC']:.3f}`.",
        f"- Current champion fused reference: `PR-AUC {champion_row['PR_AUC']:.3f}`, `AUC {champion_row['AUC']:.3f}`.",
        f"- Champion calibration protocol in this round: `{CALIBRATION_STRATEGIES['phase'].description}`.",
        f"- Direct window core interaction kernel in this round: `{', '.join(WINDOWED_CORE_INTERACTION_FEATURES)}`.",
    ]
    if prior_best_row is not None:
        lines.append(
            f"- Previous best-new motif benchmark: `{prior_best_row['Group']}` with `PR-AUC {prior_best_row['PR_AUC']:.3f}`, `AUC {prior_best_row['AUC']:.3f}`."
        )

    lines.extend(
        [
            "",
            "## Best Overall",
            "",
            f"- Best experiment: `{best_group}` with `{best_row['Best_Model']}`.",
            f"- Interval-level metrics: `PR-AUC {best_row['PR_AUC']:.3f}`, `AUC {best_row['AUC']:.3f}`, `Brier {best_row['Brier']:.3f}`.",
            f"- Versus current champion: `PR-AUC {best_row['PR_AUC'] - champion_row['PR_AUC']:+.3f}`, `AUC {best_row['AUC'] - champion_row['AUC']:+.3f}`.",
            f"- Versus direct anchor: `PR-AUC {best_row['PR_AUC'] - direct_row['PR_AUC']:+.3f}`, `AUC {best_row['AUC'] - direct_row['AUC']:+.3f}`.",
        ]
    )

    if len(delta_vs_direct):
        row = delta_vs_direct.iloc[0]
        lines.append(
            f"- Grouped bootstrap vs direct: `ΔPR-AUC {row['Delta']:+.3f}` with 95% CI `{row['CI_Low']:+.3f}` to `{row['CI_High']:+.3f}`."
        )
    if len(delta_vs_champion):
        row = delta_vs_champion.iloc[0]
        lines.append(
            f"- Grouped bootstrap vs current champion: `ΔPR-AUC {row['Delta']:+.3f}` with 95% CI `{row['CI_Low']:+.3f}` to `{row['CI_High']:+.3f}`."
        )

    lines.extend(["", "## Per-Window Where It Improved", ""])
    if window_delta_lines:
        for interval_name, delta_pr in window_delta_lines:
            lines.append(f"- `{interval_name}`: PR-AUC change vs current champion `{delta_pr:+.3f}`.")
    else:
        lines.append("- No stable per-window gains were found.")

    lines.extend(["", "## Gain Attribution", ""])
    if "nonhyper_scope" in best_group:
        lines.append("- Main gain source: **scope / data efficiency**.")
    elif "windowed_core" in best_group and "champion_calibrated" in best_group:
        lines.append("- Main gain source: **anchor + sidecar** together.")
    elif "windowed_core" in best_group:
        lines.append("- Main gain source: **direct anchor**.")
    elif "champion_sidecar_calibrated" in best_group or "champion_calibrated" in best_group:
        lines.append("- Main gain source: **champion sidecar calibration**.")
    else:
        lines.append("- Main gain source: mixed / inconclusive.")

    pass_count = int(stop_rule_df["Pass_StopRule"].sum()) if len(stop_rule_df) else 0
    lines.extend(["", "## Stop Rule", ""])
    lines.append(f"- Experiments passing the 6-of-3 stop rule: `{pass_count}`.")
    if len(stop_rule_df):
        winner_row = stop_rule_df.sort_values(["Pass_Count", "PR_AUC_Gain_vs_Champion"], ascending=[False, False]).iloc[0]
        lines.append(
            f"- Strongest stop-rule row: `{winner_row['Group']}` with `Pass_Count={int(winner_row['Pass_Count'])}` and `ΔPR-AUC={winner_row['PR_AUC_Gain_vs_Champion']:+.3f}`."
        )

    lines.extend(["", "## Stage1 Scope Counts", ""])
    count_summary = scope_count_df.groupby("Split", as_index=False)[["current_normal", "current_nonhyper", "next_hyper_from_normal", "next_hyper_from_nonhyper"]].sum()
    for row in count_summary.itertuples(index=False):
        lines.append(
            f"- `{row.Split}`: current_normal `{int(row.current_normal)}`, current_nonhyper `{int(row.current_nonhyper)}`, "
            f"next_hyper_from_normal `{int(row.next_hyper_from_normal)}`, next_hyper_from_nonhyper `{int(row.next_hyper_from_nonhyper)}`."
        )

    (Config.OUT_DIR / RESULT_REPORT_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local two-stage transition caches")
    args = parser.parse_args()
    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 92)
    print(f"  {RESULT_TITLE}")
    print("=" * 92)

    df_train, df_test = build_two_stage_long_data()

    print("\n--- Phase 2: Baseline stage-1 payload on current-Normal rows ---")
    x1_tr, x1_te, _ = make_stage1_feature_frames(df_train, df_test)
    y1_te = df_test[TARGET_COLS].values.astype(float)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values
    stage1_payload = run_stage1_oof_forecast(
        x1_tr, df_train, x1_te, df_test, groups, train_intervals, test_intervals, Config.OUT_DIR
    )
    best_delta_name = stage1_payload["delta_best_name"]
    best_delta = stage1_payload["delta_results"][best_delta_name]
    stage1_metrics = stage1_payload["delta_metrics"]
    stage1_test_metrics = evaluate_physio_predictions(y1_te, best_delta["test_next"], "Test", best_delta_name)
    stage1_all_metrics = pd.concat([stage1_metrics, stage1_test_metrics], ignore_index=True)
    stage1_all_metrics.to_csv(Config.OUT_DIR / "Physio_Forecast_Metrics.csv", index=False)
    save_stage1_metric_bar(stage1_metrics[stage1_metrics["Split"] == "Train_OOF"], Config.OUT_DIR)
    save_physio_scatter(y1_te, best_delta["test_next"], Config.OUT_DIR, best_delta_name, "Test")
    state_metrics = stage1_payload["state_metrics"].copy()
    state_metrics.to_csv(Config.OUT_DIR / "NextState_Forecast_Metrics.csv", index=False)
    print(f"  Delta-head anchor: {best_delta_name}")
    print(f"  State-head anchor family: {stage1_payload['state_anchor_name']}")

    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")

    print("\n--- Phase 3: Stage1 nonhyper-scope state payload ---")
    scope_train, scope_test, scope_count_df = build_stage1_transition_scope_data(current_scope="nonhyper")
    x_scope_tr, x_scope_te, _ = make_stage1_scope_feature_frames(scope_train, scope_test)
    scope_groups = scope_train["Patient_ID"].values
    nonhyper_best_name, nonhyper_state_train_results, nonhyper_state_metrics = run_binary_suite_oof(
        x_scope_tr, scope_train, x_scope_te, scope_test, scope_groups
    )
    nonhyper_state_metrics.to_csv(Config.OUT_DIR / "Stage1_NonHyperScope_State_Metrics.csv", index=False)
    aligned_train_state_results = align_scope_state_results_to_reference(df_train, scope_train, nonhyper_state_train_results)
    aligned_state_results = align_scope_state_results_to_test(df_test, scope_test, aligned_train_state_results, nonhyper_state_train_results)
    stage1_payload_nonhyper = {
        **stage1_payload,
        "state_best_name": nonhyper_best_name,
        "state_results": aligned_state_results,
        "state_metrics": nonhyper_state_metrics,
    }
    df_train_pred_nonhyper = add_stage1_prediction_family(df_train, stage1_payload_nonhyper, "oof")
    df_test_pred_nonhyper = add_stage1_prediction_family(df_test, stage1_payload_nonhyper, "test")

    print("\n--- Phase 4: Freeze direct anchor and champion sidecar reference ---")
    all_rows = []
    perf_frames = []
    best_groups = {}
    all_group_results = {}

    direct_df, direct_name, direct_payload, direct_results = fit_stage2_mode(
        "direct_anchor_ref",
        df_train_pred,
        df_test_pred,
        mode="direct",
        use_feature_selection=True,
        allowed_models=None,
    )
    direct_name, direct_payload = select_transparent_payload(direct_results)
    direct_df = direct_df[direct_df["Model"] == direct_name].reset_index(drop=True)
    direct_payload["c_index"] = compute_recurrent_c_index_from_intervals(df_test_pred, direct_payload["proba"])
    record_group_result(
        "direct_anchor_ref",
        direct_df,
        direct_name,
        direct_payload,
        {direct_name: direct_payload},
        df_train_pred,
        df_test_pred,
        all_rows,
        perf_frames,
        best_groups,
        all_group_results,
    )

    sidecar_df, sidecar_name, sidecar_payload, sidecar_results = fit_stage2_mode(
        "champion_sidecar_ref",
        df_train_pred,
        df_test_pred,
        mode="predicted_only_state_rule",
        use_feature_selection=False,
        allowed_models=["Logistic Reg.", "Elastic LR"],
    )
    sidecar_name, sidecar_payload = select_transparent_payload(sidecar_results)
    champion_sidecar_ref = {
        **sidecar_payload,
        "variant_name": f"predicted_only_state_rule:{sidecar_name}",
    }

    champion_calibrated = build_sidecar_calibrated_pack(
        "champion_sidecar_calibrated",
        df_train_pred,
        df_test_pred,
        champion_sidecar_ref,
        strategy_key="phase",
    )

    ref_df, ref_name, ref_payload, ref_results = fit_fixed_fusion(
        "current_champion_ref",
        df_train_pred,
        df_test_pred,
        direct_payload,
        champion_sidecar_ref,
        note=f"direct={direct_name}; sidecar={champion_sidecar_ref['variant_name']}; fusion=both_windowed+Elastic LR",
    )
    record_group_result(
        "current_champion_ref",
        ref_df,
        ref_name,
        ref_payload,
        ref_results,
        df_train_pred,
        df_test_pred,
        all_rows,
        perf_frames,
        best_groups,
        all_group_results,
    )

    print("\n--- Phase 5: Phase A experiments ---")
    phase_a_specs = [
        ExperimentSpec("champion_sidecar_calibrated", "direct_ref", "champion_calibrated", phase="A"),
        ExperimentSpec("direct_windowed_core", "direct_windowed_core", "champion_ref", phase="A"),
        ExperimentSpec("direct_windowed_core_plus_champion_calibrated", "direct_windowed_core", "champion_calibrated", phase="A"),
        ExperimentSpec(
            "stage1_nonhyper_scope_plus_champion_calibrated",
            "direct_ref",
            "champion_calibrated_nonhyper",
            stage1_scope="nonhyper",
            phase="A",
        ),
    ]

    direct_windowed_cache = None
    champion_nonhyper_sidecar_ref = None
    champion_nonhyper_calibrated = None

    for spec in phase_a_specs:
        if spec.direct_variant == "direct_ref":
            direct_pack = direct_payload
        elif spec.direct_variant == "direct_windowed_core":
            if direct_windowed_cache is None:
                direct_windowed_cache = fit_direct_windowed_core(df_train_pred, df_test_pred, direct_payload)
            _, _, direct_pack, _ = direct_windowed_cache
        else:
            raise ValueError(spec.direct_variant)

        if spec.sidecar_variant == "champion_ref":
            sidecar_pack = champion_sidecar_ref
            train_df_cur = df_train_pred
            test_df_cur = df_test_pred
        elif spec.sidecar_variant == "champion_calibrated":
            sidecar_pack = champion_calibrated
            train_df_cur = df_train_pred
            test_df_cur = df_test_pred
        elif spec.sidecar_variant == "champion_calibrated_nonhyper":
            if champion_nonhyper_sidecar_ref is None:
                nonhyper_sidecar_df, nonhyper_sidecar_name, nonhyper_sidecar_payload, nonhyper_sidecar_results = fit_stage2_mode(
                    "champion_sidecar_ref_nonhyper",
                    df_train_pred_nonhyper,
                    df_test_pred_nonhyper,
                    mode="predicted_only_state_rule",
                    use_feature_selection=False,
                    allowed_models=["Logistic Reg.", "Elastic LR"],
                )
                nonhyper_sidecar_name, nonhyper_sidecar_payload = select_transparent_payload(nonhyper_sidecar_results)
                champion_nonhyper_sidecar_ref = {
                    **nonhyper_sidecar_payload,
                    "variant_name": f"predicted_only_state_rule_nonhyper:{nonhyper_sidecar_name}",
                }
                champion_nonhyper_calibrated = build_sidecar_calibrated_pack(
                    "champion_sidecar_calibrated_nonhyper",
                    df_train_pred_nonhyper,
                    df_test_pred_nonhyper,
                    champion_nonhyper_sidecar_ref,
                    strategy_key="phase",
                )
            sidecar_pack = champion_nonhyper_calibrated
            train_df_cur = df_train_pred_nonhyper
            test_df_cur = df_test_pred_nonhyper
        else:
            raise ValueError(spec.sidecar_variant)

        group_df, best_name, payload, group_results = fit_fixed_fusion(
            spec.name,
            train_df_cur,
            test_df_cur,
            direct_pack,
            sidecar_pack,
            note=f"direct={spec.direct_variant}; sidecar={spec.sidecar_variant}; stage1_scope={spec.stage1_scope}",
        )
        record_group_result(
            spec.name,
            group_df,
            best_name,
            payload,
            group_results,
            train_df_cur,
            test_df_cur,
            all_rows,
            perf_frames,
            best_groups,
            all_group_results,
        )

    summary_rows = []
    phase_a_groups = ["direct_anchor_ref", "current_champion_ref"] + [spec.name for spec in phase_a_specs]
    for group_name in phase_a_groups:
        model_name, payload = best_groups[group_name]
        summary_rows.append(
            {
                "Group": group_name,
                "Best_Model": model_name,
                "AUC": payload["metrics"]["auc"],
                "PR_AUC": payload["metrics"]["prauc"],
                "C_Index": payload["c_index"],
                "Brier": payload["metrics"]["brier"],
                "Calibration_Intercept": payload["cal"]["intercept"],
                "Calibration_Slope": payload["cal"]["slope"],
                "Threshold": payload["threshold"],
            }
        )
    phase_a_summary = pd.DataFrame(summary_rows)
    phase_a_window_df = pd.concat([compute_window_metric_table(group_name, df_test_pred, best_groups[group_name][1]) for group_name in phase_a_groups], ignore_index=True)
    phase_a_delta_df = []
    champion_payload = best_groups["current_champion_ref"][1]
    direct_ref_payload = best_groups["direct_anchor_ref"][1]
    for group_name in phase_a_groups:
        if group_name != "direct_anchor_ref":
            phase_a_delta_df.append(build_bootstrap_delta_rows(group_name, df_test_pred, best_groups[group_name][1], "direct_anchor_ref", direct_ref_payload))
        if group_name not in {"direct_anchor_ref", "current_champion_ref"}:
            phase_a_delta_df.append(build_bootstrap_delta_rows(group_name, df_test_pred, best_groups[group_name][1], "current_champion_ref", champion_payload))
    phase_a_delta_df = pd.concat(phase_a_delta_df, ignore_index=True)
    stop_rule_df = build_anchor_champion_stop_rule(phase_a_summary, phase_a_window_df, phase_a_delta_df, "current_champion_ref")
    stop_rule_df.to_csv(Config.OUT_DIR / f"{RESULT_PREFIX}_StopRule.csv", index=False)

    if stop_rule_df["Pass_StopRule"].any():
        print("\n--- Phase 6: Phase B experiments ---")
        phase_b_specs = [
            ExperimentSpec(
                "monotonic_tree_direct_plus_champion_calibrated",
                "monotonic_tree_direct",
                "champion_calibrated",
                phase="B",
            ),
            ExperimentSpec(
                "direct_windowed_core_plus_champion_calibrated_plus_rule_sidecar",
                "direct_windowed_core",
                "champion_calibrated",
                use_rule_sidecar=True,
                phase="B",
            ),
        ]

        refs = stage1_payload["hyper_margin_reference"]
        rule_targets_tr = np.maximum(
            (stage1_payload["delta_results"][best_delta_name]["oof_next"][:, 1] - refs["FT4"]["threshold"]) * refs["FT4"]["direction"] > 0,
            (stage1_payload["delta_results"][best_delta_name]["oof_next"][:, 2] - refs["logTSH"]["threshold"]) * refs["logTSH"]["direction"] > 0,
        ).astype(int)
        rule_targets_te = np.maximum(
            (stage1_payload["delta_results"][best_delta_name]["test_next"][:, 1] - refs["FT4"]["threshold"]) * refs["FT4"]["direction"] > 0,
            (stage1_payload["delta_results"][best_delta_name]["test_next"][:, 2] - refs["logTSH"]["threshold"]) * refs["logTSH"]["direction"] > 0,
        ).astype(int)
        rule_best_name, rule_state_results, rule_metrics = run_binary_suite_oof(
            x1_tr, df_train, x1_te, df_test, groups, y_train=rule_targets_tr, y_test=rule_targets_te
        )
        rule_metrics.to_csv(Config.OUT_DIR / "Stage1_RuleSidecar_Metrics.csv", index=False)
        rule_pack = _score_pack_from_state_results(rule_state_results)
        rule_pack["variant_name"] = f"Any_hyper_rule_next:{rule_best_name}"

        monotonic_cache = None
        for spec in phase_b_specs:
            if spec.direct_variant == "direct_windowed_core":
                if direct_windowed_cache is None:
                    direct_windowed_cache = fit_direct_windowed_core(df_train_pred, df_test_pred, direct_payload)
                _, _, direct_pack, _ = direct_windowed_cache
            elif spec.direct_variant == "monotonic_tree_direct":
                if monotonic_cache is None:
                    monotonic_cache = fit_monotonic_tree_direct(df_train_pred, df_test_pred, direct_payload)
                _, _, direct_pack, _ = monotonic_cache
            else:
                raise ValueError(spec.direct_variant)

            aux_rule = rule_pack if spec.use_rule_sidecar else None
            group_df, best_name, payload, group_results = fit_fixed_fusion(
                spec.name,
                df_train_pred,
                df_test_pred,
                direct_pack,
                champion_calibrated,
                aux_rule_payload=aux_rule,
                note=f"direct={spec.direct_variant}; sidecar={spec.sidecar_variant}; rule_sidecar={spec.use_rule_sidecar}",
            )
            record_group_result(
                spec.name,
                group_df,
                best_name,
                payload,
                group_results,
                df_train_pred,
                df_test_pred,
                all_rows,
                perf_frames,
                best_groups,
                all_group_results,
            )
    else:
        print("\n--- Phase 6: Phase B skipped (no Phase A experiment passed stop rule) ---")

    print("\n--- Phase 7: Export reports ---")
    group_order = list(best_groups.keys())
    summary_rows = []
    for group_name in group_order:
        model_name, payload = best_groups[group_name]
        summary_rows.append(
            {
                "Group": group_name,
                "Best_Model": model_name,
                "AUC": payload["metrics"]["auc"],
                "PR_AUC": payload["metrics"]["prauc"],
                "C_Index": payload["c_index"],
                "Brier": payload["metrics"]["brier"],
                "Calibration_Intercept": payload["cal"]["intercept"],
                "Calibration_Slope": payload["cal"]["slope"],
                "Threshold": payload["threshold"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.concat(all_rows, ignore_index=True)
    per_window_df = pd.concat([compute_window_metric_table(group_name, df_test_pred, best_groups[group_name][1]) for group_name in group_order], ignore_index=True)

    delta_frames = []
    current_champion_payload = best_groups["current_champion_ref"][1]
    direct_anchor_payload = best_groups["direct_anchor_ref"][1]
    for group_name in group_order:
        if group_name != "direct_anchor_ref":
            delta_frames.append(build_bootstrap_delta_rows(group_name, df_test_pred, best_groups[group_name][1], "direct_anchor_ref", direct_anchor_payload))
        if group_name not in {"direct_anchor_ref", "current_champion_ref"}:
            delta_frames.append(build_bootstrap_delta_rows(group_name, df_test_pred, best_groups[group_name][1], "current_champion_ref", current_champion_payload))
    delta_boot_df = pd.concat(delta_frames, ignore_index=True)

    summary_df.to_csv(Config.OUT_DIR / f"{RESULT_PREFIX}_Best_Group_Summary.csv", index=False)
    comparison_df.to_csv(Config.OUT_DIR / f"{RESULT_PREFIX}_Model_Comparison.csv", index=False)
    per_window_df.to_csv(Config.OUT_DIR / f"{RESULT_PREFIX}_PerWindow_Metrics.csv", index=False)
    delta_boot_df.to_csv(Config.OUT_DIR / f"{RESULT_PREFIX}_DeltaBootstrap.csv", index=False)
    export_metric_matrices(pd.concat(perf_frames, ignore_index=True), Config.OUT_DIR, prefix=f"{RESULT_PREFIX}_Performance")
    save_performance_heatmap_panels(
        pd.concat(perf_frames, ignore_index=True),
        Config.OUT_DIR / f"{RESULT_PREFIX}_Performance_Heatmaps.png",
        task_order=[f"{RESULT_PREFIX} {group}" for group in group_order],
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=["prauc", "auc", "recall", "specificity", "f1"],
        title="Two-stage transition performance heatmaps",
    )
    best_mode = max(best_groups, key=lambda name: (best_groups[name][1]["metrics"]["prauc"], best_groups[name][1]["metrics"]["auc"]))
    best_name, best_payload = best_groups[best_mode]
    save_anchor_champion_comparison(summary_df)
    save_per_window_pr_auc_comparison(per_window_df, "current_champion_ref", best_mode)
    save_pr_auc_delta_forest(delta_boot_df, baseline_name="current_champion_ref")
    save_best_mode_reports_prefixed(f"{RESULT_PREFIX}_BestMode", best_mode, best_payload, df_train_pred, df_test_pred)
    if best_name in {"Logistic Reg.", "Elastic LR"}:
        prefix = f"{best_mode}_{best_name}".replace(" ", "_")
        save_logistic_regression_visuals(
            best_name,
            best_payload["model"],
            best_payload["feat_names"],
            Config.OUT_DIR,
            prefix=prefix,
            decision_threshold=best_payload["threshold"],
            output_label="P(Relapse at next window)",
        )

    prior_best_row = None
    prior_summary_path = Config.BASE_TWO_STAGE_DIR / "Stage1Only_Best_Group_Summary.csv"
    if prior_summary_path.exists():
        prior_summary = pd.read_csv(prior_summary_path)
        if "Group" in prior_summary.columns and "binary_interval_intercept_calibrated" in set(prior_summary["Group"]):
            prior_best_row = prior_summary.loc[prior_summary["Group"] == "binary_interval_intercept_calibrated"].iloc[0]

    write_anchor_champion_round_summary(
        summary_df,
        stop_rule_df,
        delta_boot_df,
        per_window_df,
        stage1_payload,
        scope_count_df,
        prior_best_row,
    )

    print("\n  Two-stage transition summary")
    for row in summary_df.itertuples(index=False):
        print(
            f"    {row.Group:<56s} {row.Best_Model:<18s} "
            f"AUC={row.AUC:.3f} PR-AUC={row.PR_AUC:.3f} Brier={row.Brier:.3f}"
        )
    print(f"    Best overall mode: {best_mode} ({best_name})")
    print(f"\n  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
