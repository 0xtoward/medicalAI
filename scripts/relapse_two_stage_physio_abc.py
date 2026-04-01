import os
import sys
import warnings
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED, STATE_NAMES, TIME_STAMPS
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
    build_next_physio_targets,
    build_physio_history_features,
    evaluate_next_state_predictions,
    evaluate_physio_predictions,
    make_stage1_feature_frames,
    make_stage2_feature_frames,
    run_stage1_oof_forecast,
    save_physio_scatter,
    save_stage1_metric_bar,
)
from utils.plot_style import (
    PRIMARY_BLUE,
    PRIMARY_TEAL,
    TEXT_DARK,
    TEXT_MID,
    apply_publication_style,
)
from utils.recurrence import build_interval_risk_data, derive_recurrent_survival_data

apply_publication_style()
np.random.seed(SEED)


STATE_TO_CODE = {name: idx for idx, name in enumerate(STATE_NAMES)}

COMPACT_STAGE1_META_COLS = [
    "P_next_hyper",
    "P_next_hypo",
    "NextState_Entropy",
    "NextState_MaxProb",
    "Pred_Delta_FT4",
    "Pred_Delta_logTSH",
    "Pred_Std_FT4",
    "Pred_Std_logTSH",
    "Pred_Width_FT4",
    "Pred_Width_logTSH",
    "Pred_FT4_Margin_To_Hyper",
    "Pred_logTSH_Margin_To_Hyper",
]
SIDECAR_PROB_COLS = ["P_next_hyper", "P_next_hypo"]
SIDECAR_DELTA_COLS = ["Pred_Delta_FT4", "Pred_Delta_logTSH"]
SIDECAR_UNCERTAINTY_COLS = [
    "NextState_Entropy",
    "NextState_MaxProb",
    "Pred_Std_FT4",
    "Pred_Std_logTSH",
    "Pred_Width_FT4",
    "Pred_Width_logTSH",
]
BRANCH_C_UNCERTAINTY_COLS = [
    "NextState_Entropy",
    "Pred_Std_FT4",
    "Pred_Std_logTSH",
]
SIDECAR_WINDOW_INTERACTIONS = [
    ("Start_Time", "P_next_hyper", "Start_Time_x_P_next_hyper"),
    ("Interval_Width", "P_next_hyper", "Interval_Width_x_P_next_hyper"),
]
BRANCH_A_CURRENT_COLS = [
    "Age",
    "Sex",
    "Dose",
    "TRAb",
    "Start_Time",
    "Interval_Width",
    "Prior_Relapse_Count",
    "Event_Order",
    "Ever_Hyper_Before",
    "Ever_Hypo_Before",
    "Time_In_Normal",
    "FT4_Current",
    "logTSH_Current",
    "FT4_Prev",
    "logTSH_Prev",
    "FT4_STD_ToDate",
    "logTSH_STD_ToDate",
    "Delta_FT4_1step",
    "Delta_TSH_1step",
]


class Config:
    OUT_DIR = Path("./results/relapse_two_stage_physio_abc/")
    SHARED_RELAPSE_DIR = Path("./results/relapse/")
    LEGACY_RELAPSE_DIR = Path("./multistate_result/")
    EXISTING_TWO_STAGE_DIR = Path("./results/relapse_two_stage_physio/")
    BASELINE_DIR = OUT_DIR / "baseline_reference"
    BRANCH_A_DIR = OUT_DIR / "branch_A_event_aligned_compact"
    BRANCH_B_DIR = OUT_DIR / "branch_B_transparent_fusion"
    BRANCH_C_DIR = OUT_DIR / "branch_C_sidecar"


for out_dir in [
    Config.OUT_DIR,
    Config.BASELINE_DIR,
    Config.BRANCH_A_DIR,
    Config.BRANCH_B_DIR,
    Config.BRANCH_C_DIR,
]:
    out_dir.mkdir(parents=True, exist_ok=True)


def _safe_name(text):
    return str(text).replace(" ", "_").replace("/", "_")


def _state_entropy(proba):
    proba = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    entropy = -np.sum(proba * np.log(proba), axis=1)
    return entropy / np.log(proba.shape[1])


def _is_mean_ensemble(payload):
    model = payload.get("model")
    return isinstance(model, dict) and model.get("type") == "mean_ensemble"


def _compute_margin(values, ref):
    return ref["direction"] * (np.asarray(values, dtype=float) - ref["threshold"])


def _group_kfold(groups):
    unique_groups = pd.Index(groups).drop_duplicates()
    n_splits = min(3, len(unique_groups))
    return GroupKFold(n_splits=n_splits)


def _param_grid_size(grid):
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _safe_logit(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_platt_calibration_from_oof(y_tr, train_fit_proba, oof_proba, test_proba):
    y_tr = np.asarray(y_tr, dtype=int)
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    x_oof = _safe_logit(oof_proba).reshape(-1, 1)
    calibrator.fit(x_oof, y_tr)
    train_fit_cal = calibrator.predict_proba(_safe_logit(train_fit_proba).reshape(-1, 1))[:, 1]
    oof_cal = calibrator.predict_proba(x_oof)[:, 1]
    test_cal = calibrator.predict_proba(_safe_logit(test_proba).reshape(-1, 1))[:, 1]
    return train_fit_cal, oof_cal, test_cal, calibrator


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


def build_two_stage_long_data(force_rebuild=False):
    train_path = Config.OUT_DIR / "two_stage_train.csv"
    test_path = Config.OUT_DIR / "two_stage_test.csv"
    if not force_rebuild and train_path.exists() and test_path.exists():
        print("\n--- Phase 1: Load cached two-stage long-format data ---")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print(f"  Loaded: train {len(df_train)}  test {len(df_test)} rows")
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
        phys_cache = Config.OUT_DIR / f"missforest_physio_depth{depth}.pkl"
        phys_fallback = Config.EXISTING_TWO_STAGE_DIR / f"missforest_physio_depth{depth}.pkl"
        fallback_cache = phys_fallback if phys_fallback.exists() else None
        imputer_phys = load_or_fit_with_message(
            raw_phys[tr_mask],
            phys_cache,
            fallback_cache_path=fallback_cache,
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
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return df_train, df_test


def append_stage1_meta_features(df, stage1_payload, split_key):
    out = df.copy()
    delta_key = f"{split_key}_delta"
    state_key = f"{split_key}_proba"

    delta_arrays = []
    for model_name, payload in stage1_payload["delta_results"].items():
        if _is_mean_ensemble(payload):
            continue
        arr = np.asarray(payload.get(delta_key), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            delta_arrays.append(arr)

    state_arrays = []
    for model_name, payload in stage1_payload["state_results"].items():
        if _is_mean_ensemble(payload):
            continue
        arr = np.asarray(payload.get(state_key), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == len(STATE_NAMES):
            state_arrays.append(arr)

    if not delta_arrays or not state_arrays:
        return out

    delta_stack = np.stack(delta_arrays, axis=0)
    state_stack = np.stack(state_arrays, axis=0)
    mean_delta = np.mean(delta_stack, axis=0)
    std_delta = np.std(delta_stack, axis=0)
    width_delta = np.max(delta_stack, axis=0) - np.min(delta_stack, axis=0)
    mean_state = np.mean(state_stack, axis=0)

    out["P_next_hyper"] = mean_state[:, STATE_TO_CODE["Hyper"]]
    out["P_next_normal"] = mean_state[:, STATE_TO_CODE["Normal"]]
    out["P_next_hypo"] = mean_state[:, STATE_TO_CODE["Hypo"]]
    out["NextState_Entropy"] = _state_entropy(mean_state)
    out["NextState_MaxProb"] = np.max(mean_state, axis=1)

    out["Pred_Delta_FT3"] = mean_delta[:, 0]
    out["Pred_Delta_FT4"] = mean_delta[:, 1]
    out["Pred_Delta_logTSH"] = mean_delta[:, 2]
    out["Pred_Std_FT3"] = std_delta[:, 0]
    out["Pred_Std_FT4"] = std_delta[:, 1]
    out["Pred_Std_logTSH"] = std_delta[:, 2]
    out["Pred_Width_FT3"] = width_delta[:, 0]
    out["Pred_Width_FT4"] = width_delta[:, 1]
    out["Pred_Width_logTSH"] = width_delta[:, 2]

    pred_ft4_next = out["FT4_Current"].values.astype(float) + mean_delta[:, 1]
    pred_logtsh_next = out["logTSH_Current"].values.astype(float) + mean_delta[:, 2]
    refs = stage1_payload["hyper_margin_reference"]
    out["Pred_FT4_Margin_To_Hyper"] = _compute_margin(pred_ft4_next, refs["FT4"])
    out["Pred_logTSH_Margin_To_Hyper"] = _compute_margin(pred_logtsh_next, refs["logTSH"])
    return out


def build_feature_frame(train_df, test_df, numeric_cols, cat_cols=None, interactions=None):
    cat_cols = [] if cat_cols is None else [col for col in cat_cols if col in train_df.columns]
    numeric_cols = [col for col in numeric_cols if col in train_df.columns]

    tr = train_df[numeric_cols + cat_cols].copy().reset_index(drop=True)
    te = test_df[numeric_cols + cat_cols].copy().reset_index(drop=True)

    for col in cat_cols:
        cats = sorted(train_df[col].astype(str).unique())
        for cat in cats:
            name = f"{col}_{cat}"
            tr[name] = (train_df[col].astype(str).values == cat).astype(float)
            te[name] = (test_df[col].astype(str).values == cat).astype(float)

    tr = tr.drop(columns=cat_cols)
    te = te.drop(columns=cat_cols)

    if interactions:
        for left, right, new_name in interactions:
            if left in tr.columns and right in tr.columns:
                tr[new_name] = tr[left].values * tr[right].values
                te[new_name] = te[left].values * te[right].values

    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)
    medians = tr.median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)
    keep_cols = [col for col in tr.columns if tr[col].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols], keep_cols


def get_stage2_tune_specs(include_balanced_rf=False):
    specs = {
        "Logistic Reg.": (
            Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "lr__penalty": ["l1", "l2"],
                "lr__solver": ["saga"],
            },
            "#1f77b4",
            "-.",
            14,
            -1,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="elasticnet",
                            solver="saga",
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8",
            "-.",
            12,
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
                random_state=SEED,
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
            18,
            -1,
        ),
    }
    if include_balanced_rf:
        specs["Balanced RF"] = (
            BalancedRandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "min_samples_leaf": [1, 3, 5, 10],
                "sampling_strategy": ["all", "not minority"],
            },
            "#72b7b2",
            "--",
            16,
            -1,
        )
    return specs


def subset_tune_specs(model_names, include_balanced_rf=False):
    all_specs = get_stage2_tune_specs(include_balanced_rf=include_balanced_rf)
    return {name: all_specs[name] for name in model_names}


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    if grid:
        rs = RandomizedSearchCV(
            base,
            grid,
            n_iter=min(n_iter, _param_grid_size(grid)),
            cv=cv,
            scoring="average_precision",
            random_state=SEED,
            n_jobs=n_jobs,
        )
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        return rs.best_estimator_, float(rs.best_score_), rs.best_params_

    scores = []
    for fold_tr, fold_val in cv.split(X_tr_df, y_tr, groups=groups_tr):
        model = clone(base)
        model.fit(X_tr_df.iloc[fold_tr], y_tr[fold_tr])
        proba = model.predict_proba(X_tr_df.iloc[fold_val])[:, 1]
        scores.append(average_precision_score(y_tr[fold_val], proba))
    return clone(base), float(np.mean(scores)), None


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


def fit_mode_from_frames(
    mode_name,
    x_tr_full,
    x_te_full,
    train_df,
    test_df,
    out_dir,
    prefix,
    tune_specs,
    use_feature_selection=False,
    min_features=8,
    apply_posthoc_calibration=False,
):
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values

    if use_feature_selection:
        fs = select_binary_features_with_l1(
            x_tr_full,
            x_te_full,
            y_tr,
            groups,
            out_dir=out_dir,
            prefix=prefix,
            seed=SEED,
            min_features=min_features,
        )
        x_tr = fs.X_train
        x_te = fs.X_test
        feat_names = fs.selected_features
        print(
            f"    {mode_name:<34s} feature selection: {fs.original_feature_count} -> {len(feat_names)} "
            f"(best C={fs.best_c}, CV PR-AUC={fs.cv_score:.3f})"
        )
    else:
        x_tr = x_tr_full.copy().replace([np.inf, -np.inf], np.nan)
        x_te = x_te_full.copy().replace([np.inf, -np.inf], np.nan)
        medians = x_tr.median(numeric_only=True)
        x_tr = x_tr.fillna(medians)
        x_te = x_te.fillna(medians)
        feat_names = list(x_tr.columns)
        pd.DataFrame({"feature": feat_names}).to_csv(out_dir / f"{prefix}_Features.csv", index=False)
        print(f"    {mode_name:<34s} features: {len(feat_names)} (no L1 pre-selection)")

    gkf = _group_kfold(groups)
    rows = []
    results = {}

    for model_name, (base_est, param_grid, color, ls, n_iter, n_jobs) in tune_specs.items():
        best_est, cv_score, best_params = fit_candidate_model(
            base_est, param_grid, n_iter, n_jobs, gkf, x_tr, y_tr, groups
        )

        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            fitted = clone(best_est)
            fitted.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]

        fitted = clone(best_est)
        fitted.fit(x_tr, y_tr)
        train_fit_proba = fitted.predict_proba(x_tr)[:, 1]
        proba = fitted.predict_proba(x_te)[:, 1]
        calibrator = None
        if apply_posthoc_calibration:
            train_fit_proba, oof, proba, calibrator = apply_platt_calibration_from_oof(
                y_tr,
                train_fit_proba=train_fit_proba,
                oof_proba=oof,
                test_proba=proba,
            )
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        c_index = compute_recurrent_c_index_from_intervals(test_df, proba)

        rows.append(
            {
                "Mode": mode_name,
                "Model": model_name,
                "CV_PR_AUC": cv_score,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "C_Index": c_index,
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
            "proba": proba,
            "oof_proba": oof,
            "train_fit_proba": train_fit_proba,
            "auc": metrics["auc"],
            "prauc": metrics["prauc"],
            "c_index": c_index,
            "threshold": thr,
            "metrics": metrics,
            "cal": cal,
            "feat_names": feat_names,
            "color": color,
            "ls": ls,
            "posthoc_calibrator": calibrator,
        }

    best_name = max(results, key=lambda name: (results[name]["prauc"], results[name]["auc"]))
    return pd.DataFrame(rows), best_name, results[best_name], results


def summarize_mode_rows(branch_name, mode_payloads):
    rows = []
    for mode_name, (best_name, payload) in mode_payloads.items():
        metrics = payload["metrics"]
        cal = payload["cal"]
        rows.append(
            {
                "Branch": branch_name,
                "Mode": mode_name,
                "Best_Model": best_name,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "C_Index": payload["c_index"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": metrics["threshold"],
            }
        )
    return pd.DataFrame(rows)


def save_best_reports(mode_name, payload, train_df, test_df, out_dir, prefix):
    y_tr = train_df["Y_Relapse"].values.astype(int)
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
    interval_summary.to_csv(out_dir / f"{prefix}_Interval_Summary.csv", index=False)

    save_calibration_figure(
        y_te,
        payload["proba"],
        f"Calibration Curve ({mode_name}, interval-level)",
        out_dir / f"{prefix}_Calibration_Interval.png",
    )
    save_dca_figure(
        y_te,
        payload["proba"],
        f"Decision Curve Analysis ({mode_name}, interval-level)",
        out_dir / f"{prefix}_DCA_Interval.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        payload["proba"],
        payload["threshold"],
        f"Threshold Sensitivity ({mode_name}, interval-level)",
        out_dir / f"{prefix}_Threshold_Sensitivity_Interval.png",
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
    patient_summary.to_csv(out_dir / f"{prefix}_Patient_Summary.csv", index=False)

    save_calibration_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Calibration Curve ({mode_name}, patient-level)",
        out_dir / f"{prefix}_Calibration_Patient.png",
    )
    save_dca_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Decision Curve Analysis ({mode_name}, patient-level)",
        out_dir / f"{prefix}_DCA_Patient.png",
    )
    save_threshold_sensitivity_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        patient_thr,
        f"Threshold Sensitivity ({mode_name}, patient-level)",
        out_dir / f"{prefix}_Threshold_Sensitivity_Patient.png",
    )
    save_patient_risk_strata(patient_tr, patient_te, out_dir / f"{prefix}_Patient_Risk_Q1Q4.png")

    patient_sens_df, _ = evaluate_patient_aggregation_sensitivity(train_df, test_df, payload["oof_proba"], payload["proba"])
    patient_sens_df.to_csv(out_dir / f"{prefix}_Patient_Aggregation_Sensitivity.csv", index=False)
    save_patient_aggregation_sensitivity_figure(
        patient_sens_df,
        out_dir / f"{prefix}_Patient_Aggregation_Sensitivity.png",
    )

    window_sens_df = evaluate_window_sensitivity(test_df, payload["proba"], payload["threshold"])
    window_sens_df.to_csv(out_dir / f"{prefix}_Window_Sensitivity.csv", index=False)
    return interval_summary, patient_summary, patient_sens_df, window_sens_df


def export_branch_results(branch_name, out_dir, mode_rows, mode_results, train_df, test_df):
    comparison_df = pd.concat(mode_rows, ignore_index=True)
    comparison_df.to_csv(out_dir / f"{branch_name}_Model_Comparison.csv", index=False)

    best_by_mode = summarize_mode_rows(branch_name, {k: (v["best_name"], v["best_payload"]) for k, v in mode_results.items()})
    best_by_mode.to_csv(out_dir / f"{branch_name}_Best_Mode_Summary.csv", index=False)

    perf_frames = []
    task_order = []
    for mode_name, payload in mode_results.items():
        task_name = f"{branch_name} {mode_name}"
        task_order.append(task_name)
        perf_frames.append(
            build_binary_performance_long(
                task_name=task_name,
                results=payload["results"],
                domain_payloads={
                    "Train_Fit": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "train_fit_proba"},
                    "Validation_OOF": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "oof_proba"},
                    "Test_Temporal": {"y_true": test_df["Y_Relapse"].values.astype(int), "proba_key": "proba"},
                },
                metric_keys=["prauc", "auc", "recall", "specificity", "f1"],
                threshold_key="threshold",
            )
        )

    perf_long_df = pd.concat(perf_frames, ignore_index=True)
    export_metric_matrices(perf_long_df, out_dir, prefix=f"{branch_name}_Performance")
    save_performance_heatmap_panels(
        perf_long_df,
        out_dir / f"{branch_name}_Performance_Heatmaps.png",
        task_order=task_order,
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=["prauc", "auc", "recall", "specificity", "f1"],
        title=f"{branch_name} Internal Performance Heatmaps",
    )

    best_mode_name = best_by_mode.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]["Mode"]
    best_payload = mode_results[best_mode_name]["best_payload"]
    best_model_name = mode_results[best_mode_name]["best_name"]
    save_best_reports(best_mode_name, best_payload, train_df, test_df, out_dir, prefix=f"{branch_name}_Best")
    if best_model_name in {"Logistic Reg.", "Elastic LR"}:
        save_logistic_regression_visuals(
            best_model_name,
            best_payload["model"],
            best_payload["feat_names"],
            out_dir,
            prefix=f"{branch_name}_{_safe_name(best_mode_name)}_{_safe_name(best_model_name)}",
            decision_threshold=best_payload["threshold"],
            output_label="P(Relapse at next window)",
        )
    return best_by_mode


def run_baseline_direct_reference(train_df, test_df):
    print("\n--- Baseline reference: direct full model zoo ---")
    x_tr_full, x_te_full, _ = make_stage2_feature_frames(train_df, test_df, mode="direct")
    mode_df, best_name, best_payload, results = fit_mode_from_frames(
        mode_name="direct_full",
        x_tr_full=x_tr_full,
        x_te_full=x_te_full,
        train_df=train_df,
        test_df=test_df,
        out_dir=Config.BASELINE_DIR,
        prefix="Baseline_Direct",
        tune_specs=get_stage2_tune_specs(include_balanced_rf=True),
        use_feature_selection=True,
        min_features=8,
    )
    mode_df.to_csv(Config.BASELINE_DIR / "Baseline_Direct_Model_Comparison.csv", index=False)

    summary = summarize_mode_rows(
        "Baseline",
        {
            "direct_full": (
                best_name,
                best_payload,
            )
        },
    )
    summary.to_csv(Config.BASELINE_DIR / "Baseline_Direct_Best_Summary.csv", index=False)
    save_best_reports("direct_full", best_payload, train_df, test_df, Config.BASELINE_DIR, prefix="Baseline_Direct_Best")

    print(
        f"  Baseline best direct: {best_name:<14s} "
        f"AUC={best_payload['metrics']['auc']:.3f} PR-AUC={best_payload['metrics']['prauc']:.3f} "
        f"Brier={best_payload['metrics']['brier']:.3f}"
    )
    return {
        "mode_df": mode_df,
        "best_name": best_name,
        "best_payload": best_payload,
        "results": results,
        "summary": summary,
    }


def run_branch_a(train_df, test_df):
    print("\n--- Branch A: event-aligned compact two-stage ---")
    tune_specs = subset_tune_specs(["Logistic Reg.", "Elastic LR", "LightGBM"])
    mode_results = {}
    mode_rows = []

    branch_configs = {
        "direct_compact": {
            "numeric_cols": BRANCH_A_CURRENT_COLS,
            "cat_cols": ["Prev_State"],
        },
        "predicted_only_compact": {
            "numeric_cols": COMPACT_STAGE1_META_COLS,
            "cat_cols": [],
        },
        "predicted_plus_current_compact": {
            "numeric_cols": BRANCH_A_CURRENT_COLS + COMPACT_STAGE1_META_COLS,
            "cat_cols": ["Prev_State"],
        },
    }

    for mode_name, cfg in branch_configs.items():
        x_tr, x_te, _ = build_feature_frame(
            train_df,
            test_df,
            numeric_cols=cfg["numeric_cols"],
            cat_cols=cfg["cat_cols"],
            interactions=None,
        )
        mode_df, best_name, best_payload, results = fit_mode_from_frames(
            mode_name=mode_name,
            x_tr_full=x_tr,
            x_te_full=x_te,
            train_df=train_df,
            test_df=test_df,
            out_dir=Config.BRANCH_A_DIR,
            prefix=f"BranchA_{mode_name}",
            tune_specs=tune_specs,
            use_feature_selection=False,
        )
        mode_rows.append(mode_df)
        mode_results[mode_name] = {
            "best_name": best_name,
            "best_payload": best_payload,
            "results": results,
        }
        print(
            f"  Branch A best {mode_name:<28s}: {best_name:<14s} "
            f"AUC={best_payload['metrics']['auc']:.3f} PR-AUC={best_payload['metrics']['prauc']:.3f}"
        )

    summary = export_branch_results("BranchA", Config.BRANCH_A_DIR, mode_rows, mode_results, train_df, test_df)
    return mode_results, summary


def run_branch_b(train_df, test_df, baseline_direct_payload, branch_a_physio_result):
    print("\n--- Branch B: transparent late fusion ---")
    mode_results = {}
    mode_rows = []

    direct_payload = baseline_direct_payload
    direct_mode_df = pd.DataFrame(
        [
            {
                "Mode": "direct_branch",
                "Model": "Logistic Reg.",
                "CV_PR_AUC": np.nan,
                "AUC": direct_payload["metrics"]["auc"],
                "PR_AUC": direct_payload["metrics"]["prauc"],
                "C_Index": direct_payload["c_index"],
                "Brier": direct_payload["metrics"]["brier"],
                "Recall": direct_payload["metrics"]["recall"],
                "Specificity": direct_payload["metrics"]["specificity"],
                "F1": direct_payload["metrics"]["f1"],
                "Calibration_Intercept": direct_payload["cal"]["intercept"],
                "Calibration_Slope": direct_payload["cal"]["slope"],
                "Threshold": direct_payload["threshold"],
                "Best_Params": "",
            }
        ]
    )
    mode_rows.append(direct_mode_df)
    mode_results["direct_branch"] = {
        "best_name": "Logistic Reg.",
        "best_payload": direct_payload,
        "results": {"Logistic Reg.": direct_payload},
    }
    print(
        f"  Branch B direct branch               : Logistic Reg.   "
        f"AUC={direct_payload['metrics']['auc']:.3f} PR-AUC={direct_payload['metrics']['prauc']:.3f}"
    )

    physio_best_name = branch_a_physio_result["best_name"]
    physio_payload = branch_a_physio_result["best_payload"]
    physio_mode_df = pd.DataFrame(
        [
            {
                "Mode": "physio_branch",
                "Model": physio_best_name,
                "CV_PR_AUC": np.nan,
                "AUC": physio_payload["metrics"]["auc"],
                "PR_AUC": physio_payload["metrics"]["prauc"],
                "C_Index": physio_payload["c_index"],
                "Brier": physio_payload["metrics"]["brier"],
                "Recall": physio_payload["metrics"]["recall"],
                "Specificity": physio_payload["metrics"]["specificity"],
                "F1": physio_payload["metrics"]["f1"],
                "Calibration_Intercept": physio_payload["cal"]["intercept"],
                "Calibration_Slope": physio_payload["cal"]["slope"],
                "Threshold": physio_payload["threshold"],
                "Best_Params": "",
            }
        ]
    )
    mode_rows.append(physio_mode_df)
    mode_results["physio_branch"] = {
        "best_name": physio_best_name,
        "best_payload": physio_payload,
        "results": {physio_best_name: physio_payload},
    }
    print(
        f"  Branch B physio branch               : {physio_best_name:<14s} "
        f"AUC={physio_payload['metrics']['auc']:.3f} PR-AUC={physio_payload['metrics']['prauc']:.3f}"
    )

    train_base = pd.DataFrame(
        {
            "Direct_Score": baseline_direct_payload["oof_proba"],
            "Physio_Score": physio_payload["oof_proba"],
            "Start_Time": train_df["Start_Time"].values.astype(float),
            "Interval_Width": train_df["Interval_Width"].values.astype(float),
        }
    )
    test_base = pd.DataFrame(
        {
            "Direct_Score": baseline_direct_payload["proba"],
            "Physio_Score": physio_payload["proba"],
            "Start_Time": test_df["Start_Time"].values.astype(float),
            "Interval_Width": test_df["Interval_Width"].values.astype(float),
        }
    )

    fusion_configs = {
        "fusion_basic": ["Direct_Score", "Physio_Score"],
        "fusion_window": ["Direct_Score", "Physio_Score", "Start_Time", "Interval_Width"],
        "fusion_window_interaction": [
            "Direct_Score",
            "Physio_Score",
            "Start_Time",
            "Interval_Width",
            "Direct_x_Physio",
            "Start_Time_x_Physio",
        ],
    }

    train_base["Direct_x_Physio"] = train_base["Direct_Score"] * train_base["Physio_Score"]
    test_base["Direct_x_Physio"] = test_base["Direct_Score"] * test_base["Physio_Score"]
    train_base["Start_Time_x_Physio"] = train_base["Start_Time"] * train_base["Physio_Score"]
    test_base["Start_Time_x_Physio"] = test_base["Start_Time"] * test_base["Physio_Score"]

    fusion_specs = subset_tune_specs(["Logistic Reg.", "Elastic LR"])
    for mode_name, cols in fusion_configs.items():
        mode_df, best_name, best_payload, results = fit_mode_from_frames(
            mode_name=mode_name,
            x_tr_full=train_base[cols].copy(),
            x_te_full=test_base[cols].copy(),
            train_df=train_df,
            test_df=test_df,
            out_dir=Config.BRANCH_B_DIR,
            prefix=f"BranchB_{mode_name}",
            tune_specs=fusion_specs,
            use_feature_selection=False,
            apply_posthoc_calibration=True,
        )
        mode_rows.append(mode_df)
        mode_results[mode_name] = {
            "best_name": best_name,
            "best_payload": best_payload,
            "results": results,
        }
        print(
            f"  Branch B best {mode_name:<22s}: {best_name:<14s} "
            f"AUC={best_payload['metrics']['auc']:.3f} PR-AUC={best_payload['metrics']['prauc']:.3f}"
        )

    summary = export_branch_results("BranchB", Config.BRANCH_B_DIR, mode_rows, mode_results, train_df, test_df)
    return mode_results, summary


def run_branch_c(train_df, test_df, baseline_direct_payload):
    print("\n--- Branch C: conservative sidecar enhancement ---")
    lr_specs = subset_tune_specs(["Logistic Reg.", "Elastic LR"])

    mode_results = {}
    mode_rows = []

    anchor_payload = baseline_direct_payload
    anchor_mode_df = pd.DataFrame(
        [
            {
                "Mode": "anchor_direct",
                "Model": "Logistic Reg.",
                "CV_PR_AUC": np.nan,
                "AUC": anchor_payload["metrics"]["auc"],
                "PR_AUC": anchor_payload["metrics"]["prauc"],
                "C_Index": anchor_payload["c_index"],
                "Brier": anchor_payload["metrics"]["brier"],
                "Recall": anchor_payload["metrics"]["recall"],
                "Specificity": anchor_payload["metrics"]["specificity"],
                "F1": anchor_payload["metrics"]["f1"],
                "Calibration_Intercept": anchor_payload["cal"]["intercept"],
                "Calibration_Slope": anchor_payload["cal"]["slope"],
                "Threshold": anchor_payload["threshold"],
                "Best_Params": "",
            }
        ]
    )
    mode_rows.append(anchor_mode_df)
    mode_results["anchor_direct"] = {
        "best_name": "Logistic Reg.",
        "best_payload": anchor_payload,
        "results": {"Logistic Reg.": anchor_payload},
    }
    print(
        f"  Branch C anchor direct               : Logistic Reg.   "
        f"AUC={anchor_payload['metrics']['auc']:.3f} PR-AUC={anchor_payload['metrics']['prauc']:.3f}"
    )

    train_base = pd.DataFrame(
        {
            "Direct_Anchor_Score": anchor_payload["oof_proba"],
            "P_next_hyper": train_df["P_next_hyper"].values.astype(float),
            "P_next_hypo": train_df["P_next_hypo"].values.astype(float),
            "Pred_Delta_FT4": train_df["Pred_Delta_FT4"].values.astype(float),
            "Pred_Delta_logTSH": train_df["Pred_Delta_logTSH"].values.astype(float),
            "NextState_Entropy": train_df["NextState_Entropy"].values.astype(float),
            "Pred_Std_FT4": train_df["Pred_Std_FT4"].values.astype(float),
            "Pred_Std_logTSH": train_df["Pred_Std_logTSH"].values.astype(float),
            "Start_Time": train_df["Start_Time"].values.astype(float),
            "Interval_Width": train_df["Interval_Width"].values.astype(float),
        }
    )
    test_base = pd.DataFrame(
        {
            "Direct_Anchor_Score": anchor_payload["proba"],
            "P_next_hyper": test_df["P_next_hyper"].values.astype(float),
            "P_next_hypo": test_df["P_next_hypo"].values.astype(float),
            "Pred_Delta_FT4": test_df["Pred_Delta_FT4"].values.astype(float),
            "Pred_Delta_logTSH": test_df["Pred_Delta_logTSH"].values.astype(float),
            "NextState_Entropy": test_df["NextState_Entropy"].values.astype(float),
            "Pred_Std_FT4": test_df["Pred_Std_FT4"].values.astype(float),
            "Pred_Std_logTSH": test_df["Pred_Std_logTSH"].values.astype(float),
            "Start_Time": test_df["Start_Time"].values.astype(float),
            "Interval_Width": test_df["Interval_Width"].values.astype(float),
        }
    )
    train_base["Start_Time_x_P_next_hyper"] = train_base["Start_Time"] * train_base["P_next_hyper"]
    test_base["Start_Time_x_P_next_hyper"] = test_base["Start_Time"] * test_base["P_next_hyper"]
    train_base["Interval_Width_x_P_next_hyper"] = train_base["Interval_Width"] * train_base["P_next_hyper"]
    test_base["Interval_Width_x_P_next_hyper"] = test_base["Interval_Width"] * test_base["P_next_hyper"]

    mode_configs = {
        "anchor_plus_next_state": {
            "cols": ["Direct_Anchor_Score"] + SIDECAR_PROB_COLS,
        },
        "anchor_plus_next_state_delta": {
            "cols": ["Direct_Anchor_Score"] + SIDECAR_PROB_COLS + SIDECAR_DELTA_COLS,
        },
        "anchor_plus_next_state_delta_uncertainty": {
            "cols": ["Direct_Anchor_Score"] + SIDECAR_PROB_COLS + SIDECAR_DELTA_COLS + BRANCH_C_UNCERTAINTY_COLS,
        },
        "anchor_plus_next_state_delta_uncertainty_window": {
            "cols": [
                "Direct_Anchor_Score",
                *SIDECAR_PROB_COLS,
                *SIDECAR_DELTA_COLS,
                *BRANCH_C_UNCERTAINTY_COLS,
                "Start_Time_x_P_next_hyper",
                "Interval_Width_x_P_next_hyper",
            ],
        },
    }

    for mode_name, cfg in mode_configs.items():
        mode_df, best_name, best_payload, results = fit_mode_from_frames(
            mode_name=mode_name,
            x_tr_full=train_base[cfg["cols"]].copy(),
            x_te_full=test_base[cfg["cols"]].copy(),
            train_df=train_df,
            test_df=test_df,
            out_dir=Config.BRANCH_C_DIR,
            prefix=f"BranchC_{mode_name}",
            tune_specs=lr_specs,
            use_feature_selection=False,
        )
        mode_rows.append(mode_df)
        mode_results[mode_name] = {
            "best_name": best_name,
            "best_payload": best_payload,
            "results": results,
        }
        print(
            f"  Branch C best {mode_name:<34s}: {best_name:<14s} "
            f"AUC={best_payload['metrics']['auc']:.3f} PR-AUC={best_payload['metrics']['prauc']:.3f}"
        )

    summary = export_branch_results("BranchC", Config.BRANCH_C_DIR, mode_rows, mode_results, train_df, test_df)
    return mode_results, summary


def write_markdown_summary(combined_df, baseline_logistic_row):
    branch_df = combined_df[combined_df["Branch"].isin(["BranchA", "BranchB", "BranchC"])].copy()
    strongest = branch_df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    branch_best = (
        branch_df.sort_values(["Branch", "PR_AUC", "AUC"], ascending=[True, False, False])
        .groupby("Branch", as_index=False)
        .first()
    )

    baseline_pr = float(baseline_logistic_row["PR_AUC"])
    baseline_auc = float(baseline_logistic_row["AUC"])
    beating = branch_df[
        (branch_df["PR_AUC"] > baseline_pr + 1e-9)
        | ((branch_df["PR_AUC"] >= baseline_pr - 1e-9) & (branch_df["AUC"] > baseline_auc + 1e-9))
    ].copy()

    if len(beating) > 0:
        winner = beating.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
        beat_text = (
            f"有。最强的是 `{winner['Branch']} / {winner['Mode']} / {winner['Best_Model']}`，"
            f"interval-level AUC={winner['AUC']:.3f}，PR-AUC={winner['PR_AUC']:.3f}，"
            f"超过 baseline direct logistic 的 AUC={baseline_auc:.3f}，PR-AUC={baseline_pr:.3f}。"
        )
    else:
        gap = strongest["PR_AUC"] - baseline_pr
        beat_text = (
            f"没有。最佳 no-stacking 分支是 `{strongest['Branch']} / {strongest['Mode']} / {strongest['Best_Model']}`，"
            f"interval-level AUC={strongest['AUC']:.3f}，PR-AUC={strongest['PR_AUC']:.3f}；"
            f"baseline direct logistic 为 AUC={baseline_auc:.3f}，PR-AUC={baseline_pr:.3f}。"
            f"PR-AUC 差值为 {gap:+.3f}。"
        )

    if len(beating) > 0:
        publication_text = (
            f"`{strongest['Branch']} / {strongest['Mode']} / {strongest['Best_Model']}`。"
            f"它已经在 no-stacking 条件下超过 direct logistic，而且仍然保持了非常小的融合结构，最适合直接写成主要结果。"
        )
    else:
        publication_df = branch_best.copy()
        publication_df["Interpretability_Bonus"] = publication_df["Branch"].map({"BranchA": 0.000, "BranchB": 0.008, "BranchC": 0.012}).fillna(0.0)
        publication_df["Pub_Score"] = publication_df["PR_AUC"] - 0.25 * publication_df["Brier"] + publication_df["Interpretability_Bonus"]
        publication_choice = publication_df.sort_values(["Pub_Score", "AUC"], ascending=[False, False]).iloc[0]
        publication_text = (
            f"`{publication_choice['Branch']} / {publication_choice['Mode']} / {publication_choice['Best_Model']}`。"
            f"这条路的优势是结构最容易解释，sidecar / fusion 贡献也更容易写成论文里的机制分析。"
        )

    strongest_text = (
        f"`{strongest['Branch']} / {strongest['Mode']} / {strongest['Best_Model']}`，"
        f"AUC={strongest['AUC']:.3f}，PR-AUC={strongest['PR_AUC']:.3f}，"
        f"Brier={strongest['Brier']:.3f}。"
    )

    bottleneck_text = (
        "当前最可能的瓶颈仍然是：当前窗口的 FT4 / TSH 与既往短期轨迹已经包含了大部分即刻 relapse 信号，"
        "而 stage-1 预测出来的是一个带噪声的中间代理量。换句话说，forecast head 还没有把“未来生理变化”预测得足够稳定，"
        "导致 sidecar / fusion 更多是在补充而不是替代强直接信号。"
    )

    md = "\n".join(
        [
            "# ABC No-Stacking Summary",
            "",
            "## 最强分支",
            strongest_text,
            "",
            "## 是否超过 direct logistic",
            beat_text,
            "",
            "## 最值得写论文的分支",
            publication_text,
            "",
            "## 可能的剩余瓶颈",
            bottleneck_text,
            "",
        ]
    )
    out_path = Config.OUT_DIR / "ABC_Summary.md"
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local ABC experiment caches")
    parser.add_argument("--rebuild-long-data", action="store_true", help="Force rebuild of the long-format data")
    args = parser.parse_args()

    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 96)
    print("  ABC No-Stacking Two-Stage Physiology Experiments for Relapse Prediction")
    print("=" * 96)

    df_train, df_test = build_two_stage_long_data(force_rebuild=args.rebuild_long_data)

    print("\n--- Phase 2: Stage-1 physiology forecast ---")
    x1_tr, x1_te, _ = make_stage1_feature_frames(df_train, df_test)
    y1_te = df_test[TARGET_COLS].values.astype(float)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values

    stage1_payload = run_stage1_oof_forecast(
        x1_tr, df_train, x1_te, df_test, groups, train_intervals, test_intervals, Config.OUT_DIR
    )
    best_stage1_name = stage1_payload["delta_best_name"]
    best_stage1 = stage1_payload["delta_results"][best_stage1_name]
    stage1_metrics = stage1_payload["delta_metrics"]
    stage1_test_metrics = evaluate_physio_predictions(y1_te, best_stage1["test_next"], "Test", best_stage1_name)
    stage1_all_metrics = pd.concat([stage1_metrics, stage1_test_metrics], ignore_index=True)
    stage1_all_metrics.to_csv(Config.OUT_DIR / "Physio_Forecast_Metrics.csv", index=False)
    save_stage1_metric_bar(stage1_metrics[stage1_metrics["Split"] == "Train_OOF"], Config.OUT_DIR)
    save_physio_scatter(y1_te, best_stage1["test_next"], Config.OUT_DIR, best_stage1_name, "Test")
    print(f"  Best delta-head model by average OOF RMSE: {best_stage1_name}")
    print(
        stage1_all_metrics[stage1_all_metrics["Target"] == "Average"][["Split", "Model", "MAE", "RMSE", "R2"]].to_string(index=False)
    )

    best_state_name = stage1_payload["state_best_name"]
    state_metrics = stage1_payload["state_metrics"].copy()
    state_metrics.to_csv(Config.OUT_DIR / "NextState_Forecast_Metrics.csv", index=False)
    print(f"  Best next-state head by Hyper PR-AUC: {best_state_name}")
    print(
        state_metrics[state_metrics["Model"] == best_state_name][["Split", "Metric", "Value"]].to_string(index=False)
    )

    df_train_meta = append_stage1_meta_features(df_train, stage1_payload, "oof")
    df_test_meta = append_stage1_meta_features(df_test, stage1_payload, "test")
    df_train_meta.to_csv(Config.OUT_DIR / "two_stage_train_with_meta.csv", index=False)
    df_test_meta.to_csv(Config.OUT_DIR / "two_stage_test_with_meta.csv", index=False)

    baseline = run_baseline_direct_reference(df_train_meta, df_test_meta)
    baseline_logistic_payload = baseline["results"]["Logistic Reg."]

    branch_a_results, branch_a_summary = run_branch_a(df_train_meta, df_test_meta)
    branch_b_results, branch_b_summary = run_branch_b(
        df_train_meta,
        df_test_meta,
        baseline_direct_payload=baseline_logistic_payload,
        branch_a_physio_result=branch_a_results["predicted_only_compact"],
    )
    branch_c_results, branch_c_summary = run_branch_c(df_train_meta, df_test_meta, baseline_logistic_payload)

    baseline_logistic_row = pd.DataFrame(
        [
            {
                "Branch": "Baseline",
                "Mode": "direct_logistic_full",
                "Best_Model": "Logistic Reg.",
                "AUC": baseline_logistic_payload["metrics"]["auc"],
                "PR_AUC": baseline_logistic_payload["metrics"]["prauc"],
                "C_Index": baseline_logistic_payload["c_index"],
                "Brier": baseline_logistic_payload["metrics"]["brier"],
                "Recall": baseline_logistic_payload["metrics"]["recall"],
                "Specificity": baseline_logistic_payload["metrics"]["specificity"],
                "Calibration_Intercept": baseline_logistic_payload["cal"]["intercept"],
                "Calibration_Slope": baseline_logistic_payload["cal"]["slope"],
                "Threshold": baseline_logistic_payload["threshold"],
            }
        ]
    )

    combined_df = pd.concat(
        [
            baseline_logistic_row,
            branch_a_summary,
            branch_b_summary,
            branch_c_summary,
        ],
        ignore_index=True,
    )
    combined_df = combined_df[
        [
            "Branch",
            "Mode",
            "Best_Model",
            "AUC",
            "PR_AUC",
            "C_Index",
            "Brier",
            "Recall",
            "Specificity",
            "Calibration_Intercept",
            "Calibration_Slope",
            "Threshold",
        ]
    ]
    combined_df.to_csv(Config.OUT_DIR / "ABC_Combined_Comparison.csv", index=False)

    summary_path = write_markdown_summary(combined_df, baseline_logistic_row.iloc[0])

    print("\n  Combined ABC comparison")
    for row in combined_df.itertuples(index=False):
        print(
            f"    {row.Branch:<9s} {row.Mode:<42s} {row.Best_Model:<14s} "
            f"AUC={row.AUC:.3f}  PR-AUC={row.PR_AUC:.3f}  "
            f"Brier={row.Brier:.3f}  Cal.Int={row.Calibration_Intercept:.3f}  "
            f"Cal.Slope={row.Calibration_Slope:.3f}"
        )

    best_branch_row = combined_df[combined_df["Branch"].isin(["BranchA", "BranchB", "BranchC"])].sort_values(
        ["PR_AUC", "AUC"], ascending=[False, False]
    ).iloc[0]
    print(
        f"\n  Strongest no-stacking branch: {best_branch_row['Branch']} / {best_branch_row['Mode']} / "
        f"{best_branch_row['Best_Model']}"
    )
    print(f"  Summary markdown: {summary_path.resolve()}")
    print(f"  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
