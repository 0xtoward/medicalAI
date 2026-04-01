"""Helpers for two-stage physiology forecasting and relapse prediction."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from utils.config import SEED, STATIC_NAMES, STATE_NAMES
from utils.plot_style import PRIMARY_BLUE, PRIMARY_TEAL, SOFT_BLUE, TEXT_DARK, TEXT_MID

TARGET_COLS = ["FT3_Next", "FT4_Next", "logTSH_Next"]
DELTA_TARGET_COLS = ["Delta_FT3_Next", "Delta_FT4_Next", "Delta_logTSH_Next"]
CURRENT_BY_TARGET = {
    "FT3_Next": "FT3_Current",
    "FT4_Next": "FT4_Current",
    "logTSH_Next": "logTSH_Current",
}
CURRENT_LAB_COLS = [CURRENT_BY_TARGET[target] for target in TARGET_COLS]
STATE_TO_CODE = {name: idx for idx, name in enumerate(STATE_NAMES)}
CODE_TO_STATE = {idx: name for name, idx in STATE_TO_CODE.items()}
BINARY_STATE_NAMES = ["NonHyper", "Hyper"]
STAGE1_META_COLS = [
    "P_next_hyper",
    "P_next_nonhyper",
    "NextState_Entropy",
    "NextState_MaxProb",
    "Pred_Delta_FT3",
    "Pred_Delta_FT4",
    "Pred_Delta_logTSH",
    "Pred_Std_FT3",
    "Pred_Std_FT4",
    "Pred_Std_logTSH",
    "Pred_FT4_Margin_To_Hyper",
    "Pred_logTSH_Margin_To_Hyper",
    "Pred_FT4_HyperZone",
    "Pred_logTSH_HyperZone",
    "Pred_AnyHyperZone",
    "Pred_BothHyperZone",
    "Pred_HyperEvidence",
]
META_FEATURE_PRIORITY = [
    "P_next_hyper",
    "NextState_Entropy",
    "NextState_MaxProb",
    "Pred_HyperEvidence",
    "Pred_Delta_FT4",
    "Pred_Delta_logTSH",
    "Pred_FT4_Margin_To_Hyper",
    "Pred_logTSH_Margin_To_Hyper",
]
META_VARIANTS = {
    "state_only": META_FEATURE_PRIORITY[:4],
    "state_delta": META_FEATURE_PRIORITY[:6],
    "state_delta_uncertainty": META_FEATURE_PRIORITY[:8] + ["Pred_Std_FT4", "Pred_Std_logTSH"],
    "state_rule": [
        "P_next_hyper",
        "NextState_MaxProb",
        "Pred_FT4_Margin_To_Hyper",
        "Pred_logTSH_Margin_To_Hyper",
        "Pred_AnyHyperZone",
        "Pred_HyperEvidence",
    ],
}

HISTORY_FEATURE_COLS = [
    "FT3_Baseline",
    "FT4_Baseline",
    "logTSH_Baseline",
    "FT3_Prev",
    "FT4_Prev",
    "logTSH_Prev",
    "FT3_Mean_ToDate",
    "FT4_Mean_ToDate",
    "logTSH_Mean_ToDate",
    "FT3_STD_ToDate",
    "FT4_STD_ToDate",
    "logTSH_STD_ToDate",
]

CURRENT_CONTEXT_COLS = STATIC_NAMES + [
    "Start_Time",
    "Stop_Time",
    "Interval_Width",
    "Prior_Relapse_Count",
    "Event_Order",
    "FT3_Current",
    "FT4_Current",
    "logTSH_Current",
    *HISTORY_FEATURE_COLS,
    "Ever_Hyper_Before",
    "Ever_Hypo_Before",
    "Time_In_Normal",
    "Delta_FT4_k0",
    "Delta_TSH_k0",
    "Delta_FT4_1step",
    "Delta_TSH_1step",
]


def _format_p_value(p_value):
    if p_value < 1e-3:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _safe_model_tag(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def _existing_columns(df, columns):
    return [col for col in columns if col in df.columns]


def _state_entropy(proba):
    proba = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    ent = -np.sum(proba * np.log(proba), axis=1)
    return ent / np.log(proba.shape[1])


def _binary_state_targets(df):
    return (df["Next_State"].astype(str).values == "Hyper").astype(int)


def _binary_proba_2col(proba):
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    if arr.shape[1] == 1:
        arr = np.column_stack([1.0 - arr[:, 0], arr[:, 0]])
    return np.clip(arr, 1e-8, 1 - 1e-8)


def _build_delta_targets(df):
    return np.column_stack(
        [
            df["FT3_Next"].values.astype(float) - df["FT3_Current"].values.astype(float),
            df["FT4_Next"].values.astype(float) - df["FT4_Current"].values.astype(float),
            df["logTSH_Next"].values.astype(float) - df["logTSH_Current"].values.astype(float),
        ]
    )


def _reconstruct_next_labs(df, delta_pred):
    delta_pred = np.asarray(delta_pred, dtype=float)
    current = df[CURRENT_LAB_COLS].values.astype(float)
    return current + delta_pred


def _fit_empirical_hyper_margin_reference(train_df):
    refs = {}
    y_hyper = (train_df["Next_State"].astype(str).values == "Hyper").astype(int)
    for source_col, short_name in [("FT4_Next", "FT4"), ("logTSH_Next", "logTSH")]:
        values = train_df[source_col].values.astype(float)
        pos = values[y_hyper == 1]
        neg = values[y_hyper == 0]
        if len(pos) == 0 or len(neg) == 0:
            refs[short_name] = {"threshold": float(np.nanmedian(values)), "direction": 1.0}
            continue
        pos_median = float(np.nanmedian(pos))
        neg_median = float(np.nanmedian(neg))
        direction = 1.0 if pos_median >= neg_median else -1.0
        refs[short_name] = {"threshold": 0.5 * (pos_median + neg_median), "direction": direction}
    return refs


def _compute_margin(values, ref):
    return ref["direction"] * (np.asarray(values, dtype=float) - ref["threshold"])


def get_meta_feature_subset(df, variant="state_delta_uncertainty", max_features=None):
    """Return a compact priority-ordered subset of stage-1 meta-features."""
    if variant == "all_priority":
        priority = META_FEATURE_PRIORITY
    else:
        priority = META_VARIANTS.get(variant, META_VARIANTS["state_delta_uncertainty"])
    cols = [col for col in priority if col in df.columns]
    if max_features is not None:
        cols = cols[: int(max_features)]
    return cols


def build_next_physio_targets(X_d_next, S_matrix, pids, target_k=None, row_ids=None):
    """Build next-window physiology targets for rows currently in Normal."""
    rows = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)
    if row_ids is None:
        row_ids = np.arange(n_samples)

    for i in range(n_samples):
        pid = pids[i]
        row_id = int(row_ids[i])
        for k in k_range:
            if int(S_matrix[i, k]) != 1:
                continue
            labs_next = X_d_next[i, k + 1, :]
            rows.append(
                {
                    "Patient_ID": pid,
                    "Source_Row": row_id,
                    "Interval_ID": k,
                    "FT3_Next": float(labs_next[0]),
                    "FT4_Next": float(labs_next[1]),
                    "logTSH_Next": float(np.log1p(np.clip(labs_next[2], 0, None))),
                }
            )
    cols = ["Patient_ID", "Source_Row", "Interval_ID", "FT3_Next", "FT4_Next", "logTSH_Next"]
    return pd.DataFrame(rows, columns=cols)


def build_physio_history_features(X_d, S_matrix, pids, target_k=None, row_ids=None):
    """Attach raw trajectory summaries for the current landmark."""
    rows = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)
    if row_ids is None:
        row_ids = np.arange(n_samples)

    for i in range(n_samples):
        pid = pids[i]
        row_id = int(row_ids[i])
        for k in k_range:
            if int(S_matrix[i, k]) != 1:
                continue

            hist = X_d[i, : k + 1, :]
            prev = X_d[i, k - 1, :] if k > 0 else X_d[i, k, :]
            base = X_d[i, 0, :]
            hist_log_tsh = np.log1p(np.clip(hist[:, 2], 0, None))

            rows.append(
                {
                    "Patient_ID": pid,
                    "Source_Row": row_id,
                    "Interval_ID": k,
                    "FT3_Baseline": float(base[0]),
                    "FT4_Baseline": float(base[1]),
                    "logTSH_Baseline": float(np.log1p(np.clip(base[2], 0, None))),
                    "FT3_Prev": float(prev[0]),
                    "FT4_Prev": float(prev[1]),
                    "logTSH_Prev": float(np.log1p(np.clip(prev[2], 0, None))),
                    "FT3_Mean_ToDate": float(np.mean(hist[:, 0])),
                    "FT4_Mean_ToDate": float(np.mean(hist[:, 1])),
                    "logTSH_Mean_ToDate": float(np.mean(hist_log_tsh)),
                    "FT3_STD_ToDate": float(np.std(hist[:, 0])),
                    "FT4_STD_ToDate": float(np.std(hist[:, 1])),
                    "logTSH_STD_ToDate": float(np.std(hist_log_tsh)),
                }
            )

    cols = ["Patient_ID", "Source_Row", "Interval_ID", *HISTORY_FEATURE_COLS]
    return pd.DataFrame(rows, columns=cols)


def make_stage1_feature_frames(train_df, test_df):
    """Create stage-1 regression features from current known information."""
    feat_cols = _existing_columns(train_df, CURRENT_CONTEXT_COLS)
    cat_cols = [col for col in ["Interval_Name", "Prev_State"] if col in train_df.columns]
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


def get_stage1_models():
    """Candidate delta-regression models for future physiology."""
    return {
        "Ridge": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=1.0, random_state=SEED)),
                ]
            ),
            "per_interval": False,
        },
        "ElasticNet": {
            "model": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "reg",
                        MultiOutputRegressor(
                            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=5000)
                        ),
                    ),
                ]
            ),
            "per_interval": False,
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            ),
            "per_interval": False,
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            ),
            "per_interval": False,
        },
        "RandomForest_ByWindow": {
            "model": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            ),
            "per_interval": True,
        },
        "ExtraTrees_ByWindow": {
            "model": ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=1,
            ),
            "per_interval": True,
        },
    }


def get_next_state_models():
    """Candidate binary Hyper-vs-nonHyper classifiers."""
    return {
        "Binary LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "LightGBM": LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
        ),
    }


def evaluate_physio_predictions(y_true, y_pred, split_name, model_name):
    """Summarize per-target and average regression error."""
    rows = []
    for idx, target in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        rows.append(
            {
                "Split": split_name,
                "Model": model_name,
                "Target": target,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )
    df = pd.DataFrame(rows)
    avg = {
        "Split": split_name,
        "Model": model_name,
        "Target": "Average",
        "MAE": df["MAE"].mean(),
        "RMSE": df["RMSE"].mean(),
        "R2": df["R2"].mean(),
    }
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def evaluate_next_state_predictions(y_true, proba, split_name, model_name):
    """Summarize Hyper-vs-nonHyper state classification quality."""
    y_true = np.asarray(y_true, dtype=int)
    proba = _binary_proba_2col(proba)
    p_hyper = proba[:, 1]
    try:
        auc = roc_auc_score(y_true, p_hyper)
    except Exception:
        auc = np.nan
    hyper_pr_auc = average_precision_score(y_true, p_hyper)
    rows = [
        {"Split": split_name, "Model": model_name, "Metric": "AUC", "Value": auc},
        {"Split": split_name, "Model": model_name, "Metric": "Hyper_PR_AUC", "Value": hyper_pr_auc},
        {"Split": split_name, "Model": model_name, "Metric": "LogLoss", "Value": log_loss(y_true, proba, labels=[0, 1])},
        {
            "Split": split_name,
            "Model": model_name,
            "Metric": "Mean_MaxProb",
            "Value": float(np.mean(np.maximum(p_hyper, 1.0 - p_hyper))),
        },
        {"Split": split_name, "Model": model_name, "Metric": "Mean_P_Hyper", "Value": float(np.mean(p_hyper))},
    ]
    return pd.DataFrame(rows)


EARLY_INTERVAL_NAMES = {"1M->3M", "3M->6M", "6M->12M"}


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _safe_logit_1d(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibration_model():
    return LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)


def _predict_binary_from_fitted(fitted, X_pred):
    if isinstance(fitted, dict) and fitted.get("type") == "constant":
        return np.full(len(X_pred), float(fitted["p"]), dtype=float)
    return _binary_proba_2col(fitted.predict_proba(X_pred))[:, 1]


def _fit_predict_binary(base_model, X_fit, y_fit, X_pred):
    y_fit = np.asarray(y_fit, dtype=int)
    prevalence = float(np.clip(np.mean(y_fit), 1e-6, 1 - 1e-6))
    if len(np.unique(y_fit)) < 2:
        return {"type": "constant", "p": prevalence}, np.full(len(X_pred), prevalence, dtype=float)
    try:
        model = clone(base_model)
        model.fit(X_fit, y_fit)
        return model, _binary_proba_2col(model.predict_proba(X_pred))[:, 1]
    except Exception:
        return {"type": "constant", "p": prevalence}, np.full(len(X_pred), prevalence, dtype=float)


def _run_binary_head_oof(base_model, X_train, y_train, X_test, groups):
    y_train = np.asarray(y_train, dtype=int)
    groups = np.asarray(groups)
    n_splits = min(3, len(pd.Index(groups).drop_duplicates()))
    if n_splits < 2:
        fitted, test_proba = _fit_predict_binary(base_model, X_train, y_train, X_test)
        train_fit = _predict_binary_from_fitted(fitted, X_train)
        return {
            "model": fitted,
            "train_fit_proba": train_fit,
            "oof_proba": train_fit.copy(),
            "test_proba": test_proba,
            "raw_logit_oof": _safe_logit_1d(train_fit),
            "raw_logit_test": _safe_logit_1d(test_proba),
            "calibrated_oof_proba": train_fit.copy(),
            "calibrated_test_proba": test_proba.copy(),
        }

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
        _, fold_proba = _fit_predict_binary(base_model, X_train.iloc[tr_idx], y_train[tr_idx], X_train.iloc[val_idx])
        oof[val_idx] = fold_proba

    fitted, _ = _fit_predict_binary(base_model, X_train, y_train, X_train)
    train_fit = _predict_binary_from_fitted(fitted, X_train)
    test_proba = _predict_binary_from_fitted(fitted, X_test)
    return {
        "model": fitted,
        "train_fit_proba": train_fit,
        "oof_proba": oof,
        "test_proba": test_proba,
        "raw_logit_oof": _safe_logit_1d(oof),
        "raw_logit_test": _safe_logit_1d(test_proba),
        "calibrated_oof_proba": oof.copy(),
        "calibrated_test_proba": test_proba.copy(),
    }


def _route_phase_labels(intervals):
    intervals = pd.Series(intervals).astype(str)
    return np.where(intervals.isin(EARLY_INTERVAL_NAMES), "Early", "Late")


def _build_score_design(train_df, test_df, train_logit, test_logit, mode="interval_intercept"):
    tr = pd.DataFrame({"Stage1_Logit": np.asarray(train_logit, dtype=float)})
    te = pd.DataFrame({"Stage1_Logit": np.asarray(test_logit, dtype=float)})

    if mode.startswith("phase_"):
        train_labels = _route_phase_labels(train_df["Interval_Name"].values)
        test_labels = _route_phase_labels(test_df["Interval_Name"].values)
        prefix = "Phase"
    else:
        train_labels = train_df["Interval_Name"].astype(str).values
        test_labels = test_df["Interval_Name"].astype(str).values
        prefix = "Interval"

    dummy_cols = []
    for label in sorted(pd.Index(train_labels).drop_duplicates()):
        name = f"{prefix}_{label}"
        tr[name] = (train_labels == label).astype(float)
        te[name] = (test_labels == label).astype(float)
        dummy_cols.append(name)

    if mode in {"interval_slope", "phase_slope"}:
        for dummy in dummy_cols:
            inter_name = f"{dummy}_x_Stage1_Logit"
            tr[inter_name] = tr[dummy].values * tr["Stage1_Logit"].values
            te[inter_name] = te[dummy].values * te["Stage1_Logit"].values

    keep_cols = [col for col in tr.columns if tr[col].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols]


def _run_score_correction(train_design, y_train, test_design, groups):
    y_train = np.asarray(y_train, dtype=int)
    groups = np.asarray(groups)
    prevalence = float(np.clip(np.mean(y_train), 1e-6, 1 - 1e-6))
    n_splits = min(3, len(pd.Index(groups).drop_duplicates()))
    if n_splits < 2:
        if len(np.unique(y_train)) < 2:
            const = np.full(len(train_design), prevalence, dtype=float)
            return {"type": "constant", "p": prevalence}, const.copy(), const, np.full(len(test_design), prevalence, dtype=float)
        model = _calibration_model()
        try:
            model.fit(train_design, y_train)
            train_fit = model.predict_proba(train_design)[:, 1]
            test_proba = model.predict_proba(test_design)[:, 1]
            return model, train_fit.copy(), train_fit, test_proba
        except Exception:
            const = np.full(len(train_design), prevalence, dtype=float)
            return {"type": "constant", "p": prevalence}, const.copy(), const, np.full(len(test_design), prevalence, dtype=float)

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in gkf.split(train_design, y_train, groups=groups):
        if len(np.unique(y_train[tr_idx])) < 2:
            oof[val_idx] = float(np.clip(np.mean(y_train[tr_idx]), 1e-6, 1 - 1e-6))
            continue
        model = _calibration_model()
        try:
            model.fit(train_design.iloc[tr_idx], y_train[tr_idx])
            oof[val_idx] = model.predict_proba(train_design.iloc[val_idx])[:, 1]
        except Exception:
            oof[val_idx] = float(np.clip(np.mean(y_train[tr_idx]), 1e-6, 1 - 1e-6))

    if len(np.unique(y_train)) < 2:
        const = np.full(len(train_design), prevalence, dtype=float)
        return {"type": "constant", "p": prevalence}, const.copy(), oof, np.full(len(test_design), prevalence, dtype=float)
    fitted = _calibration_model()
    try:
        fitted.fit(train_design, y_train)
        train_fit = fitted.predict_proba(train_design)[:, 1]
        test_proba = fitted.predict_proba(test_design)[:, 1]
        return fitted, train_fit, oof, test_proba
    except Exception:
        const = np.full(len(train_design), prevalence, dtype=float)
        return {"type": "constant", "p": prevalence}, const.copy(), oof, np.full(len(test_design), prevalence, dtype=float)


def _run_partitioned_binary_head(base_model, X_train, y_train, X_test, groups, train_partitions, test_partitions, fallback_pack):
    train_partitions = np.asarray(train_partitions)
    test_partitions = np.asarray(test_partitions)
    y_train = np.asarray(y_train, dtype=int)
    groups = np.asarray(groups)

    train_fit = np.asarray(fallback_pack["train_fit_proba"], dtype=float).copy()
    oof = np.asarray(fallback_pack["oof_proba"], dtype=float).copy()
    test_proba = np.asarray(fallback_pack["test_proba"], dtype=float).copy()
    models = {"fallback": fallback_pack.get("model")}

    for part in sorted(pd.Index(train_partitions).drop_duplicates()):
        fit_mask = train_partitions == part
        pred_mask = test_partitions == part
        if int(np.sum(fit_mask)) < 24:
            continue
        if len(pd.Index(groups[fit_mask]).drop_duplicates()) < 2:
            continue
        if len(np.unique(y_train[fit_mask])) < 2:
            continue
        sub_pack = _run_binary_head_oof(
            base_model,
            X_train.iloc[fit_mask],
            y_train[fit_mask],
            X_test.iloc[pred_mask],
            groups[fit_mask],
        )
        train_fit[fit_mask] = np.asarray(sub_pack["train_fit_proba"], dtype=float)
        oof[fit_mask] = np.asarray(sub_pack["oof_proba"], dtype=float)
        test_proba[pred_mask] = np.asarray(sub_pack["test_proba"], dtype=float)
        models[str(part)] = sub_pack["model"]

    return {
        "model": models,
        "train_fit_proba": train_fit,
        "oof_proba": oof,
        "test_proba": test_proba,
        "raw_logit_oof": _safe_logit_1d(oof),
        "raw_logit_test": _safe_logit_1d(test_proba),
        "calibrated_oof_proba": oof.copy(),
        "calibrated_test_proba": test_proba.copy(),
    }


def _build_rule_targets(df, refs):
    ft4_high = (_compute_margin(df["FT4_Next"].values.astype(float), refs["FT4"]) > 0).astype(int)
    tsh_suppressed = (_compute_margin(df["logTSH_Next"].values.astype(float), refs["logTSH"]) > 0).astype(int)
    any_hyper_rule = np.maximum(ft4_high, tsh_suppressed)
    return {
        "Any_hyper_rule_next": any_hyper_rule,
        "FT4_high_next": ft4_high,
        "TSH_suppressed_next": tsh_suppressed,
    }


def _combine_binary_packs(main_pack, aux_packs, aux_weights, tag="combined"):
    main_weight = 1.0 - float(np.sum(aux_weights))
    aux_weights = [float(w) for w in aux_weights]
    oof_logit = main_weight * _safe_logit_1d(main_pack["oof_proba"])
    test_logit = main_weight * _safe_logit_1d(main_pack["test_proba"])
    train_fit_logit = main_weight * _safe_logit_1d(main_pack["train_fit_proba"])

    for weight, aux_pack in zip(aux_weights, aux_packs):
        oof_logit += weight * _safe_logit_1d(aux_pack["oof_proba"])
        test_logit += weight * _safe_logit_1d(aux_pack["test_proba"])
        train_fit_logit += weight * _safe_logit_1d(aux_pack["train_fit_proba"])

    oof = _sigmoid(oof_logit)
    test_proba = _sigmoid(test_logit)
    train_fit = _sigmoid(train_fit_logit)
    return {
        "model": {
            "type": tag,
            "main_model": main_pack.get("model"),
            "aux_models": [pack.get("model") for pack in aux_packs],
            "aux_weights": aux_weights,
        },
        "train_fit_proba": train_fit,
        "oof_proba": oof,
        "test_proba": test_proba,
        "raw_logit_oof": oof_logit,
        "raw_logit_test": test_logit,
        "calibrated_oof_proba": oof.copy(),
        "calibrated_test_proba": test_proba.copy(),
    }


def _pack_transition_variant(name, model_name, pack, y_train, y_test, notes=""):
    return {
        **pack,
        "variant_name": name,
        "base_model_name": model_name,
        "notes": notes,
        "metrics": pd.concat(
            [
                evaluate_next_state_predictions(y_train, pack["oof_proba"], "Train_OOF", name),
                evaluate_next_state_predictions(y_test, pack["test_proba"], "Test", name),
            ],
            ignore_index=True,
        ),
    }


def _select_state_anchor_model(state_metrics):
    score_df = (
        state_metrics[state_metrics["Split"] == "Train_OOF"]
        .pivot_table(index="Model", columns="Metric", values="Value")
        .reset_index()
    )
    score_df = score_df[score_df["Model"] != "EnsembleMean"].copy()
    if len(score_df) == 0:
        return list(get_next_state_models().keys())[0]
    score_df = score_df.sort_values(["Hyper_PR_AUC", "AUC"], ascending=[False, False])
    return str(score_df.iloc[0]["Model"])


def _run_transition_variant_suite(
    X_train,
    train_df,
    X_test,
    test_df,
    groups,
    train_intervals,
    test_intervals,
    state_anchor_name,
    state_results,
    hyper_margin_reference,
):
    y_train = _binary_state_targets(train_df)
    y_test = _binary_state_targets(test_df)
    refs = hyper_margin_reference
    base_model = get_next_state_models()[state_anchor_name]
    baseline_state = state_results[state_anchor_name]
    baseline_pack = {
        "model": baseline_state["model"],
        "train_fit_proba": np.asarray(baseline_state["train_fit_proba"], dtype=float),
        "oof_proba": np.asarray(_binary_proba_2col(baseline_state["oof_proba"])[:, 1], dtype=float),
        "test_proba": np.asarray(_binary_proba_2col(baseline_state["test_proba"])[:, 1], dtype=float),
        "raw_logit_oof": _safe_logit_1d(_binary_proba_2col(baseline_state["oof_proba"])[:, 1]),
        "raw_logit_test": _safe_logit_1d(_binary_proba_2col(baseline_state["test_proba"])[:, 1]),
        "calibrated_oof_proba": np.asarray(_binary_proba_2col(baseline_state["oof_proba"])[:, 1], dtype=float),
        "calibrated_test_proba": np.asarray(_binary_proba_2col(baseline_state["test_proba"])[:, 1], dtype=float),
    }

    results = {
        "binary_interval_intercept": _pack_transition_variant(
            "binary_interval_intercept",
            state_anchor_name,
            baseline_pack,
            y_train,
            y_test,
            notes="shared binary head with interval intercepts from the pooled stage-1 frame",
        )
    }

    slope_tr, slope_te = _build_score_design(
        train_df,
        test_df,
        baseline_pack["raw_logit_oof"],
        baseline_pack["raw_logit_test"],
        mode="interval_slope",
    )
    slope_model, slope_train_fit, slope_oof, slope_test = _run_score_correction(slope_tr, y_train, slope_te, groups)
    results["binary_interval_slope"] = _pack_transition_variant(
        "binary_interval_slope",
        state_anchor_name,
        {
            "model": slope_model,
            "train_fit_proba": slope_train_fit,
            "oof_proba": slope_oof,
            "test_proba": slope_test,
            "raw_logit_oof": baseline_pack["raw_logit_oof"],
            "raw_logit_test": baseline_pack["raw_logit_test"],
            "calibrated_oof_proba": slope_oof,
            "calibrated_test_proba": slope_test,
        },
        y_train,
        y_test,
        notes="interval-specific score slope correction on the shared binary head",
    )

    cal_tr, cal_te = _build_score_design(
        train_df,
        test_df,
        baseline_pack["raw_logit_oof"],
        baseline_pack["raw_logit_test"],
        mode="interval_intercept",
    )
    cal_model, cal_train_fit, cal_oof, cal_test = _run_score_correction(cal_tr, y_train, cal_te, groups)
    results["binary_interval_intercept_calibrated"] = _pack_transition_variant(
        "binary_interval_intercept_calibrated",
        state_anchor_name,
        {
            "model": cal_model,
            "train_fit_proba": cal_train_fit,
            "oof_proba": cal_oof,
            "test_proba": cal_test,
            "raw_logit_oof": baseline_pack["raw_logit_oof"],
            "raw_logit_test": baseline_pack["raw_logit_test"],
            "calibrated_oof_proba": cal_oof,
            "calibrated_test_proba": cal_test,
        },
        y_train,
        y_test,
        notes="shared-slope interval-specific intercept calibration on the shared binary head",
    )

    phase_train = _route_phase_labels(train_intervals)
    phase_test = _route_phase_labels(test_intervals)
    early_late_pack = _run_partitioned_binary_head(
        base_model,
        X_train,
        y_train,
        X_test,
        groups,
        phase_train,
        phase_test,
        baseline_pack,
    )
    results["binary_early_late"] = _pack_transition_variant(
        "binary_early_late",
        state_anchor_name,
        early_late_pack,
        y_train,
        y_test,
        notes="two routed binary heads for early and late follow-up windows",
    )

    rule_targets = _build_rule_targets(train_df, refs)
    rule_pack = _run_binary_head_oof(base_model, X_train, rule_targets["Any_hyper_rule_next"], X_test, groups)
    combined_rule_pack = _combine_binary_packs(baseline_pack, [rule_pack], [0.35], tag="rule_aux")
    results["binary_rule_aux"] = _pack_transition_variant(
        "binary_rule_aux",
        state_anchor_name,
        combined_rule_pack,
        y_train,
        y_test,
        notes="main hyper head plus Any_hyper_rule_next auxiliary score, weight=0.35",
    )

    full_window_pack = _run_partitioned_binary_head(
        base_model,
        X_train,
        y_train,
        X_test,
        groups,
        np.asarray(train_intervals),
        np.asarray(test_intervals),
        baseline_pack,
    )
    results["binary_full_5window"] = _pack_transition_variant(
        "binary_full_5window",
        state_anchor_name,
        full_window_pack,
        y_train,
        y_test,
        notes="five routed binary heads, one per follow-up window with shared fallback",
    )

    phase_cal_tr, phase_cal_te = _build_score_design(
        train_df,
        test_df,
        early_late_pack["raw_logit_oof"],
        early_late_pack["raw_logit_test"],
        mode="phase_intercept",
    )
    phase_cal_model, phase_cal_train_fit, phase_cal_oof, phase_cal_test = _run_score_correction(
        phase_cal_tr, y_train, phase_cal_te, groups
    )
    results["binary_early_late_calibrated"] = _pack_transition_variant(
        "binary_early_late_calibrated",
        state_anchor_name,
        {
            "model": phase_cal_model,
            "train_fit_proba": phase_cal_train_fit,
            "oof_proba": phase_cal_oof,
            "test_proba": phase_cal_test,
            "raw_logit_oof": early_late_pack["raw_logit_oof"],
            "raw_logit_test": early_late_pack["raw_logit_test"],
            "calibrated_oof_proba": phase_cal_oof,
            "calibrated_test_proba": phase_cal_test,
        },
        y_train,
        y_test,
        notes="early/late routed heads plus phase-specific intercept calibration",
    )

    for weight in [0.2, 0.35, 0.5]:
        results[f"binary_rule_aux_w{str(weight).replace('.', '_')}"] = _pack_transition_variant(
            f"binary_rule_aux_w{str(weight).replace('.', '_')}",
            state_anchor_name,
            _combine_binary_packs(baseline_pack, [rule_pack], [weight], tag="rule_aux_weight_sweep"),
            y_train,
            y_test,
            notes=f"main hyper head plus Any_hyper_rule_next auxiliary score, weight={weight:.2f}",
        )

    ft4_pack = _run_binary_head_oof(base_model, X_train, rule_targets["FT4_high_next"], X_test, groups)
    tsh_pack = _run_binary_head_oof(base_model, X_train, rule_targets["TSH_suppressed_next"], X_test, groups)
    results["binary_threshold_twin_aux"] = _pack_transition_variant(
        "binary_threshold_twin_aux",
        state_anchor_name,
        _combine_binary_packs(baseline_pack, [ft4_pack, tsh_pack], [0.175, 0.175], tag="threshold_twin_aux"),
        y_train,
        y_test,
        notes="main hyper head plus FT4_high_next and TSH_suppressed_next auxiliary scores",
    )

    metrics_df = pd.concat([payload["metrics"] for payload in results.values()], ignore_index=True)
    return results, metrics_df


def _fit_stage1_model(
    base_model,
    X_fit,
    y_fit,
    X_pred,
    fit_intervals=None,
    pred_intervals=None,
    per_interval=False,
):
    """Fit a global or interval-specific regression model and return predictions."""
    if not per_interval:
        model = clone(base_model)
        model.fit(X_fit, y_fit)
        return model, model.predict(X_pred)

    global_model = clone(base_model)
    global_model.fit(X_fit, y_fit)
    pred = global_model.predict(X_pred)
    fitted = {"__global__": global_model}

    fit_intervals = np.asarray(fit_intervals)
    pred_intervals = np.asarray(pred_intervals)
    for interval_name in sorted(pd.Index(fit_intervals).dropna().unique()):
        fit_mask = fit_intervals == interval_name
        pred_mask = pred_intervals == interval_name
        if int(fit_mask.sum()) < 24 or not np.any(pred_mask):
            continue
        local_model = clone(base_model)
        local_model.fit(X_fit.iloc[fit_mask], y_fit[fit_mask])
        pred[pred_mask] = local_model.predict(X_pred.iloc[pred_mask])
        fitted[str(interval_name)] = local_model
    return fitted, pred


def _run_delta_oof_forecast(X_train, train_df, X_test, test_df, groups, train_intervals, test_intervals):
    """Train stage-1 delta models and evaluate on reconstructed next labs."""
    y_delta_train = _build_delta_targets(train_df)
    y_next_train = train_df[TARGET_COLS].values.astype(float)
    n_splits = min(3, len(pd.Index(groups).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)
    metric_frames = []
    results = {}

    for name, payload in get_stage1_models().items():
        base_model = payload["model"]
        per_interval = bool(payload.get("per_interval", False))
        oof_delta = np.zeros_like(y_delta_train, dtype=float)
        for tr_idx, val_idx in gkf.split(X_train, y_delta_train, groups=groups):
            _, fold_pred = _fit_stage1_model(
                base_model,
                X_train.iloc[tr_idx],
                y_delta_train[tr_idx],
                X_train.iloc[val_idx],
                fit_intervals=np.asarray(train_intervals)[tr_idx],
                pred_intervals=np.asarray(train_intervals)[val_idx],
                per_interval=per_interval,
            )
            oof_delta[val_idx] = fold_pred
        fitted_model, test_delta = _fit_stage1_model(
            base_model,
            X_train,
            y_delta_train,
            X_test,
            fit_intervals=train_intervals,
            pred_intervals=test_intervals,
            per_interval=per_interval,
        )
        oof_next = _reconstruct_next_labs(train_df, oof_delta)
        test_next = _reconstruct_next_labs(test_df, test_delta)
        metric_frames.append(evaluate_physio_predictions(y_next_train, oof_next, "Train_OOF", name))
        results[name] = {
            "model": fitted_model,
            "oof_delta": oof_delta,
            "test_delta": test_delta,
            "oof_next": oof_next,
            "test_next": test_next,
            "per_interval": per_interval,
        }

    base_names = list(results.keys())
    if base_names:
        oof_delta_stack = np.stack([results[name]["oof_delta"] for name in base_names], axis=0)
        test_delta_stack = np.stack([results[name]["test_delta"] for name in base_names], axis=0)
        ensemble_name = "EnsembleMean"
        ensemble_oof_delta = np.mean(oof_delta_stack, axis=0)
        ensemble_test_delta = np.mean(test_delta_stack, axis=0)
        results[ensemble_name] = {
            "model": {"members": base_names, "type": "mean_ensemble"},
            "oof_delta": ensemble_oof_delta,
            "test_delta": ensemble_test_delta,
            "oof_next": _reconstruct_next_labs(train_df, ensemble_oof_delta),
            "test_next": _reconstruct_next_labs(test_df, ensemble_test_delta),
            "per_interval": False,
        }
        metric_frames.append(
            evaluate_physio_predictions(y_next_train, results[ensemble_name]["oof_next"], "Train_OOF", ensemble_name)
        )

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    best_name = (
        metrics_df.loc[metrics_df["Target"] == "Average"]
        .sort_values(["RMSE", "MAE"], ascending=[True, True])
        .iloc[0]["Model"]
    )
    return best_name, results, metrics_df


def _run_next_state_oof_forecast(X_train, train_df, X_test, test_df, groups):
    """Train binary Hyper-vs-nonHyper classifiers with grouped OOF probabilities."""
    y_state_train = _binary_state_targets(train_df)
    y_state_test = _binary_state_targets(test_df)
    n_splits = min(3, len(pd.Index(groups).drop_duplicates()))
    gkf = GroupKFold(n_splits=n_splits)
    metric_frames = []
    results = {}

    for name, base_model in get_next_state_models().items():
        oof_proba = np.zeros((len(y_state_train), 2), dtype=float)
        for tr_idx, val_idx in gkf.split(X_train, y_state_train, groups=groups):
            model = clone(base_model)
            model.fit(X_train.iloc[tr_idx], y_state_train[tr_idx])
            oof_proba[val_idx] = _binary_proba_2col(model.predict_proba(X_train.iloc[val_idx]))
        fitted = clone(base_model)
        fitted.fit(X_train, y_state_train)
        train_fit_proba = _binary_proba_2col(fitted.predict_proba(X_train))
        test_proba = _binary_proba_2col(fitted.predict_proba(X_test))
        metric_frames.append(evaluate_next_state_predictions(y_state_train, oof_proba, "Train_OOF", name))
        results[name] = {
            "model": fitted,
            "train_fit_proba": train_fit_proba[:, 1],
            "oof_proba": oof_proba,
            "test_proba": test_proba,
        }

    base_names = list(results.keys())
    if base_names:
        oof_stack = np.stack([results[name]["oof_proba"] for name in base_names], axis=0)
        test_stack = np.stack([results[name]["test_proba"] for name in base_names], axis=0)
        results["EnsembleMean"] = {
            "model": {"members": base_names, "type": "mean_ensemble"},
            "train_fit_proba": np.mean(
                np.stack([_binary_proba_2col(results[name]["train_fit_proba"]) for name in base_names], axis=0),
                axis=0,
            )[:, 1],
            "oof_proba": np.mean(oof_stack, axis=0),
            "test_proba": np.mean(test_stack, axis=0),
        }
        metric_frames.append(evaluate_next_state_predictions(y_state_train, results["EnsembleMean"]["oof_proba"], "Train_OOF", "EnsembleMean"))

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    score_df = metrics_df.pivot_table(index="Model", columns="Metric", values="Value")
    best_name = score_df.sort_values(["Hyper_PR_AUC", "AUC"], ascending=[False, False]).index[0]
    test_metrics = evaluate_next_state_predictions(y_state_test, results[best_name]["test_proba"], "Test", best_name)
    return best_name, results, pd.concat([metrics_df, test_metrics], ignore_index=True)


def run_stage1_oof_forecast(X_train, train_df, X_test, test_df, groups, train_intervals, test_intervals, out_dir):
    """Train stage-1 delta and next-state heads, then return compact meta-feature payloads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_best_name, delta_results, delta_metrics = _run_delta_oof_forecast(
        X_train, train_df, X_test, test_df, groups, train_intervals, test_intervals
    )
    state_best_name, state_results, state_metrics = _run_next_state_oof_forecast(
        X_train, train_df, X_test, test_df, groups
    )
    state_anchor_name = _select_state_anchor_model(state_metrics)
    transition_variant_results, transition_variant_metrics = _run_transition_variant_suite(
        X_train,
        train_df,
        X_test,
        test_df,
        groups,
        train_intervals,
        test_intervals,
        state_anchor_name,
        state_results,
        _fit_empirical_hyper_margin_reference(train_df),
    )
    return {
        "delta_best_name": delta_best_name,
        "delta_results": delta_results,
        "delta_metrics": delta_metrics,
        "state_best_name": state_best_name,
        "state_anchor_name": state_anchor_name,
        "state_results": state_results,
        "state_metrics": state_metrics,
        "hyper_margin_reference": _fit_empirical_hyper_margin_reference(train_df),
        "transition_variant_results": transition_variant_results,
        "transition_variant_metrics": transition_variant_metrics,
    }


def add_predicted_physio_columns(df, pred_array, prefix="Pred"):
    """Append one set of predicted next-window physiology columns to a dataframe copy."""
    out = df.copy()
    out[f"{prefix}_FT3_Next"] = pred_array[:, 0]
    out[f"{prefix}_FT4_Next"] = pred_array[:, 1]
    out[f"{prefix}_logTSH_Next"] = pred_array[:, 2]
    return out


def add_stage1_prediction_family(df, stage1_payload, split_key):
    """Append compact state-aware, delta-aware, and uncertainty-aware meta-features."""
    out = df.copy()
    delta_key = f"{split_key}_delta"
    state_key = f"{split_key}_proba"
    delta_arrays = []
    state_arrays = []

    for model_name in sorted(stage1_payload["delta_results"]):
        payload = stage1_payload["delta_results"][model_name]
        arr = np.asarray(payload.get(delta_key), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == len(DELTA_TARGET_COLS):
            delta_arrays.append(arr)

    for model_name in sorted(stage1_payload["state_results"]):
        payload = stage1_payload["state_results"][model_name]
        arr = np.asarray(payload.get(state_key), dtype=float)
        arr = _binary_proba_2col(arr)
        if arr.ndim == 2 and arr.shape[1] == 2:
            state_arrays.append(arr)

    if not delta_arrays or not state_arrays:
        return out

    delta_stack = np.stack(delta_arrays, axis=0)
    state_stack = np.stack(state_arrays, axis=0)
    mean_delta = np.mean(delta_stack, axis=0)
    std_delta = np.std(delta_stack, axis=0)
    mean_state = np.mean(state_stack, axis=0)

    out["P_next_hyper"] = mean_state[:, 1]
    out["P_next_nonhyper"] = mean_state[:, 0]
    out["NextState_Entropy"] = _state_entropy(mean_state)
    out["NextState_MaxProb"] = np.max(mean_state, axis=1)

    out["Pred_Delta_FT3"] = mean_delta[:, 0]
    out["Pred_Delta_FT4"] = mean_delta[:, 1]
    out["Pred_Delta_logTSH"] = mean_delta[:, 2]
    out["Pred_Std_FT3"] = std_delta[:, 0]
    out["Pred_Std_FT4"] = std_delta[:, 1]
    out["Pred_Std_logTSH"] = std_delta[:, 2]

    pred_ft4_next = out["FT4_Current"].values.astype(float) + mean_delta[:, 1]
    pred_logtsh_next = out["logTSH_Current"].values.astype(float) + mean_delta[:, 2]
    refs = stage1_payload["hyper_margin_reference"]
    out["Pred_FT4_Margin_To_Hyper"] = _compute_margin(pred_ft4_next, refs["FT4"])
    out["Pred_logTSH_Margin_To_Hyper"] = _compute_margin(pred_logtsh_next, refs["logTSH"])
    out["Pred_FT4_HyperZone"] = (out["Pred_FT4_Margin_To_Hyper"].values > 0).astype(float)
    out["Pred_logTSH_HyperZone"] = (out["Pred_logTSH_Margin_To_Hyper"].values > 0).astype(float)
    out["Pred_AnyHyperZone"] = np.maximum(out["Pred_FT4_HyperZone"].values, out["Pred_logTSH_HyperZone"].values)
    out["Pred_BothHyperZone"] = (
        out["Pred_FT4_HyperZone"].values * out["Pred_logTSH_HyperZone"].values
    )
    out["Pred_HyperEvidence"] = out["P_next_hyper"].values * (1.0 + out["Pred_AnyHyperZone"].values)
    return out


def make_stage2_feature_frames(train_df, test_df, mode, base_current_features=None):
    """Create compact stage-2 feature frames for direct, ablation, and fusion-style branches."""
    mode_alias = {
        "predicted_only": "predicted_only_compact",
        "predicted_only_state_delta_uncertainty": "predicted_only_compact",
        "predicted_plus_current": "predicted_plus_current_compact",
    }
    mode = mode_alias.get(mode, mode)

    include_current = mode in {
        "direct",
        "predicted_plus_current_compact",
        "direct_plus_state_only",
        "direct_plus_state_delta",
        "direct_plus_state_delta_uncertainty",
    }
    include_interactions = mode == "predicted_plus_current_compact"

    if mode in {
        "predicted_only_compact",
        "predicted_plus_current_compact",
        "direct_plus_state_delta_uncertainty",
    }:
        meta_variant = "state_delta_uncertainty"
    elif mode == "predicted_only_state_rule":
        meta_variant = "state_rule"
    elif mode == "predicted_only_state_delta":
        meta_variant = "state_delta"
    elif mode == "predicted_only_state_only":
        meta_variant = "state_only"
    elif mode == "direct_plus_state_delta":
        meta_variant = "state_delta"
    elif mode == "direct_plus_state_only":
        meta_variant = "state_only"
    elif mode == "direct":
        meta_variant = None
    else:
        raise ValueError(f"Unknown stage-2 mode: {mode}")

    current_cols = _existing_columns(train_df, CURRENT_CONTEXT_COLS)
    cat_cols = [col for col in ["Interval_Name", "Prev_State"] if col in train_df.columns] if include_current else []

    tr_cur = train_df[current_cols + cat_cols].copy().reset_index(drop=True)
    te_cur = test_df[current_cols + cat_cols].copy().reset_index(drop=True)
    for col in cat_cols:
        cats = sorted(train_df[col].astype(str).unique())
        for cat in cats:
            name = f"{col}_{cat}"
            tr_cur[name] = (train_df[col].astype(str).values == cat).astype(float)
            te_cur[name] = (test_df[col].astype(str).values == cat).astype(float)
    tr_cur = tr_cur.drop(columns=cat_cols).replace([np.inf, -np.inf], np.nan)
    te_cur = te_cur.drop(columns=cat_cols).replace([np.inf, -np.inf], np.nan)
    med_cur = tr_cur.median(numeric_only=True)
    tr_cur = tr_cur.fillna(med_cur)
    te_cur = te_cur.fillna(med_cur)

    if include_current:
        if base_current_features is None:
            current_keep = [c for c in tr_cur.columns if tr_cur[c].nunique(dropna=False) > 1]
        else:
            current_keep = [c for c in base_current_features if c in tr_cur.columns]
        tr_parts = [tr_cur[current_keep].copy()]
        te_parts = [te_cur[current_keep].copy()]
    else:
        tr_parts = []
        te_parts = []

    if meta_variant is not None:
        meta_cols = get_meta_feature_subset(train_df, variant=meta_variant)
        tr_meta = train_df[meta_cols].copy().reset_index(drop=True)
        te_meta = test_df[meta_cols].copy().reset_index(drop=True)
        tr_parts.append(tr_meta)
        te_parts.append(te_meta)

    tr = pd.concat(tr_parts, axis=1) if tr_parts else pd.DataFrame(index=np.arange(len(train_df)))
    te = pd.concat(te_parts, axis=1) if te_parts else pd.DataFrame(index=np.arange(len(test_df)))

    if include_interactions and "P_next_hyper" in tr.columns:
        for base_col in ["FT4_Current", "logTSH_Current", "Time_In_Normal"]:
            if base_col in tr.columns:
                inter_name = f"{base_col}_x_P_next_hyper"
                tr[inter_name] = tr[base_col].values * tr["P_next_hyper"].values
                te[inter_name] = te[base_col].values * te["P_next_hyper"].values
        for dummy_col in [c for c in tr.columns if c.startswith("Interval_Name_")]:
            inter_name = f"{dummy_col}_x_P_next_hyper"
            tr[inter_name] = tr[dummy_col].values * tr["P_next_hyper"].values
            te[inter_name] = te[dummy_col].values * te["P_next_hyper"].values

    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)
    medians = tr.median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)
    keep_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols], keep_cols


def get_stage2_models():
    """Backward-compatible simple classifiers for stage-2 relapse prediction."""
    return {
        "Logistic Reg.": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]
        ),
        "Elastic LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=5000,
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=0.5,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def save_physio_scatter(y_true, y_pred, out_dir, model_name, split_name):
    """Save true-vs-pred scatter plots for the best stage-1 model."""
    out_dir = Path(out_dir)
    fig = plt.figure(figsize=(15.2, 5.0))
    outer = fig.add_gridspec(1, 3, wspace=0.26)
    actual_color = "#f59e0b"
    predicted_color = PRIMARY_BLUE
    ref_color = "#64748b"
    for idx, target in enumerate(TARGET_COLS):
        inner = outer[0, idx].subgridspec(
            2,
            2,
            height_ratios=[0.9, 4.0],
            width_ratios=[4.0, 0.9],
            hspace=0.0,
            wspace=0.0,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax = fig.add_subplot(inner[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax)
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        rmse = float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        corr_r, corr_p = pearsonr(y_true[:, idx], y_pred[:, idx])
        actual = y_true[:, idx]
        predicted = y_pred[:, idx]

        actual_handle = ax.scatter(
            actual,
            actual,
            alpha=0.32,
            s=22,
            facecolors="none",
            edgecolors=actual_color,
            linewidths=0.9,
            label="Actual",
            zorder=2,
        )
        predicted_handle = ax.scatter(
            actual,
            predicted,
            alpha=0.62,
            s=22,
            color=predicted_color,
            edgecolors="white",
            linewidths=0.25,
            label="Predicted",
            zorder=4,
        )
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        pad = 0.03 * (high - low if high > low else 1.0)
        line = ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            "--",
            color=ref_color,
            lw=1.2,
            label="Ideal agreement",
            zorder=1,
        )[0]
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.grid(alpha=0.22)
        legend = ax.legend(
            handles=[actual_handle, predicted_handle, line],
            loc="upper left",
            frameon=False,
            fontsize=6.3,
        )
        ax_top.hist(
            actual,
            bins=18,
            range=(low - pad, high + pad),
            color=actual_color,
            alpha=0.28,
            edgecolor=actual_color,
            linewidth=0.8,
        )
        ax_top.hist(
            predicted,
            bins=18,
            range=(low - pad, high + pad),
            color=predicted_color,
            alpha=0.20,
            edgecolor=predicted_color,
            linewidth=0.8,
        )
        ax_right.hist(
            actual,
            bins=18,
            range=(low - pad, high + pad),
            orientation="horizontal",
            color=actual_color,
            alpha=0.28,
            edgecolor=actual_color,
            linewidth=0.8,
        )
        ax_right.hist(
            predicted,
            bins=18,
            range=(low - pad, high + pad),
            orientation="horizontal",
            color=predicted_color,
            alpha=0.20,
            edgecolor=predicted_color,
            linewidth=0.8,
        )
        ax_top.set_title(target, fontsize=9, pad=4)
        for marginal_ax in (ax_top, ax_right):
            marginal_ax.grid(False)
            marginal_ax.set_facecolor("white")
        ax_top.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_top.tick_params(axis="y", labelleft=False, left=False)
        ax_right.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_right.tick_params(axis="y", labelleft=False, left=False)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["left"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        metric_text = "\n".join(
            [
                f"MAE = {mae:.3f}",
                f"RMSE = {rmse:.3f}",
                f"R^2 = {r2:.3f}",
                f"r = {corr_r:.3f}",
                _format_p_value(corr_p),
            ]
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        metric_x = max(0.04, float(legend_bbox.x0) + 0.01)
        metric_y = max(0.12, float(legend_bbox.y0) - 0.03)
        ax.text(
            metric_x,
            metric_y,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=TEXT_DARK,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=SOFT_BLUE, alpha=0.92),
            zorder=5,
        )
    fig.suptitle(f"{model_name}: actual vs predicted physiology ({split_name})", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.12)
    fig.savefig(out_dir / f"{model_name.replace(' ', '_')}_{split_name}_Scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stage1_metric_bar(metrics_df, out_dir):
    """Save stage-1 forecast error comparison."""
    out_dir = Path(out_dir)
    avg_df = metrics_df[metrics_df["Target"] == "Average"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    rmse_bars = axes[0].bar(avg_df["Model"], avg_df["RMSE"], color=PRIMARY_BLUE, alpha=0.88)
    axes[0].set_title("Average RMSE by stage-1 model", fontsize=9)
    axes[0].tick_params(axis="x", rotation=20, labelsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    r2_bars = axes[1].bar(avg_df["Model"], avg_df["R2"], color=PRIMARY_TEAL, alpha=0.88)
    axes[1].set_title("Average R^2 by stage-1 model", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20, labelsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, float(avg_df["RMSE"].max()) * 1.12)
    axes[1].set_ylim(min(-0.05, float(avg_df["R2"].min()) - 0.03), max(0.08, float(avg_df["R2"].max()) * 1.30))
    for bars, ax in [(rmse_bars, axes[0]), (r2_bars, axes[1])]:
        y_offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=TEXT_MID,
            )
    fig.tight_layout()
    fig.savefig(out_dir / "Stage1_Forecast_Bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
