"""Helpers for two-stage physiology forecasting and relapse prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED, STATIC_NAMES, TIME_STAMPS
from utils.data import apply_missforest, fit_missforest, load_or_fit_depth_imputer, split_imputed

TARGET_COLS = ["FT3_Next", "FT4_Next", "logTSH_Next"]


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


def prepare_physio_target_tables(X_s_raw, ft3_raw, ft4_raw, tsh_raw, pids, tr_mask, te_mask, out_dir):
    """Impute horizon-safe next-window physiology labels without using future eval labels."""
    out_dir = Path(out_dir)
    n_static = X_s_raw.shape[1]
    train_parts = []
    test_parts = []

    for depth in range(1, 7):
        # For interval k=depth-1 -> k+1, targets require lab sequence up to depth+1.
        if depth + 1 > len(TIME_STAMPS):
            continue
        k = depth - 1
        n_lab = depth + 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"
        raw = np.hstack([X_s_raw, ft3_raw[:, :n_lab], ft4_raw[:, :n_lab], tsh_raw[:, :n_lab]])
        imp_path = out_dir / f"missforest_physio_depth{depth}.pkl"
        imputer = load_or_fit_depth_imputer(raw[tr_mask], imp_path)

        filled_tr = apply_missforest(raw[tr_mask], imputer, 0)
        filled_te = apply_missforest(raw[te_mask], imputer, 0)
        _, ft3_tr, ft4_tr, tsh_tr = split_imputed(filled_tr, n_static, n_lab, 0)
        _, ft3_te, ft4_te, tsh_te = split_imputed(filled_te, n_static, n_lab, 0)
        x_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        x_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        train_parts.append((interval_name, x_d_tr))
        test_parts.append((interval_name, x_d_te))

    return train_parts, test_parts


def make_stage1_feature_frames(train_df, test_df):
    """Create stage-1 regression features from current known information."""
    feat_cols = STATIC_NAMES + [
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
    cat_cols = ["Interval_Name", "Prev_State"]
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
    """Candidate next-physiology forecasting models."""
    return {
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=SEED)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", MultiOutputRegressor(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=5000))),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=1,
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


def run_stage1_oof_forecast(X_train, y_train, X_test, groups, out_dir):
    """Train stage-1 models with OOF predictions and return best model outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gkf = GroupKFold(n_splits=3)
    metric_frames = []
    results = {}

    for name, base_model in get_stage1_models().items():
        oof_pred = np.zeros_like(y_train, dtype=float)
        for tr_idx, val_idx in gkf.split(X_train, groups=groups):
            model = base_model
            model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof_pred[val_idx] = model.predict(X_train.iloc[val_idx])
        test_model = base_model
        test_model.fit(X_train, y_train)
        test_pred = test_model.predict(X_test)
        oof_metrics = evaluate_physio_predictions(y_train, oof_pred, "Train_OOF", name)
        metric_frames.append(oof_metrics)
        results[name] = {"model": test_model, "oof_pred": oof_pred, "test_pred": test_pred}

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    best_name = (
        metrics_df.loc[metrics_df["Target"] == "Average"]
        .sort_values(["RMSE", "MAE"], ascending=[True, True])
        .iloc[0]["Model"]
    )
    return best_name, results, metrics_df


def add_predicted_physio_columns(df, pred_array, prefix="Pred"):
    """Append predicted next-window physiology columns to a dataframe copy."""
    out = df.copy()
    out[f"{prefix}_FT3_Next"] = pred_array[:, 0]
    out[f"{prefix}_FT4_Next"] = pred_array[:, 1]
    out[f"{prefix}_logTSH_Next"] = pred_array[:, 2]
    return out


def make_stage2_feature_frames(train_df, test_df, mode):
    """Create stage-2 classification features for three comparison groups."""
    pred_cols = ["Pred_FT3_Next", "Pred_FT4_Next", "Pred_logTSH_Next"]
    current_cols = STATIC_NAMES + [
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
    cat_cols = ["Interval_Name", "Prev_State"]

    if mode == "direct":
        feat_cols = current_cols
    elif mode == "predicted_only":
        feat_cols = pred_cols
        cat_cols = []
    elif mode == "predicted_plus_current":
        feat_cols = current_cols + pred_cols
    else:
        raise ValueError(f"Unknown stage-2 mode: {mode}")

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


def get_stage2_models():
    """Simple interpretable classifiers for stage-2 relapse prediction."""
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
                ("lr", LogisticRegression(max_iter=5000, penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=SEED)),
            ]
        ),
    }


def save_physio_scatter(y_true, y_pred, out_dir, model_name, split_name):
    """Save true-vs-pred scatter plots for the best stage-1 model."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.35, s=14)
        low = float(min(y_true[:, idx].min(), y_pred[:, idx].min()))
        high = float(max(y_true[:, idx].max(), y_pred[:, idx].max()))
        ax.plot([low, high], [low, high], "--", color="gray", lw=1.2)
        ax.set_title(target)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{model_name}: true vs predicted physiology ({split_name})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name.replace(' ', '_')}_{split_name}_Scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stage1_metric_bar(metrics_df, out_dir):
    """Save stage-1 forecast error comparison."""
    out_dir = Path(out_dir)
    avg_df = metrics_df[metrics_df["Target"] == "Average"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(avg_df["Model"], avg_df["RMSE"], color="steelblue")
    axes[0].set_title("Average RMSE by stage-1 model")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(avg_df["Model"], avg_df["R2"], color="coral")
    axes[1].set_title("Average R2 by stage-1 model")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "Stage1_Forecast_Bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
