import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.config import SEED, STATIC_NAMES, STATE_NAMES, TIME_STAMPS
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    clear_pkl_cache,
    fit_missforest,
    load_data as _load_data,
    split_imputed,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import shap
except ImportError:
    shap = None

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:
    BalancedRandomForestClassifier = None

plt.rcParams["font.family"] = "DejaVu Sans"
np.random.seed(SEED)


class Config:
    OUT_DIR = Path("./markov_relapse_result/")
    SHARED_MF_DIR = Path("./multistate_result/")


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=None, label=""):
    """Reuse an existing MissForest cache when the training matrix is identical."""
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None

    if primary.exists():
        print(f"  MissForest {label}: loaded from {primary}")
        return joblib.load(primary)

    if fallback is not None and fallback.exists():
        print(f"  MissForest {label}: reused shared cache {fallback}")
        return joblib.load(fallback)

    print(f"  MissForest {label}: fitting on train ({len(raw_train)} records)...")
    imputer = fit_missforest(raw_train)
    joblib.dump(imputer, primary)
    return imputer


def plot_transition_heatmaps(S_matrix):
    """Plot state transition heatmaps for each consecutive interval."""
    n_intervals = S_matrix.shape[1] - 1
    ncols = min(n_intervals, 3)
    nrows = (n_intervals + ncols - 1) // ncols
    total = np.zeros((3, 3), dtype=int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n_intervals == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_intervals:
            ax.set_visible(False)
            continue
        label = f"{TIME_STAMPS[i]} -> {TIME_STAMPS[i + 1]}"
        mat = np.zeros((3, 3), dtype=int)
        for src in range(3):
            for dst in range(3):
                mat[src, dst] = np.sum((S_matrix[:, i] == src) & (S_matrix[:, i + 1] == dst))
        total += mat
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=STATE_NAMES,
            yticklabels=STATE_NAMES,
            ax=ax,
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("State at t+1")
        ax.set_ylabel("State at t")
        print(
            f"  [{label}] transition: {(mat.sum() - np.trace(mat)) / mat.sum():.1%}  "
            f"relapse(Normal->Hyper): {mat[1, 0]}"
        )

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close(fig)
    print(
        f"\n  Total transitions: {total.sum()}  Stable: {np.trace(total)}  "
        f"Relapse(N->H): {total[1, 0]}  Normal->Hypo: {total[1, 2]}"
    )


def build_transition_rows(S_matrix):
    """Build all one-step transitions for Markov summaries."""
    rows = []
    n_samples, n_timepoints = S_matrix.shape
    for i in range(n_samples):
        for k in range(n_timepoints - 1):
            rows.append(
                {
                    "Interval_ID": k,
                    "Interval_Name": f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}",
                    "From_State": STATE_NAMES[int(S_matrix[i, k])],
                    "To_State": STATE_NAMES[int(S_matrix[i, k + 1])],
                }
            )
    return pd.DataFrame(rows)


def save_markov_transition_table(df_transitions):
    """Save smoothed interval-specific transition probabilities."""
    states = STATE_NAMES
    rows = []
    for interval_name, g in df_transitions.groupby("Interval_Name", sort=False):
        for from_state, g_from in g.groupby("From_State", sort=False):
            counts = g_from["To_State"].value_counts().reindex(states, fill_value=0)
            smoothed = (counts + 1) / (counts.sum() + len(states))
            rows.append(
                {
                    "Interval_Name": interval_name,
                    "From_State": from_state,
                    "N": int(counts.sum()),
                    "P_to_Hyper": float(smoothed["Hyper"]),
                    "P_to_Normal": float(smoothed["Normal"]),
                    "P_to_Hypo": float(smoothed["Hypo"]),
                }
            )
    pd.DataFrame(rows).to_csv(Config.OUT_DIR / "Interval_Markov_Transition_Table.csv", index=False)


def build_markov_relapse_data(X_s, X_d, S_matrix, pids, target_k=None):
    """Build first-order Markov relapse rows: current state must be Normal."""
    records = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)

    for i in range(n_samples):
        pid = pids[i]
        xs = X_s[i]
        for k in k_range:
            current_state = int(S_matrix[i, k])
            if current_state != 1:
                continue

            labs_k = X_d[i, k, :]
            y_relapse = int(S_matrix[i, k + 1] == 0)
            records.append(
                {
                    "Patient_ID": pid,
                    "Interval_ID": k,
                    "Interval_Name": f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}",
                    "Current_State": "Normal",
                    "Y_Relapse": y_relapse,
                    **dict(zip(STATIC_NAMES, xs)),
                    "FT3_Current": labs_k[0],
                    "FT4_Current": labs_k[1],
                    "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                }
            )
    return pd.DataFrame(records)


def fit_markov_baseline(df_train):
    """Fit interval-specific Markov relapse prior P(H at t+1 | Normal at t, interval)."""
    interval_stats = (
        df_train.groupby("Interval_Name")["Y_Relapse"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "events", "count": "n"})
    )
    interval_stats["proba"] = (interval_stats["events"] + 1.0) / (interval_stats["n"] + 2.0)
    mapping = dict(zip(interval_stats["Interval_Name"], interval_stats["proba"]))
    global_rate = float((df_train["Y_Relapse"].sum() + 1.0) / (len(df_train) + 2.0))
    return mapping, global_rate


def predict_markov_baseline(df, mapping, global_rate):
    return df["Interval_Name"].map(mapping).fillna(global_rate).values.astype(float)


def get_tune_specs():
    """Return tuned candidate models for Markov transition prediction."""
    s = SEED
    specs = {
        "Logistic Reg.": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2500, solver="saga", random_state=s)),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "lr__penalty": ["l1", "l2"],
            },
            "#1f77b4",
            "-.",
            12,
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
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5],
                "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "#4c78a8",
            "-.",
            10,
            -1,
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {"svc__C": [0.1, 0.5, 1, 2, 5], "svc__gamma": ["scale", 0.01, 0.05, 0.1]},
            "#f58518",
            ":",
            12,
            -1,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=s, n_jobs=1, class_weight="balanced"),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [3, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5],
            },
            "#54a24b",
            "--",
            16,
            -1,
        ),
        "MLP": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            max_iter=1000,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=50,
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "mlp__alpha": [0.001, 0.01, 0.05, 0.1],
                "mlp__learning_rate_init": [0.0005, 0.001, 0.005],
            },
            "#9d755d",
            ":",
            10,
            -1,
        ),
    }
    if BalancedRandomForestClassifier is not None:
        specs["Balanced RF"] = (
            BalancedRandomForestClassifier(random_state=s, n_jobs=1),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "min_samples_leaf": [1, 3, 5, 10],
                "sampling_strategy": ["all", "not minority"],
            },
            "#72b7b2",
            "--",
            14,
            -1,
        )
    if lgb is not None:
        specs["LightGBM"] = (
            lgb.LGBMClassifier(random_state=s, n_jobs=1, verbosity=-1),
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5, -1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_samples": [5, 10, 20],
            },
            "#b279a2",
            "--",
            16,
            -1,
        )
    return specs


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    rs = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    rs.fit(X_tr_df, y_tr, groups=groups_tr)
    return rs.best_estimator_, rs.best_score_, rs.best_params_


def run_shap_analysis(model_name, model, X_tr_df, feat_names):
    """Run SHAP on the chosen best model when available."""
    if shap is None:
        print("  SHAP skipped: shap is not installed")
        return
    print(f"\n  SHAP analysis on: {model_name}")
    try:
        if model_name in {"Logistic Reg.", "Elastic LR"} and isinstance(model, Pipeline):
            scaler = model.named_steps["scaler"]
            lr = model.named_steps["lr"]
            X_scaled = scaler.transform(X_tr_df)
            explainer = shap.LinearExplainer(lr, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_scaled,
                feature_names=feat_names,
                max_display=20,
                show=False,
            )
        elif hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_tr_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_tr_df,
                feature_names=feat_names,
                max_display=20,
                show=False,
            )
        else:
            bg = shap.sample(X_tr_df, min(100, len(X_tr_df)), random_state=SEED)
            eval_x = X_tr_df.iloc[: min(300, len(X_tr_df))]
            explainer = shap.Explainer(
                lambda data: model.predict_proba(pd.DataFrame(data, columns=feat_names))[:, 1],
                bg,
            )
            shap_values = explainer(eval_x)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                eval_x,
                feature_names=feat_names,
                max_display=20,
                show=False,
            )
        plt.title(f"Discrete-Time Markov Drivers of Relapse ({model_name})", fontsize=14, pad=20)
        plt.savefig(Config.OUT_DIR / "Markov_SHAP_Relapse.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")


def compute_binary_metrics(y_true, proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, proba),
        "prauc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "acc": accuracy_score(y_true, pred),
        "bacc": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }


def compute_calibration_stats(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return {"intercept": np.nan, "slope": np.nan}
    logit_p = np.log(proba / (1 - proba)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        lr.fit(logit_p, y_true)
        return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0])}
    except Exception:
        return {"intercept": np.nan, "slope": np.nan}


def bootstrap_group_cis(eval_df, threshold, group_col="Patient_ID", n_boot=800):
    group_values = eval_df[group_col].dropna().unique()
    grouped = {g: eval_df.loc[eval_df[group_col] == g] for g in group_values}
    rng = np.random.default_rng(SEED)
    store = {k: [] for k in ["auc", "prauc", "brier", "recall", "specificity", "cal_intercept", "cal_slope"]}

    for _ in range(n_boot):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        sampled_df = pd.concat([grouped[g] for g in sampled_groups], ignore_index=True)
        y_boot = sampled_df["Y"].values
        if len(np.unique(y_boot)) < 2:
            continue
        m = compute_binary_metrics(y_boot, sampled_df["proba"].values, threshold)
        cal = compute_calibration_stats(y_boot, sampled_df["proba"].values)
        m["cal_intercept"] = cal["intercept"]
        m["cal_slope"] = cal["slope"]
        for key in store:
            if np.isfinite(m[key]):
                store[key].append(m[key])

    cis = {}
    for key, values in store.items():
        if len(values) < 20:
            cis[key] = (np.nan, np.nan)
        else:
            cis[key] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
    return cis


def format_ci(cis, key):
    low, high = cis.get(key, (np.nan, np.nan))
    if np.isnan(low) or np.isnan(high):
        return "NA"
    return f"[{low:.3f}, {high:.3f}]"


def save_calibration_figure(y_true, proba, title, out_path, n_bins=6):
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, marker="o", lw=2, color="navy", label="Model")
    ax1.set_title(title, fontsize=13)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed event rate")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.hist(proba, bins=20, color="steelblue", alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_dca(y_true, proba, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    if thresholds is None:
        thresholds = np.arange(0.02, 0.51, 0.01)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        odds = thr / (1 - thr)
        rows.append(
            {
                "threshold": thr,
                "model": tp / n - fp / n * odds,
                "treat_all": prevalence - (1 - prevalence) * odds,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_dca_figure(y_true, proba, title, out_path):
    dca_df = compute_dca(y_true, proba)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dca_df["threshold"], dca_df["model"], lw=2.2, color="darkgreen", label="Model")
    ax.plot(dca_df["threshold"], dca_df["treat_all"], lw=1.6, color="crimson", linestyle="--", label="Treat all")
    ax.plot(dca_df["threshold"], dca_df["treat_none"], lw=1.6, color="black", linestyle=":", label="Treat none")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sensitivity_figure(y_true, proba, best_thr, title, out_path):
    thresholds = np.arange(0.02, 0.61, 0.01)
    rows = []
    for thr in thresholds:
        m = compute_binary_metrics(y_true, proba, thr)
        rows.append(
            {"threshold": thr, "recall": m["recall"], "specificity": m["specificity"], "f1": m["f1"]}
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df["threshold"], df["recall"], lw=2, label="Recall")
    ax.plot(df["threshold"], df["specificity"], lw=2, label="Specificity")
    ax.plot(df["threshold"], df["f1"], lw=2, label="F1")
    ax.axvline(best_thr, color="black", linestyle="--", alpha=0.7, label=f"Chosen threshold={best_thr:.2f}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_patient_level(df_long, proba, method="product"):
    tmp = df_long[["Patient_ID", "Interval_Name", "Y_Relapse"]].copy().reset_index(drop=True)
    tmp["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    patient_rows = []
    for pid, g in tmp.groupby("Patient_ID", sort=False):
        probs = g["proba"].values
        if method == "product":
            agg_proba = float(1 - np.prod(1 - probs))
        elif method == "max":
            agg_proba = float(np.max(probs))
        elif method == "mean":
            agg_proba = float(np.mean(probs))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        patient_rows.append({"Patient_ID": pid, "Y": int(g["Y_Relapse"].max()), "proba": agg_proba})
    return pd.DataFrame(patient_rows)


def select_best_threshold(y_true, proba, low=0.02, high=0.95, step=0.01):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(low, high + 1e-9, step):
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def evaluate_window_sensitivity(df_long_test, proba, threshold):
    tmp = df_long_test[["Patient_ID", "Interval_Name", "Y_Relapse"]].copy().reset_index(drop=True)
    tmp["Y"] = tmp["Y_Relapse"].astype(int)
    tmp["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    scenarios = {
        "All_Windows": np.ones(len(tmp), dtype=bool),
        "Exclude_1M_3M": tmp["Interval_Name"].values != "1M->3M",
        "Exclude_12Mplus": ~tmp["Interval_Name"].isin(["12M->18M", "18M->24M"]).values,
        "Only_3Mplus": tmp["Interval_Name"].isin(["3M->6M", "6M->12M", "12M->18M", "18M->24M"]).values,
    }
    rows = []
    for label, mask in scenarios.items():
        sub = tmp.loc[mask].copy()
        if len(sub) == 0 or len(np.unique(sub["Y"])) < 2:
            continue
        m = compute_binary_metrics(sub["Y"].values, sub["proba"].values, threshold)
        rows.append(
            {
                "Scenario": label,
                "Intervals": len(sub),
                "Events": int(sub["Y"].sum()),
                "AUC": m["auc"],
                "PR_AUC": m["prauc"],
                "Brier": m["brier"],
                "Recall": m["recall"],
                "Specificity": m["specificity"],
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate_markov(df_train, df_test):
    feat_cols = STATIC_NAMES + ["FT3_Current", "FT4_Current", "logTSH_Current"]
    interval_cats = sorted(df_train["Interval_Name"].unique())

    def make_features(df):
        out = df[feat_cols].copy().reset_index(drop=True)
        for cat in interval_cats:
            out[f"Window_{cat}"] = (df["Interval_Name"].values == cat).astype(float)
        return out

    X_tr_df = make_features(df_train)
    X_te_df = make_features(df_test)
    feat_names = list(X_tr_df.columns)
    y_tr = df_train["Y_Relapse"].values.astype(int)
    y_te = df_test["Y_Relapse"].values.astype(int)
    groups_tr = df_train["Patient_ID"].values

    print(f"\n  Train: {len(y_tr)} intervals (relapse: {int(y_tr.sum())})")
    print(f"  Test : {len(y_te)} intervals (relapse: {int(y_te.sum())})")
    print(f"  Features: {len(feat_names)}")
    print("  Markov assumption: P(relapse at t+1 | state at t=Normal, X_t, interval)")

    gkf = GroupKFold(n_splits=3)
    results = {}

    print("\n  Pure Markov baseline (interval-specific transition prior)...")
    oof_markov = np.zeros(len(df_train), dtype=float)
    for fold_tr, fold_val in gkf.split(df_train, y_tr, groups=groups_tr):
        mapping, global_rate = fit_markov_baseline(df_train.iloc[fold_tr].copy())
        oof_markov[fold_val] = predict_markov_baseline(df_train.iloc[fold_val].copy(), mapping, global_rate)
    mapping_full, global_full = fit_markov_baseline(df_train)
    test_markov = predict_markov_baseline(df_test, mapping_full, global_full)
    thr_markov = select_best_threshold(y_tr, oof_markov, low=0.02, high=0.60, step=0.01)
    m_markov = compute_binary_metrics(y_te, test_markov, thr_markov)
    results["Pure Markov"] = {
        **m_markov,
        "model": None,
        "color": "#7f7f7f",
        "ls": "-",
        "proba": test_markov,
        "oof_proba": oof_markov,
    }
    print(f"    Pure Markov prior by interval: threshold={thr_markov:.2f}")

    print("\n  Tuning covariate-augmented Markov models...")
    tuned_models = {}
    for name, (base_est, param_grid, color, ls, n_iter, n_jobs) in get_tune_specs().items():
        best_est, best_score, best_params = fit_candidate_model(
            base_est, param_grid, n_iter, n_jobs, gkf, X_tr_df, y_tr, groups_tr
        )
        tuned_models[name] = (best_est, color, ls)
        print(f"    {name:<18s} CV PR-AUC={best_score:.3f}  {best_params}")

    stacking_estimators = [
        ("lr", clone(tuned_models["Logistic Reg."][0])),
        ("rf", clone(tuned_models["Random Forest"][0])),
    ]
    if "LightGBM" in tuned_models:
        stacking_estimators.append(("lgbm", clone(tuned_models["LightGBM"][0])))
    elif "Balanced RF" in tuned_models:
        stacking_estimators.append(("brf", clone(tuned_models["Balanced RF"][0])))

    stacking_est = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )
    tuned_models["Stacking"] = (stacking_est, "#17becf", "-")
    print(f"    {'Stacking':<18s} Built from tuned base learners")

    print("\n  OOF threshold selection + evaluation...")
    print(f"\n{'=' * 108}")
    print(
        f"  {'Model':<18s} {'AUC':>6} {'PR-AUC':>8} {'Thr':>5} {'Acc':>6} "
        f"{'BalAcc':>8} {'F1':>6} {'TP':>4} {'FP':>5} {'FN':>4} {'TN':>5}"
    )
    print(f"{'=' * 108}")

    for name, r in results.items():
        print(
            f"  {name:<18s} {r['auc']:>5.3f} {r['prauc']:>7.3f} {r['threshold']:>4.2f} "
            f"{r['acc']:>5.3f} {r['bacc']:>7.3f} {r['f1']:>5.3f} "
            f"{r['tp']:>4} {r['fp']:>5} {r['fn']:>4} {r['tn']:>5}"
        )

    for name, (model, color, ls) in tuned_models.items():
        oof_proba = np.zeros(len(y_tr), dtype=float)
        for fold_tr, fold_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[fold_tr], y_tr[fold_tr])
            oof_proba[fold_val] = m.predict_proba(X_tr_df.iloc[fold_val])[:, 1]
        best_thr = select_best_threshold(y_tr, oof_proba, low=0.02, high=0.60, step=0.01)
        model.fit(X_tr_df, y_tr)
        proba = model.predict_proba(X_te_df)[:, 1]
        m = compute_binary_metrics(y_te, proba, best_thr)
        results[name] = {
            **m,
            "model": model,
            "color": color,
            "ls": ls,
            "proba": proba,
            "oof_proba": oof_proba,
        }
        print(
            f"  {name:<18s} {m['auc']:>5.3f} {m['prauc']:>7.3f} {best_thr:>4.2f} "
            f"{m['acc']:>5.3f} {m['bacc']:>7.3f} {m['f1']:>5.3f} "
            f"{m['tp']:>4} {m['fp']:>5} {m['fn']:>4} {m['tn']:>5}"
        )

    print(f"{'=' * 108}")
    best_auc_name = max(results, key=lambda k: results[k]["auc"])
    best_pr_name = max(results, key=lambda k: results[k]["prauc"])
    print(f"  Best AUC: {best_auc_name} ({results[best_auc_name]['auc']:.3f})")
    print(f"  Best PR-AUC: {best_pr_name} ({results[best_pr_name]['prauc']:.3f})")

    best_eval_name = best_pr_name
    best_eval = results[best_eval_name]
    eval_df = pd.DataFrame(
        {"Patient_ID": df_test["Patient_ID"].values, "Y": y_te, "proba": best_eval["proba"]}
    )
    cis = bootstrap_group_cis(eval_df, best_eval["threshold"], group_col="Patient_ID")
    cal = compute_calibration_stats(y_te, best_eval["proba"])

    print(f"\n  Interval-level report on best PR-AUC model: {best_eval_name}")
    print(f"    AUC         = {best_eval['auc']:.3f}  95% CI {format_ci(cis, 'auc')}")
    print(f"    PR-AUC      = {best_eval['prauc']:.3f}  95% CI {format_ci(cis, 'prauc')}")
    print(f"    Brier       = {best_eval['brier']:.3f}  95% CI {format_ci(cis, 'brier')}")
    print(f"    Recall      = {best_eval['recall']:.3f}  95% CI {format_ci(cis, 'recall')}")
    print(f"    Specificity = {best_eval['specificity']:.3f}  95% CI {format_ci(cis, 'specificity')}")
    print(f"    Cal. Intcp  = {cal['intercept']:.3f}  95% CI {format_ci(cis, 'cal_intercept')}")
    print(f"    Cal. Slope  = {cal['slope']:.3f}  95% CI {format_ci(cis, 'cal_slope')}")
    print(f"    Threshold   = {best_eval['threshold']:.2f}")

    interval_order = {"0M->1M": 0, "1M->3M": 1, "3M->6M": 2, "6M->12M": 3, "12M->18M": 4, "18M->24M": 5}
    empirical = (
        df_test.groupby("Interval_Name")
        .agg(N=("Y_Relapse", "count"), Events=("Y_Relapse", "sum"), Actual=("Y_Relapse", "mean"))
        .reset_index()
    )
    empirical["sk"] = empirical["Interval_Name"].map(interval_order)
    empirical = empirical.sort_values("sk").reset_index(drop=True)
    x_labels = empirical["Interval_Name"].tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_labels, empirical["Actual"], marker="o", lw=2.5, color="crimson", label="Actual Relapse Rate")
    for idx, row in empirical.iterrows():
        ax.text(idx, row["Actual"] + 0.006, f"{row['Events']:.0f}/{row['N']:.0f}", ha="center", fontsize=8)
    for name, r in results.items():
        tmp = df_test.copy()
        tmp["_pred"] = r["proba"]
        pred_hz = tmp.groupby("Interval_Name")["_pred"].mean().reset_index()
        pred_hz["sk"] = pred_hz["Interval_Name"].map(interval_order)
        pred_hz = pred_hz.sort_values("sk")
        ax.plot(
            x_labels,
            pred_hz["_pred"].values,
            marker="s",
            lw=1.5,
            color=r["color"],
            linestyle=r["ls"],
            alpha=0.85,
            label=f"{name} (AUC={r['auc']:.3f})",
        )
    ax.set_title("Discrete-Time Markov Relapse Curve Comparison", fontsize=13, pad=12)
    ax.set_xlabel("Follow-up Intervals")
    ax.set_ylabel("P(Normal -> Hyper at next interval)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.savefig(Config.OUT_DIR / "Markov_Relapse_Curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    praucs = [results[n]["prauc"] for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, aucs, w, label="ROC-AUC", color="steelblue")
    ax.bar(x + w / 2, praucs, w, label="PR-AUC", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Discrete-Time Markov Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Model_Comparison_Bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_calibration_figure(
        y_te,
        best_eval["proba"],
        f"Calibration Curve ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "Calibration_Interval.png",
    )
    save_dca_figure(
        y_te,
        best_eval["proba"],
        f"Decision Curve Analysis ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "DCA_Interval.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        best_eval["proba"],
        best_eval["threshold"],
        f"Threshold Sensitivity ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "Threshold_Sensitivity_Interval.png",
    )

    patient_tr = aggregate_patient_level(df_train, best_eval["oof_proba"])
    patient_te = aggregate_patient_level(df_test, best_eval["proba"])
    patient_thr = select_best_threshold(patient_tr["Y"].values, patient_tr["proba"].values, low=0.05, high=0.95, step=0.01)
    patient_metrics = compute_binary_metrics(patient_te["Y"].values, patient_te["proba"].values, patient_thr)
    patient_cis = bootstrap_group_cis(patient_te, patient_thr, group_col="Patient_ID")
    patient_cal = compute_calibration_stats(patient_te["Y"].values, patient_te["proba"].values)

    print(f"\n  Patient-level report (endpoint: any relapse during follow-up)")
    print(f"    Patients    = train {len(patient_tr)} / test {len(patient_te)}")
    print(f"    AUC         = {patient_metrics['auc']:.3f}  95% CI {format_ci(patient_cis, 'auc')}")
    print(f"    PR-AUC      = {patient_metrics['prauc']:.3f}  95% CI {format_ci(patient_cis, 'prauc')}")
    print(f"    Brier       = {patient_metrics['brier']:.3f}  95% CI {format_ci(patient_cis, 'brier')}")
    print(f"    Recall      = {patient_metrics['recall']:.3f}  95% CI {format_ci(patient_cis, 'recall')}")
    print(f"    Specificity = {patient_metrics['specificity']:.3f}  95% CI {format_ci(patient_cis, 'specificity')}")
    print(f"    Cal. Intcp  = {patient_cal['intercept']:.3f}  95% CI {format_ci(patient_cis, 'cal_intercept')}")
    print(f"    Cal. Slope  = {patient_cal['slope']:.3f}  95% CI {format_ci(patient_cis, 'cal_slope')}")
    print(f"    Threshold   = {patient_thr:.2f}")

    window_sens_df = evaluate_window_sensitivity(df_test, best_eval["proba"], best_eval["threshold"])
    print(f"\n  Window sensitivity (interval-level best model)")
    for _, row in window_sens_df.iterrows():
        print(
            f"    {row['Scenario']:<18s} n={int(row['Intervals'])}  events={int(row['Events'])}  "
            f"AUC={row['AUC']:.3f}  PR-AUC={row['PR_AUC']:.3f}  Brier={row['Brier']:.3f}"
        )
    window_sens_df.to_csv(Config.OUT_DIR / "Window_Sensitivity.csv", index=False)

    best_shap_name = best_pr_name
    if results[best_shap_name]["model"] is not None:
        run_shap_analysis(best_shap_name, results[best_shap_name]["model"], X_tr_df, feat_names)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached .pkl files")
    args = parser.parse_args()
    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 88)
    print("  Discrete-Time Markov Relapse Prediction (Temporal-Safe MissForest)")
    print("=" * 88)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    print(f"  Records: {len(pids)}")

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    print(f"  Train: {int(tr_mask.sum())} records")
    print(f"  Test:  {int(te_mask.sum())} records")

    print(f"\n--- Phase 1: State Transition Heatmaps ---")
    max_depth = eval_raw.shape[1]
    hm_raw = np.hstack([X_s_raw, ft3_raw[:, :max_depth], ft4_raw[:, :max_depth], tsh_raw[:, :max_depth], eval_raw])
    hm_imp_path = Config.OUT_DIR / f"missforest_depth{max_depth}.pkl"
    hm_shared_path = Config.SHARED_MF_DIR / f"missforest_depth{max_depth}.pkl"
    hm_imputer = load_or_fit_depth_imputer(
        hm_raw[tr_mask],
        hm_imp_path,
        fallback_cache_path=hm_shared_path,
        label=f"depth-{max_depth}",
    )
    hm_filled = apply_missforest(hm_raw, hm_imputer, max_depth)
    _, _, _, _, ev_hm = split_imputed(hm_filled, n_static, max_depth, max_depth)
    S_full = build_states_from_labels(ev_hm)
    plot_transition_heatmaps(S_full)
    save_markov_transition_table(build_transition_rows(S_full[tr_mask]))

    print(f"\n--- Phase 2: Building first-order Markov relapse data ---")
    df_tr_parts, df_te_parts = [], []
    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"
        raw = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        imp_path = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        shared_imp_path = Config.SHARED_MF_DIR / f"missforest_depth{depth}.pkl"
        imputer = load_or_fit_depth_imputer(
            raw[tr_mask],
            imp_path,
            fallback_cache_path=shared_imp_path,
            label=f"depth-{depth} ({interval_name})",
        )

        filled_tr = apply_missforest(raw[tr_mask], imputer, depth)
        filled_te = apply_missforest(raw[te_mask], imputer, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        S_tr = build_states_from_labels(ev_tr)
        S_te = build_states_from_labels(ev_te)
        X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)
        df_tr_k = build_markov_relapse_data(xs_tr, X_d_tr, S_tr, pids[tr_mask], target_k=k)
        df_te_k = build_markov_relapse_data(xs_te, X_d_te, S_te, pids[te_mask], target_k=k)
        df_tr_parts.append(df_tr_k)
        df_te_parts.append(df_te_k)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr_k)}  test {len(df_te_k)} rows")

    df_tr = pd.concat(df_tr_parts, ignore_index=True)
    df_te = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_tr)}  test {len(df_te)} rows")

    print(f"\n--- Phase 3: Discrete-Time Markov Relapse Modeling ---")
    train_and_evaluate_markov(df_tr, df_te)
    print(f"\n  All plots saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
