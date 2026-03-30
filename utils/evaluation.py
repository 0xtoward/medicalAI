"""Shared evaluation and plotting helpers for relapse prediction scripts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
from utils.plot_style import PRIMARY_BLUE, PRIMARY_TEAL


def compute_binary_metrics(y_true, proba, threshold):
    """Compute discrimination, calibration, and classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan
    prauc = average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "auc": auc,
        "prauc": prauc,
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
    """Compute calibration intercept and slope."""
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


def bootstrap_group_cis(eval_df, threshold, group_col="Patient_ID", n_boot=800, seed=42):
    """Cluster bootstrap confidence intervals by patient/group."""
    group_values = eval_df[group_col].dropna().unique()
    if len(group_values) == 0:
        return {}

    grouped = {g: eval_df.loc[eval_df[group_col] == g] for g in group_values}
    rng = np.random.default_rng(seed)
    store = {
        k: []
        for k in ["auc", "prauc", "brier", "recall", "specificity", "cal_intercept", "cal_slope"]
    }

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
    """Format CI for display."""
    low, high = cis.get(key, (np.nan, np.nan))
    if np.isnan(low) or np.isnan(high):
        return "NA"
    return f"[{low:.3f}, {high:.3f}]"


def save_calibration_figure(y_true, proba, title, out_path, n_bins=6):
    """Save calibration curve and prediction histogram."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")

    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, marker="o", lw=2, color="navy", label="Model")
    ax1.set_title(title)
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


def save_threshold_sensitivity_figure(y_true, proba, best_thr, title, out_path):
    """Show recall/specificity/F1 across thresholds."""
    thresholds = np.arange(0.02, 0.61, 0.01)
    rows = []
    for thr in thresholds:
        m = compute_binary_metrics(y_true, proba, thr)
        rows.append({"threshold": thr, "recall": m["recall"], "specificity": m["specificity"], "f1": m["f1"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df["threshold"], df["recall"], lw=2, label="Recall")
    ax.plot(df["threshold"], df["specificity"], lw=2, label="Specificity")
    ax.plot(df["threshold"], df["f1"], lw=2, label="F1")
    ax.axvline(best_thr, color="black", linestyle="--", alpha=0.7, label=f"Chosen threshold={best_thr:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_dca(y_true, proba, thresholds=None):
    """Decision curve analysis net benefit."""
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
    """Save DCA figure."""
    dca_df = compute_dca(y_true, proba)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dca_df["threshold"], dca_df["model"], lw=2.2, color="darkgreen", label="Model")
    ax.plot(dca_df["threshold"], dca_df["treat_all"], lw=1.6, color="crimson", linestyle="--", label="Treat all")
    ax.plot(dca_df["threshold"], dca_df["treat_none"], lw=1.6, color="black", linestyle=":", label="Treat none")
    ax.set_title(title)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_patient_level(df_long, proba, method="product"):
    """Aggregate interval risks into patient-level relapse probability."""
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
        patient_rows.append(
            {
                "Patient_ID": pid,
                "Y": int(g["Y_Relapse"].max()),
                "proba": agg_proba,
                "Max_Interval_Risk": float(np.max(probs)),
                "Mean_Interval_Risk": float(np.mean(probs)),
                "Intervals": int(len(g)),
                "First_Window": g["Interval_Name"].iloc[0],
                "Last_Window": g["Interval_Name"].iloc[-1],
                "Aggregation": method,
            }
        )
    return pd.DataFrame(patient_rows)


def select_best_threshold(y_true, proba, low=0.02, high=0.95, step=0.01):
    """Select threshold by best F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(low, high + 1e-9, step):
        f1 = f1_score(y_true, (np.asarray(proba) >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def save_patient_risk_strata(train_patient_df, test_patient_df, out_path):
    """Use train-set quartiles for patient-level risk stratification."""
    train_scores = np.clip(train_patient_df["proba"].values, 0, 1)
    bins = np.quantile(train_scores, [0, 0.25, 0.5, 0.75, 1.0])
    bins[0] = min(bins[0], 0.0)
    bins[-1] = max(bins[-1], 1.0)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6

    labels = ["Q1", "Q2", "Q3", "Q4"]
    strat_df = test_patient_df.copy()
    strat_df["Risk_Group"] = pd.cut(strat_df["proba"], bins=bins, include_lowest=True, labels=labels).astype(str)

    summary = (
        strat_df.groupby("Risk_Group")
        .agg(N=("Y", "size"), Observed=("Y", "mean"), Predicted=("proba", "mean"))
        .reindex(labels)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(summary["Risk_Group"], summary["Observed"], color="cornflowerblue", alpha=0.85, label="Observed relapse rate")
    ax.plot(summary["Risk_Group"], summary["Predicted"], marker="o", lw=2, color="darkorange", label="Mean predicted risk")
    for bar, n in zip(bars, summary["N"].fillna(0).astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, min(1.0, float(np.nanmax([summary["Observed"].max(), summary["Predicted"].max()]) + 0.12)))
    ax.set_title("Patient-Level Risk Stratification (Q1-Q4)")
    ax.set_xlabel("Risk groups from train-set quartiles")
    ax.set_ylabel("Relapse probability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_patient_aggregation_sensitivity(df_long_train, df_long_test, oof_proba, test_proba):
    """Compare patient-level conclusions across aggregation rules."""
    methods = {"Product_AnyRisk": "product", "Max_Interval_Risk": "max", "Mean_Interval_Risk": "mean"}
    rows = []
    curves = {}
    for label, method in methods.items():
        train_df = aggregate_patient_level(df_long_train, oof_proba, method=method)
        test_df = aggregate_patient_level(df_long_test, test_proba, method=method)
        thr = select_best_threshold(train_df["Y"].values, train_df["proba"].values, low=0.05, high=0.95, step=0.01)
        m = compute_binary_metrics(test_df["Y"].values, test_df["proba"].values, thr)
        cal = compute_calibration_stats(test_df["Y"].values, test_df["proba"].values)
        rows.append(
            {
                "Aggregation": label,
                "Threshold": thr,
                "AUC": m["auc"],
                "PR_AUC": m["prauc"],
                "Brier": m["brier"],
                "Recall": m["recall"],
                "Specificity": m["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
            }
        )
        curves[label] = test_df
    return pd.DataFrame(rows), curves


def save_patient_aggregation_sensitivity_figure(summary_df, out_path):
    """Visualize patient-level aggregation sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(summary_df))
    labels = summary_df["Aggregation"].tolist()

    axes[0].bar(x - 0.17, summary_df["AUC"], 0.34, label="AUC", color=PRIMARY_BLUE)
    axes[0].bar(x + 0.17, summary_df["PR_AUC"], 0.34, label="PR-AUC", color=PRIMARY_TEAL)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Patient-Level Aggregation Sensitivity")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.17, summary_df["Calibration_Intercept"], 0.34, label="Intercept", color="slateblue")
    axes[1].bar(x + 0.17, summary_df["Calibration_Slope"], 0.34, label="Slope", color="darkorange")
    axes[1].axhline(0.0, color="gray", linestyle="--", lw=1)
    axes[1].axhline(1.0, color="gray", linestyle=":", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_title("Calibration Statistics by Aggregation")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_window_sensitivity(df_long_test, proba, threshold):
    """Check whether results depend heavily on specific follow-up windows."""
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


def concordance_index_simple(time, score, event):
    """Simple concordance index without external dependencies."""
    time = np.asarray(time, dtype=float)
    score = np.asarray(score, dtype=float)
    event = np.asarray(event, dtype=int)
    conc = 0.0
    ties = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j] and event[i] == event[j]:
                continue
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                if score[i] > score[j]:
                    conc += 1
                elif score[i] == score[j]:
                    ties += 1
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1
                if score[j] > score[i]:
                    conc += 1
                elif score[i] == score[j]:
                    ties += 1
    if comparable == 0:
        return np.nan
    return float((conc + 0.5 * ties) / comparable)
