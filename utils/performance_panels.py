"""Shared performance heatmap helpers for binary modeling scripts."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.evaluation import compute_binary_metrics


DOMAIN_LABELS = {
    "Train_Fit": "Train Fit",
    "Validation_OOF": "Validation OOF",
    "Test_Temporal": "Temporal Test",
    "External": "External",
    "Prospective": "Prospective",
}

METRIC_LABELS = {
    "auc": "AUC",
    "prauc": "PR-AUC",
    "acc": "Accuracy",
    "bacc": "Balanced Acc",
    "f1": "F1",
    "recall": "Sensitivity",
    "specificity": "Specificity",
    "brier": "Brier",
}


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def build_binary_performance_long(
    task_name,
    results,
    domain_payloads,
    metric_keys,
    threshold_key,
):
    """Convert per-model predictions into a long metric table."""
    rows = []
    for model_name, payload in results.items():
        threshold = float(payload[threshold_key])
        for domain_name, domain_info in domain_payloads.items():
            y_true = np.asarray(domain_info["y_true"]).astype(int)
            proba = np.asarray(payload[domain_info["proba_key"]], dtype=float)
            metrics = compute_binary_metrics(y_true, proba, threshold)
            for metric_key in metric_keys:
                rows.append(
                    {
                        "Task": task_name,
                        "Domain": domain_name,
                        "Domain_Label": DOMAIN_LABELS.get(domain_name, domain_name),
                        "Model": model_name,
                        "Metric_Key": metric_key,
                        "Metric": METRIC_LABELS.get(metric_key, metric_key),
                        "Value": float(metrics[metric_key]),
                        "Threshold": threshold,
                        "N": int(len(y_true)),
                        "Events": int(y_true.sum()),
                    }
                )
    return pd.DataFrame(rows)


def export_metric_matrices(long_df, out_dir, prefix="Performance"):
    """Save long and wide metric tables for downstream use."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = long_df.copy()
    long_df.to_csv(out_dir / f"{prefix}_Long.csv", index=False)

    task_order = list(dict.fromkeys(long_df["Task"].tolist()))
    domain_order = list(dict.fromkeys(long_df["Domain"].tolist()))
    metric_order = list(dict.fromkeys(long_df["Metric"].tolist()))

    for task_name in task_order:
        for domain_name in domain_order:
            sub = long_df[(long_df["Task"] == task_name) & (long_df["Domain"] == domain_name)].copy()
            if sub.empty:
                continue
            pivot = (
                sub.pivot(index="Metric", columns="Model", values="Value")
                .reindex(metric_order)
            )
            out_name = f"{prefix}_{_safe_name(task_name)}_{_safe_name(domain_name)}.csv"
            pivot.to_csv(out_dir / out_name)


def save_performance_heatmap_panels(
    long_df,
    out_path,
    task_order,
    domain_order,
    metric_order,
    title,
    cmap="YlOrRd",
):
    """Save panel heatmaps for one or more tasks across performance domains."""
    if len(task_order) == 0 or len(domain_order) == 0:
        return

    nrows = len(task_order)
    ncols = len(domain_order)
    fig_w = max(14, 4.4 * ncols)
    fig_h = max(4.6, 2.8 * nrows + 1.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    panel_idx = 0
    cbar_drawn = False
    for row_idx, task_name in enumerate(task_order):
        for col_idx, domain_name in enumerate(domain_order):
            ax = axes[row_idx, col_idx]
            sub = long_df[(long_df["Task"] == task_name) & (long_df["Domain"] == domain_name)].copy()
            panel_letter = chr(ord("A") + panel_idx)
            panel_idx += 1

            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{panel_letter}. No data", ha="center", va="center", fontsize=12)
                continue

            pivot = (
                sub.pivot(index="Metric", columns="Model", values="Value")
                .reindex([METRIC_LABELS.get(k, k) for k in metric_order])
            )
            sns.heatmap(
                pivot,
                ax=ax,
                annot=True,
                fmt=".3f",
                cmap=cmap,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="white",
                cbar=not cbar_drawn,
                annot_kws={"fontsize": 7},
            )
            cbar_drawn = True
            ax.set_title(f"{panel_letter}. {DOMAIN_LABELS.get(domain_name, domain_name)}", fontsize=11, pad=8)
            ax.set_xlabel("")
            ax.set_ylabel(task_name if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=28, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
