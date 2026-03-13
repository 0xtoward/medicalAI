"""Visualization helpers for interpretable model structure summaries."""

from collections import Counter
from pathlib import Path
import re
from textwrap import fill

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import _tree, plot_tree

from utils.config import STATIC_NAMES


# ── shared drawing primitives ────────────────────────────────────────────────

def _rounded_box(ax, x, y, w, h, title, subtitle, fc="#eef3fb", ec="#4c78a8"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.58, title, ha="center", va="center",
            fontsize=13, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center",
                fontsize=8.5, color="#555", family="monospace")


# ── Logistic Regression flow ─────────────────────────────────────────────────

LR_FAMILY_ORDER = [
    "Current physiology",
    "State memory",
    "Trajectory",
    "Follow-up window",
    "Previous state",
    "Baseline clinic",
    "Other",
]

LR_FAMILY_COLORS = {
    "Current physiology": "#d97706",
    "State memory": "#0f766e",
    "Trajectory": "#7c3aed",
    "Follow-up window": "#2563eb",
    "Previous state": "#dc2626",
    "Baseline clinic": "#64748b",
    "Other": "#475569",
}


def _lr_family(feature):
    if feature in STATIC_NAMES:
        return "Baseline clinic"
    if feature.endswith("_Current"):
        return "Current physiology"
    if feature in {"Ever_Hyper_Before", "Ever_Hypo_Before", "Time_In_Normal", "Prior_Relapse_Count"}:
        return "State memory"
    if feature.startswith("Delta_"):
        return "Trajectory"
    if feature.startswith("Window_"):
        return "Follow-up window"
    if feature.startswith("PrevState_") or feature == "Prev_State":
        return "Previous state"
    return "Other"


def _pretty_lr_feature(feature):
    prev_map = {"0": "Prev Hyper", "1": "Prev Normal", "2": "Prev Hypo", "-1": "No previous state"}
    mapping = {
        "FT3_Current": "Current FT3",
        "FT4_Current": "Current FT4",
        "logTSH_Current": "Current logTSH",
        "Ever_Hyper_Before": "Had Hyper before",
        "Ever_Hypo_Before": "Had Hypo before",
        "Time_In_Normal": "Time spent in Normal",
        "Prior_Relapse_Count": "Prior relapse count",
        "Delta_FT4_k0": "FT4 change vs 0M",
        "Delta_TSH_k0": "logTSH change vs 0M",
        "Delta_FT4_1step": "FT4 change vs prior visit",
        "Delta_TSH_1step": "logTSH change vs prior visit",
        "ThyroidW": "Thyroid weight",
        "RAI3d": "RAI 3d uptake",
        "Uptake24h": "24h uptake",
        "TreatCount": "Prior treatment count",
        "MaxUptake": "Max uptake",
        "HalfLife": "Effective half-life",
        "Exophthalmos": "Exophthalmos",
    }
    if feature in mapping:
        return mapping[feature]
    if feature.startswith("Window_"):
        return f"Window {feature.split('_', 1)[1]}"
    if feature.startswith("PrevState_"):
        return prev_map.get(feature.split("_", 1)[1], feature.replace("_", " "))
    return feature.replace("_", " ")


def _lighten(color, weight=0.86):
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(1 - (1 - rgb) * (1 - weight))


def _draw_text_card(ax, x, y, w, h, title, lines, fc, ec, title_color="#0f172a"):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(x + 0.04 * w, y + h - 0.16 * h, title, ha="left", va="center",
            fontsize=12, weight="bold", color=title_color)
    if not lines:
        return
    ax.text(x + 0.04 * w, y + h - 0.33 * h, "\n".join(lines), ha="left", va="top",
            fontsize=9.2, color="#334155", linespacing=1.35)


def _build_lr_frame(feature_names, coef):
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df["family"] = coef_df["feature"].map(_lr_family)
    coef_df["pretty_feature"] = coef_df["feature"].map(_pretty_lr_feature)
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["is_active"] = coef_df["abs_coef"] > 1e-10
    coef_df["direction"] = np.where(
        coef_df["coef"] > 0,
        "Risk up",
        np.where(coef_df["coef"] < 0, "Risk down", "Zeroed"),
    )
    coef_df["family_order"] = coef_df["family"].map({k: i for i, k in enumerate(LR_FAMILY_ORDER)}).fillna(len(LR_FAMILY_ORDER))
    coef_df.sort_values(["abs_coef", "feature"], ascending=[False, True], inplace=True)
    return coef_df


def _save_lr_model_structure(model_name, lr, coef_df, intercept, out_path, decision_threshold=None, output_label="Predicted risk"):
    total_features = len(coef_df)
    active_df = coef_df[coef_df["is_active"]].copy()
    active_count = len(active_df)
    zero_count = total_features - active_count
    pos_count = int((active_df["coef"] > 0).sum())
    neg_count = int((active_df["coef"] < 0).sum())
    family_summary = (
        coef_df.groupby("family", sort=False)
        .agg(
            total_features=("feature", "size"),
            active_features=("is_active", "sum"),
            l1_mass=("abs_coef", "sum"),
        )
        .reset_index()
    )
    family_summary["family_order"] = family_summary["family"].map({k: i for i, k in enumerate(LR_FAMILY_ORDER)})
    family_summary.sort_values("family_order", inplace=True)
    family_summary = family_summary[family_summary["total_features"] > 0]

    top_pos = active_df[active_df["coef"] > 0].head(3)
    top_neg = active_df[active_df["coef"] < 0].head(3)

    fig, ax = plt.subplots(figsize=(15.2, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, f"{model_name} - structural blueprint", ha="left", va="top",
            fontsize=18, weight="bold", color="#0f172a")
    ax.text(
        0.03,
        0.928,
        "This figure explains how the fitted sparse LR converts rolling-landmark features into next-window relapse risk.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#475569",
    )

    family_y = np.linspace(0.69, 0.12, len(family_summary))
    for y, row in zip(family_y, family_summary.itertuples(index=False)):
        fam = row.family
        fam_color = LR_FAMILY_COLORS.get(fam, "#475569")
        fam_active = active_df[active_df["family"] == fam]
        example_text = ", ".join(fam_active["pretty_feature"].head(2).tolist()) if len(fam_active) else ""
        extra = max(0, len(fam_active) - 2)
        if example_text and extra > 0:
            example_text = f"{example_text} +{extra} more"
        lines = [
            f"{int(row.active_features)}/{int(row.total_features)} active coefficients",
            fill(f"Key signals: {example_text or 'all coefficients zeroed'}", 28),
        ]
        _draw_text_card(
            ax, 0.03, y, 0.22, 0.12, fam, lines,
            fc=_lighten(fam_color, 0.90), ec=fam_color,
        )

    _draw_text_card(
        ax, 0.31, 0.63, 0.15, 0.18, "1. Standardize inputs",
        ["Each feature is converted to a z-score.", "Coefficients therefore compare effect size", "on a shared standardized scale."],
        fc="#eef2ff", ec="#4f46e5",
    )
    _draw_text_card(
        ax, 0.52, 0.60, 0.18, 0.24, "2. Sparse linear score",
        [
            f"active = {active_count}/{total_features}",
            f"positive = {pos_count}, negative = {neg_count}, zeroed = {zero_count}",
            f"intercept = {intercept:+.3f}",
            f"penalty = {getattr(lr, 'penalty', 'na')}, C = {getattr(lr, 'C', np.nan):.3g}",
        ],
        fc="#eff6ff", ec="#2563eb",
    )
    _draw_text_card(
        ax, 0.75, 0.60, 0.20, 0.24, "3. Logistic decision rule",
        [
            "logit(p) = b + sum(beta_j * z_j)",
            "p = sigmoid(logit)",
            f"{output_label}",
            f"decision threshold = {decision_threshold:.2f}" if decision_threshold is not None else "decision threshold selected outside plot",
        ],
        fc="#fff7ed", ec="#ea580c",
    )

    for x0, x1, y in [(0.25, 0.31, 0.69), (0.46, 0.52, 0.69), (0.70, 0.75, 0.69)]:
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", lw=2.6, color="#334155"),
        )

    risk_lines = [
        f"{row.pretty_feature}  beta={row.coef:+.3f}  OR={row.odds_ratio:.2f}"
        for row in top_pos.itertuples(index=False)
    ] or ["no positive coefficients retained"]
    protect_lines = [
        f"{row.pretty_feature}  beta={row.coef:+.3f}  OR={row.odds_ratio:.2f}"
        for row in top_neg.itertuples(index=False)
    ] or ["no negative coefficients retained"]
    _draw_text_card(
        ax, 0.52, 0.20, 0.20, 0.24, "Largest risk-up weights", risk_lines,
        fc="#fff1f2", ec="#dc2626",
    )
    _draw_text_card(
        ax, 0.75, 0.20, 0.20, 0.24, "Largest protective weights", protect_lines,
        fc="#ecfeff", ec="#0f766e",
    )

    ax.text(
        0.31,
        0.49,
        "Interpretation shortcut:\npositive beta pushes relapse odds up;\nnegative beta pushes relapse odds down.\nL1 shrinkage removes weak or redundant signals.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#334155",
        linespacing=1.45,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_lr_parameter_shapes(model_name, lr, coef_df, intercept, out_path, decision_threshold=None):
    active_df = coef_df[coef_df["is_active"]].copy()
    family_mass = (
        active_df.groupby("family", sort=False)
        .agg(
            active_features=("feature", "size"),
            l1_mass=("abs_coef", "sum"),
            mean_abs=("abs_coef", "mean"),
        )
        .reset_index()
    )
    family_mass["family_order"] = family_mass["family"].map({k: i for i, k in enumerate(LR_FAMILY_ORDER)})
    family_mass.sort_values("family_order", inplace=True)

    ordered_blocks = []
    family_centers = []
    cursor = 0
    for family in LR_FAMILY_ORDER:
        part = active_df[active_df["family"] == family].copy()
        if part.empty:
            continue
        part.sort_values("coef", inplace=True)
        part["y"] = np.arange(cursor, cursor + len(part))
        family_centers.append((family, float(part["y"].mean())))
        cursor = int(part["y"].max()) + 2
        ordered_blocks.append(part)
    plot_df = pd.concat(ordered_blocks, ignore_index=True) if ordered_blocks else active_df.head(0).copy()

    fig = plt.figure(figsize=(15.5, 10.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.22)
    ax_coef = fig.add_subplot(gs[:, 0])
    ax_family = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    if len(plot_df):
        y_vals = plot_df["y"].values
        colors = plot_df["family"].map(LR_FAMILY_COLORS).values
        coef_min = min(float(plot_df["coef"].min()), -0.05)
        coef_max = max(float(plot_df["coef"].max()), 0.05)
        span = coef_max - coef_min
        x_left = coef_min - 0.52 * span
        x_right = coef_max + 0.24 * span
        label_x = x_left + 0.05 * (x_right - x_left)
        for family, center_y in family_centers:
            fam_rows = plot_df[plot_df["family"] == family]
            ax_coef.axhspan(
                fam_rows["y"].min() - 0.5,
                fam_rows["y"].max() + 0.5,
                color=_lighten(LR_FAMILY_COLORS[family], 0.93),
                alpha=0.85,
                zorder=0,
            )
            ax_coef.text(
                label_x,
                center_y,
                family,
                ha="left",
                va="center",
                fontsize=9.4,
                weight="bold",
                color=LR_FAMILY_COLORS[family],
                bbox=dict(
                    boxstyle="round,pad=0.20,rounding_size=0.05",
                    facecolor=_lighten(LR_FAMILY_COLORS[family], 0.88),
                    edgecolor="none",
                    alpha=0.98,
                ),
                zorder=4,
            )

        ax_coef.barh(y_vals, plot_df["coef"], color=colors, alpha=0.92, height=0.72, zorder=2)
        ax_coef.axvline(0.0, color="#0f172a", lw=1.2, zorder=3)
        ax_coef.set_xlim(x_left, x_right)
        ax_coef.set_yticks(y_vals)
        ax_coef.set_yticklabels(plot_df["pretty_feature"], fontsize=9)
        ax_coef.tick_params(axis="y", pad=10)
        ax_coef.set_xlabel("Standardized coefficient")
        ax_coef.set_title("Non-zero coefficient atlas", fontsize=13, pad=10, weight="bold")
        ax_coef.grid(axis="x", alpha=0.25)

        top_labels = plot_df.reindex(plot_df["abs_coef"].sort_values(ascending=False).head(8).index)
        x_span = max(0.08, float(plot_df["abs_coef"].max()) * 0.035)
        for row in top_labels.itertuples(index=False):
            x_text = row.coef + (x_span if row.coef >= 0 else -x_span)
            ha = "left" if row.coef >= 0 else "right"
            ax_coef.text(
                x_text,
                row.y,
                f"OR {row.odds_ratio:.2f}",
                ha=ha,
                va="center",
                fontsize=8.5,
                color="#0f172a",
            )
    else:
        ax_coef.text(0.5, 0.5, "No non-zero coefficients retained", ha="center", va="center")
        ax_coef.set_axis_off()

    if len(family_mass):
        ax_family.barh(
            family_mass["family"],
            family_mass["l1_mass"],
            color=family_mass["family"].map(LR_FAMILY_COLORS),
            alpha=0.9,
        )
        for row in family_mass.itertuples(index=False):
            ax_family.text(
                row.l1_mass + max(family_mass["l1_mass"]) * 0.02,
                row.family,
                f"{int(row.active_features)} active",
                ha="left",
                va="center",
                fontsize=9,
                color="#334155",
            )
        ax_family.set_title("Absolute weight mass by family", fontsize=12.5, pad=10, weight="bold")
        ax_family.set_xlabel("Sum of |beta|")
        ax_family.grid(axis="x", alpha=0.25)
        ax_family.invert_yaxis()
    else:
        ax_family.set_axis_off()

    ax_info.axis("off")
    total_features = len(coef_df)
    active_count = int(coef_df["is_active"].sum())
    zero_count = total_features - active_count
    info_lines = [
        f"Intercept: {intercept:+.3f}",
        f"Threshold: {decision_threshold:.2f}" if decision_threshold is not None else "Threshold: external selection",
        f"Penalty / C: {getattr(lr, 'penalty', 'na')} / {getattr(lr, 'C', np.nan):.3g}",
        f"Active coefficients: {active_count}/{total_features}",
        f"Zeroed coefficients: {zero_count}/{total_features}",
        "",
        "How to read this figure:",
        "1. Left panel shows every non-zero beta on the standardized scale.",
        "2. Right-top panel shows which feature families carry most total weight.",
        "3. Positive beta pushes relapse odds up; negative beta pushes them down.",
    ]
    ax_info.text(
        0.0,
        0.98,
        "Parameter summary",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )
    ax_info.text(
        0.0,
        0.90,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=10.2,
        color="#334155",
        linespacing=1.45,
    )
    fig.suptitle(f"{model_name} - parameter shapes", fontsize=17, weight="bold", y=0.985)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_lr_coefficient_flow(model_name, coef_df, intercept, out_path, top_n=8, decision_threshold=None, output_label="Predicted risk"):
    active_df = coef_df[coef_df["is_active"]].copy()
    top_neg = active_df[active_df["coef"] < 0].head(top_n).copy()
    top_pos = active_df[active_df["coef"] > 0].head(top_n).copy()
    max_rows = max(len(top_neg), len(top_pos), 1)
    y_vals = np.arange(max_rows)[::-1]

    fig, ax = plt.subplots(figsize=(15.0, 8.6))
    ax.axvline(0.0, color="#0f172a", lw=1.4, zorder=2)
    ax.axvspan(-1.0, 0.0, color="#eff6ff", alpha=0.65, zorder=0)
    ax.axvspan(0.0, 1.0, color="#fff1f2", alpha=0.55, zorder=0)

    if len(top_neg):
        neg_rows = list(top_neg.itertuples(index=False))
        for y, row in zip(y_vals[:len(neg_rows)], neg_rows):
            edge = LR_FAMILY_COLORS.get(row.family, "#2563eb")
            ax.barh(y, row.coef, color="#93c5fd", edgecolor=edge, linewidth=2.0, height=0.72, zorder=3)
            ax.text(row.coef - 0.02, y, row.pretty_feature, ha="right", va="center", fontsize=10, color="#0f172a")
            ax.text(-0.02, y - 0.26, f"beta {row.coef:+.3f} | OR {row.odds_ratio:.2f}", ha="right", va="center",
                    fontsize=8.8, color="#475569")

    if len(top_pos):
        pos_rows = list(top_pos.itertuples(index=False))
        for y, row in zip(y_vals[:len(pos_rows)], pos_rows):
            edge = LR_FAMILY_COLORS.get(row.family, "#dc2626")
            ax.barh(y, row.coef, color="#fda4af", edgecolor=edge, linewidth=2.0, height=0.72, zorder=3)
            ax.text(row.coef + 0.02, y, row.pretty_feature, ha="left", va="center", fontsize=10, color="#0f172a")
            ax.text(0.02, y - 0.26, f"beta {row.coef:+.3f} | OR {row.odds_ratio:.2f}", ha="left", va="center",
                    fontsize=8.8, color="#475569")

    ax.text(-0.55, max_rows - 0.15, "Protective direction\n(lower relapse odds)", ha="center", va="bottom",
            fontsize=12, weight="bold", color="#1d4ed8")
    ax.text(0.55, max_rows - 0.15, "Risk-up direction\n(higher relapse odds)", ha="center", va="bottom",
            fontsize=12, weight="bold", color="#dc2626")
    center_lines = [
        "Center line = no change in log-odds",
        f"Intercept = {intercept:+.3f}",
        f"Threshold = {decision_threshold:.2f}" if decision_threshold is not None else "Threshold selected outside plot",
        output_label,
    ]
    ax.text(0.0, -0.95, "\n".join(center_lines), ha="center", va="top",
            fontsize=10.2, color="#334155", linespacing=1.45)

    legend_items = [
        Line2D([0], [0], color="#93c5fd", lw=8, label="Negative beta"),
        Line2D([0], [0], color="#fda4af", lw=8, label="Positive beta"),
    ]
    for family in LR_FAMILY_ORDER:
        if family in active_df["family"].values:
            legend_items.append(
                Line2D([0], [0], color=LR_FAMILY_COLORS[family], lw=2.5, label=family)
            )
    ax.legend(handles=legend_items, ncol=2, fontsize=8.8, loc="lower center", bbox_to_anchor=(0.5, -0.23), frameon=False)

    lim = max(0.15, float(active_df["abs_coef"].head(top_n).max()) * 1.45 if len(active_df) else 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.25, max_rows - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Standardized coefficient (beta)")
    ax.set_title(f"{model_name} - coefficient flow for top retained signals", fontsize=17, weight="bold", pad=14)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_logistic_regression_visuals(
    model_name, model, feature_names, out_dir, prefix="LR", top_n=12,
    decision_threshold=None, output_label="Predicted risk"
):
    """Create publication-oriented LR structure and coefficient figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, Pipeline) or "lr" not in model.named_steps:
        return

    lr = model.named_steps["lr"]
    coef = np.asarray(lr.coef_, dtype=float).ravel()
    intercept = float(lr.intercept_[0])

    coef_df = _build_lr_frame(feature_names, coef)
    coef_df.to_csv(out_dir / f"{prefix}_Coefficients.csv", index=False)
    if not coef_df["is_active"].any():
        return
    _save_lr_model_structure(
        model_name, lr, coef_df, intercept,
        out_dir / f"{prefix}_Model_Structure.png",
        decision_threshold=decision_threshold,
        output_label=output_label,
    )
    _save_lr_parameter_shapes(
        model_name, lr, coef_df, intercept,
        out_dir / f"{prefix}_Parameter_Shapes.png",
        decision_threshold=decision_threshold,
    )
    _save_lr_coefficient_flow(
        model_name, coef_df, intercept,
        out_dir / f"{prefix}_Coefficient_Flow.png",
        top_n=top_n,
        decision_threshold=decision_threshold,
        output_label=output_label,
    )


# ── Tree / ensemble importance flow ──────────────────────────────────────────

def save_tree_importance_flow(
    model_name, model, feature_names, out_dir, prefix="Tree",
    top_n=12, output_label="P(Hyper)",
):
    """Feature-importance flow diagram for any model with feature_importances_."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imp = _get_importances(model)
    if imp is None:
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    imp_df.to_csv(out_dir / f"{prefix}_Importances.csv", index=False)

    top = imp_df[imp_df["importance"] > 0].head(top_n).reset_index(drop=True)
    if top.empty:
        return
    n = len(top)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=top["importance"].max())

    fig_h = max(5.5, 0.55 * n + 2.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.8, n + 0.8)
    ax.axis("off")

    feat_x, feat_w = 0.0, 3.6
    model_x = 6.2
    out_x = 9.0
    max_imp = top["importance"].max()

    model_type = _model_type_label(model)

    for i, row in top.iterrows():
        y = n - 1 - i
        val = row["importance"]
        rgba = cmap(norm(val))
        edge_c = mcolors.to_hex(rgba)
        fill_c = mcolors.to_hex((*rgba[:3], 0.15))

        rect = mpatches.FancyBboxPatch(
            (feat_x, y - 0.3), feat_w, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=fill_c, edgecolor=edge_c, linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(feat_x + 0.15, y, row["feature"],
                va="center", ha="left", fontsize=9, weight="bold")
        ax.text(feat_x + feat_w - 0.15, y, f"{val:.4f}",
                va="center", ha="right", fontsize=8.5, color="#333", family="monospace")

        pct = val / max_imp
        bar_max_w = feat_w - 0.3
        bar_w = bar_max_w * pct
        bar = mpatches.FancyBboxPatch(
            (feat_x + 0.1, y - 0.38), bar_w, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=edge_c, edgecolor="none", alpha=0.35,
        )
        ax.add_patch(bar)

        lw = 0.6 + 3.0 * pct
        ax.annotate("", xy=(model_x, n / 2 - 0.15), xytext=(feat_x + feat_w + 0.08, y),
                    arrowprops=dict(arrowstyle="->,head_width=0.16,head_length=0.10",
                                    lw=lw, color=edge_c,
                                    alpha=0.45 + 0.55 * pct))

    _rounded_box(ax, model_x, n/2 - 0.65, 2.2, 1.3, model_type, _model_detail(model),
                 fc="#eef3fb", ec="#4c78a8")
    _rounded_box(ax, out_x, n/2 - 0.45, 1.7, 0.9, output_label, "",
                 fc="#fff3e6", ec="#e6550d")

    ax.annotate("", xy=(out_x, n/2), xytext=(model_x + 2.2, n/2),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#333"))

    ax.text(5.25, n + 0.5,
            f"{model_name}  —  Top {n} features by importance",
            ha="center", va="bottom", fontsize=12, weight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, aspect=30)
    cbar.set_label("Feature importance", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_Importance_Flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_importances(model):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if isinstance(model, Pipeline):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                return np.asarray(step.feature_importances_, dtype=float)
    return None


def _model_type_label(model):
    name = type(model).__name__
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            name = type(step).__name__
    mapping = {
        "RandomForestClassifier": "Random Forest",
        "BalancedRandomForestClassifier": "Balanced RF",
        "GradientBoostingClassifier": "GBDT",
        "LGBMClassifier": "LightGBM",
        "XGBClassifier": "XGBoost",
    }
    return mapping.get(name, name)


def _model_detail(model):
    est = model
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            est = step
    parts = []
    if hasattr(est, "n_estimators"):
        parts.append(f"trees={est.n_estimators}")
    depth = getattr(est, "max_depth", None)
    if depth is not None and depth != -1:
        parts.append(f"depth={depth}")
    lr = getattr(est, "learning_rate", None)
    if lr is not None:
        parts.append(f"lr={lr}")
    return ", ".join(parts) if parts else ""


def save_forest_anatomy(
    model_name,
    model,
    feature_names,
    out_dir,
    prefix="Forest",
    sample_X=None,
    sample_y=None,
    max_trees=4,
    max_depth=2,
    top_n=6,
    output_label="P(Hyper)",
):
    """Render separate RF anatomy figures for readability."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    est = model
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            est = step

    estimators = list(getattr(est, "estimators_", []))
    if len(estimators) == 0:
        return
    if isinstance(estimators[0], (list, tuple, np.ndarray)):
        estimators = [tree for block in estimators for tree in np.ravel(block).tolist()]
    if len(estimators) == 0 or not hasattr(estimators[0], "tree_"):
        return

    root_counter, depth1_counter, _ = _collect_forest_patterns(estimators, feature_names, max_depth=max_depth)
    root_df = _counter_to_frame(root_counter, "feature", "count").head(top_n)
    depth1_df = _counter_to_frame(depth1_counter, "feature", "count").head(top_n)

    chosen_idx = _select_representative_trees(estimators, feature_names, max_depth=max_depth, n_select=max_trees)
    chosen_trees = [estimators[i] for i in chosen_idx]
    n_show = len(chosen_trees)
    shortest_local_idx = int(np.argmin([tree.tree_.node_count for tree in chosen_trees]))
    detail = _model_detail(est)
    subtitle = f"4 representative trees selected by consensus + simplicity" + (f" | {detail}" if detail else "")

    full_tree = chosen_trees[shortest_local_idx]
    full_idx = chosen_idx[shortest_local_idx] + 1
    fig, ax = plt.subplots(figsize=(15, 8.8))
    _draw_tree_custom(ax, full_tree, feature_names, max_depth=None, show_samples=True)
    fig.suptitle(f"{model_name} — Full Shortest Tree", fontsize=17, weight="bold", y=0.98)
    fig.text(0.5, 0.952, f"Tree {full_idx} | {subtitle}", ha="center", va="center", fontsize=10, color="#555", family="monospace")
    fig.savefig(out_dir / f"{prefix}_Forest_Anatomy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    shallow_fig = plt.figure(figsize=(16, 5.2))
    shallow_gs = shallow_fig.add_gridspec(1, 3, wspace=0.12)
    shallow_rank = 0
    for idx in range(n_show):
        if idx == shortest_local_idx or shallow_rank >= 3:
            continue
        ax = shallow_fig.add_subplot(shallow_gs[0, shallow_rank])
        _draw_tree_custom(
            ax,
            chosen_trees[idx],
            feature_names,
            max_depth=max_depth,
            show_samples=False,
            show_omission=True,
        )
        ax.set_title(f"Tree {chosen_idx[idx] + 1} — shown to depth={max_depth}", fontsize=9.5, pad=4)
        shallow_rank += 1
    shallow_fig.suptitle(f"{model_name} — Representative Shallow Trees", fontsize=15, weight="bold", y=0.99)
    shallow_fig.text(0.5, 0.035, "Subtrees below the depth limit are omitted and marked with '...'", ha="center", va="center", fontsize=9.5, color="#666")
    shallow_fig.savefig(out_dir / f"{prefix}_Representative_Subtrees.png", dpi=300, bbox_inches="tight")
    plt.close(shallow_fig)

    summary_fig = plt.figure(figsize=(11.6, 4.8))
    summary_gs = summary_fig.add_gridspec(1, 2, wspace=0.28)
    ax_root = summary_fig.add_subplot(summary_gs[0, 0])
    _plot_split_frequency(ax_root, root_df, "Most common root splits")
    ax_vote = summary_fig.add_subplot(summary_gs[0, 1])
    _plot_ensemble_vote(ax_vote, est, sample_X, sample_y, output_label=output_label)
    summary_fig.suptitle(f"{model_name} — Forest Summary", fontsize=15, weight="bold", y=0.98)
    summary_fig.savefig(out_dir / f"{prefix}_Forest_Summary.png", dpi=300, bbox_inches="tight")
    plt.close(summary_fig)


def _collect_forest_patterns(estimators, feature_names, max_depth=2):
    root_counter = Counter()
    depth1_counter = Counter()
    path_counter = Counter()

    for est in estimators:
        tree = est.tree_
        feature = tree.feature
        children_left = tree.children_left
        children_right = tree.children_right

        if feature[0] != _tree.TREE_UNDEFINED:
            root_counter[feature_names[feature[0]]] += 1
        for child in (children_left[0], children_right[0]):
            if child != _tree.TREE_LEAF and feature[child] != _tree.TREE_UNDEFINED:
                depth1_counter[feature_names[feature[child]]] += 1

        def walk(node_id, depth, path_feats):
            feat_idx = feature[node_id]
            if feat_idx == _tree.TREE_UNDEFINED or depth > max_depth:
                if path_feats:
                    path_counter[" -> ".join(path_feats)] += 1
                return
            next_path = path_feats + [feature_names[feat_idx]]
            if depth == max_depth:
                path_counter[" -> ".join(next_path)] += 1
                return
            walk(children_left[node_id], depth + 1, next_path)
            walk(children_right[node_id], depth + 1, next_path)

        walk(0, 0, [])

    return root_counter, depth1_counter, path_counter


def _counter_to_frame(counter, label_col, value_col):
    if not counter:
        return pd.DataFrame(columns=[label_col, value_col])
    return (
        pd.DataFrame([{label_col: key, value_col: value} for key, value in counter.items()])
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def _plot_split_frequency(ax, freq_df, title):
    if len(freq_df) == 0:
        ax.set_axis_off()
        return
    plot_df = freq_df.iloc[::-1].copy()
    ax.barh(plot_df.iloc[:, 0], plot_df.iloc[:, 1], color="#4c78a8", alpha=0.85)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Number of trees")
    ax.grid(axis="x", alpha=0.25)


def _plot_ensemble_vote(ax, est, sample_X, sample_y=None, output_label="P(Hyper)"):
    ax.set_title("Ensemble voting consensus", fontsize=12, pad=8)
    if sample_X is None or len(getattr(est, "estimators_", [])) == 0:
        ax.text(0.5, 0.5, "Prediction sample unavailable", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    X_vals = sample_X.values if hasattr(sample_X, "values") else np.asarray(sample_X)
    per_tree = np.column_stack([tree.predict_proba(X_vals)[:, 1] for tree in est.estimators_[: min(100, len(est.estimators_))]])
    mean_prob = per_tree.mean(axis=1)
    vote_rate = (per_tree >= 0.5).mean(axis=1)

    if sample_y is not None:
        sample_y = np.asarray(sample_y).astype(int)
        ax.hist(mean_prob[sample_y == 0], bins=20, alpha=0.65, label="N", color="#4c78a8")
        ax.hist(mean_prob[sample_y == 1], bins=20, alpha=0.65, label="H", color="#e45756")
        ax.legend(fontsize=9)
    else:
        ax.hist(mean_prob, bins=20, alpha=0.85, color="#4c78a8")

    ax2 = ax.twinx()
    order = np.argsort(mean_prob)
    ax2.plot(np.arange(len(vote_rate)), vote_rate[order], color="#f58518", lw=2.0, alpha=0.85)
    ax2.set_ylabel("Share of trees voting Hyper", color="#f58518")
    ax.set_xlabel(output_label)
    ax.set_ylabel("Sample count")
    ax.grid(alpha=0.20)


def _select_representative_trees(estimators, feature_names, max_depth=2, n_select=4):
    profiles = [_tree_profile(idx, est, feature_names, max_depth=max_depth) for idx, est in enumerate(estimators)]
    root_counts = Counter(p["root_feature"] for p in profiles if p["root_feature"] is not None)
    path_counts = Counter(p["path_signature"] for p in profiles if p["path_signature"])

    selected = []
    used_roots = set()
    used_paths = set()
    remaining = profiles.copy()

    for _ in range(min(n_select, len(remaining))):
        best = None
        best_score = -1e18
        for prof in remaining:
            rep = 2.5 * root_counts.get(prof["root_feature"], 0) + 1.5 * path_counts.get(prof["path_signature"], 0)
            diversity = (1.5 if prof["root_feature"] not in used_roots else 0.0) + (1.0 if prof["path_signature"] not in used_paths else 0.0)
            simplicity = -0.025 * prof["node_count"] - 0.25 * prof["depth"]
            score = rep + diversity + simplicity
            if score > best_score:
                best_score = score
                best = prof
        if best is None:
            break
        selected.append(best["idx"])
        used_roots.add(best["root_feature"])
        used_paths.add(best["path_signature"])
        remaining = [p for p in remaining if p["idx"] != best["idx"]]
    return selected


def _tree_profile(idx, estimator, feature_names, max_depth=2):
    tree = estimator.tree_
    feature = tree.feature
    node_samples = tree.n_node_samples
    children_left = tree.children_left
    children_right = tree.children_right
    root_feature = feature_names[feature[0]] if feature[0] != _tree.TREE_UNDEFINED else None

    path = []
    node = 0
    depth = 0
    while node != _tree.TREE_LEAF and feature[node] != _tree.TREE_UNDEFINED and depth <= max_depth:
        path.append(feature_names[feature[node]])
        left = children_left[node]
        right = children_right[node]
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            break
        node = left if node_samples[left] >= node_samples[right] else right
        depth += 1

    return {
        "idx": idx,
        "root_feature": root_feature,
        "path_signature": " -> ".join(path),
        "node_count": int(tree.node_count),
        "depth": int(tree.max_depth),
    }


def _draw_tree_custom(ax, estimator, feature_names, max_depth=None, show_samples=False, show_omission=False):
    ax.set_axis_off()
    tree = estimator.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value.squeeze(axis=1) if tree.value.ndim == 3 else tree.value
    n_samples = tree.n_node_samples

    def subtree_width(node_id, depth):
        is_leaf = children_left[node_id] == _tree.TREE_LEAF and children_right[node_id] == _tree.TREE_LEAF
        cut_here = max_depth is not None and depth >= max_depth and not is_leaf
        if is_leaf or cut_here:
            return 1.0
        return subtree_width(children_left[node_id], depth + 1) + subtree_width(children_right[node_id], depth + 1)

    def assign_positions(node_id, depth, x_left, pos):
        width = subtree_width(node_id, depth)
        pos[node_id] = (x_left + width / 2, -depth)
        is_leaf = children_left[node_id] == _tree.TREE_LEAF and children_right[node_id] == _tree.TREE_LEAF
        cut_here = max_depth is not None and depth >= max_depth and not is_leaf
        if is_leaf or cut_here:
            return
        left_w = subtree_width(children_left[node_id], depth + 1)
        assign_positions(children_left[node_id], depth + 1, x_left, pos)
        assign_positions(children_right[node_id], depth + 1, x_left + left_w, pos)

    positions = {}
    assign_positions(0, 0, 0.0, positions)
    total_w = subtree_width(0, 0)
    max_seen_depth = max(int(-y) for _, y in positions.values()) if positions else 0
    ax.set_xlim(-0.4, total_w + 0.4)
    ax.set_ylim(-(max_seen_depth + 1.1), 0.8)

    def label_for_node(node_id, depth):
        is_leaf = children_left[node_id] == _tree.TREE_LEAF and children_right[node_id] == _tree.TREE_LEAF
        cut_here = max_depth is not None and depth >= max_depth and not is_leaf
        class_idx = int(np.argmax(value[node_id]))
        pred = "H" if class_idx == 1 else "N"
        if is_leaf:
            label = pred
            if show_samples:
                label += f"\n(n={int(n_samples[node_id])})"
            return label, ("#dbeafe" if class_idx == 0 else "#fde2cf"), ("#4c78a8" if class_idx == 0 else "#e07a2f")
        if cut_here:
            split = f"{feature_names[feature[node_id]]}\n<= {threshold[node_id]:.1f}"
            omit = "\n..." if show_omission else ""
            return split + omit, "#fbf3df", "#d09a2f"
        split = f"{feature_names[feature[node_id]]}\n<= {threshold[node_id]:.1f}"
        return split, "#fbf3df", "#d09a2f"

    for node_id, (x, y) in positions.items():
        is_leaf = children_left[node_id] == _tree.TREE_LEAF and children_right[node_id] == _tree.TREE_LEAF
        cut_here = max_depth is not None and (-y) >= max_depth and not is_leaf
        if not (is_leaf or cut_here):
            for child in (children_left[node_id], children_right[node_id]):
                cx, cy = positions[child]
                ax.plot([x, cx], [y - 0.1, cy + 0.15], color="#999", lw=1.0)
        label, fc, ec = label_for_node(node_id, int(-y))
        rect = mpatches.FancyBboxPatch(
            (x - 0.42, y - 0.18),
            0.84,
            0.36,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.7, linespacing=1.05)
