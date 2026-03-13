"""Paper-style methodology framework figure for the manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _panel(ax, x, y, w, h, title, fc="#e9eef1", ec="#6a7f8f", title_size=17):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h - 0.032,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color="#0f172a",
    )


def _card(ax, x, y, w, h, title, lines, fc="#f8fbfb", ec="#8da2b1", title_size=12.5, body_size=10):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.018 * w,
        y + h - 0.17 * h,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        weight="bold",
        color="#111827",
    )
    ax.text(
        x + 0.018 * w,
        y + h - 0.30 * h,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=body_size,
        color="#334155",
        linespacing=1.35,
    )


def _chip(ax, x, y, w, h, label, fc, ec, txt="#0f172a", size=9.3):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, weight="bold", color=txt)


def _arrow(ax, x0, y0, x1, y1, color="#5b6b75", lw=2.0):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0),
    )


def save_paper_method_framework(out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(17.2, 10.6))
    fig.patch.set_facecolor("#f5f7f6")
    ax.set_facecolor("#f5f7f6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = mpatches.FancyBboxPatch(
        (0.015, 0.947),
        0.97,
        0.04,
        boxstyle="round,pad=0.01,rounding_size=0.012",
        facecolor="#dfe7ea",
        edgecolor="#6b7f8d",
        linewidth=1.8,
    )
    ax.add_patch(title)
    ax.text(
        0.5,
        0.967,
        "Temporal Landmark Framework for Graves Disease After RAI",
        ha="center",
        va="center",
        fontsize=22,
        weight="bold",
        color="#111827",
    )

    _panel(ax, 0.02, 0.08, 0.46, 0.84, "a. Cohort and Variables")
    _panel(ax, 0.51, 0.08, 0.47, 0.84, "b. Temporal-safe Development")

    _card(
        ax,
        0.03,
        0.73,
        0.44,
        0.16,
        "i. Study cohort",
        [
            "RAI-treated Graves cohort",
            "889 patients",
            "1003 records",
            "Visits: 0M, 1M, 3M, 6M, 12M, 18M, 24M",
            "Chronological split",
            "Train: 711 patients / 795 records",
            "Test: 178 patients / 208 records",
        ],
        body_size=9.6,
    )

    _card(
        ax,
        0.03,
        0.50,
        0.44,
        0.19,
        "ii. Variables",
        [
            "Demographics",
            "Baseline thyroid characteristics",
            "FT3 / FT4 / TSH trajectories",
            "Antibodies, dose and uptake",
            "Doctor-assessed state",
        ],
    )
    _chip(ax, 0.33, 0.565, 0.11, 0.035, "Hyper", "#dcfce7", "#16a34a")
    _chip(ax, 0.33, 0.525, 0.11, 0.035, "Normal", "#dbeafe", "#2563eb")
    _chip(ax, 0.33, 0.485, 0.11, 0.035, "Hypo", "#fef3c7", "#d97706")

    _card(
        ax,
        0.03,
        0.16,
        0.44,
        0.28,
        "iii. Clinical prediction targets",
        [
            "Fixed landmark: Hyper vs Non-Hyper at 3M / 6M",
            "Rolling landmark: Normal at k  ->  relapse at k+1",
            "Recurrent-survival benchmark: AG / PWP / RSF",
            "Two-stage physiology extension: predict labs  ->  predict relapse",
            "Supplement: fixed landmark multi-state routing",
        ],
        body_size=9.4,
    )
    _chip(ax, 0.05, 0.20, 0.12, 0.05, "Fixed landmark", "#dbeafe", "#3b82f6")
    _chip(ax, 0.19, 0.20, 0.14, 0.05, "Rolling landmark", "#dcfce7", "#16a34a")
    _chip(ax, 0.35, 0.20, 0.10, 0.05, "Extensions", "#fef3c7", "#d97706")

    _card(
        ax,
        0.525,
        0.71,
        0.20,
        0.16,
        "iv. Preprocessing",
        [
            "Split before preprocessing",
            "Train-only MissForest",
            "Landmark truncation",
            "Scale on training data only",
        ],
        body_size=9.4,
    )
    _chip(ax, 0.64, 0.76, 0.06, 0.033, "No", "#f8fafc", "#94a3b8")
    _chip(ax, 0.64, 0.725, 0.06, 0.033, "future", "#f8fafc", "#94a3b8")
    _chip(ax, 0.64, 0.69, 0.06, 0.033, "leakage", "#f8fafc", "#94a3b8")

    _card(
        ax,
        0.745,
        0.71,
        0.22,
        0.16,
        "v. Representation",
        [
            "Fixed landmark",
            "Wide-format features",
            "Current labs + deltas",
            "",
            "Rolling / recurrent",
            "Wide -> long",
            "Patient-interval rows",
            "Target = next-window relapse",
        ],
        body_size=9.0,
    )

    _card(
        ax,
        0.525,
        0.41,
        0.44,
        0.24,
        "vi. Model development",
        [],
        body_size=8.8,
    )

    chips = [
        ("LR", "#dbeafe", "#3b82f6"),
        ("Elastic LR", "#dbeafe", "#3b82f6"),
        ("SVM", "#ffedd5", "#ea580c"),
        ("RF", "#dcfce7", "#16a34a"),
        ("Balanced RF", "#dcfce7", "#16a34a"),
        ("LightGBM", "#bbf7d0", "#15803d"),
        ("MLP", "#f5d0fe", "#a21caf"),
        ("Cox PH", "#fef3c7", "#ca8a04"),
        ("Stacking", "#e0f2fe", "#0284c7"),
    ]
    chip_x0, chip_y0, chip_w, chip_h = 0.55, 0.535, 0.075, 0.036
    for idx, (label, fc, ec) in enumerate(chips):
        row = idx // 5
        col = idx % 5
        _chip(ax, chip_x0 + col * 0.085, chip_y0 - row * 0.055, chip_w, chip_h, label, fc, ec, size=8.6)

    funnel = [
        (0.61, 0.455, 0.27, 0.028, "RandomizedSearchCV"),
        (0.63, 0.417, 0.23, 0.028, "GroupKFold + OOF threshold"),
        (0.655, 0.379, 0.18, 0.028, "Best model for evaluation"),
    ]
    for x, y, w, h, label in funnel:
        poly = mpatches.Polygon(
            [[x, y + h], [x + w, y + h], [x + w - 0.02, y], [x + 0.02, y]],
            closed=True,
            facecolor="#dbe7ea",
            edgecolor="#68808b",
            linewidth=1.25,
        )
        ax.add_patch(poly)
        ax.text(x + w / 2, y + h * 0.53, label, ha="center", va="center", fontsize=9.7, weight="bold", color="#102a43")

    _card(
        ax,
        0.525,
        0.19,
        0.44,
        0.15,
        "vii. Outputs",
        [
            "Figures and tables saved under results/",
            "Fixed landmark, rolling relapse warning, and patient-level risk outputs",
        ],
        body_size=9.4,
    )
    _chip(ax, 0.56, 0.208, 0.14, 0.038, "Fixed landmark", "#dbeafe", "#3b82f6")
    _chip(ax, 0.72, 0.208, 0.13, 0.038, "Dynamic relapse", "#dcfce7", "#16a34a")
    _chip(ax, 0.87, 0.208, 0.08, 0.038, "Risk strata", "#fef3c7", "#d97706")

    web = mpatches.FancyBboxPatch(
        (0.525, 0.08),
        0.44,
        0.075,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#f8fafc",
        edgecolor="#8da2b1",
        linewidth=1.35,
    )
    ax.add_patch(web)
    ax.text(0.545, 0.137, "viii. WebApp", ha="left", va="center", fontsize=12.5, weight="bold", color="#111827")
    ax.text(
        0.545,
        0.110,
        "Interactive relapse-risk web application (planned)",
        ha="left",
        va="center",
        fontsize=10,
        color="#334155",
        style="italic",
    )
    ax.text(0.545, 0.088, "Legend", ha="left", va="center", fontsize=9.2, weight="bold", color="#475569")
    _chip(ax, 0.60, 0.075, 0.10, 0.03, "Input data", "#dbeafe", "#3b82f6", size=8.2)
    _chip(ax, 0.715, 0.075, 0.11, 0.03, "Risk score", "#dcfce7", "#16a34a", size=8.2)
    _chip(ax, 0.84, 0.075, 0.10, 0.03, "Risk tier", "#fef3c7", "#d97706", size=8.2)

    _arrow(ax, 0.47, 0.30, 0.51, 0.30, color="#64748b", lw=2.1)
    _arrow(ax, 0.625, 0.71, 0.625, 0.65, color="#64748b", lw=1.8)
    _arrow(ax, 0.855, 0.71, 0.855, 0.65, color="#64748b", lw=1.8)
    _arrow(ax, 0.745, 0.41, 0.745, 0.34, color="#64748b", lw=1.8)
    _arrow(ax, 0.745, 0.19, 0.745, 0.155, color="#16a34a", lw=2.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_paper_method_framework(Path("results/repluse/Paper_Method_Framework.png"))
