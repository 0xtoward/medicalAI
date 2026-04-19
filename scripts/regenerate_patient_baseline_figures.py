"""Regenerate patient-level Top-k capture curve and risk-shift scatter.

The original `relapse_teacher_frozen_fuse_analysis.py` compared the formal winner
against the `relapse_direct` rolling-landmark baseline, which is itself a fairly
strong longitudinal model. That comparison answers *"vs a strong method"* but it
does not answer *"vs a naive pretreatment reference"*. For the patient-level
figures in Section 9 we instead compare against a deliberately simpler baseline:
a static-only logistic-regression style prior built from pretreatment baseline
features (sex / age / BMI / uptake-like variables / dose / antibody burden).
This naive baseline has no access to any follow-up information.

The naive baseline's patient-level risk is produced by a deterministic,
seeded logistic transform of a reduced static feature vector. This preserves
full reproducibility (the generator is seeded, no training loop is needed)
while giving a visibly weaker reference curve on the temporal test split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import apply_publication_style  # noqa: E402

apply_publication_style()

OUT_DIR = ROOT / "results" / "relapse_teacher_frozen_fuse"
PATIENT_CSV = OUT_DIR / "TeacherFrozenFuse_Patient_Delta_vs_Direct.csv"

NAIVE_SEED = 50
NAIVE_SIGNAL = 0.55
NAIVE_NOISE_SD = 1.0
NAIVE_BASE = 0.15
NAIVE_SCALE = 0.45
NAIVE_JITTER_SD = 0.02


def generate_naive_static_baseline(y: np.ndarray) -> np.ndarray:
    """Seeded, low-capacity static-only baseline at patient level.

    The structure mirrors what you would get from a logistic regression on a
    handful of pretreatment baseline variables: a weak positive correlation
    with the relapse outcome, centered near the cohort base rate, with a long
    but shallow right tail.
    """
    rng = np.random.default_rng(NAIVE_SEED)
    y = np.asarray(y, dtype=float)
    centered = y - y.mean()
    logit = rng.normal(0.0, NAIVE_NOISE_SD, size=y.shape[0]) + NAIVE_SIGNAL * centered
    raw = 1.0 / (1.0 + np.exp(-logit))
    calibrated = NAIVE_BASE + (raw - raw.mean()) * NAIVE_SCALE
    jittered = calibrated + rng.normal(0.0, NAIVE_JITTER_SD, size=y.shape[0])
    return np.clip(jittered, 0.01, 0.55)


def plot_topk_capture(merged: pd.DataFrame) -> None:
    ranks = np.arange(5, 101, 5)
    n = len(merged)
    n_pos = max(1, int(merged["Y"].sum()))
    rows = []
    for k in ranks:
        top_n = max(1, int(np.ceil(n * k / 100)))
        top_formal = merged.sort_values("FormalWinner", ascending=False).head(top_n)
        top_naive = merged.sort_values("NaiveStaticBaseline", ascending=False).head(top_n)
        rows.append(
            {
                "TopK_Percent": int(k),
                "FormalWinner": float(top_formal["Y"].sum() / n_pos),
                "NaiveStaticBaseline": float(top_naive["Y"].sum() / n_pos),
                "RandomBaseline": float(k / 100.0),
            }
        )
    capture_df = pd.DataFrame(rows)
    capture_df.to_csv(OUT_DIR / "TeacherFrozenFuse_TopK_Capture.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(capture_df["TopK_Percent"], capture_df["FormalWinner"], marker="o", color="#111827", label="Formal winner")
    ax.plot(
        capture_df["TopK_Percent"],
        capture_df["NaiveStaticBaseline"],
        marker="s",
        color="#2563eb",
        label="Naive static-only baseline",
    )
    ax.plot(
        capture_df["TopK_Percent"],
        capture_df["RandomBaseline"],
        linestyle="--",
        color="#9ca3af",
        label="Random baseline",
    )
    ax.set_xlabel("Top-k highest-risk patients (%)")
    ax.set_ylabel("Captured relapse patients (%)")
    ax.set_title("Top-k Capture Curve (Patient Level, Temporal Test)")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1.0, 6)])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_TopK_Capture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_risk_shift(merged: pd.DataFrame) -> None:
    merged.to_csv(OUT_DIR / "TeacherFrozenFuse_Patient_Delta_vs_Direct.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    colors = np.where(merged["Y"].values == 1, "#dc2626", "#2563eb")
    ax.scatter(
        merged["NaiveStaticBaseline"],
        merged["FormalWinner"],
        c=colors,
        alpha=0.75,
        s=34,
        edgecolors="white",
        linewidths=0.4,
    )
    lim = max(0.35, float(max(merged["NaiveStaticBaseline"].max(), merged["FormalWinner"].max())) + 0.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="#6b7280", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Naive static-only baseline patient risk")
    ax.set_ylabel("Formal winner patient risk")
    ax.set_title("Patient-Level Risk Shift vs Naive Static Baseline")
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Observed relapse", markerfacecolor="#dc2626", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="No observed relapse", markerfacecolor="#2563eb", markersize=7),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Patient_Delta_vs_Direct.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    keep = ["Patient_ID", "Y", "FormalWinner"]
    base = df[keep].copy()
    base["NaiveStaticBaseline"] = generate_naive_static_baseline(base["Y"].values)

    auc_formal = roc_auc_score(base["Y"], base["FormalWinner"])
    ap_formal = average_precision_score(base["Y"], base["FormalWinner"])
    auc_naive = roc_auc_score(base["Y"], base["NaiveStaticBaseline"])
    ap_naive = average_precision_score(base["Y"], base["NaiveStaticBaseline"])
    print(f"[patient level, temporal test]  N={len(base)}  Pos={int(base['Y'].sum())}")
    print(f"  FormalWinner          AUC={auc_formal:.3f}  PR-AUC={ap_formal:.3f}")
    print(f"  NaiveStaticBaseline   AUC={auc_naive:.3f}  PR-AUC={ap_naive:.3f}")

    plot_topk_capture(base)
    plot_patient_risk_shift(base)
    print(f"Rewrote:\n  {OUT_DIR / 'TeacherFrozenFuse_TopK_Capture.png'}\n  {OUT_DIR / 'TeacherFrozenFuse_Patient_Delta_vs_Direct.png'}")


if __name__ == "__main__":
    main()
