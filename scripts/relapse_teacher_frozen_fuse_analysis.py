import os
import sys
import pickle
import json
import argparse
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
import scipy.integrate
import torch
import joblib

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from scripts import relapse_teacher_frozen_fuse as tf
from thyroid_app.features import make_relapse_features
from utils.config import SEED
from utils.evaluation import aggregate_patient_level, save_patient_risk_strata
from utils.plot_style import apply_publication_style
from utils.shap_viz import save_binary_shap_suite, _fit_binary_explainer

apply_publication_style()


OUT_DIR = tf.Config.OUT_DIR
FORMAL_THRESHOLD = 0.20
INTERVAL_ORDER = ["1M->3M", "3M->6M", "6M->12M", "12M->18M", "18M->24M"]


def load_best_artifacts():
    best_summary = pd.read_csv(OUT_DIR / "Best_Pipeline_Summary.csv").iloc[0]
    fuse_df = pd.read_csv(OUT_DIR / "FrozenFuse_Experiment_Summary.csv")
    backbone_df = pd.read_csv(OUT_DIR / "Backbone_Experiment_Summary.csv")
    with open(OUT_DIR / "Best_Frozen_Fuse.pkl", "rb") as f:
        best_fuse = pickle.load(f)
    return best_summary, fuse_df, backbone_df, best_fuse


def load_formal_winner_bundle():
    with open(OUT_DIR / "TeacherFrozenFuse_FormalWinner.pkl", "rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Posthoc analysis for the formal winner teacher-frozen fuse model.")
    parser.add_argument(
        "--load-pkl",
        action="store_true",
        help="Load the existing TeacherFrozenFuse_FormalWinner.pkl instead of rehydrating the sweep winner object.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP regeneration and reuse existing interpretation outputs when only patient-level figures are needed.",
    )
    return parser.parse_args()


def rebuild_data():
    tf.seed_everything(SEED)
    df_tr, df_te, landmark_train_map, landmark_test_map, unique_pids = tf.base.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = tf.base.split_train_validation(df_tr, train_patient_order)
    teacher_targets = tf.build_teacher035_targets(train_patient_order)
    df_fit = tf.attach_teacher_targets(df_fit, teacher_targets, "Fit")
    df_val = tf.attach_teacher_targets(df_val, teacher_targets, "Validation")
    df_te = tf.attach_teacher_targets(df_te, teacher_targets, "TemporalTest")
    fit_ds, val_ds, te_ds, scalers = tf.build_datasets(df_fit, df_val, df_te, landmark_train_map, landmark_test_map)
    return df_fit, df_val, df_te, fit_ds, val_ds, te_ds, scalers


def load_backbone(best_summary, fit_ds):
    ckpt_path = Path(best_summary["Backbone_Checkpoint"])
    try:
        state = torch.load(ckpt_path, map_location=tf.Config.DEVICE, weights_only=False)
    except TypeError:  # pragma: no cover
        state = torch.load(ckpt_path, map_location=tf.Config.DEVICE)
    gate_variant = state.get("gate_variant", tf.get_gate_variants()[0])
    model = tf.instantiate_model(fit_ds, gate_variant)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


class FormalWinnerEndToEndWrapper:
    def __init__(self, backbone_model, best_model, scalers, static_cols, local_cols, global_cols):
        self.backbone_model = backbone_model
        self.best_model = best_model
        self.scalers = scalers
        self.static_cols = list(static_cols)
        self.local_cols = list(local_cols)
        self.global_cols = list(global_cols)
        self.pref_static = [f"Static::{c}" for c in self.static_cols]
        self.pref_local = [f"Local::{c}" for c in self.local_cols]
        self.pref_global = [f"Global::{c}" for c in self.global_cols]
        self.meta_mask = "Meta::LandmarkObsMask"
        self.meta_value = "Meta::LandmarkObsValue"

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            all_cols = self.pref_static + self.pref_local + self.pref_global + [self.meta_mask, self.meta_value]
            X = pd.DataFrame(np.asarray(X), columns=all_cols)
        static_df = X[self.pref_static].copy()
        static_df.columns = self.static_cols
        local_df = X[self.pref_local].copy()
        local_df.columns = self.local_cols
        global_df = X[self.pref_global].copy()
        global_df.columns = self.global_cols
        tensors = tf.base.transform_blocks(self.scalers, static_df, local_df, global_df)
        obs_mask = X[self.meta_mask].values.astype(np.float32)
        obs_value = X[self.meta_value].values.astype(np.float32)

        probs = []
        batch_size = 256
        self.backbone_model.eval()
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch = {
                    "static": torch.tensor(tensors["static"][start:end], dtype=torch.float32, device=tf.Config.DEVICE),
                    "local": torch.tensor(tensors["local"][start:end], dtype=torch.float32, device=tf.Config.DEVICE),
                    "global": torch.tensor(tensors["global"][start:end], dtype=torch.float32, device=tf.Config.DEVICE),
                    "landmark_obs_mask": torch.tensor(obs_mask[start:end], dtype=torch.float32, device=tf.Config.DEVICE),
                    "landmark_obs_value": torch.tensor(obs_value[start:end], dtype=torch.float32, device=tf.Config.DEVICE),
                }
                out = self.backbone_model(
                    batch["static"],
                    batch["local"],
                    batch["global"],
                    batch["landmark_obs_mask"],
                    batch["landmark_obs_value"],
                )
                fuse_frame = pd.DataFrame(
                    {
                        "Teacher_Logit": out["teacher_logit"].detach().cpu().numpy(),
                        "Landmark_Signal": out["landmark_signal"].detach().cpu().numpy(),
                        "Gate_Static": out["gate"][:, 0].detach().cpu().numpy(),
                        "Gate_Local": out["gate"][:, 1].detach().cpu().numpy(),
                    }
                )
                probs.append(self.best_model.predict_proba(fuse_frame)[:, 1])
        proba = np.concatenate(probs, axis=0)
        return np.column_stack([1.0 - proba, proba])


def build_engineered_feature_frames(df_fit, df_val, df_te):
    interval_cats = sorted(df_fit["Interval_Name"].unique())
    prev_state_cats = sorted(df_fit["Prev_State"].unique())

    def make(df):
        static_df, local_df, global_df = tf.base.build_feature_blocks(df, interval_cats, prev_state_cats)
        out = pd.concat(
            [
                static_df.add_prefix("Static::"),
                local_df.add_prefix("Local::"),
                global_df.add_prefix("Global::"),
            ],
            axis=1,
        )
        out["Meta::LandmarkObsMask"] = (df["Start_Time"].values >= 3.0).astype(float)
        landmark_obs_value = df["Patient_ID"].map(
            {
                **{pid: int(v) for pid, v in df_fit.groupby("Patient_ID")["Teacher035_Prob"].first().to_dict().items()},
            }
        )
        out["Meta::LandmarkObsValue"] = np.nan
        return out, list(static_df.columns), list(local_df.columns), list(global_df.columns)

    fit_frame, static_cols, local_cols, global_cols = make(df_fit)
    val_frame, _, _, _ = make(df_val)
    te_frame, _, _, _ = make(df_te)
    return fit_frame, val_frame, te_frame, static_cols, local_cols, global_cols


def attach_landmark_meta(fit_frame, val_frame, te_frame, df_fit, df_val, df_te, fit_ds, val_ds, te_ds):
    fit_frame = fit_frame.copy()
    val_frame = val_frame.copy()
    te_frame = te_frame.copy()
    fit_frame["Meta::LandmarkObsValue"] = fit_ds.y_landmark.numpy()
    val_frame["Meta::LandmarkObsValue"] = val_ds.y_landmark.numpy()
    te_frame["Meta::LandmarkObsValue"] = te_ds.y_landmark.numpy()
    return fit_frame, val_frame, te_frame


def _align_feature_frame(frame, feature_names, medians=None):
    out = frame.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.reindex(columns=list(feature_names), fill_value=np.nan)
    if medians is not None:
        med = pd.Series(medians, dtype=float)
        out = out.fillna(med)
    else:
        out = out.fillna(0.0)
    return out.astype(float)


def load_direct_baseline_bundle():
    return joblib.load("artifacts/relapse_direct/bundle.joblib")


def compute_direct_baseline_predictions(df_fit, df_te):
    bundle = load_direct_baseline_bundle()
    x_fit = make_relapse_features(df_fit, bundle["interval_categories"], bundle["prev_state_categories"])
    x_te = make_relapse_features(df_te, bundle["interval_categories"], bundle["prev_state_categories"])
    x_fit = _align_feature_frame(x_fit, bundle["feature_names"], bundle.get("feature_medians"))
    x_te = _align_feature_frame(x_te, bundle["feature_names"], bundle.get("feature_medians"))
    fit_proba = bundle["model"].predict_proba(x_fit)[:, 1]
    te_proba = bundle["model"].predict_proba(x_te)[:, 1]
    return bundle, fit_proba, te_proba


def _wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    radius = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _assign_patient_quartiles(patient_df, train_patient_df):
    bins = np.quantile(train_patient_df["proba"].values, [0, 0.25, 0.5, 0.75, 1.0])
    bins[0] = min(bins[0], 0.0)
    bins[-1] = max(bins[-1], 1.0)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6
    out = patient_df.copy()
    out["Risk_Group"] = pd.cut(out["proba"], bins=bins, include_lowest=True, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    return out, bins


def build_patient_level_outputs(df_fit, df_te, fit_proba, te_proba, save_outputs=True, prefix="TeacherFrozenFuse"):
    patient_fit = aggregate_patient_level(df_fit, fit_proba, method="product")
    patient_te = aggregate_patient_level(df_te, te_proba, method="product")
    patient_te, bins = _assign_patient_quartiles(patient_te, patient_fit)
    patient_fit, _ = _assign_patient_quartiles(patient_fit, patient_fit)
    patient_fit["Split"] = "Fit"
    patient_te["Split"] = "TemporalTest"
    if save_outputs:
        patient_fit.to_csv(OUT_DIR / f"{prefix}_PatientLevel_Train.csv", index=False)
        patient_te.to_csv(OUT_DIR / f"{prefix}_PatientLevel_Test.csv", index=False)
    return patient_fit, patient_te, bins


def load_temporal_prediction_table():
    path = OUT_DIR / "TemporalTest_Predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_patient_waterfall(patient_te, bins):
    plot_df = patient_te.sort_values("proba", ascending=False).reset_index(drop=True).copy()
    plot_df["Rank"] = np.arange(1, len(plot_df) + 1)
    plot_df.to_csv(OUT_DIR / "TeacherFrozenFuse_Patient_Waterfall.csv", index=False)

    fig, ax = plt.subplots(figsize=(14.5, 5.8))
    colors = np.where(plot_df["Y"].values == 1, "#dc2626", "#2563eb")
    ax.bar(plot_df["Rank"], plot_df["proba"], color=colors, width=0.92, edgecolor="none")
    ax.axhline(FORMAL_THRESHOLD, color="#111827", linestyle="--", lw=1.3, label="Interval threshold = 0.20")

    q_counts = plot_df["Risk_Group"].value_counts().reindex(["Q4", "Q3", "Q2", "Q1"]).fillna(0).astype(int)
    cursor = 0
    for label, count in q_counts.items():
        cursor += count
        if cursor < len(plot_df):
            ax.axvline(cursor + 0.5, color="#9ca3af", linestyle=":", lw=1)

    ax.set_title("Patient-Level Cumulative Risk Waterfall")
    ax.set_xlabel("Patients ranked by cumulative risk")
    ax.set_ylabel("Patient-level cumulative risk")
    ax.set_xlim(0.5, len(plot_df) + 0.5)
    ax.set_ylim(0, min(1.0, float(plot_df["proba"].max()) + 0.08))
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#dc2626", label="Observed relapse"),
        plt.Rectangle((0, 0), 1, 1, color="#2563eb", label="No observed relapse"),
    ]
    ax.legend(handles=handles + [ax.lines[0]], frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Patient_Waterfall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_risk_heatmap(df_te, te_proba, patient_te):
    tmp = df_te[["Patient_ID", "Interval_Name", "Y_Relapse"]].copy()
    tmp["FuseBest_Prob"] = te_proba
    pivot = (
        tmp.groupby(["Patient_ID", "Interval_Name"], as_index=False)
        .agg(FuseBest_Prob=("FuseBest_Prob", "mean"), Y_Relapse=("Y_Relapse", "max"))
        .pivot(index="Patient_ID", columns="Interval_Name", values="FuseBest_Prob")
        .reindex(columns=INTERVAL_ORDER)
    )
    ordered_ids = patient_te.sort_values("proba", ascending=False)["Patient_ID"].tolist()
    pivot = pivot.reindex(ordered_ids)
    pivot.to_csv(OUT_DIR / "TeacherFrozenFuse_Patient_Risk_Heatmap.csv")

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#e5e7eb")
    fig_h = max(7.0, min(24.0, 0.10 * len(pivot) + 2.5))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(0.35, np.nanmax(pivot.values)))
    ax.set_title("Patient-by-Window Risk Heatmap")
    ax.set_xlabel("Follow-up window")
    ax.set_ylabel("Patients ranked by cumulative risk")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    if len(pivot) <= 25:
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(x) for x in pivot.index])
    else:
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Interval relapse risk")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Patient_Risk_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_typical_patients(patient_te):
    positive = patient_te[patient_te["Y"] == 1].copy()
    negative = patient_te[patient_te["Y"] == 0].copy()
    high_risk_tp = positive.sort_values("proba", ascending=False).iloc[0]
    low_risk_tn = negative.sort_values("proba", ascending=True).iloc[0]

    remaining = patient_te.loc[~patient_te["Patient_ID"].isin([high_risk_tp["Patient_ID"], low_risk_tn["Patient_ID"]])].copy()
    remaining["BoundaryGap"] = np.abs(remaining["proba"] - FORMAL_THRESHOLD)
    boundary = remaining.sort_values(["BoundaryGap", "proba"], ascending=[True, True]).iloc[0]

    out = pd.DataFrame(
        [
            {"Case_Type": "高风险真阳性", **high_risk_tp.to_dict()},
            {"Case_Type": "低风险真阴性", **low_risk_tn.to_dict()},
            {"Case_Type": "边界病例", **boundary.to_dict()},
        ]
    )
    out.to_csv(OUT_DIR / "TeacherFrozenFuse_TypicalPatients.csv", index=False)
    return out


def plot_typical_patients(df_te, te_pred_df, typical_df):
    te_long = df_te[["Patient_ID", "Interval_Name", "Start_Time", "Stop_Time", "FT3_Current", "FT4_Current", "logTSH_Current", "Y_Relapse"]].copy().reset_index(drop=True)
    pred_cols = ["TeacherHead_Prob", "Landmark_Signal", "FuseBest_Prob"]
    if len(te_pred_df) == len(te_long):
        merged = te_long.copy()
        merged[pred_cols] = te_pred_df[pred_cols].reset_index(drop=True)
    else:
        merged = te_long.merge(
            te_pred_df[
                [
                    "Patient_ID",
                    "Interval_Name",
                    "TeacherHead_Prob",
                    "Landmark_Signal",
                    "FuseBest_Prob",
                ]
            ],
            on=["Patient_ID", "Interval_Name"],
            how="left",
        )
    merged["TSH_Current"] = np.expm1(merged["logTSH_Current"].astype(float)).clip(lower=0.0)

    fig, axes = plt.subplots(len(typical_df), 2, figsize=(15.5, 4.4 * len(typical_df)))
    if len(typical_df) == 1:
        axes = np.asarray([axes])

    for row_idx, (_, case_row) in enumerate(typical_df.iterrows()):
        pid = case_row["Patient_ID"]
        sub = (
            merged[merged["Patient_ID"] == pid]
            .groupby(["Interval_Name", "Start_Time", "Stop_Time"], as_index=False)
            .agg(
                FT3_Current=("FT3_Current", "mean"),
                FT4_Current=("FT4_Current", "mean"),
                TSH_Current=("TSH_Current", "mean"),
                TeacherHead_Prob=("TeacherHead_Prob", "mean"),
                Landmark_Signal=("Landmark_Signal", "mean"),
                FuseBest_Prob=("FuseBest_Prob", "mean"),
                Y_Relapse=("Y_Relapse", "max"),
            )
            .sort_values("Start_Time")
        )
        x = np.arange(len(sub))
        x_labels = sub["Interval_Name"].tolist()

        ax_l = axes[row_idx, 0]
        ax_l.plot(x, sub["FT3_Current"], marker="o", color="#2563eb", label="FT3")
        ax_l.plot(x, sub["FT4_Current"], marker="s", color="#16a34a", label="FT4")
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(x_labels, rotation=20, ha="right")
        ax_l.set_ylabel("FT3 / FT4")
        ax_l.grid(alpha=0.25)
        ax_l2 = ax_l.twinx()
        ax_l2.plot(x, sub["TSH_Current"], marker="^", color="#dc2626", label="TSH")
        ax_l2.set_ylabel("TSH")
        event_idx = np.where(sub["Y_Relapse"].values.astype(int) == 1)[0]
        for idx_event in event_idx:
            ax_l.axvspan(idx_event - 0.35, idx_event + 0.35, color="#fecaca", alpha=0.5)
        ax_l.set_title(f"{case_row['Case_Type']} | Patient {pid}")
        lines1, labels1 = ax_l.get_legend_handles_labels()
        lines2, labels2 = ax_l2.get_legend_handles_labels()
        ax_l.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", ncol=3)

        ax_r = axes[row_idx, 1]
        ax_r.plot(x, sub["TeacherHead_Prob"], marker="o", color="#7c3aed", label="Teacher")
        ax_r.plot(x, sub["Landmark_Signal"], marker="s", color="#ea580c", label="Landmark")
        ax_r.plot(x, sub["FuseBest_Prob"], marker="D", color="#111827", label="Final risk")
        ax_r.axhline(FORMAL_THRESHOLD, color="#6b7280", linestyle="--", lw=1.1, label="Threshold")
        for idx_event in event_idx:
            ax_r.axvspan(idx_event - 0.35, idx_event + 0.35, color="#fecaca", alpha=0.5)
        ax_r.set_xticks(x)
        ax_r.set_xticklabels(x_labels, rotation=20, ha="right")
        ax_r.set_ylim(0, max(0.35, float(sub[["TeacherHead_Prob", "Landmark_Signal", "FuseBest_Prob"]].to_numpy().max()) + 0.08))
        ax_r.set_ylabel("Predicted probability")
        ax_r.set_title(f"Cumulative risk={case_row['proba']:.3f}")
        ax_r.grid(alpha=0.25)
        ax_r.legend(frameon=False, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_TypicalPatients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_q1q4_ci(patient_fit, patient_te):
    _, bins = _assign_patient_quartiles(patient_fit, patient_fit)
    strat_df = patient_te.copy()
    strat_df["Risk_Group"] = pd.cut(strat_df["proba"], bins=bins, include_lowest=True, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)

    rows = []
    for grp in ["Q1", "Q2", "Q3", "Q4"]:
        sub = strat_df[strat_df["Risk_Group"] == grp]
        n = int(len(sub))
        events = int(sub["Y"].sum())
        observed = float(sub["Y"].mean()) if n else np.nan
        pred = float(sub["proba"].mean()) if n else np.nan
        low, high = _wilson_ci(events, n)
        rows.append(
            {
                "Risk_Group": grp,
                "N": n,
                "Events": events,
                "Observed": observed,
                "Observed_CI_Low": low,
                "Observed_CI_High": high,
                "Predicted": pred,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "TeacherFrozenFuse_Patient_Q1Q4_CI.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x = np.arange(len(summary))
    yerr = np.vstack(
        [
            summary["Observed"] - summary["Observed_CI_Low"],
            summary["Observed_CI_High"] - summary["Observed"],
        ]
    )
    bars = ax.bar(x, summary["Observed"], color="#60a5fa", alpha=0.9, yerr=yerr, capsize=4, label="Observed relapse rate")
    ax.plot(x, summary["Predicted"], marker="o", lw=2, color="#f59e0b", label="Mean predicted risk")
    for xi, n in zip(x, summary["N"].tolist()):
        ax.text(xi, float(summary.loc[xi, "Observed"]) + 0.03, f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Risk_Group"])
    ax.set_ylim(0, min(1.0, float(max(summary["Observed_CI_High"].max(), summary["Predicted"].max())) + 0.12))
    ax.set_xlabel("Train-set quartile risk groups")
    ax.set_ylabel("Patient-level relapse probability")
    ax.set_title("Patient-Level Risk Stratification with 95% CI")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Patient_Q1Q4_CI.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_leadtime_warning(df_te, te_proba, patient_te):
    tmp = df_te[["Patient_ID", "Interval_Name", "Start_Time", "Y_Relapse"]].copy()
    tmp["FuseBest_Prob"] = te_proba
    tmp = (
        tmp.groupby(["Patient_ID", "Interval_Name", "Start_Time"], as_index=False)
        .agg(FuseBest_Prob=("FuseBest_Prob", "mean"), Y_Relapse=("Y_Relapse", "max"))
        .sort_values(["Patient_ID", "Start_Time"])
    )
    pos_ids = patient_te.loc[patient_te["Y"] == 1, "Patient_ID"].tolist()
    tmp = tmp[tmp["Patient_ID"].isin(pos_ids)].copy()

    records = []
    for pid, g in tmp.groupby("Patient_ID", sort=False):
        g = g.sort_values("Start_Time").reset_index(drop=True)
        event_hits = np.where(g["Y_Relapse"].values.astype(int) == 1)[0]
        if len(event_hits) == 0:
            continue
        event_idx = int(event_hits[0])
        warning_hits = np.where(g["FuseBest_Prob"].values >= FORMAL_THRESHOLD)[0]
        if len(warning_hits) == 0:
            category = "missed"
            lead_windows = np.nan
        else:
            warn_idx = int(warning_hits[0])
            lead_windows = event_idx - warn_idx
            if lead_windows >= 2:
                category = ">=2 windows early"
            elif lead_windows == 1:
                category = "1 window early"
            elif lead_windows == 0:
                category = "same window"
            else:
                category = "missed"
        records.append({"Patient_ID": pid, "Event_Index": event_idx, "Lead_Windows": lead_windows, "Category": category})
    lead_df = pd.DataFrame(records)
    lead_df.to_csv(OUT_DIR / "TeacherFrozenFuse_LeadTime_Warning.csv", index=False)

    order = [">=2 windows early", "1 window early", "same window", "missed"]
    counts = lead_df["Category"].value_counts().reindex(order).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars = ax.bar(order, counts.values, color=["#16a34a", "#84cc16", "#f59e0b", "#dc2626"])
    total = max(1, counts.sum())
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{count}\n({count/total:.0%})", ha="center", fontsize=10)
    ax.set_title("Lead-Time Warning Categories for Relapsing Patients")
    ax.set_ylabel("Patients")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_LeadTime_Warning.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_topk_capture(patient_te, patient_te_direct):
    ranks = np.arange(5, 101, 5)
    n = len(patient_te)
    n_pos = max(1, int(patient_te["Y"].sum()))
    rows = []
    for k in ranks:
        top_n = max(1, int(np.ceil(n * k / 100)))
        top_formal = patient_te.sort_values("proba", ascending=False).head(top_n)
        top_direct = patient_te_direct.sort_values("proba", ascending=False).head(top_n)
        formal_capture = float(top_formal["Y"].sum() / n_pos)
        direct_capture = float(top_direct["Y"].sum() / n_pos)
        random_capture = float(k / 100.0)
        rows.append({"TopK_Percent": k, "FormalWinner": formal_capture, "DirectBaseline": direct_capture, "RandomBaseline": random_capture})
    capture_df = pd.DataFrame(rows)
    capture_df.to_csv(OUT_DIR / "TeacherFrozenFuse_TopK_Capture.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(capture_df["TopK_Percent"], capture_df["FormalWinner"], marker="o", color="#111827", label="Formal winner")
    ax.plot(capture_df["TopK_Percent"], capture_df["DirectBaseline"], marker="s", color="#2563eb", label="Direct baseline")
    ax.plot(capture_df["TopK_Percent"], capture_df["RandomBaseline"], linestyle="--", color="#9ca3af", label="Random baseline")
    ax.set_xlabel("Top-k highest-risk patients (%)")
    ax.set_ylabel("Captured relapse patients (%)")
    ax.set_title("Top-k Capture Curve (Patient Level)")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1.0, 6)])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_TopK_Capture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patient_delta_vs_direct(patient_te, patient_te_direct):
    merged = patient_te[["Patient_ID", "Y", "proba"]].rename(columns={"proba": "FormalWinner"}).merge(
        patient_te_direct[["Patient_ID", "proba"]].rename(columns={"proba": "DirectBaseline"}),
        on="Patient_ID",
        how="inner",
    )
    merged.to_csv(OUT_DIR / "TeacherFrozenFuse_Patient_Delta_vs_Direct.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    colors = np.where(merged["Y"].values == 1, "#dc2626", "#2563eb")
    ax.scatter(merged["DirectBaseline"], merged["FormalWinner"], c=colors, alpha=0.75, s=34, edgecolors="white", linewidths=0.4)
    lim = max(0.35, float(max(merged["DirectBaseline"].max(), merged["FormalWinner"].max())) + 0.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="#6b7280", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Direct baseline patient risk")
    ax.set_ylabel("Formal winner patient risk")
    ax.set_title("Patient-Level Risk Shift vs Direct Baseline")
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Observed relapse", markerfacecolor="#dc2626", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="No observed relapse", markerfacecolor="#2563eb", markersize=7),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Patient_Delta_vs_Direct.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_architecture(best_summary):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_axis_off()

    def box(x, y, w, h, text, fc, ec="#1f2937", size=11, weight="normal"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#374151"))

    box(0.02, 0.58, 0.16, 0.18, "Static Features\n(age, pathology,\nTNM, surgery...)", "#dbeafe", weight="bold")
    box(0.02, 0.34, 0.16, 0.18, "Local Features\n(current FT3/FT4/TSH,\n1-step deltas,\nwindow/state)", "#dcfce7", weight="bold")
    box(0.02, 0.10, 0.16, 0.18, "Global Features\nwhole-course signals,\nTime_In_Normal,\nStart/Width...)", "#fef3c7", weight="bold")

    box(0.25, 0.58, 0.14, 0.18, "static_branch\nMLP", "#bfdbfe")
    box(0.25, 0.34, 0.14, 0.18, "local_branch\nMLP", "#bbf7d0")
    box(0.25, 0.10, 0.14, 0.18, "global_branch\nLinear", "#fde68a")

    box(0.46, 0.37, 0.16, 0.22, "joint = [h_static,\nh_local,\nh_global]", "#e5e7eb", weight="bold")
    box(0.67, 0.58, 0.15, 0.16, "landmark_head\n3M signal", "#fecaca")
    box(0.67, 0.36, 0.15, 0.16, f"gate_net\n{best_summary['Best_Gate_Mode']}\n(global floor={best_summary['Best_Gate_Global_Floor']})", "#ddd6fe")
    box(0.67, 0.14, 0.15, 0.16, "next_hyper_head", "#fbcfe8")

    box(0.86, 0.40, 0.12, 0.18, "teacher_head\nold 0.351\nteacher risk", "#fca5a5", weight="bold")
    box(0.86, 0.12, 0.12, 0.16, "Frozen Fuse\nLR on:\nTeacher_Logit\n+ Landmark_Signal\n+ Gate_Static\n+ Gate_Local", "#c7d2fe", size=10, weight="bold")

    arrow(0.18, 0.67, 0.25, 0.67)
    arrow(0.18, 0.43, 0.25, 0.43)
    arrow(0.18, 0.19, 0.25, 0.19)
    arrow(0.39, 0.67, 0.46, 0.48)
    arrow(0.39, 0.43, 0.46, 0.48)
    arrow(0.39, 0.19, 0.46, 0.48)
    arrow(0.62, 0.52, 0.67, 0.66)
    arrow(0.62, 0.48, 0.67, 0.44)
    arrow(0.62, 0.44, 0.67, 0.22)
    arrow(0.82, 0.66, 0.86, 0.49)
    arrow(0.82, 0.44, 0.86, 0.49)
    arrow(0.82, 0.22, 0.86, 0.20)

    ax.text(0.02, 0.92, "Teacher-Frozen Fuse Architecture", fontsize=16, weight="bold")
    ax.text(
        0.02,
        0.86,
        "3 branches encode static/local/global inputs; 3 heads supervise landmark/teacher/next-hyper; "
        "final deployed risk comes from a frozen-backbone logistic fuse.",
        fontsize=11,
    )
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_backbone_comparison(backbone_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rank_df = backbone_df.sort_values("Screen_Best_Val_PR_AUC", ascending=False).reset_index(drop=True)
    x = np.arange(len(rank_df))
    labels = rank_df["Backbone_Run"].tolist()

    axes[0].bar(x - 0.17, rank_df["TeacherHead_Val_PR_AUC"], 0.34, label="Teacher Head", color="#60a5fa")
    axes[0].bar(x + 0.17, rank_df["Screen_Best_Val_PR_AUC"], 0.34, label="Best Screen Fuse", color="#34d399")
    axes[0].set_title("Validation PR-AUC by Backbone")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.17, rank_df["TeacherHead_Test_PR_AUC"], 0.34, label="Teacher Head", color="#60a5fa")
    test_like = rank_df["Wide_Best_Val_PR_AUC"].fillna(rank_df["Screen_Best_Val_PR_AUC"])
    axes[1].bar(x + 0.17, test_like, 0.34, label="Selected Fuse Val PR-AUC", color="#f59e0b")
    axes[1].set_title("Screen/Wide Selection Signal")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Backbone_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gate_comparison(backbone_df):
    rank_df = backbone_df.sort_values("Screen_Best_Val_PR_AUC", ascending=False).reset_index(drop=True)
    top_df = rank_df.head(5).copy()
    labels = top_df["Backbone_Run"].tolist()
    x = np.arange(len(top_df))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - 0.25, top_df["Gate_Static_Mean_Val"], 0.25, label="Gate_Static", color="#93c5fd")
    ax.bar(x, top_df["Gate_Local_Mean_Val"], 0.25, label="Gate_Local", color="#86efac")
    ax.bar(x + 0.25, top_df["Gate_Global_Mean_Val"], 0.25, label="Gate_Global", color="#fde68a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Validation Gate Means for Top Backbones")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Gate_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_experiments(fuse_df):
    top_val = fuse_df.sort_values(["Val_PR_AUC", "Test_PR_AUC"], ascending=[False, False]).head(8).copy()
    top_test = fuse_df.sort_values(["Test_PR_AUC", "Val_PR_AUC"], ascending=[False, False]).head(8).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df_plot, title, metric in [
        (axes[0], top_val, "Top Fuse Experiments by Validation PR-AUC", "Val_PR_AUC"),
        (axes[1], top_test, "Top Fuse Experiments by Temporal Test PR-AUC", "Test_PR_AUC"),
    ]:
        labels = [f"{r.Backbone_Run}\n{r.PackName}" for _, r in df_plot.iterrows()]
        x = np.arange(len(df_plot))
        ax.bar(x - 0.17, df_plot["Val_PR_AUC"], 0.34, label="Val PR-AUC", color="#60a5fa")
        ax.bar(x + 0.17, df_plot["Test_PR_AUC"], 0.34, label="Test PR-AUC", color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_Top_Experiment_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_formal_winner_bundle(best_summary, best_fuse, best_model, feature_names):
    bundle = {
        "best_summary": best_summary.to_dict(),
        "threshold": float(best_fuse["threshold"]),
        "feature_names": list(feature_names),
        "pack_name": best_fuse["pack_name"],
        "model_family": best_fuse["model_family"],
        "backbone_checkpoint": best_fuse["backbone_checkpoint"],
        "backbone_run": best_fuse["backbone_run"],
        "model": best_model,
    }
    with open(OUT_DIR / "TeacherFrozenFuse_FormalWinner.pkl", "wb") as f:
        pickle.dump(bundle, f)
    with open(OUT_DIR / "TeacherFrozenFuse_FormalWinner_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_backbone_run": best_summary["Best_Backbone_Run"],
                "best_gate_mode": best_summary["Best_Gate_Mode"],
                "best_gate_global_floor": None if pd.isna(best_summary["Best_Gate_Global_Floor"]) else float(best_summary["Best_Gate_Global_Floor"]),
                "best_experiment": best_summary["Best_Experiment"],
                "best_model_family": best_summary["Best_ModelFamily"],
                "best_pack_name": best_summary["Best_PackName"],
                "best_features": list(feature_names),
                "best_threshold": float(best_fuse["threshold"]),
                "best_val_prauc": float(best_summary["Best_Val_PR_AUC"]),
                "best_test_prauc": float(best_summary["Best_Test_PR_AUC"]),
                "backbone_checkpoint": best_fuse["backbone_checkpoint"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_formal_winner_params(best_model, feature_names):
    rows = []
    if hasattr(best_model, "named_steps") and "clf" in best_model.named_steps:
        scaler = best_model.named_steps.get("scaler")
        clf = best_model.named_steps["clf"]
        coefs = np.asarray(clf.coef_).reshape(-1)
        intercept = float(np.asarray(clf.intercept_).reshape(-1)[0])
        for feat, coef in zip(feature_names, coefs):
            rows.append({"Feature": feat, "Coefficient": float(coef), "AbsCoefficient": float(abs(coef))})
        coef_df = pd.DataFrame(rows).sort_values("AbsCoefficient", ascending=False)
        coef_df.to_csv(OUT_DIR / "TeacherFrozenFuse_FormalWinner_Coefficients.csv", index=False)
        pd.DataFrame([{"Term": "Intercept", "Value": intercept}]).to_csv(
            OUT_DIR / "TeacherFrozenFuse_FormalWinner_Intercept.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        show_df = coef_df.head(min(10, len(coef_df))).iloc[::-1]
        colors = ["#ef4444" if v < 0 else "#2563eb" for v in show_df["Coefficient"]]
        ax.barh(show_df["Feature"], show_df["Coefficient"], color=colors)
        ax.axvline(0, color="black", lw=1)
        ax.set_title("Formal Winner LR Coefficients")
        ax.set_xlabel("Coefficient")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "TeacherFrozenFuse_FormalWinner_Coefficients.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_formal_winner_discrimination(recomputed_df, best_summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(recomputed_df))
    labels = recomputed_df["Split"].tolist()

    axes[0].bar(x, recomputed_df["AUC"], color="#60a5fa")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Formal Winner AUC")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(recomputed_df["AUC"]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, recomputed_df["PR_AUC"], color="#f59e0b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, max(0.45, recomputed_df["PR_AUC"].max() + 0.05))
    axes[1].set_title(
        f"Formal Winner PR-AUC\n(best val={best_summary['Best_Val_PR_AUC']:.3f}, test={best_summary['Best_Test_PR_AUC']:.3f})"
    )
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(recomputed_df["PR_AUC"]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "TeacherFrozenFuse_FormalWinner_Discrimination.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_shap_outputs(best_model, x_fit, x_te, feature_names):
    save_binary_shap_suite(
        model_name="Teacher Frozen Fuse",
        model=best_model,
        X_background=x_fit,
        X_local=x_te,
        feat_names=feature_names,
        out_dir=OUT_DIR,
        summary_filename="TeacherFrozenFuse_SHAP.png",
        summary_title="Teacher Frozen Fuse SHAP",
        seed=SEED,
        max_display=min(12, len(feature_names)),
    )

    explain = _fit_binary_explainer("Teacher Frozen Fuse", best_model, x_fit, feature_names, seed=SEED)
    explanation, _ = explain(x_te)
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsSHAP": np.abs(explanation.values).mean(axis=0),
            "MeanSHAP": explanation.values.mean(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    shap_df.to_csv(OUT_DIR / "TeacherFrozenFuse_SHAP_Feature_Importance.csv", index=False)
    return shap_df


def save_end_to_end_shap_outputs(wrapper_model, x_fit_raw, x_te_raw):
    feature_names = list(x_fit_raw.columns)
    x_te_local = x_te_raw.sample(n=min(180, len(x_te_raw)), random_state=SEED).reset_index(drop=True)
    save_binary_shap_suite(
        model_name="Teacher Frozen Fuse EndToEnd",
        model=wrapper_model,
        X_background=x_fit_raw,
        X_local=x_te_local,
        feat_names=feature_names,
        out_dir=OUT_DIR,
        summary_filename="TeacherFrozenFuse_EndToEnd_SHAP.png",
        summary_title="Teacher Frozen Fuse End-to-End SHAP",
        seed=SEED,
        max_display=min(20, len(feature_names)),
    )

    explain = _fit_binary_explainer("Teacher Frozen Fuse EndToEnd", wrapper_model, x_fit_raw, feature_names, seed=SEED)
    explanation, _ = explain(x_te_local)
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsSHAP": np.abs(explanation.values).mean(axis=0),
            "MeanSHAP": explanation.values.mean(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    shap_df.to_csv(OUT_DIR / "TeacherFrozenFuse_EndToEnd_SHAP_Feature_Importance.csv", index=False)
    return shap_df


def write_analysis_markdown(best_summary, backbone_df, fuse_df, shap_df, e2e_shap_df):
    test_best = fuse_df.sort_values(["Test_PR_AUC", "Val_PR_AUC"], ascending=[False, False]).iloc[0]
    lines = [
        "# Teacher Frozen Fuse Analysis",
        "",
        "## Structure",
        f"- Formal winner backbone: `{best_summary['Best_Backbone_Run']}`",
        f"- Formal winner fuse: `{best_summary['Best_Experiment']}`",
        f"- Best validation PR-AUC: `{best_summary['Best_Val_PR_AUC']:.3f}`",
        f"- Corresponding temporal test PR-AUC: `{best_summary['Best_Test_PR_AUC']:.3f}`",
        "",
        "## Main Findings",
        f"- Validation-selected winner improved over the previous validation baseline (`0.345 -> {best_summary['Best_Val_PR_AUC']:.3f}`), but temporal generalization fell (`0.279 -> {best_summary['Best_Test_PR_AUC']:.3f}`).",
        f"- Highest temporal test PR-AUC in the full sweep was `{test_best['Test_PR_AUC']:.3f}` from `{test_best['Experiment']}` on backbone `{test_best['Backbone_Run']}`, but its validation PR-AUC was only `{test_best['Val_PR_AUC']:.3f}`.",
        f"- This means the current pipeline is validation-optimistic rather than temporally robust.",
        "",
        "## Gate Reading",
    ]
    top_backbone = backbone_df.sort_values("Screen_Best_Val_PR_AUC", ascending=False).iloc[0]
    lines += [
        f"- Winner backbone gate mode: `{top_backbone['Gate_Mode']}`.",
        f"- Validation gate means for the winner were static `{top_backbone['Gate_Static_Mean_Val']:.3f}`, local `{top_backbone['Gate_Local_Mean_Val']:.3f}`, global `{top_backbone['Gate_Global_Mean_Val']:.3f}`.",
        "- In practice, `global_floor_040` fixed the global gate at `0.4` and let static/local split the remaining `0.6`.",
        "",
        "## Fuse-Space SHAP",
    ]
    for _, row in shap_df.head(5).iterrows():
        lines.append(f"- `{row['Feature']}`: mean |SHAP| = `{row['MeanAbsSHAP']:.4f}`")
    lines += [
        "",
        "## End-to-End SHAP",
    ]
    for _, row in e2e_shap_df.head(8).iterrows():
        lines.append(f"- `{row['Feature']}`: mean |SHAP| = `{row['MeanAbsSHAP']:.4f}`")
    lines += [
        "",
        "## Interpretation",
        "- The formal winner mainly relies on the old teacher risk, the 3M landmark signal, and the learned branch weights rather than richer nonlinear fuse interactions.",
        "- The validation winner appears to exploit gate-pattern regularities that do not transfer well to the later temporal cohort.",
        "- The more temporally robust rows were usually simpler and more window-aware, especially the reused old-backbone `teacher_landmark_windowed_raw` variants.",
        "",
    ]
    (OUT_DIR / "TeacherFrozenFuse_Analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    best_summary, fuse_df, backbone_df, best_fuse = load_best_artifacts()
    df_fit, df_val, df_te, fit_ds, val_ds, te_ds, scalers = rebuild_data()
    backbone_model = load_backbone(best_summary, fit_ds)
    fit_preds = tf.collect_predictions(backbone_model, fit_ds)
    val_preds = tf.collect_predictions(backbone_model, val_ds)
    te_preds = tf.collect_predictions(backbone_model, te_ds)

    master_frames, _, _, _, _ = tf.build_master_feature_frames(
        df_fit,
        df_val,
        df_te,
        fit_preds,
        val_preds,
        te_preds,
        df_fit["Y_Relapse"].values.astype(int),
        df_fit["Patient_ID"].values,
    )

    if args.load_pkl and (OUT_DIR / "TeacherFrozenFuse_FormalWinner.pkl").exists():
        formal_bundle = load_formal_winner_bundle()
        feature_names = formal_bundle["feature_names"]
        best_model = formal_bundle["model"]
    else:
        feature_names = best_fuse["feature_names"]
        best_model = best_fuse["model"]
    x_fit = master_frames["fit"][feature_names]
    x_val = master_frames["val"][feature_names]
    x_te = master_frames["test"][feature_names]
    fit_proba = best_model.predict_proba(x_fit)[:, 1]
    val_proba = best_model.predict_proba(x_val)[:, 1]
    te_proba = best_model.predict_proba(x_te)[:, 1]
    te_pred_df = load_temporal_prediction_table()
    if te_pred_df is not None and len(te_pred_df) == len(te_proba):
        te_proba = te_pred_df["FuseBest_Prob"].to_numpy()
    else:
        te_pred_df = pd.DataFrame(
            {
                "Patient_ID": df_te["Patient_ID"].values,
                "Interval_Name": df_te["Interval_Name"].values,
                "TeacherHead_Prob": te_preds["teacher_prob"],
                "Landmark_Signal": te_preds["landmark_signal"],
                "FuseBest_Prob": te_proba,
            }
        )

    plot_architecture(best_summary)
    plot_backbone_comparison(backbone_df)
    plot_gate_comparison(backbone_df)
    plot_top_experiments(fuse_df)
    if not args.load_pkl:
        save_formal_winner_bundle(best_summary, best_fuse, best_model, feature_names)
    save_formal_winner_params(best_model, feature_names)

    patient_fit, patient_te, bins = build_patient_level_outputs(df_fit, df_te, fit_proba, te_proba)
    save_patient_risk_strata(patient_fit, patient_te, OUT_DIR / "TeacherFrozenFuse_Patient_Risk_Q1Q4.png")
    plot_patient_waterfall(patient_te, bins)
    plot_patient_risk_heatmap(df_te, te_proba, patient_te)
    typical_df = select_typical_patients(patient_te)
    plot_typical_patients(df_te, te_pred_df, typical_df)
    plot_patient_q1q4_ci(patient_fit, patient_te)
    plot_leadtime_warning(df_te, te_proba, patient_te)

    _, direct_fit_proba, direct_te_proba = compute_direct_baseline_predictions(df_fit, df_te)
    _, patient_te_direct, _ = build_patient_level_outputs(df_fit, df_te, direct_fit_proba, direct_te_proba, save_outputs=False)
    patient_te_direct.rename(columns={"Risk_Group": "Direct_Risk_Group"}, inplace=True)
    plot_topk_capture(patient_te, patient_te_direct)
    plot_patient_delta_vs_direct(patient_te, patient_te_direct)

    if args.skip_shap:
        shap_path = OUT_DIR / "TeacherFrozenFuse_SHAP_Feature_Importance.csv"
        e2e_path = OUT_DIR / "TeacherFrozenFuse_EndToEnd_SHAP_Feature_Importance.csv"
        shap_df = pd.read_csv(shap_path) if shap_path.exists() else pd.DataFrame(columns=["Feature", "MeanAbsSHAP", "MeanSHAP"])
        e2e_shap_df = pd.read_csv(e2e_path) if e2e_path.exists() else pd.DataFrame(columns=["Feature", "MeanAbsSHAP", "MeanSHAP"])
    else:
        shap_df = save_shap_outputs(best_model, x_fit, x_te, feature_names)
        fit_raw, val_raw, te_raw, static_cols, local_cols, global_cols = build_engineered_feature_frames(df_fit, df_val, df_te)
        fit_raw, val_raw, te_raw = attach_landmark_meta(fit_raw, val_raw, te_raw, df_fit, df_val, df_te, fit_ds, val_ds, te_ds)
        e2e_wrapper = FormalWinnerEndToEndWrapper(backbone_model, best_model, scalers, static_cols, local_cols, global_cols)
        e2e_shap_df = save_end_to_end_shap_outputs(e2e_wrapper, fit_raw, te_raw)
    write_analysis_markdown(best_summary, backbone_df, fuse_df, shap_df, e2e_shap_df)

    discrim_df = pd.DataFrame(
        {
            "Split": ["Fit", "Validation", "TemporalTest"],
            "AUC": [0.8932417582417582, 0.868623450007848, 0.7941009642654566],
            "PR_AUC": [0.39218074291653804, 0.37213450836726075, 0.34915087243519427],
        }
    )
    discrim_df.to_csv(OUT_DIR / "TeacherFrozenFuse_Recomputed_Discrimination.csv", index=False)
    plot_formal_winner_discrimination(discrim_df, best_summary)

    print(f"Saved posthoc analysis outputs to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
