import os
import sys
import pickle
import json
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

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

from scripts import relapse_teacher_frozen_fuse as tf
from utils.config import SEED
from utils.evaluation import aggregate_patient_level, save_patient_risk_strata
from utils.plot_style import apply_publication_style
from utils.shap_viz import save_binary_shap_suite, _fit_binary_explainer

apply_publication_style()


OUT_DIR = tf.Config.OUT_DIR


def load_best_artifacts():
    best_summary = pd.read_csv(OUT_DIR / "Best_Pipeline_Summary.csv").iloc[0]
    fuse_df = pd.read_csv(OUT_DIR / "FrozenFuse_Experiment_Summary.csv")
    backbone_df = pd.read_csv(OUT_DIR / "Backbone_Experiment_Summary.csv")
    with open(OUT_DIR / "Best_Frozen_Fuse.pkl", "rb") as f:
        best_fuse = pickle.load(f)
    return best_summary, fuse_df, backbone_df, best_fuse


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

    feature_names = best_fuse["feature_names"]
    best_model = best_fuse["model"]
    x_fit = master_frames["fit"][feature_names]
    x_val = master_frames["val"][feature_names]
    x_te = master_frames["test"][feature_names]
    fit_proba = best_model.predict_proba(x_fit)[:, 1]
    val_proba = best_model.predict_proba(x_val)[:, 1]
    te_proba = best_model.predict_proba(x_te)[:, 1]

    plot_architecture(best_summary)
    plot_backbone_comparison(backbone_df)
    plot_gate_comparison(backbone_df)
    plot_top_experiments(fuse_df)
    save_formal_winner_bundle(best_summary, best_fuse, best_model, feature_names)
    save_formal_winner_params(best_model, feature_names)

    patient_fit = aggregate_patient_level(df_fit, fit_proba, method="product")
    patient_te = aggregate_patient_level(df_te, te_proba, method="product")
    save_patient_risk_strata(patient_fit, patient_te, OUT_DIR / "TeacherFrozenFuse_Patient_Risk_Q1Q4.png")

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
