import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import relapse_two_stage_transition as base
from utils.config import SEED
from utils.feature_selection import select_binary_features_with_l1
from utils.physio_forecast import CURRENT_CONTEXT_COLS, make_stage1_feature_frames, add_stage1_prediction_family
from utils.plot_style import PRIMARY_BLUE, PRIMARY_TEAL, TEXT_MID, apply_publication_style

apply_publication_style()
np.random.seed(SEED)


OUT_DIR = Path("./results/relapse_two_stage_transition_intermediate/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Redirect all reused transition helpers to the new sidecar experiment directory.
base.Config.OUT_DIR = OUT_DIR

DELTA_COLS = ["FT3", "FT4", "logTSH"]
HIGH_RISK_WINDOWS = {"1M->3M", "3M->6M", "6M->12M"}
EPS = 1e-6


@dataclass(frozen=True)
class LatentVariant:
    key: str
    title: str
    description: str


LATENT_VARIANTS = [
    LatentVariant(
        key="g1_delta_latent",
        title="Gen1 Delta Manifold",
        description="continuous physiology forecast geometry only; no state/rule target proxy",
    ),
    LatentVariant(
        key="g2_pca_manifold",
        title="Gen2 PCA Manifold",
        description="compressed low-dimensional manifold of stage-1 continuous forecasts",
    ),
    LatentVariant(
        key="g3_cluster_soft_state",
        title="Gen3 Soft Archetypes",
        description="unsupervised physiology archetype memberships from predicted future labs",
    ),
    LatentVariant(
        key="g4_residual_pca",
        title="Gen4 Residual Manifold",
        description="residualized future physiology after removing current-window context",
    ),
    LatentVariant(
        key="g5_residual_cluster_hybrid",
        title="Gen5 Residual Hybrid",
        description="residual manifold plus residual archetype memberships",
    ),
]


def _base_delta_model_names(stage1_payload):
    names = []
    for name in sorted(stage1_payload["delta_results"]):
        if name == "EnsembleMean":
            continue
        payload = stage1_payload["delta_results"][name]
        if "oof_delta" in payload and "test_delta" in payload:
            names.append(name)
    return names


def _collect_delta_stack(stage1_payload, split_key):
    key = f"{split_key}_delta"
    arrays = []
    names = _base_delta_model_names(stage1_payload)
    for name in names:
        arr = np.asarray(stage1_payload["delta_results"][name][key], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == len(DELTA_COLS):
            arrays.append(arr)
    if not arrays:
        raise RuntimeError("No usable stage-1 delta arrays found.")
    return names, np.stack(arrays, axis=0)


def _safe_norm(mat):
    mat = np.asarray(mat, dtype=float)
    return np.sqrt(np.sum(mat * mat, axis=1))


def _window_onehot(df):
    out = pd.DataFrame(index=np.arange(len(df)))
    for name in sorted(df["Interval_Name"].astype(str).unique()):
        out[f"Window_{name}"] = (df["Interval_Name"].astype(str).values == name).astype(float)
    out["Window_HighRisk"] = df["Interval_Name"].astype(str).isin(HIGH_RISK_WINDOWS).astype(float)
    return out


def _fill_and_keep(train_df, test_df):
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    med = train_df.median(numeric_only=True)
    train_df = train_df.fillna(med)
    test_df = test_df.fillna(med)
    keep_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) > 1]
    return train_df[keep_cols].copy(), test_df[keep_cols].copy(), keep_cols


def build_delta_latent_frames(train_df, test_df, stage1_payload):
    _, train_stack = _collect_delta_stack(stage1_payload, "oof")
    _, test_stack = _collect_delta_stack(stage1_payload, "test")

    mean_delta_tr = np.mean(train_stack, axis=0)
    mean_delta_te = np.mean(test_stack, axis=0)
    std_delta_tr = np.std(train_stack, axis=0)
    std_delta_te = np.std(test_stack, axis=0)
    sign_agree_tr = np.abs(np.mean(np.sign(train_stack), axis=0))
    sign_agree_te = np.abs(np.mean(np.sign(test_stack), axis=0))

    cur_tr = train_df[["FT3_Current", "FT4_Current", "logTSH_Current"]].values.astype(float)
    cur_te = test_df[["FT3_Current", "FT4_Current", "logTSH_Current"]].values.astype(float)
    next_tr = cur_tr + mean_delta_tr
    next_te = cur_te + mean_delta_te

    hist_mean_tr = train_df[["FT3_Mean_ToDate", "FT4_Mean_ToDate", "logTSH_Mean_ToDate"]].values.astype(float)
    hist_mean_te = test_df[["FT3_Mean_ToDate", "FT4_Mean_ToDate", "logTSH_Mean_ToDate"]].values.astype(float)
    hist_std_tr = train_df[["FT3_STD_ToDate", "FT4_STD_ToDate", "logTSH_STD_ToDate"]].values.astype(float)
    hist_std_te = test_df[["FT3_STD_ToDate", "FT4_STD_ToDate", "logTSH_STD_ToDate"]].values.astype(float)

    tr = pd.DataFrame(index=np.arange(len(train_df)))
    te = pd.DataFrame(index=np.arange(len(test_df)))

    for idx, lab in enumerate(DELTA_COLS):
        tr[f"Latent_Delta_{lab}"] = mean_delta_tr[:, idx]
        te[f"Latent_Delta_{lab}"] = mean_delta_te[:, idx]
        tr[f"Latent_Uncertainty_{lab}"] = std_delta_tr[:, idx]
        te[f"Latent_Uncertainty_{lab}"] = std_delta_te[:, idx]
        tr[f"Latent_SignAgreement_{lab}"] = sign_agree_tr[:, idx]
        te[f"Latent_SignAgreement_{lab}"] = sign_agree_te[:, idx]
        tr[f"Latent_Next_{lab}"] = next_tr[:, idx]
        te[f"Latent_Next_{lab}"] = next_te[:, idx]
        tr[f"Latent_NextMinusHistMean_{lab}"] = next_tr[:, idx] - hist_mean_tr[:, idx]
        te[f"Latent_NextMinusHistMean_{lab}"] = next_te[:, idx] - hist_mean_te[:, idx]
        tr[f"Latent_NextZ_{lab}"] = (next_tr[:, idx] - hist_mean_tr[:, idx]) / (hist_std_tr[:, idx] + 0.10)
        te[f"Latent_NextZ_{lab}"] = (next_te[:, idx] - hist_mean_te[:, idx]) / (hist_std_te[:, idx] + 0.10)

    tr["Latent_DeltaNorm"] = _safe_norm(mean_delta_tr)
    te["Latent_DeltaNorm"] = _safe_norm(mean_delta_te)
    tr["Latent_UncertaintyNorm"] = _safe_norm(std_delta_tr)
    te["Latent_UncertaintyNorm"] = _safe_norm(std_delta_te)
    tr["Latent_ForecastAgreement"] = 1.0 / (1.0 + tr["Latent_UncertaintyNorm"].values)
    te["Latent_ForecastAgreement"] = 1.0 / (1.0 + te["Latent_UncertaintyNorm"].values)
    tr["Latent_FT4_minus_logTSH"] = tr["Latent_Delta_FT4"].values - tr["Latent_Delta_logTSH"].values
    te["Latent_FT4_minus_logTSH"] = te["Latent_Delta_FT4"].values - te["Latent_Delta_logTSH"].values
    tr["Latent_FT4_over_logTSH_abs"] = tr["Latent_Delta_FT4"].values / (np.abs(tr["Latent_Delta_logTSH"].values) + 0.05)
    te["Latent_FT4_over_logTSH_abs"] = te["Latent_Delta_FT4"].values / (np.abs(te["Latent_Delta_logTSH"].values) + 0.05)

    tr_windows = _window_onehot(train_df)
    te_windows = _window_onehot(test_df)
    for base_col in ["Latent_Delta_FT4", "Latent_Delta_logTSH", "Latent_DeltaNorm", "Latent_UncertaintyNorm"]:
        for win_col in [c for c in tr_windows.columns if c in {"Window_1M->3M", "Window_3M->6M", "Window_6M->12M", "Window_HighRisk"}]:
            inter = f"{base_col}_x_{win_col}"
            tr[inter] = tr[base_col].values * tr_windows[win_col].values
            te[inter] = te[base_col].values * te_windows[win_col].values

    return _fill_and_keep(tr, te)


def _fit_pca(train_df, test_df, prefix, max_components=6, var_cutoff=0.90):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_df.values.astype(float))
    x_te = scaler.transform(test_df.values.astype(float))
    max_k = int(min(max_components, x_tr.shape[1], x_tr.shape[0] - 1))
    if max_k < 1:
        raise RuntimeError(f"{prefix}: invalid PCA dimension")
    pca_full = PCA(n_components=max_k, random_state=SEED)
    pca_full.fit(x_tr)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, var_cutoff) + 1)
    n_comp = max(2, min(max_k, n_comp))
    pca = PCA(n_components=n_comp, random_state=SEED)
    z_tr = pca.fit_transform(x_tr)
    z_te = pca.transform(x_te)
    recon_tr = scaler.inverse_transform(pca.inverse_transform(z_tr))
    recon_te = scaler.inverse_transform(pca.inverse_transform(z_te))
    rec_err_tr = np.sqrt(np.mean((train_df.values.astype(float) - recon_tr) ** 2, axis=1))
    rec_err_te = np.sqrt(np.mean((test_df.values.astype(float) - recon_te) ** 2, axis=1))

    tr = pd.DataFrame({f"{prefix}_PC{i + 1}": z_tr[:, i] for i in range(z_tr.shape[1])})
    te = pd.DataFrame({f"{prefix}_PC{i + 1}": z_te[:, i] for i in range(z_te.shape[1])})
    tr[f"{prefix}_ReconRMSE"] = rec_err_tr
    te[f"{prefix}_ReconRMSE"] = rec_err_te

    loading_df = pd.DataFrame(
        pca.components_.T,
        index=train_df.columns,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    loading_df.to_csv(OUT_DIR / f"{prefix}_Loadings.csv")
    pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(pca.n_components_)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(OUT_DIR / f"{prefix}_ExplainedVariance.csv", index=False)
    return _fill_and_keep(tr, te)


def build_pca_manifold(train_df, test_df):
    return _fit_pca(train_df, test_df, prefix="G2_PCA")


def _softmax_neg_distance(dist):
    dist = np.asarray(dist, dtype=float)
    shifted = -dist - np.max(-dist, axis=1, keepdims=True)
    expv = np.exp(shifted)
    return expv / np.clip(np.sum(expv, axis=1, keepdims=True), EPS, None)


def _fit_best_gmm(train_df, prefix, k_values=(3, 4, 5)):
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_df.values.astype(float))
    candidates = []
    for k in k_values:
        if k >= len(x_tr):
            continue
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, n_init=5)
        model.fit(x_tr)
        bic = float(model.bic(x_tr))
        labels = model.predict(x_tr)
        sil = float(silhouette_score(x_tr, labels)) if len(np.unique(labels)) > 1 else np.nan
        candidates.append({"k": k, "bic": bic, "silhouette": sil, "model": model})
    if not candidates:
        raise RuntimeError(f"{prefix}: no valid GMM candidate")
    cand_df = pd.DataFrame([{k: v for k, v in row.items() if k != "model"} for row in candidates])
    cand_df.to_csv(OUT_DIR / f"{prefix}_GMM_Search.csv", index=False)
    best_row = sorted(
        candidates,
        key=lambda row: (row["bic"], -np.nan_to_num(row["silhouette"], nan=-999.0)),
    )[0]
    return scaler, best_row["model"], cand_df


def _gmm_feature_frame(train_df, test_df, prefix):
    scaler, gmm, _ = _fit_best_gmm(train_df, prefix)
    x_tr = scaler.transform(train_df.values.astype(float))
    x_te = scaler.transform(test_df.values.astype(float))
    proba_tr = gmm.predict_proba(x_tr)
    proba_te = gmm.predict_proba(x_te)
    centers = gmm.means_
    dist_tr = np.sqrt(((x_tr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    dist_te = np.sqrt(((x_te[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    soft_dist_tr = _softmax_neg_distance(dist_tr)
    soft_dist_te = _softmax_neg_distance(dist_te)

    tr = pd.DataFrame(index=np.arange(len(train_df)))
    te = pd.DataFrame(index=np.arange(len(test_df)))
    for idx in range(proba_tr.shape[1]):
        tr[f"{prefix}_Prob_{idx}"] = proba_tr[:, idx]
        te[f"{prefix}_Prob_{idx}"] = proba_te[:, idx]
        tr[f"{prefix}_SoftDist_{idx}"] = soft_dist_tr[:, idx]
        te[f"{prefix}_SoftDist_{idx}"] = soft_dist_te[:, idx]
        tr[f"{prefix}_Dist_{idx}"] = dist_tr[:, idx]
        te[f"{prefix}_Dist_{idx}"] = dist_te[:, idx]
    tr[f"{prefix}_Entropy"] = -np.sum(proba_tr * np.log(np.clip(proba_tr, EPS, 1.0)), axis=1)
    te[f"{prefix}_Entropy"] = -np.sum(proba_te * np.log(np.clip(proba_te, EPS, 1.0)), axis=1)
    tr[f"{prefix}_MaxProb"] = np.max(proba_tr, axis=1)
    te[f"{prefix}_MaxProb"] = np.max(proba_te, axis=1)
    tr[f"{prefix}_MinDist"] = np.min(dist_tr, axis=1)
    te[f"{prefix}_MinDist"] = np.min(dist_te, axis=1)
    return _fill_and_keep(tr, te)


def build_cluster_soft_state(train_df, test_df):
    use_cols = [
        col
        for col in train_df.columns
        if col.startswith("Latent_Next_") or col.startswith("Latent_Uncertainty_") or col in {"Latent_DeltaNorm", "Latent_UncertaintyNorm"}
    ]
    return _gmm_feature_frame(train_df[use_cols], test_df[use_cols], prefix="G3_Archetype")


def _build_residual_context(train_df, test_df):
    use_cols = [col for col in CURRENT_CONTEXT_COLS if col in train_df.columns]
    cat_cols = [col for col in ["Interval_Name", "Prev_State"] if col in train_df.columns]
    tr = train_df[use_cols + cat_cols].copy().reset_index(drop=True)
    te = test_df[use_cols + cat_cols].copy().reset_index(drop=True)
    for col in cat_cols:
        cats = sorted(train_df[col].astype(str).unique())
        for cat in cats:
            name = f"{col}_{cat}"
            tr[name] = (train_df[col].astype(str).values == cat).astype(float)
            te[name] = (test_df[col].astype(str).values == cat).astype(float)
    tr = tr.drop(columns=cat_cols)
    te = te.drop(columns=cat_cols)
    return _fill_and_keep(tr, te)


def build_residual_pca(train_df, test_df, delta_latent_tr, delta_latent_te):
    context_tr, context_te, _ = _build_residual_context(train_df, test_df)
    target_cols = ["Latent_Next_FT3", "Latent_Next_FT4", "Latent_Next_logTSH"]
    multi_ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(Ridge(alpha=2.0, random_state=SEED))),
        ]
    )
    multi_ridge.fit(context_tr, delta_latent_tr[target_cols].values.astype(float))
    pred_tr = multi_ridge.predict(context_tr)
    pred_te = multi_ridge.predict(context_te)
    resid_tr = delta_latent_tr[target_cols].values.astype(float) - pred_tr
    resid_te = delta_latent_te[target_cols].values.astype(float) - pred_te

    resid_df_tr = pd.DataFrame(
        {
            "Residual_FT3": resid_tr[:, 0],
            "Residual_FT4": resid_tr[:, 1],
            "Residual_logTSH": resid_tr[:, 2],
            "Residual_Norm": _safe_norm(resid_tr),
            "Residual_UncertaintyNorm": delta_latent_tr["Latent_UncertaintyNorm"].values,
            "Residual_DeltaNorm": delta_latent_tr["Latent_DeltaNorm"].values,
        }
    )
    resid_df_te = pd.DataFrame(
        {
            "Residual_FT3": resid_te[:, 0],
            "Residual_FT4": resid_te[:, 1],
            "Residual_logTSH": resid_te[:, 2],
            "Residual_Norm": _safe_norm(resid_te),
            "Residual_UncertaintyNorm": delta_latent_te["Latent_UncertaintyNorm"].values,
            "Residual_DeltaNorm": delta_latent_te["Latent_DeltaNorm"].values,
        }
    )
    resid_df_tr.to_csv(OUT_DIR / "G4_Residual_Base_Train.csv", index=False)
    resid_df_te.to_csv(OUT_DIR / "G4_Residual_Base_Test.csv", index=False)
    return _fit_pca(resid_df_tr, resid_df_te, prefix="G4_ResidualPCA")


def build_residual_cluster_hybrid(train_df, test_df, delta_latent_tr, delta_latent_te):
    context_tr, context_te, _ = _build_residual_context(train_df, test_df)
    target_cols = ["Latent_Next_FT3", "Latent_Next_FT4", "Latent_Next_logTSH"]
    multi_ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(Ridge(alpha=2.0, random_state=SEED))),
        ]
    )
    multi_ridge.fit(context_tr, delta_latent_tr[target_cols].values.astype(float))
    pred_tr = multi_ridge.predict(context_tr)
    pred_te = multi_ridge.predict(context_te)
    resid_tr = delta_latent_tr[target_cols].values.astype(float) - pred_tr
    resid_te = delta_latent_te[target_cols].values.astype(float) - pred_te

    resid_df_tr = pd.DataFrame(
        {
            "Residual_FT3": resid_tr[:, 0],
            "Residual_FT4": resid_tr[:, 1],
            "Residual_logTSH": resid_tr[:, 2],
            "Residual_Norm": _safe_norm(resid_tr),
            "Residual_UncertaintyNorm": delta_latent_tr["Latent_UncertaintyNorm"].values,
        }
    )
    resid_df_te = pd.DataFrame(
        {
            "Residual_FT3": resid_te[:, 0],
            "Residual_FT4": resid_te[:, 1],
            "Residual_logTSH": resid_te[:, 2],
            "Residual_Norm": _safe_norm(resid_te),
            "Residual_UncertaintyNorm": delta_latent_te["Latent_UncertaintyNorm"].values,
        }
    )
    pca_tr, pca_te, _ = _fit_pca(resid_df_tr, resid_df_te, prefix="G5_ResidualPCA", max_components=4, var_cutoff=0.92)
    clu_tr, clu_te, _ = _gmm_feature_frame(resid_df_tr, resid_df_te, prefix="G5_ResidualArchetype")
    combo_tr = pd.concat([pca_tr.reset_index(drop=True), clu_tr.reset_index(drop=True)], axis=1)
    combo_te = pd.concat([pca_te.reset_index(drop=True), clu_te.reset_index(drop=True)], axis=1)
    return _fill_and_keep(combo_tr, combo_te)


def fit_custom_classifier(group_name, x_tr, x_te, train_df, test_df, note="", use_feature_selection=False):
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values

    if use_feature_selection:
        fs = select_binary_features_with_l1(
            x_tr,
            x_te,
            y_tr,
            groups,
            out_dir=OUT_DIR,
            prefix=f"Stage2_{group_name}",
            seed=SEED,
            min_features=6,
        )
        x_tr_fit = fs.X_train
        x_te_fit = fs.X_test
        feat_names = fs.selected_features
    else:
        x_tr_fit = x_tr.copy()
        x_te_fit = x_te.copy()
        feat_names = list(x_tr_fit.columns)
        base.write_fixed_feature_manifest(prefix=f"Stage2_{group_name}", feat_names=feat_names, note=note)

    group_df, best_name, best_payload, results = base.fit_from_frames(
        group_name,
        x_tr_fit,
        x_te_fit,
        y_tr,
        y_te,
        groups,
        base.get_linear_tune_specs(),
        note=note,
    )
    for payload in results.values():
        payload["feat_names"] = feat_names
        payload["c_index"] = base.compute_recurrent_c_index_from_intervals(test_df, payload["proba"])
    group_df["C_Index"] = group_df["Model"].map({k: v["c_index"] for k, v in results.items()})
    best_payload["c_index"] = results[best_name]["c_index"]
    return group_df, best_name, best_payload, results


def save_result_barplot(summary_df):
    plot_df = summary_df.copy().sort_values(["PR_AUC", "AUC"], ascending=[True, True]).reset_index(drop=True)
    fig_h = max(4.2, 0.36 * len(plot_df) + 1.3)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, fig_h), sharey=True)
    y = np.arange(len(plot_df))
    auc_bars = axes[0].barh(y, plot_df["AUC"], color=PRIMARY_BLUE, alpha=0.88)
    pr_bars = axes[1].barh(y, plot_df["PR_AUC"], color=PRIMARY_TEAL, alpha=0.88)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["Group"], fontsize=7)
    axes[1].set_yticks(y)
    axes[0].set_title("AUC", fontsize=9)
    axes[1].set_title("PR-AUC", fontsize=9)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)
    for bars, ax in [(auc_bars, axes[0]), (pr_bars, axes[1])]:
        for bar in bars:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                ha="left",
                va="center",
                fontsize=6.4,
                color=TEXT_MID,
            )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Intermediate_Generation_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(summary_df, notes_df):
    fusion_df = summary_df[summary_df["Family"] == "fusion"].sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    sidecar_df = summary_df[summary_df["Family"] == "sidecar"].sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    lines = [
        "# Transition Intermediate Representation Sweep",
        "",
        "## Goal",
        "",
        "- Keep the two-stage transition setup, but make stage-1 output more intermediate and less directly aligned to relapse / hyper rule targets.",
        "- All outputs here were generated in a brand-new experiment directory; no existing files were modified.",
        "",
        "## Best Fusions",
        "",
    ]
    for row in fusion_df.head(8).itertuples(index=False):
        lines.append(
            f"- `{row.Group}`: `{row.Best_Model}` with `PR-AUC {row.PR_AUC:.3f}`, `AUC {row.AUC:.3f}`, `Brier {row.Brier:.3f}`."
        )
    lines.extend(["", "## Best Sidecars", ""])
    for row in sidecar_df.head(8).itertuples(index=False):
        lines.append(
            f"- `{row.Group}`: `{row.Best_Model}` with `PR-AUC {row.PR_AUC:.3f}`, `AUC {row.AUC:.3f}`, `Brier {row.Brier:.3f}`."
        )
    lines.extend(["", "## Generation Notes", ""])
    for row in notes_df.itertuples(index=False):
        lines.append(f"- `{row.key}`: {row.description}")
    (OUT_DIR / "Intermediate_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_group_rows(rows):
    return pd.DataFrame(rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)


def append_summary(rows, group, family, best_name, payload, note):
    rows.append(
        {
            "Group": group,
            "Family": family,
            "Best_Model": best_name,
            "AUC": payload["metrics"]["auc"],
            "PR_AUC": payload["metrics"]["prauc"],
            "C_Index": payload.get("c_index", np.nan),
            "Brier": payload["metrics"]["brier"],
            "Calibration_Intercept": payload["cal"]["intercept"],
            "Calibration_Slope": payload["cal"]["slope"],
            "Threshold": payload["threshold"],
            "Note": note,
        }
    )


def attach_eval_fields(payload, train_df, test_df):
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    thr = base.select_best_threshold(y_tr, np.asarray(payload["oof_proba"], dtype=float), low=0.02, high=0.60, step=0.01)
    payload["threshold"] = thr
    payload["metrics"] = base.compute_binary_metrics(y_te, np.asarray(payload["proba"], dtype=float), thr)
    payload["cal"] = base.compute_calibration_stats(y_te, np.asarray(payload["proba"], dtype=float))
    payload["c_index"] = base.compute_recurrent_c_index_from_intervals(test_df, np.asarray(payload["proba"], dtype=float))
    return payload


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local caches in the new intermediate experiment directory")
    args = parser.parse_args()
    if args.clear_cache:
        base.clear_pkl_cache(OUT_DIR)

    print("=" * 92)
    print("  Two-Stage Transition Intermediate Representation Sweep")
    print("=" * 92)

    df_train, df_test = base.build_two_stage_long_data()
    x1_tr, x1_te, _ = make_stage1_feature_frames(df_train, df_test)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values

    print("\n--- Stage 1: forecast only once, then reuse it for all latent generations ---")
    stage1_payload = base.run_stage1_oof_forecast(
        x1_tr,
        df_train,
        x1_te,
        df_test,
        groups,
        train_intervals,
        test_intervals,
        OUT_DIR,
    )

    # Current transition-task baseline for comparison only.
    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")

    summary_rows = []
    notes_rows = [{"key": row.key, "title": row.title, "description": row.description} for row in LATENT_VARIANTS]

    print("\n--- Baseline anchors ---")
    direct_df, direct_name, direct_payload, direct_results = base.fit_stage2_mode(
        "baseline_direct_anchor",
        df_train,
        df_test,
        mode="direct",
        use_feature_selection=True,
        allowed_models=None,
    )
    direct_name, direct_payload = base.select_transparent_payload(direct_results)
    append_summary(summary_rows, "baseline_direct_anchor", "direct", direct_name, direct_payload, "direct-only transparent anchor")

    window_df, window_name, window_payload, window_results = base.fit_direct_windowed_core(df_train, df_test, direct_payload)
    append_summary(summary_rows, "baseline_direct_windowed_anchor", "direct", window_name, window_payload, "direct-only windowed anchor")

    champion_df, champion_name, champion_payload, champion_results = base.fit_stage2_mode(
        "baseline_champion_sidecar_ref",
        df_train_pred,
        df_test_pred,
        mode="predicted_only_state_rule",
        use_feature_selection=False,
        allowed_models=["Logistic Reg.", "Elastic LR"],
    )
    champion_name, champion_payload = base.select_transparent_payload(champion_results)
    champion_sidecar_ref = {
        **champion_payload,
        "variant_name": f"predicted_only_state_rule:{champion_name}",
    }
    append_summary(summary_rows, "baseline_champion_sidecar_ref", "sidecar", champion_name, champion_sidecar_ref, "current task-aligned sidecar")

    champion_cal = base.build_sidecar_calibrated_pack(
        "baseline_champion_sidecar_cal",
        df_train,
        df_test,
        champion_sidecar_ref,
        strategy_key="phase",
    )
    champion_cal = attach_eval_fields(champion_cal, df_train, df_test)
    append_summary(summary_rows, "baseline_champion_sidecar_cal", "sidecar", "Calibrated", champion_cal, "phase calibrated task-like sidecar")

    best_like_df, best_like_name, best_like_payload, _ = base.fit_fixed_fusion(
        "baseline_windowed_plus_champion_cal",
        df_train,
        df_test,
        window_payload,
        champion_cal,
        note="windowed direct anchor + current task-like champion sidecar (phase calibrated)",
    )
    append_summary(summary_rows, "baseline_windowed_plus_champion_cal", "fusion", best_like_name, best_like_payload, "closest current transition best")

    print("\n--- Build intermediate stage-1 representations ---")
    g1_tr, g1_te, g1_cols = build_delta_latent_frames(df_train, df_test, stage1_payload)
    g1_tr.to_csv(OUT_DIR / "G1_DeltaLatent_Train.csv", index=False)
    g1_te.to_csv(OUT_DIR / "G1_DeltaLatent_Test.csv", index=False)

    g2_tr, g2_te, g2_cols = build_pca_manifold(g1_tr, g1_te)
    g3_tr, g3_te, g3_cols = build_cluster_soft_state(g1_tr, g1_te)
    g4_tr, g4_te, g4_cols = build_residual_pca(df_train, df_test, g1_tr, g1_te)
    g5_tr, g5_te, g5_cols = build_residual_cluster_hybrid(df_train, df_test, g1_tr, g1_te)

    latent_frames = {
        "g1_delta_latent": (g1_tr, g1_te, g1_cols),
        "g2_pca_manifold": (g2_tr, g2_te, g2_cols),
        "g3_cluster_soft_state": (g3_tr, g3_te, g3_cols),
        "g4_residual_pca": (g4_tr, g4_te, g4_cols),
        "g5_residual_cluster_hybrid": (g5_tr, g5_te, g5_cols),
    }

    print("\n--- Train latent sidecars and fuse them with the windowed direct anchor ---")
    for variant in LATENT_VARIANTS:
        x_lat_tr, x_lat_te, _ = latent_frames[variant.key]
        sidecar_group = f"{variant.key}_sidecar_ref"
        sidecar_df, sidecar_name, sidecar_payload, sidecar_results = fit_custom_classifier(
            sidecar_group,
            x_lat_tr,
            x_lat_te,
            df_train,
            df_test,
            note=variant.description,
            use_feature_selection=False,
        )
        sidecar_name, sidecar_payload = base.select_transparent_payload(sidecar_results)
        sidecar_ref = {**sidecar_payload, "variant_name": f"{variant.key}:{sidecar_name}"}
        append_summary(summary_rows, sidecar_group, "sidecar", sidecar_name, sidecar_ref, variant.description)

        sidecar_cal = base.build_sidecar_calibrated_pack(
            f"{variant.key}_sidecar_cal",
            df_train,
            df_test,
            sidecar_ref,
            strategy_key="phase",
        )
        sidecar_cal = attach_eval_fields(sidecar_cal, df_train, df_test)
        append_summary(summary_rows, f"{variant.key}_sidecar_cal", "sidecar", "Calibrated", sidecar_cal, f"{variant.description}; phase-calibrated")

        fusion_group = f"{variant.key}_windowed_plus_latent_cal"
        fusion_df, fusion_name, fusion_payload, _ = base.fit_fixed_fusion(
            fusion_group,
            df_train,
            df_test,
            window_payload,
            sidecar_cal,
            note=f"windowed direct anchor + {variant.key} intermediate sidecar",
        )
        append_summary(summary_rows, fusion_group, "fusion", fusion_name, fusion_payload, f"fusion with {variant.key}")

    summary_df = summarize_group_rows(summary_rows)
    summary_df.to_csv(OUT_DIR / "Intermediate_Best_Group_Summary.csv", index=False)
    pd.DataFrame(notes_rows).to_csv(OUT_DIR / "Intermediate_Generation_Notes.csv", index=False)
    save_result_barplot(summary_df)
    write_markdown_summary(summary_df, pd.DataFrame(notes_rows))

    print("\n  Top results")
    for row in summary_df.head(12).itertuples(index=False):
        print(
            f"    {row.Group:<42s} {row.Best_Model:<14s} "
            f"PR-AUC={row.PR_AUC:.3f} AUC={row.AUC:.3f} Brier={row.Brier:.3f}"
        )
    print(f"\n  Outputs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
