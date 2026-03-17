import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import shap
import joblib

from utils.config import SEED, STATIC_NAMES
from utils.data import (load_data, fit_missforest, split_imputed,
                        extract_flat_features)
from utils.evaluation import (
    bootstrap_group_cis,
    compute_binary_metrics,
    compute_calibration_stats,
    format_ci,
    save_calibration_figure,
    save_dca_figure,
    save_threshold_sensitivity_figure,
    select_best_threshold,
)

# ==========================================
# 1. Config & 目录设置
# ==========================================
class Config:
    OUT_DIR = Path("./results/fixed_landmark_multiclass/")

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
CLASS_NAMES = ["Hyper", "Normal", "Hypo"]
FOCUS_CLASSES = [("Hyper", 0), ("Hypo", 2)]

# ==========================================
# 3. 级联分类器 (Cascade Framework)
# ==========================================
class CascadeClassifier:
    """万能级联包装器：支持包入任何 sklearn 分类器"""
    def __init__(self, stage1_model, stage2_model):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
    def fit(self, X, y):
        # Stage 1: 0(甲亢) vs 非0(正常+甲减)
        y_stage1 = (y != 0).astype(int)
        self.stage1.fit(X, y_stage1)
        
        # Stage 2: 在非甲亢样本中, 1(正常) vs 2(甲减)
        mask = (y != 0)
        X_s2, y_s2 = X[mask], y[mask]
        y_stage2 = (y_s2 == 2).astype(int) 
        if len(np.unique(y_stage2)) > 1:
            self.stage2.fit(X_s2, y_stage2)
        
    def predict_proba(self, X):
        p_non_hyper = self.stage1.predict_proba(X)[:, 1]
        p_hyper = 1 - p_non_hyper
        
        try: p_hypo_given_non = self.stage2.predict_proba(X)[:, 1]
        except: p_hypo_given_non = np.zeros(len(X))
        
        prob_normal = p_non_hyper * (1 - p_hypo_given_non)
        prob_hypo = p_non_hyper * p_hypo_given_non
        return np.column_stack([p_hyper, prob_normal, prob_hypo])

# ==========================================
# 4. 评估与可视化
# ==========================================
def evaluate_probs(probs, labels, model_name):
    preds = probs.argmax(axis=1)
    acc3 = accuracy_score(labels, preds)
    
    y_hyper = (labels == 0).astype(int)
    pred_hyper = (preds == 0).astype(int)
    acc_hyper = accuracy_score(y_hyper, pred_hyper)
    auc_hyper = roc_auc_score(y_hyper, probs[:, 0]) if len(np.unique(y_hyper))>1 else 0.5
    
    y_hypo = (labels == 2).astype(int)
    pred_hypo = (preds == 2).astype(int)
    acc_hypo = accuracy_score(y_hypo, pred_hypo)
    auc_hypo = roc_auc_score(y_hypo, probs[:, 2]) if len(np.unique(y_hypo))>1 else 0.5
    
    print(f"[{model_name:<18}] 3分类ACC: {acc3:.3f} | 🔴 甲亢截获(AUC:{auc_hyper:.3f}/ACC:{acc_hyper:.3f}) | 🔵 甲减风险(AUC:{auc_hypo:.3f}/ACC:{acc_hypo:.3f})")
    return probs[:, 0]

def plot_roc_curves(y_te, roc_data_dict, landmark_name):
    y_hyper = (y_te == 0).astype(int)
    plt.figure(figsize=(8, 6))
    for name, probs in roc_data_dict.items():
        fpr, tpr, _ = roc_curve(y_hyper, probs)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_score(y_hyper, probs):.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC — {landmark_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.savefig(Config.OUT_DIR / f'ROC_{landmark_name[:4]}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap(cascade_model, X_tr, feat_names, title, filename):
    print(f"🔍 正在生成 SHAP 解释图: {title}...")
    explainer = shap.TreeExplainer(cascade_model.stage1)
    shap_values = explainer.shap_values(X_tr)
    # 兼容不同版本 sklearn/shap 输出
    if isinstance(shap_values, list): shap_values = shap_values[1] 
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_tr, feature_names=feat_names, show=False)
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(Config.OUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def compute_multiclass_metrics(y_true, probs):
    preds = probs.argmax(axis=1)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(len(CLASS_NAMES))[y_true]

    per_class_auc = {}
    per_class_prauc = {}
    auc_vals, pr_vals = [], []
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == idx).astype(int)
        if len(np.unique(y_bin)) > 1:
            auc = roc_auc_score(y_bin, probs[:, idx])
            prauc = average_precision_score(y_bin, probs[:, idx])
            auc_vals.append(auc)
            pr_vals.append(prauc)
        else:
            auc = np.nan
            prauc = np.nan
        per_class_auc[name] = auc
        per_class_prauc[name] = prauc

    return {
        "acc3": accuracy_score(y_true, preds),
        "macro_auc_ovr": float(np.nanmean(auc_vals)),
        "macro_prauc_ovr": float(np.nanmean(pr_vals)),
        "brier_multi": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "per_class_auc": per_class_auc,
        "per_class_prauc": per_class_prauc,
        "preds": preds,
    }


def bootstrap_multiclass_cis(eval_df, group_col="Patient_ID", n_boot=800, seed=SEED):
    group_values = eval_df[group_col].dropna().unique()
    grouped = {g: eval_df.loc[eval_df[group_col] == g] for g in group_values}
    rng = np.random.default_rng(seed)
    store = {
        "acc3": [],
        "macro_auc_ovr": [],
        "macro_prauc_ovr": [],
        "brier_multi": [],
        "hyper_auc": [],
        "hypo_auc": [],
    }

    for _ in range(n_boot):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        sampled_df = pd.concat([grouped[g] for g in sampled_groups], ignore_index=True)
        y_boot = sampled_df["Y"].values.astype(int)
        probs_boot = sampled_df[[f"proba_{i}" for i in range(len(CLASS_NAMES))]].values.astype(float)
        m = compute_multiclass_metrics(y_boot, probs_boot)
        store["acc3"].append(m["acc3"])
        store["macro_auc_ovr"].append(m["macro_auc_ovr"])
        store["macro_prauc_ovr"].append(m["macro_prauc_ovr"])
        store["brier_multi"].append(m["brier_multi"])
        if np.isfinite(m["per_class_auc"]["Hyper"]):
            store["hyper_auc"].append(m["per_class_auc"]["Hyper"])
        if np.isfinite(m["per_class_auc"]["Hypo"]):
            store["hypo_auc"].append(m["per_class_auc"]["Hypo"])

    cis = {}
    for key, values in store.items():
        if len(values) < 20:
            cis[key] = (np.nan, np.nan)
        else:
            cis[key] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
    return cis

# ==========================================
# 5. 主执行
# ==========================================
def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*105}\n  Multi-class Classification — {landmark_name}\n{'='*105}")
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, _eval, y_outcome, pids = load_data()
    y = np.vectorize({1: 0, 2: 1, 3: 2}.get)(y_outcome)
    n_static = X_s_raw.shape[1]

    # --- Step 1: Temporal split FIRST (by enrollment order) ---
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    train_idx = np.where(np.array([pid in train_pids for pid in pids]))[0]
    test_idx = np.where(np.array([pid not in train_pids for pid in pids]))[0]
    n_train_records = len(train_idx)
    n_test_records = len(test_idx)
    print(f"  Train: {n_train_records} records")
    print(f"  Test:  {n_test_records} records")

    # --- Step 2: Temporal-safe MissForest: only columns ≤ seq_len ---
    ft3_cut = ft3_raw[:, :seq_len]
    ft4_cut = ft4_raw[:, :seq_len]
    tsh_cut = tsh_raw[:, :seq_len]
    raw_all = np.hstack([X_s_raw, ft3_cut, ft4_cut, tsh_cut])

    imputer_path = Config.OUT_DIR / f"missforest_seqlen{seq_len}.pkl"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        print(f"  MissForest (seq≤{seq_len}): loaded from cache")
    else:
        print(f"  MissForest (seq≤{seq_len}): fitting on train ({n_train_records} records)...")
        imputer = fit_missforest(raw_all[train_idx], seed=SEED)
        joblib.dump(imputer, imputer_path)
        print(f"  MissForest (seq≤{seq_len}): saved to {imputer_path}")

    filled_tr = imputer.transform(raw_all[train_idx])
    filled_te = imputer.transform(raw_all[test_idx])
    X_s_tr, ft3_tr, ft4_tr, tsh_tr = split_imputed(filled_tr, n_static, seq_len, n_eval=0)
    X_s_te, ft3_te, ft4_te, tsh_te = split_imputed(filled_te, n_static, seq_len, n_eval=0)

    X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
    X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

    # --- Step 3: Feature extraction ---
    X_tr_feat, feat_names = extract_flat_features(X_s_tr, X_d_tr, seq_len, static_names=STATIC_NAMES)
    X_te_feat, _          = extract_flat_features(X_s_te, X_d_te, seq_len, static_names=STATIC_NAMES)

    # --- Step 4: Scale on TRAIN ONLY ---
    scaler = StandardScaler().fit(X_tr_feat)
    X_tr, X_te = scaler.transform(X_tr_feat), scaler.transform(X_te_feat)
    y_tr, y_te = y[train_idx], y[test_idx]
    groups_tr = pids[train_idx]
    gkf = GroupKFold(n_splits=3)

    model_specs = {
        "Cascade RF": lambda: CascadeClassifier(
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED),
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED),
        ),
        "Cascade GBDT": lambda: CascadeClassifier(
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        ),
    }

    print("\n  OOF evaluation + final refit...")
    results = {}
    roc_data = {}
    for name, builder in model_specs.items():
        oof = np.zeros((len(y_tr), len(CLASS_NAMES)), dtype=float)
        for fold_tr, fold_val in gkf.split(X_tr, y_tr, groups=groups_tr):
            m = builder()
            m.fit(X_tr[fold_tr], y_tr[fold_tr])
            oof[fold_val] = m.predict_proba(X_tr[fold_val])

        model = builder()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        metrics = compute_multiclass_metrics(y_te, proba)
        hyper_thr = select_best_threshold((y_tr == 0).astype(int), oof[:, 0], low=0.05, high=0.95, step=0.01)
        hypo_thr = select_best_threshold((y_tr == 2).astype(int), oof[:, 2], low=0.05, high=0.95, step=0.01)

        results[name] = {
            "model": model,
            "oof": oof,
            "proba": proba,
            "metrics": metrics,
            "thr_hyper": hyper_thr,
            "thr_hypo": hypo_thr,
        }
        roc_data[name] = proba[:, 0]
        evaluate_probs(proba, y_te, name)

    plot_roc_curves(y_te, roc_data, landmark_name)

    best_name = max(results, key=lambda k: results[k]["metrics"]["macro_auc_ovr"])
    best = results[best_name]
    tag = landmark_name[:4]
    eval_df = pd.DataFrame({
        "Patient_ID": pids[test_idx],
        "Y": y_te,
        "proba_0": best["proba"][:, 0],
        "proba_1": best["proba"][:, 1],
        "proba_2": best["proba"][:, 2],
    })
    cis = bootstrap_multiclass_cis(eval_df)
    print(f"\n  Multi-class report on best macro-AUC model: {best_name}")
    print(f"    3-class ACC    = {best['metrics']['acc3']:.3f}  95% CI {format_ci(cis, 'acc3')}")
    print(f"    Macro ROC-AUC  = {best['metrics']['macro_auc_ovr']:.3f}  95% CI {format_ci(cis, 'macro_auc_ovr')}")
    print(f"    Macro PR-AUC   = {best['metrics']['macro_prauc_ovr']:.3f}  95% CI {format_ci(cis, 'macro_prauc_ovr')}")
    print(f"    Multi Brier    = {best['metrics']['brier_multi']:.3f}  95% CI {format_ci(cis, 'brier_multi')}")
    print(f"    Hyper ROC-AUC  = {best['metrics']['per_class_auc']['Hyper']:.3f}  95% CI {format_ci(cis, 'hyper_auc')}")
    print(f"    Hypo ROC-AUC   = {best['metrics']['per_class_auc']['Hypo']:.3f}  95% CI {format_ci(cis, 'hypo_auc')}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = confusion_matrix(y_te, best["metrics"]["preds"], labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"3-Class CM — {best_name} ({landmark_name})")
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / f"CM_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for class_name, class_idx in FOCUS_CLASSES:
        y_bin_te = (y_te == class_idx).astype(int)
        class_proba = best["proba"][:, class_idx]
        class_thr = best["thr_hyper"] if class_idx == 0 else best["thr_hypo"]
        class_metrics = compute_binary_metrics(y_bin_te, class_proba, class_thr)
        class_eval_df = pd.DataFrame({
            "Patient_ID": pids[test_idx],
            "Y": y_bin_te,
            "proba": class_proba,
        })
        class_cis = bootstrap_group_cis(class_eval_df, class_thr, group_col="Patient_ID", seed=SEED)
        class_cal = compute_calibration_stats(y_bin_te, class_proba)
        print(f"\n  {class_name} vs rest report ({best_name})")
        print(f"    AUC         = {class_metrics['auc']:.3f}  95% CI {format_ci(class_cis, 'auc')}")
        print(f"    PR-AUC      = {class_metrics['prauc']:.3f}  95% CI {format_ci(class_cis, 'prauc')}")
        print(f"    Brier       = {class_metrics['brier']:.3f}  95% CI {format_ci(class_cis, 'brier')}")
        print(f"    Recall      = {class_metrics['recall']:.3f}  95% CI {format_ci(class_cis, 'recall')}")
        print(f"    Specificity = {class_metrics['specificity']:.3f}  95% CI {format_ci(class_cis, 'specificity')}")
        print(f"    Cal. Intcp  = {class_cal['intercept']:.3f}  95% CI {format_ci(class_cis, 'cal_intercept')}")
        print(f"    Cal. Slope  = {class_cal['slope']:.3f}  95% CI {format_ci(class_cis, 'cal_slope')}")
        print(f"    Threshold   = {class_thr:.2f}")

        suffix = f"{tag}_{class_name}"
        save_calibration_figure(
            y_bin_te,
            class_proba,
            f"Calibration Curve ({best_name}, {class_name} vs rest, {landmark_name})",
            Config.OUT_DIR / f"Calibration_{suffix}.png",
        )
        save_dca_figure(
            y_bin_te,
            class_proba,
            f"Decision Curve Analysis ({best_name}, {class_name} vs rest, {landmark_name})",
            Config.OUT_DIR / f"DCA_{suffix}.png",
        )
        save_threshold_sensitivity_figure(
            y_bin_te,
            class_proba,
            class_thr,
            f"Threshold Sensitivity ({best_name}, {class_name} vs rest, {landmark_name})",
            Config.OUT_DIR / f"Threshold_Sensitivity_{suffix}.png",
        )

    plot_shap(
        best["model"],
        X_tr,
        feat_names,
        title=f"Multi-class SHAP — {best_name} ({landmark_name[:4]})",
        filename=f"SHAP_{landmark_name[:4]}_Hyper.png",
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached MissForest .pkl files')
    args = parser.parse_args()
    if args.clear_cache:
        from utils.data import clear_pkl_cache
        clear_pkl_cache(Config.OUT_DIR)

    run_experiment("3-Month Assessment", seq_len=3)
    run_experiment("6-Month Assessment", seq_len=4)
    print(f"\n🎉 运行完毕！所有结果图表已保存至: {Config.OUT_DIR.resolve()}")
