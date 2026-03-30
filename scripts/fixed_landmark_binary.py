"""
fixed_landmark_binary.py — 纯二分类甲亢截获模型
目标: P(Hyper | 3M or 6M 截面数据)
  • MissForest (train-only)
  • Diverse model zoo + RandomizedSearchCV + OOF threshold
  • Stacking on selected tuned baselines
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             f1_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# import lightgbm as lgb  # Disabled to keep the binary manuscript focused on RF-style interpretable ensembles.
from imblearn.ensemble import BalancedRandomForestClassifier
# from tabpfn import TabPFNClassifier  # Disabled for the current manuscript version.

from utils.config import SEED
from utils.data import load_data, fit_missforest, split_imputed, extract_flat_features
from utils.evaluation import (
    bootstrap_group_cis,
    compute_binary_metrics,
    compute_calibration_stats,
    format_ci,
    save_calibration_figure,
    save_dca_figure,
    save_threshold_sensitivity_figure,
)
from utils.model_viz import save_forest_anatomy
from utils.performance_panels import (
    build_binary_performance_long,
    export_metric_matrices,
    save_performance_heatmap_panels,
)
from utils.plot_style import PRIMARY_BLUE, PRIMARY_TEAL, apply_publication_style
from utils.shap_viz import save_binary_shap_suite

apply_publication_style()


# ==========================================
# 1. Config
# ==========================================
class Config:
    SEED = SEED
    OUT_DIR = Path("./results/fixed_landmark_binary/")


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.SEED)


# ==========================================
# 2. Tuning specs for ALL models
# ==========================================
def get_tune_specs():
    """Each entry: (base_estimator, param_grid, color, linestyle, n_iter, n_jobs)."""
    S = Config.SEED
    return {
        "Logistic Reg.": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, random_state=S))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__penalty': ['l1', 'l2'],
             'lr__solver': ['saga']},
            "#1f77b4", "-.", 14, -1
        ),
        "Elastic LR": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(
                          max_iter=4000, penalty='elasticnet', solver='saga', random_state=S
                      ))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8", "-.", 12, -1
        ),
        "SVM": (
            Pipeline([('scaler', StandardScaler()),
                      ('svc', SVC(
                          kernel='rbf', probability=True, class_weight='balanced',
                          random_state=S
                      ))]),
            {'svc__C': [0.1, 0.5, 1, 2, 5, 10],
             'svc__gamma': ['scale', 0.01, 0.05, 0.1, 0.5]},
            "#f58518", ":", 14, -1
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=S, n_jobs=1, class_weight='balanced'),
            {'n_estimators': [100, 200, 300, 500],
             'max_depth': [3, 5, 7, 10, None],
             'min_samples_leaf': [3, 5, 10, 20],
             'max_features': ['sqrt', 'log2', 0.5]},
            "#54a24b", "--", 20, -1
        ),
        "Balanced RF": (
            BalancedRandomForestClassifier(random_state=S, n_jobs=1),
            {'n_estimators': [100, 200, 300, 500],
             'max_depth': [2, 3, 4, 5],
             'min_samples_leaf': [1, 3, 5, 10],
             'sampling_strategy': ['all', 'not minority']},
            "#72b7b2", "--", 16, -1
        ),
        # "LightGBM": (
        #     lgb.LGBMClassifier(random_state=S, n_jobs=1, verbosity=-1),
        #     {'n_estimators': [100, 200, 300, 500],
        #      'learning_rate': [0.01, 0.03, 0.05, 0.1],
        #      'max_depth': [2, 3, 4, 5, -1],
        #      'subsample': [0.6, 0.7, 0.8, 1.0],
        #      'colsample_bytree': [0.6, 0.8, 1.0],
        #      'min_child_samples': [5, 10, 20]},
        #     "#b279a2", "--", 20, -1
        # ),
        "MLP": (
            Pipeline([('scaler', StandardScaler()),
                      ('mlp', MLPClassifier(max_iter=1000, early_stopping=True,
                                            validation_fraction=0.15,
                                            n_iter_no_change=50, random_state=S))]),
            {'mlp__hidden_layer_sizes': [(32,), (64, 32), (128, 64), (64, 32, 16)],
             'mlp__alpha': [0.001, 0.01, 0.05, 0.1],
             'mlp__learning_rate_init': [0.0005, 0.001, 0.005, 0.01]},
            "#9d755d", ":", 16, -1
        ),
        # "TabPFN": (
        #     TabPFNClassifier(
        #         device='cpu',
        #         random_state=S,
        #         n_estimators=8,
        #         n_preprocessing_jobs=1,
        #         ignore_pretraining_limits=True
        #     ),
        #     None,
        #     "#bab0ac", "-", 0, 1
        # ),
    }


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    """Tune model with GroupKFold, or evaluate a fixed-config model."""
    if grid:
        rs = RandomizedSearchCV(
            base, grid, n_iter=n_iter, cv=cv,
            scoring='average_precision', random_state=Config.SEED, n_jobs=n_jobs
        )
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        return rs.best_estimator_, rs.best_score_, rs.best_params_

    scores = []
    for f_tr, f_val in cv.split(X_tr_df, y_tr, groups=groups_tr):
        m = clone(base)
        m.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
        proba = m.predict_proba(X_tr_df.iloc[f_val])[:, 1]
        scores.append(average_precision_score(y_tr[f_val], proba))
    return clone(base), float(np.mean(scores)), None


# ==========================================
# 4. Evaluation + plots
# ==========================================
def plot_shap_binary(model_name, model, X_tr_df, X_te_df, feat_names, title, filename):
    print(f"  SHAP: {title}...")
    try:
        save_binary_shap_suite(
            model_name=model_name,
            model=model,
            X_background=X_tr_df,
            X_local=X_te_df,
            feat_names=feat_names,
            out_dir=Config.OUT_DIR,
            summary_filename=filename,
            summary_title=title,
            seed=Config.SEED,
            max_display=20,
        )
    except Exception as e:
        print(f"    SHAP failed: {e}")


def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*100}")
    print(f"  Binary Classification — {landmark_name}")
    print(f"{'='*100}")

    # ---- Data ----
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, _, y_raw, pids = load_data()
    n_static = X_s_raw.shape[1]

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    train_idx = np.where(np.array([pid in train_pids for pid in pids]))[0]
    test_idx = np.where(np.array([pid not in train_pids for pid in pids]))[0]
    n_train_records = len(train_idx)
    n_test_records = len(test_idx)

    # Temporal-safe MissForest: only use columns ≤ seq_len (no future data)
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
        imputer = fit_missforest(raw_all[train_idx])
        joblib.dump(imputer, imputer_path)
        print(f"  MissForest (seq≤{seq_len}): saved to {imputer_path}")

    filled_tr = imputer.transform(raw_all[train_idx])
    filled_te = imputer.transform(raw_all[test_idx])
    xs_tr, f3_tr, f4_tr, ts_tr = split_imputed(filled_tr, n_static, seq_len)
    xs_te, f3_te, f4_te, ts_te = split_imputed(filled_te, n_static, seq_len)

    Xd_tr = np.stack([f3_tr, f4_tr, ts_tr], axis=-1)
    Xd_te = np.stack([f3_te, f4_te, ts_te], axis=-1)
    X_tr_feat, feat_names = extract_flat_features(xs_tr, Xd_tr, seq_len)
    X_te_feat, _ = extract_flat_features(xs_te, Xd_te, seq_len)

    scaler = StandardScaler().fit(X_tr_feat)
    X_tr = scaler.transform(X_tr_feat)
    X_te = scaler.transform(X_te_feat)

    # Binary label: 1=Hyper (outcome==1), 0=Non-Hyper
    y_tr = (y_raw[train_idx] == 1).astype(int)
    y_te = (y_raw[test_idx] == 1).astype(int)
    groups_tr = pids[train_idx]

    print(f"  Train: {n_train_records} records  (Hyper={y_tr.sum()}, Non-Hyper={len(y_tr)-y_tr.sum()})")
    print(f"  Test:  {n_test_records} records  (Hyper={y_te.sum()}, Non-Hyper={len(y_te)-y_te.sum()})")
    print(f"  Features: {len(feat_names)}")

    X_tr_df = pd.DataFrame(X_tr, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, columns=feat_names)

    # ---- Hyperparameter tuning: ALL models (GroupKFold) ----
    print("\n  Tuning ALL models (RandomizedSearchCV + GroupKFold)...")
    S = Config.SEED
    gkf = GroupKFold(n_splits=3)
    model_zoo = {}

    for name, (base, grid, color, ls, n_iter, n_jobs) in get_tune_specs().items():
        best_est, best_score, best_params = fit_candidate_model(
            base, grid, n_iter, n_jobs, gkf, X_tr_df, y_tr, groups_tr
        )
        model_zoo[name] = (best_est, color, ls)
        suffix = "[fixed config]" if best_params is None else str(best_params)
        print(f"    {name:<18s} CV PR-AUC={best_score:.3f}  {suffix}")

    # Stacking commented out: meta-learner makes SHAP uninterpretable for paper.
    # stacking = StackingClassifier(
    #     estimators=[
    #         ('lr', clone(model_zoo['Logistic Reg.'][0])),
    #         ('rf', clone(model_zoo['Random Forest'][0])),
    #         ('lgbm', clone(model_zoo['LightGBM'][0])),
    #     ],
    #     final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=S),
    #     cv=3, n_jobs=1, passthrough=False
    # )
    # model_zoo["Stacking"] = (stacking, "#17becf", "-")

    # ---- OOF threshold + evaluate ----
    print("\n  OOF threshold selection + evaluation...")
    results = {}

    hdr = (f"  {'Model':<18s} {'AUC':>5} {'PR-AUC':>7} {'Thr':>4} {'Acc':>5} "
           f"{'BalAcc':>6} {'F1':>5}  {'TP':>3} {'FP':>4} {'FN':>3} {'TN':>4}")
    print(f"\n{'='*85}")
    print(hdr)
    print(f"{'='*85}")

    for name, (model, color, ls) in model_zoo.items():
        oof = np.zeros(len(y_tr))
        for f_tr, f_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
            oof[f_val] = m.predict_proba(X_tr_df.iloc[f_val])[:, 1]

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.05, 0.60, 0.01):
            f = f1_score(y_tr, (oof >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

        model.fit(X_tr_df, y_tr)
        train_fit_proba = model.predict_proba(X_tr_df)[:, 1]
        proba = model.predict_proba(X_te_df)[:, 1]
        pred = (proba >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba)
        prauc = average_precision_score(y_te, proba)
        acc = accuracy_score(y_te, pred)
        bacc = balanced_accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        cm = confusion_matrix(y_te, pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = dict(proba=proba, auc=auc, prauc=prauc, acc=acc, bacc=bacc,
                             f1=f1, thr=best_thr, tp=tp, fp=fp, fn=fn, tn=tn,
                             model=model, color=color, ls=ls,
                             oof_proba=oof, train_fit_proba=train_fit_proba)

        star = " *" if auc == max(r['auc'] for r in results.values()) else "  "
        print(f"{star}{name:<18s} {auc:>4.3f} {prauc:>6.3f} {best_thr:>3.2f} {acc:>4.3f} "
              f"{bacc:>5.3f} {f1:>4.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}")

    print(f"{'='*85}")
    best_auc_name = max(results, key=lambda k: results[k]['auc'])
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    print(f"  Best AUC:    {best_auc_name} ({results[best_auc_name]['auc']:.3f})")
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")

    perf_domains = {
        "Train_Fit": {"y_true": y_tr, "proba_key": "train_fit_proba"},
        "Validation_OOF": {"y_true": y_tr, "proba_key": "oof_proba"},
        "Test_Temporal": {"y_true": y_te, "proba_key": "proba"},
    }
    perf_metric_keys = ["auc", "recall", "specificity", "f1", "bacc"]
    perf_long_df = build_binary_performance_long(
        task_name=landmark_name,
        results=results,
        domain_payloads=perf_domains,
        metric_keys=perf_metric_keys,
        threshold_key="thr",
    )

    tag = landmark_name.replace(" ", "_")[:4]
    best_eval_name = best_prauc_name
    best_eval = results[best_eval_name]

    eval_df = pd.DataFrame({
        "Patient_ID": pids[test_idx],
        "Y": y_te,
        "proba": best_eval["proba"],
    })
    cis = bootstrap_group_cis(eval_df, best_eval["thr"], group_col="Patient_ID", seed=Config.SEED)
    metrics = compute_binary_metrics(y_te, best_eval["proba"], best_eval["thr"])
    cal = compute_calibration_stats(y_te, best_eval["proba"])

    print(f"\n  Binary report on best PR-AUC model: {best_eval_name}")
    print(f"    AUC         = {metrics['auc']:.3f}  95% CI {format_ci(cis, 'auc')}")
    print(f"    PR-AUC      = {metrics['prauc']:.3f}  95% CI {format_ci(cis, 'prauc')}")
    print(f"    Brier       = {metrics['brier']:.3f}  95% CI {format_ci(cis, 'brier')}")
    print(f"    Recall      = {metrics['recall']:.3f}  95% CI {format_ci(cis, 'recall')}")
    print(f"    Specificity = {metrics['specificity']:.3f}  95% CI {format_ci(cis, 'specificity')}")
    print(f"    Cal. Intcp  = {cal['intercept']:.3f}  95% CI {format_ci(cis, 'cal_intercept')}")
    print(f"    Cal. Slope  = {cal['slope']:.3f}  95% CI {format_ci(cis, 'cal_slope')}")
    print(f"    Threshold   = {metrics['threshold']:.2f}")

    # ================= Plot 1: ROC curves =================
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_te, r['proba'])
        ax.plot(fpr, tpr, lw=2, color=r['color'], linestyle=r['ls'],
                label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Binary Classification ROC — {landmark_name}", fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    fig.savefig(Config.OUT_DIR / f"ROC_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 2: AUC + PR-AUC bar chart =================
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    praucs = [results[n]['prauc'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, aucs, w, label='ROC-AUC', color=PRIMARY_BLUE)
    b2 = ax.bar(x + w/2, praucs, w, label='PR-AUC', color=PRIMARY_TEAL)
    for b in b1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha='center', fontsize=7.5)
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha='center', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Binary Model Comparison — {landmark_name}", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(aucs) + 0.12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / f"Bar_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 3: Confusion matrix of best model =================
    best_name = best_auc_name
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_best = confusion_matrix(y_te, (results[best_name]['proba'] >= results[best_name]['thr']).astype(int))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Hyper', 'Hyper'], yticklabels=['Non-Hyper', 'Hyper'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"CM: {best_name}  AUC={results[best_name]['auc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / f"CM_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    save_calibration_figure(
        y_te,
        best_eval["proba"],
        f"Calibration Curve ({best_eval_name}, {landmark_name})",
        Config.OUT_DIR / f"Calibration_{tag}.png",
    )
    save_dca_figure(
        y_te,
        best_eval["proba"],
        f"Decision Curve Analysis ({best_eval_name}, {landmark_name})",
        Config.OUT_DIR / f"DCA_{tag}.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        best_eval["proba"],
        best_eval["thr"],
        f"Threshold Sensitivity ({best_eval_name}, {landmark_name})",
        Config.OUT_DIR / f"Threshold_Sensitivity_{tag}.png",
    )

    # ================= Plot 4: SHAP suite (best tree model) =================
    tree_ok = {n: r for n, r in results.items() if hasattr(r['model'], 'feature_importances_')}
    if tree_ok:
        bt = max(tree_ok, key=lambda k: tree_ok[k]['auc'])
        plot_shap_binary(bt, tree_ok[bt]['model'], X_tr_df, X_te_df, feat_names,
                         title=f"Binary SHAP — {bt} ({landmark_name})",
                         filename=f"SHAP_{tag}.png")

    # ================= Plot 5: Forest anatomy (replace old structure-flow figures) =================
    forest_ok = {}
    for name, r in results.items():
        est_name = type(r["model"]).__name__
        if "Forest" in est_name and hasattr(r["model"], "estimators_"):
            forest_ok[name] = r
    if forest_ok:
        best_forest_name = max(forest_ok, key=lambda k: forest_ok[k]["auc"])
        safe = best_forest_name.replace(" ", "_").replace(".", "")
        prefix = f"{safe}_{tag}"
        save_forest_anatomy(
            best_forest_name,
            forest_ok[best_forest_name]["model"],
            feat_names,
            Config.OUT_DIR,
            prefix=prefix,
            sample_X=X_te_df,
            sample_y=y_te,
        )

    return perf_long_df


# ==========================================
# 5. Main
# ==========================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached .pkl files')
    args = parser.parse_args()
    if args.clear_cache:
        from utils.data import clear_pkl_cache
        clear_pkl_cache(Config.OUT_DIR)

    perf_frames = [
        run_experiment("3-Month", seq_len=3),
        run_experiment("6-Month", seq_len=4),
    ]
    perf_long_df = pd.concat(perf_frames, ignore_index=True)
    export_metric_matrices(perf_long_df, Config.OUT_DIR, prefix="Performance")
    save_performance_heatmap_panels(
        perf_long_df,
        Config.OUT_DIR / "Performance_Heatmaps.png",
        task_order=["3-Month", "6-Month"],
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=["auc", "recall", "specificity", "f1", "bacc"],
        title="Fixed Landmark Internal Performance Heatmaps",
    )
    print(f"\n  Done. Results in {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
