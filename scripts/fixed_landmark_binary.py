"""
fixed_landmark_binary.py — 纯二分类甲亢截获模型
目标: P(Hyper | 3M or 6M 截面数据)
  • MissForest (train-only)
  • Diverse model zoo + RandomizedSearchCV + OOF threshold
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
                             f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from utils.config import SEED, STATIC_NAMES
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
# 2. Landmark feature engineering
# ==========================================
def build_landmark_features(X_s, X_d, seq_len):
    """Augment base flatten+deltas with compact trajectory and response features."""
    X_base, feat_names = extract_flat_features(X_s, X_d, seq_len)
    labs = X_d[:, :seq_len, :]
    final = labs[:, -1, :]
    prev = labs[:, -2, :] if seq_len > 1 else labs[:, -1, :]
    first = labs[:, 0, :]
    eps = 1e-6

    thyroid_idx = STATIC_NAMES.index("ThyroidW")
    trab_idx = STATIC_NAMES.index("TRAb")

    extra_blocks = [
        np.column_stack(
            [
                final[:, 0] / (first[:, 0] + eps),
                final[:, 1] / (first[:, 1] + eps),
                (final[:, 2] + 1.0) / (first[:, 2] + 1.0),
            ]
        ),
        np.column_stack(
            [
                final[:, 0] / (final[:, 1] + eps),
                final[:, 1] / (final[:, 2] + 1.0),
            ]
        ),
        labs.mean(axis=1),
        labs.std(axis=1),
        final - prev,
        np.column_stack(
            [
                X_s[:, thyroid_idx] * final[:, 0],
                X_s[:, thyroid_idx] * final[:, 1],
                X_s[:, trab_idx] * final[:, 0],
                X_s[:, trab_idx] * final[:, 1],
            ]
        ),
    ]
    extra_names = [
        "FT3_last_over_0M",
        "FT4_last_over_0M",
        "TSH_last_over_0M_plus1",
        "FT3_last_over_FT4_last",
        "FT4_last_over_TSH_last_plus1",
        "FT3_mean",
        "FT4_mean",
        "TSH_mean",
        "FT3_std",
        "FT4_std",
        "TSH_std",
        "FT3_last_minus_prev",
        "FT4_last_minus_prev",
        "TSH_last_minus_prev",
        "ThyroidW_x_FT3_last",
        "ThyroidW_x_FT4_last",
        "TRAb_x_FT3_last",
        "TRAb_x_FT4_last",
    ]
    X_feat = np.concatenate([X_base] + extra_blocks, axis=1)
    return X_feat, feat_names + extra_names


# ==========================================
# 3. Tuning specs for ALL models
# ==========================================
def get_tune_specs():
    """Each entry: (base_estimator, param_grid, color, linestyle, n_iter, n_jobs)."""
    S = Config.SEED
    specs = {
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
    if LGBMClassifier is not None:
        specs["LightGBM"] = (
            LGBMClassifier(
                random_state=S,
                objective="binary",
                verbosity=-1,
                n_jobs=1,
                class_weight="balanced",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=3,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
            ),
            None,
            "#e45756",
            "--",
            0,
            1,
        )
    return specs


def search_best_threshold(y_true, proba, objective="f1", low=0.05, high=0.80, step=0.005):
    """Pick a threshold using train OOF predictions only."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    thresholds = np.arange(low, high + 1e-9, step)

    def score_for(thr):
        metrics = compute_binary_metrics(y_true, proba, thr)
        if objective == "acc":
            return (metrics["acc"], metrics["bacc"], metrics["f1"], metrics["auc"])
        if objective == "bacc":
            return (metrics["bacc"], metrics["f1"], metrics["acc"], metrics["auc"])
        return (metrics["f1"], metrics["bacc"], metrics["acc"], metrics["auc"])

    return max(thresholds, key=score_for)


def build_weighted_blend(positive_model, anchor_model, y_tr):
    """Blend two already-evaluated models using OOF accuracy on train."""
    best_payload = None
    for positive_weight in np.arange(0.0, 1.0 + 1e-9, 0.05):
        anchor_weight = 1.0 - positive_weight
        oof = (
            positive_weight * positive_model["oof_proba"]
            + anchor_weight * anchor_model["oof_proba"]
        )
        thr = search_best_threshold(y_tr, oof, objective="acc")
        train_metrics = compute_binary_metrics(y_tr, oof, thr)
        key = (
            train_metrics["acc"],
            train_metrics["bacc"],
            train_metrics["f1"],
            train_metrics["auc"],
        )
        payload = {
            "weights": (positive_weight, anchor_weight),
            "thr": thr,
            "oof_proba": oof,
            "train_fit_proba": (
                positive_weight * positive_model["train_fit_proba"]
                + anchor_weight * anchor_model["train_fit_proba"]
            ),
            "proba": (
                positive_weight * positive_model["proba"]
                + anchor_weight * anchor_model["proba"]
            ),
            "key": key,
        }
        if best_payload is None or payload["key"] > best_payload["key"]:
            best_payload = payload
    return best_payload


def rank_error_gate_features(X_df, y_true, proba, thr, top_k=8):
    """Rank features whose values shift most on train OOF mistakes."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba, dtype=float) >= thr).astype(int)
    error_mask = pred != y_true

    rows = []
    if error_mask.sum() == 0 or error_mask.sum() == len(error_mask):
        return list(X_df.columns[:top_k]), pd.DataFrame(rows)

    for col in X_df.columns:
        values = pd.to_numeric(X_df[col], errors="coerce").to_numpy(dtype=float)
        std = float(np.nanstd(values))
        if std < 1e-8:
            continue
        error_mean = float(np.nanmean(values[error_mask]))
        correct_mean = float(np.nanmean(values[~error_mask]))
        score = abs(error_mean - correct_mean) / (std + 1e-6)
        rows.append({
            "feature": col,
            "error_shift": score,
            "error_mean": error_mean,
            "correct_mean": correct_mean,
        })

    feature_df = pd.DataFrame(rows).sort_values("error_shift", ascending=False).reset_index(drop=True)
    ranked = feature_df["feature"].head(top_k).tolist()

    clinical_anchors = [
        "ThyroidW_x_FT3_last",
        "ThyroidW_x_FT4_last",
        "TGAb",
        "TPOAb",
        "TRAb_x_FT3_last",
        "TRAb_x_FT4_last",
        "ThyroidW",
    ]
    for feat in clinical_anchors:
        if feat in X_df.columns and feat not in ranked:
            ranked.append(feat)

    return ranked, feature_df


def build_specialist_candidates(results):
    """Reuse tuned global models and add a stronger tree specialist."""
    specs = {}
    if "Elastic LR" in results and results["Elastic LR"]["model"] is not None:
        specs["Elastic"] = clone(results["Elastic LR"]["model"])
    if "Random Forest" in results and results["Random Forest"]["model"] is not None:
        specs["RF"] = clone(results["Random Forest"]["model"])
    if "LightGBM" in results and results["LightGBM"]["model"] is not None:
        specs["LGBM"] = clone(results["LightGBM"]["model"])

    specs["ET"] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced",
        random_state=Config.SEED,
        n_jobs=1,
    )
    return specs


def build_routed_specialist_model(
    main_name,
    main_result,
    results,
    X_tr_df,
    y_tr,
    groups_tr,
    X_te_df,
    cv,
    objective="acc",
):
    """
    Search a gate+specialist route using train OOF errors only.

    The main model handles all samples by default; gated samples are overwritten
    by a specialist model trained only on that harder subgroup.
    """
    candidate_features, feature_rank_df = rank_error_gate_features(
        X_tr_df, y_tr, main_result["oof_proba"], main_result["thr"]
    )
    specialist_candidates = build_specialist_candidates(results)
    if not candidate_features or not specialist_candidates:
        return None, pd.DataFrame(), feature_rank_df

    main_train_metrics = compute_binary_metrics(y_tr, main_result["oof_proba"], main_result["thr"])
    main_key = (
        main_train_metrics["acc"],
        main_train_metrics["bacc"],
        main_train_metrics["f1"],
        main_train_metrics["auc"],
    )

    quantiles = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
    search_rows = []
    best_payload = None

    for feat in candidate_features:
        feat_tr = X_tr_df[feat].to_numpy(dtype=float)
        feat_te = X_te_df[feat].to_numpy(dtype=float)

        for q in quantiles:
            cutoff = float(np.nanquantile(feat_tr, q))
            gate_tr = feat_tr >= cutoff
            gate_te = feat_te >= cutoff
            gate_count = int(gate_tr.sum())
            gate_rate = float(gate_tr.mean())

            if gate_count < 24 or gate_rate < 0.10 or gate_rate > 0.40:
                continue
            if len(np.unique(y_tr[gate_tr])) < 2:
                continue

            for spec_name, spec_model in specialist_candidates.items():
                routed_oof = main_result["oof_proba"].copy()
                valid_folds = 0

                for f_tr, f_val in cv.split(X_tr_df, y_tr, groups=groups_tr):
                    fold_train_idx = f_tr[gate_tr[f_tr]]
                    fold_val_idx = f_val[gate_tr[f_val]]
                    if len(fold_val_idx) == 0:
                        continue
                    if len(fold_train_idx) < 18 or len(np.unique(y_tr[fold_train_idx])) < 2:
                        valid_folds = -1
                        break

                    m = clone(spec_model)
                    m.fit(X_tr_df.iloc[fold_train_idx], y_tr[fold_train_idx])
                    routed_oof[fold_val_idx] = m.predict_proba(X_tr_df.iloc[fold_val_idx])[:, 1]
                    valid_folds += 1

                if valid_folds <= 0:
                    continue

                best_thr = search_best_threshold(y_tr, routed_oof, objective=objective)
                train_metrics = compute_binary_metrics(y_tr, routed_oof, best_thr)
                key = (
                    train_metrics["acc"],
                    train_metrics["bacc"],
                    train_metrics["f1"],
                    train_metrics["auc"],
                )

                full_model = clone(spec_model)
                full_model.fit(X_tr_df.loc[gate_tr], y_tr[gate_tr])

                routed_train_fit = main_result["train_fit_proba"].copy()
                routed_train_fit[gate_tr] = full_model.predict_proba(X_tr_df.loc[gate_tr])[:, 1]

                routed_test = main_result["proba"].copy()
                if gate_te.any():
                    routed_test[gate_te] = full_model.predict_proba(X_te_df.loc[gate_te])[:, 1]

                row = {
                    "main_model": main_name,
                    "gate_feature": feat,
                    "quantile": q,
                    "direction": "high",
                    "cutoff": cutoff,
                    "gate_rate": gate_rate,
                    "specialist": spec_name,
                    "train_acc": train_metrics["acc"],
                    "train_bacc": train_metrics["bacc"],
                    "train_f1": train_metrics["f1"],
                    "train_auc": train_metrics["auc"],
                    "threshold": best_thr,
                }
                search_rows.append(row)

                payload = {
                    "key": key,
                    "thr": best_thr,
                    "oof_proba": routed_oof,
                    "train_fit_proba": routed_train_fit,
                    "proba": routed_test,
                    "gate_feature": feat,
                    "gate_quantile": q,
                    "gate_cutoff": cutoff,
                    "gate_rate": gate_rate,
                    "specialist_name": spec_name,
                    "specialist_model": full_model,
                }
                if best_payload is None or payload["key"] > best_payload["key"]:
                    best_payload = payload

    search_df = pd.DataFrame(search_rows).sort_values(
        ["train_acc", "train_bacc", "train_f1", "train_auc"],
        ascending=False,
    ).reset_index(drop=True) if search_rows else pd.DataFrame()

    if best_payload is None or best_payload["key"] <= main_key:
        return None, search_df, feature_rank_df
    return best_payload, search_df, feature_rank_df


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
    X_tr_feat, feat_names = build_landmark_features(xs_tr, Xd_tr, seq_len)
    X_te_feat, _ = build_landmark_features(xs_te, Xd_te, seq_len)

    # Binary label: 1=Hyper (outcome==1), 0=Non-Hyper
    y_tr = (y_raw[train_idx] == 1).astype(int)
    y_te = (y_raw[test_idx] == 1).astype(int)
    groups_tr = pids[train_idx]

    print(f"  Train: {n_train_records} records  (Hyper={y_tr.sum()}, Non-Hyper={len(y_tr)-y_tr.sum()})")
    print(f"  Test:  {n_test_records} records  (Hyper={y_te.sum()}, Non-Hyper={len(y_te)-y_te.sum()})")
    print(f"  Features: {len(feat_names)}")

    X_tr_df = pd.DataFrame(X_tr_feat, columns=feat_names)
    X_te_df = pd.DataFrame(X_te_feat, columns=feat_names)

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

    tag = landmark_name.replace(" ", "_")[:4]

    # ---- OOF threshold + evaluate ----
    print("\n  OOF threshold selection + evaluation...")
    results = {}

    hdr = (f"  {'Model':<26s} {'AUC':>5} {'PR-AUC':>7} {'Thr':>5} {'Acc':>5} "
           f"{'Recall':>6} {'Spec':>6} {'F1':>5}  {'TP':>3} {'FP':>4} {'FN':>3} {'TN':>4}")
    print(f"\n{'='*104}")
    print(hdr)
    print(f"{'='*104}")

    for name, (model, color, ls) in model_zoo.items():
        oof = np.zeros(len(y_tr))
        for f_tr, f_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
            oof[f_val] = m.predict_proba(X_tr_df.iloc[f_val])[:, 1]

        best_thr = search_best_threshold(y_tr, oof, objective="f1")

        model.fit(X_tr_df, y_tr)
        train_fit_proba = model.predict_proba(X_tr_df)[:, 1]
        proba = model.predict_proba(X_te_df)[:, 1]
        pred = (proba >= best_thr).astype(int)

        metrics = compute_binary_metrics(y_te, proba, best_thr)
        cm = confusion_matrix(y_te, pred)
        tn, fp, fn, tp = cm.ravel()
        results[name] = dict(
            proba=proba,
            auc=metrics["auc"],
            prauc=metrics["prauc"],
            acc=metrics["acc"],
            bacc=metrics["bacc"],
            recall=metrics["recall"],
            specificity=metrics["specificity"],
            f1=metrics["f1"],
            thr=best_thr,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            model=model,
            color=color,
            ls=ls,
            oof_proba=oof,
            train_fit_proba=train_fit_proba,
        )
        print(f"  {name:<26s} {metrics['auc']:>4.3f} {metrics['prauc']:>6.3f} {best_thr:>4.2f} "
              f"{metrics['acc']:>4.3f} {metrics['recall']:>5.3f} {metrics['specificity']:>5.3f} "
              f"{metrics['f1']:>4.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}")

    if "Elastic LR" in results and "LightGBM" in results:
        blend = build_weighted_blend(results["Elastic LR"], results["LightGBM"], y_tr)
        blend_metrics = compute_binary_metrics(y_te, blend["proba"], blend["thr"])
        pred = (blend["proba"] >= blend["thr"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        pos_w, anchor_w = blend["weights"]
        blend_name = "Elastic+LGBM Blend"
        results[blend_name] = dict(
            proba=blend["proba"],
            auc=blend_metrics["auc"],
            prauc=blend_metrics["prauc"],
            acc=blend_metrics["acc"],
            bacc=blend_metrics["bacc"],
            recall=blend_metrics["recall"],
            specificity=blend_metrics["specificity"],
            f1=blend_metrics["f1"],
            thr=blend["thr"],
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            model=None,
            color="#2f6f4f",
            ls="-",
            oof_proba=blend["oof_proba"],
            train_fit_proba=blend["train_fit_proba"],
            blend_weights=(pos_w, anchor_w),
        )
        print(f"  {blend_name:<26s} {blend_metrics['auc']:>4.3f} {blend_metrics['prauc']:>6.3f} {blend['thr']:>4.2f} "
              f"{blend_metrics['acc']:>4.3f} {blend_metrics['recall']:>5.3f} {blend_metrics['specificity']:>5.3f} "
              f"{blend_metrics['f1']:>4.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}"
              f"   [w={pos_w:.2f}/{anchor_w:.2f}]")

    if landmark_name == "3-Month":
        route_anchor_name = "Elastic+LGBM Blend" if "Elastic+LGBM Blend" in results else max(
            results, key=lambda k: results[k]["acc"]
        )
        routed, routed_search_df, routed_feature_df = build_routed_specialist_model(
            route_anchor_name,
            results[route_anchor_name],
            results,
            X_tr_df,
            y_tr,
            groups_tr,
            X_te_df,
            gkf,
            objective="acc",
        )
        if not routed_feature_df.empty:
            routed_feature_df.to_csv(
                Config.OUT_DIR / f"Routed_Error_Features_{tag}.csv",
                index=False,
            )
            top_feats = ", ".join(routed_feature_df["feature"].head(5).tolist())
            print(f"    Routed error features: {top_feats}")
        if not routed_search_df.empty:
            routed_search_df.to_csv(
                Config.OUT_DIR / f"Routed_Search_{tag}.csv",
                index=False,
            )
        if routed is not None:
            routed_metrics = compute_binary_metrics(y_te, routed["proba"], routed["thr"])
            pred = (routed["proba"] >= routed["thr"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            routed_name = f"{route_anchor_name} Routed"
            results[routed_name] = dict(
                proba=routed["proba"],
                auc=routed_metrics["auc"],
                prauc=routed_metrics["prauc"],
                acc=routed_metrics["acc"],
                bacc=routed_metrics["bacc"],
                recall=routed_metrics["recall"],
                specificity=routed_metrics["specificity"],
                f1=routed_metrics["f1"],
                thr=routed["thr"],
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                model=None,
                color="#1b4332",
                ls="-",
                oof_proba=routed["oof_proba"],
                train_fit_proba=routed["train_fit_proba"],
                route_main=route_anchor_name,
                route_feature=routed["gate_feature"],
                route_quantile=routed["gate_quantile"],
                route_cutoff=routed["gate_cutoff"],
                route_gate_rate=routed["gate_rate"],
                route_specialist=routed["specialist_name"],
            )
            print(
                f"  {routed_name:<26s} {routed_metrics['auc']:>4.3f} {routed_metrics['prauc']:>6.3f} {routed['thr']:>4.2f} "
                f"{routed_metrics['acc']:>4.3f} {routed_metrics['recall']:>5.3f} {routed_metrics['specificity']:>5.3f} "
                f"{routed_metrics['f1']:>4.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}"
                f"   [gate={routed['gate_feature']}>=q{routed['gate_quantile']:.2f}, "
                f"spec={routed['specialist_name']}, rate={routed['gate_rate']:.2f}]"
            )

    print(f"{'='*104}")
    best_auc_name = max(results, key=lambda k: results[k]['auc'])
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    best_acc_name = max(results, key=lambda k: results[k]['acc'])
    print(f"  Best AUC:    {best_auc_name} ({results[best_auc_name]['auc']:.3f})")
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")
    print(f"  Best Acc:    {best_acc_name} ({results[best_acc_name]['acc']:.3f})")

    perf_domains = {
        "Train_Fit": {"y_true": y_tr, "proba_key": "train_fit_proba"},
        "Validation_OOF": {"y_true": y_tr, "proba_key": "oof_proba"},
        "Test_Temporal": {"y_true": y_te, "proba_key": "proba"},
    }
    perf_metric_keys = ["auc", "acc", "recall", "specificity", "f1"]
    perf_long_df = build_binary_performance_long(
        task_name=landmark_name,
        results=results,
        domain_payloads=perf_domains,
        metric_keys=perf_metric_keys,
        threshold_key="thr",
    )

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
    print(f"    Accuracy    = {metrics['acc']:.3f}")
    print(f"    Bal. Acc    = {metrics['bacc']:.3f}")
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
        metric_order=["auc", "acc", "recall", "specificity", "f1"],
        title="Fixed Landmark Internal Performance Heatmaps",
    )
    print(f"\n  Done. Results in {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
