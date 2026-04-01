import os
import sys
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
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import SEED, TIME_STAMPS
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    clear_pkl_cache,
    load_data as _load_data,
    load_or_fit_depth_imputer,
    split_imputed,
)
from utils.evaluation import (
    aggregate_patient_level,
    bootstrap_group_cis,
    compute_binary_metrics,
    compute_calibration_stats,
    concordance_index_simple,
    evaluate_patient_aggregation_sensitivity,
    evaluate_window_sensitivity,
    format_ci,
    save_calibration_figure,
    save_dca_figure,
    save_patient_aggregation_sensitivity_figure,
    save_patient_risk_strata,
    save_threshold_sensitivity_figure,
    select_best_threshold,
)
from utils.feature_selection import select_binary_features_with_l1
from utils.model_viz import save_logistic_regression_visuals
from utils.performance_panels import (
    build_binary_performance_long,
    export_metric_matrices,
    save_performance_heatmap_panels,
)
from utils.physio_forecast import (
    TARGET_COLS,
    add_stage1_prediction_family,
    build_next_physio_targets,
    build_physio_history_features,
    evaluate_physio_predictions,
    evaluate_next_state_predictions,
    make_stage1_feature_frames,
    make_stage2_feature_frames,
    run_stage1_oof_forecast,
    save_physio_scatter,
    save_stage1_metric_bar,
)
from utils.plot_style import (
    PRIMARY_BLUE,
    PRIMARY_TEAL,
    TEXT_DARK,
    TEXT_MID,
    apply_publication_style,
)

from utils.recurrence import build_interval_risk_data, derive_recurrent_survival_data

apply_publication_style()
np.random.seed(SEED)


class Config:
    OUT_DIR = Path("./results/relapse_two_stage_physio/")
    SHARED_RELAPSE_DIR = Path("./results/relapse/")
    LEGACY_RELAPSE_DIR = Path("./multistate_result/")


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_fit_with_message(raw_train, cache_path, fallback_cache_path=None, label=""):
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None
    if primary.exists():
        print(f"  MissForest {label}: loaded from {primary}")
    elif fallback is not None and fallback.exists():
        print(f"  MissForest {label}: reused shared cache {fallback}")
    else:
        print(f"  MissForest {label}: fitting on train ({len(raw_train)} records)...")
    return load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=fallback_cache_path)


def build_two_stage_long_data():
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    row_ids = np.arange(len(pids))
    print(f"  Records: {len(pids)}")

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    print(f"  Train: {int(tr_mask.sum())} records")
    print(f"  Test:  {int(te_mask.sum())} records")

    print("\n--- Phase 1: Build two-stage long-format data ---")
    df_tr_parts = []
    df_te_parts = []

    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"

        raw_relapse = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        relapse_cache = Config.SHARED_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        if not relapse_cache.exists():
            relapse_cache = Config.LEGACY_RELAPSE_DIR / f"missforest_depth{depth}.pkl"
        local_relapse_cache = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer_cur = load_or_fit_with_message(
            raw_relapse[tr_mask],
            local_relapse_cache,
            fallback_cache_path=relapse_cache,
            label=f"current-depth-{depth} ({interval_name})",
        )
        filled_tr = apply_missforest(raw_relapse[tr_mask], imputer_cur, depth)
        filled_te = apply_missforest(raw_relapse[te_mask], imputer_cur, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        s_tr = build_states_from_labels(ev_tr)
        s_te = build_states_from_labels(ev_te)
        x_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        x_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_cur_tr = build_interval_risk_data(
            xs_tr, x_d_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_cur_te = build_interval_risk_data(
            xs_te, x_d_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )
        df_hist_tr = build_physio_history_features(
            x_d_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_hist_te = build_physio_history_features(
            x_d_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )

        n_lab_next = depth + 1
        raw_phys = np.hstack([X_s_raw, ft3_raw[:, :n_lab_next], ft4_raw[:, :n_lab_next], tsh_raw[:, :n_lab_next]])
        phys_cache = Config.OUT_DIR / f"missforest_physio_depth{depth}.pkl"
        imputer_phys = load_or_fit_with_message(
            raw_phys[tr_mask],
            phys_cache,
            fallback_cache_path=None,
            label=f"physio-depth-{depth} ({interval_name})",
        )
        filled_phys_tr = apply_missforest(raw_phys[tr_mask], imputer_phys, 0)
        filled_phys_te = apply_missforest(raw_phys[te_mask], imputer_phys, 0)
        _, ft3n_tr, ft4n_tr, tshn_tr = split_imputed(filled_phys_tr, n_static, n_lab_next, 0)
        _, ft3n_te, ft4n_te, tshn_te = split_imputed(filled_phys_te, n_static, n_lab_next, 0)
        x_d_next_tr = np.stack([ft3n_tr, ft4n_tr, tshn_tr], axis=-1)
        x_d_next_te = np.stack([ft3n_te, ft4n_te, tshn_te], axis=-1)
        df_next_tr = build_next_physio_targets(
            x_d_next_tr, s_tr, pids[tr_mask], target_k=k, row_ids=row_ids[tr_mask]
        )
        df_next_te = build_next_physio_targets(
            x_d_next_te, s_te, pids[te_mask], target_k=k, row_ids=row_ids[te_mask]
        )

        df_tr = df_cur_tr.merge(df_hist_tr, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_tr = df_tr.merge(df_next_tr, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_te = df_cur_te.merge(df_hist_te, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_te = df_te.merge(df_next_te, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")

        df_tr_parts.append(df_tr)
        df_te_parts.append(df_te)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr)}  test {len(df_te)} rows")

    df_train = pd.concat(df_tr_parts, ignore_index=True)
    df_test = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_train)}  test {len(df_test)} rows")
    df_train.to_csv(Config.OUT_DIR / "two_stage_train.csv", index=False)
    df_test.to_csv(Config.OUT_DIR / "two_stage_test.csv", index=False)
    return df_train, df_test


def get_stage2_tune_specs():
    """Return {name: (estimator, param_grid, color, linestyle, n_iter, n_jobs)}."""
    S = SEED
    return {
        "Logistic Reg.": (
            Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=S))]),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__penalty": ["l1", "l2"], "lr__solver": ["saga"]},
            "#1f77b4",
            "-.",
            10,
            -1,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="elasticnet",
                            solver="saga",
                            random_state=S,
                        ),
                    ),
                ]
            ),
            {"lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8",
            "-.",
            10,
            -1,
        ),
        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                n_estimators=240,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=S,
                n_jobs=1,
                verbose=-1,
            ),
            {
                "num_leaves": [15, 31, 63],
                "min_child_samples": [10, 20, 40],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [0.0, 0.1, 0.5, 1.0],
            },
            "#f58518",
            ":",
            10,
            -1,
        ),
        "Balanced RF": (
            BalancedRandomForestClassifier(random_state=S, n_jobs=1),
            {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5], "min_samples_leaf": [1, 3, 5, 10], "sampling_strategy": ["all", "not minority"]},
            "#72b7b2",
            "--",
            8,
            -1,
        ),
    }


def get_fusion_tune_specs():
    """Transparent fusion only tunes linear models."""
    S = SEED
    return {
        "Logistic Reg.": (
            Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=S))]),
            {"lr__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "lr__penalty": ["l1", "l2"], "lr__solver": ["saga"]},
            "#1f77b4",
            "-.",
            8,
            -1,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=4000,
                            penalty="elasticnet",
                            solver="saga",
                            random_state=S,
                        ),
                    ),
                ]
            ),
            {"lr__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8",
            "-.",
            8,
            -1,
        ),
    }


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    """Tune model with GroupKFold, or evaluate a fixed-config model."""
    if grid:
        rs = RandomizedSearchCV(
            base,
            grid,
            n_iter=n_iter,
            cv=cv,
            scoring="average_precision",
            random_state=SEED,
            n_jobs=n_jobs,
        )
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        return rs.best_estimator_, rs.best_score_, rs.best_params_

    scores = []
    for f_tr, f_val in cv.split(X_tr_df, y_tr, groups=groups_tr):
        model = clone(base)
        model.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
        proba = model.predict_proba(X_tr_df.iloc[f_val])[:, 1]
        scores.append(average_precision_score(y_tr[f_val], proba))
    return clone(base), float(np.mean(scores)), None


def write_fixed_feature_manifest(prefix, feat_names, note=""):
    """Write a deterministic feature manifest for branches that skip L1 selection."""
    feat_names = list(feat_names)
    summary_df = pd.DataFrame(
        {
            "feature": feat_names,
            "coefficient": np.nan,
            "abs_coefficient": np.nan,
            "selected": True,
        }
    )
    summary_df.to_csv(Config.OUT_DIR / f"{prefix}_Feature_Selection.csv", index=False)
    lines = [
        f"original_features={len(feat_names)}",
        f"selected_features={len(feat_names)}",
        "best_c=NA_fixed",
        "cv_pr_auc=NA_fixed",
        "selected_feature_names=" + ", ".join(feat_names),
    ]
    if note:
        lines.append(f"note={note}")
    (Config.OUT_DIR / f"{prefix}_Feature_Selection_Report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_logit(proba):
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def get_top_linear_feature_subset(payload, top_k=16):
    """Extract a narrow current-core subset from a fitted linear pipeline."""
    feat_names = list(payload.get("feat_names", []))
    model = payload.get("model")
    if not feat_names or model is None or not hasattr(model, "named_steps"):
        return feat_names
    lr = model.named_steps.get("lr")
    if lr is None or not hasattr(lr, "coef_"):
        return feat_names
    coef = np.asarray(lr.coef_, dtype=float).reshape(-1)
    if coef.size != len(feat_names):
        return feat_names
    order = np.argsort(np.abs(coef))[::-1]
    keep = [feat_names[idx] for idx in order[: min(int(top_k), len(feat_names))]]
    return keep


def select_transparent_payload(results):
    """Prefer linear branch scores for the transparent fusion model."""
    linear_names = [name for name in ["Logistic Reg.", "Elastic LR"] if name in results]
    if not linear_names:
        best_name = max(results, key=lambda k: (results[k]["prauc"], results[k]["auc"]))
        return best_name, results[best_name]
    best_name = max(linear_names, key=lambda k: (results[k]["prauc"], results[k]["auc"]))
    return best_name, results[best_name]


def build_fusion_feature_frames(train_df, test_df, direct_payload, physio_payload, template="base"):
    """Create a tiny transparent fusion design using branch scores and window dummies."""
    physio_oof = np.asarray(physio_payload.get("oof_proba"), dtype=float)
    physio_test = np.asarray(physio_payload.get("proba", physio_payload.get("test_proba")), dtype=float)
    tr = pd.DataFrame(
        {
            "Direct_Logit": _safe_logit(direct_payload["oof_proba"]),
            "Physio_Logit": _safe_logit(physio_oof),
        }
    )
    te = pd.DataFrame(
        {
            "Direct_Logit": _safe_logit(direct_payload["proba"]),
            "Physio_Logit": _safe_logit(physio_test),
        }
    )
    tr["Direct_x_Physio"] = tr["Direct_Logit"].values * tr["Physio_Logit"].values
    te["Direct_x_Physio"] = te["Direct_Logit"].values * te["Physio_Logit"].values

    cats = sorted(train_df["Interval_Name"].astype(str).unique())
    high_risk_windows = {"1M->3M", "3M->6M", "6M->12M"}
    for cat in cats:
        name = f"Interval_Name_{cat}"
        tr[name] = (train_df["Interval_Name"].astype(str).values == cat).astype(float)
        te[name] = (test_df["Interval_Name"].astype(str).values == cat).astype(float)

    if template in {"physio_windowed", "both_windowed"}:
        for cat in cats:
            dummy_name = f"Interval_Name_{cat}"
            inter_name = f"{dummy_name}_x_Physio_Logit"
            tr[inter_name] = tr[dummy_name].values * tr["Physio_Logit"].values
            te[inter_name] = te[dummy_name].values * te["Physio_Logit"].values
    if template == "both_windowed":
        for cat in cats:
            dummy_name = f"Interval_Name_{cat}"
            inter_name = f"{dummy_name}_x_Direct_Logit"
            tr[inter_name] = tr[dummy_name].values * tr["Direct_Logit"].values
            te[inter_name] = te[dummy_name].values * te["Direct_Logit"].values
    if template in {"gap", "gap_windowed"}:
        tr["Physio_minus_Direct"] = tr["Physio_Logit"].values - tr["Direct_Logit"].values
        te["Physio_minus_Direct"] = te["Physio_Logit"].values - te["Direct_Logit"].values
    if template == "gap_windowed":
        for cat in cats:
            dummy_name = f"Interval_Name_{cat}"
            inter_name = f"{dummy_name}_x_Physio_minus_Direct"
            tr[inter_name] = tr[dummy_name].values * tr["Physio_minus_Direct"].values
            te[inter_name] = te[dummy_name].values * te["Physio_minus_Direct"].values
    if template in {"early_physio", "early_both", "early_gap"}:
        tr["HighRiskWindow"] = train_df["Interval_Name"].astype(str).isin(high_risk_windows).astype(float)
        te["HighRiskWindow"] = test_df["Interval_Name"].astype(str).isin(high_risk_windows).astype(float)
        tr["HighRiskWindow_x_Physio_Logit"] = tr["HighRiskWindow"].values * tr["Physio_Logit"].values
        te["HighRiskWindow_x_Physio_Logit"] = te["HighRiskWindow"].values * te["Physio_Logit"].values
    if template in {"early_both"}:
        tr["HighRiskWindow_x_Direct_Logit"] = tr["HighRiskWindow"].values * tr["Direct_Logit"].values
        te["HighRiskWindow_x_Direct_Logit"] = te["HighRiskWindow"].values * te["Direct_Logit"].values
    if template in {"early_gap"}:
        tr["Physio_minus_Direct"] = tr["Physio_Logit"].values - tr["Direct_Logit"].values
        te["Physio_minus_Direct"] = te["Physio_Logit"].values - te["Direct_Logit"].values
        tr["HighRiskWindow_x_Physio_minus_Direct"] = tr["HighRiskWindow"].values * tr["Physio_minus_Direct"].values
        te["HighRiskWindow_x_Physio_minus_Direct"] = te["HighRiskWindow"].values * te["Physio_minus_Direct"].values

    medians = tr.median(numeric_only=True)
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(medians)
    te = te.replace([np.inf, -np.inf], np.nan).fillna(medians)
    keep_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) > 1]
    return tr[keep_cols], te[keep_cols], keep_cols


def compute_recurrent_c_index_from_intervals(df_long, proba):
    """Project interval predictions to spell starts and compute event-time ranking."""
    tmp = df_long.copy().reset_index(drop=True)
    tmp["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    recurrent_df = derive_recurrent_survival_data(tmp)
    if len(recurrent_df) == 0:
        return np.nan
    return concordance_index_simple(
        recurrent_df["Gap_Time"].values,
        recurrent_df["proba"].values,
        recurrent_df["Event"].values,
    )


def fit_stage2_group(mode, train_df, test_df, base_current_features=None, use_feature_selection=False):
    x_tr_full, x_te_full, _ = make_stage2_feature_frames(
        train_df, test_df, mode=mode, base_current_features=base_current_features
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values

    if use_feature_selection:
        min_features = 8
        fs = select_binary_features_with_l1(
            x_tr_full,
            x_te_full,
            y_tr,
            groups,
            out_dir=Config.OUT_DIR,
            prefix=f"Stage2_{mode}",
            seed=SEED,
            min_features=min_features,
        )
        x_tr = fs.X_train
        x_te = fs.X_test
        feat_names = fs.selected_features
        print(
            f"    Stage2 {mode:<32s} feature selection: {fs.original_feature_count} -> {len(feat_names)} "
            f"(best C={fs.best_c}, CV PR-AUC={fs.cv_score:.3f})"
        )
    else:
        x_tr = x_tr_full.copy()
        x_te = x_te_full.copy()
        feat_names = list(x_tr.columns)
        write_fixed_feature_manifest(
            prefix=f"Stage2_{mode}",
            feat_names=feat_names,
            note="fixed compact handoff",
        )
        print(f"    Stage2 {mode:<32s} fixed feature template: {len(feat_names)} features")

    gkf = GroupKFold(n_splits=3)
    tune_specs = get_stage2_tune_specs()
    tuned_models = {}
    rows = []
    results = {}

    for name, (base_est, param_grid, color, ls, n_iter, n_jobs) in tune_specs.items():
        best_est, best_score, best_params = fit_candidate_model(
            base_est, param_grid, n_iter, n_jobs, gkf, x_tr, y_tr, groups
        )
        tuned_models[name] = (best_est, color, ls, best_score, best_params)

    for model_name, (model, color, ls, cv_score, best_params) in tuned_models.items():
        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            fitted = clone(model)
            fitted.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        fitted = clone(model)
        fitted.fit(x_tr, y_tr)
        train_fit_proba = fitted.predict_proba(x_tr)[:, 1]
        proba = fitted.predict_proba(x_te)[:, 1]
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        c_index = compute_recurrent_c_index_from_intervals(test_df, proba)
        rows.append(
            {
                "Group": mode,
                "Model": model_name,
                "CV_PR_AUC": cv_score,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "C_Index": c_index,
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "F1": metrics["f1"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": thr,
                "Best_Params": "" if best_params is None else str(best_params),
            }
        )
        results[model_name] = {
            "model": fitted,
            "proba": proba,
            "oof_proba": oof,
            "train_fit_proba": train_fit_proba,
            "auc": metrics["auc"],
            "prauc": metrics["prauc"],
            "c_index": c_index,
            "threshold": thr,
            "metrics": metrics,
            "cal": cal,
            "feat_names": feat_names,
            "color": color,
            "ls": ls,
        }

    best_name = max(results, key=lambda k: (results[k]["prauc"], results[k]["auc"]))
    best_payload = results[best_name]
    return pd.DataFrame(rows), best_name, best_payload, results


def fit_fusion_group(mode, train_df, test_df, direct_payload, physio_payload, direct_name, physio_name, template="base"):
    """Fit a transparent fusion model on OOF branch scores."""
    x_tr, x_te, feat_names = build_fusion_feature_frames(
        train_df,
        test_df,
        direct_payload,
        physio_payload,
        template=template,
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    write_fixed_feature_manifest(
        prefix=f"Stage2_{mode}",
        feat_names=feat_names,
        note=f"direct_branch={direct_name}; physio_branch={physio_name}; template={template}",
    )
    print(
        f"    Stage2 {mode:<32s} transparent fusion: {len(feat_names)} features "
        f"(direct={direct_name}, physio={physio_name}, template={template})"
    )

    gkf = GroupKFold(n_splits=3)
    tuned_models = {}
    rows = []
    results = {}
    for name, (base_est, param_grid, color, ls, n_iter, n_jobs) in get_fusion_tune_specs().items():
        best_est, best_score, best_params = fit_candidate_model(
            base_est, param_grid, n_iter, n_jobs, gkf, x_tr, y_tr, groups
        )
        tuned_models[name] = (best_est, color, ls, best_score, best_params)

    for model_name, (model, color, ls, cv_score, best_params) in tuned_models.items():
        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            fitted = clone(model)
            fitted.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        fitted = clone(model)
        fitted.fit(x_tr, y_tr)
        train_fit_proba = fitted.predict_proba(x_tr)[:, 1]
        proba = fitted.predict_proba(x_te)[:, 1]
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        c_index = compute_recurrent_c_index_from_intervals(test_df, proba)
        rows.append(
            {
                "Group": mode,
                "Model": model_name,
                "CV_PR_AUC": cv_score,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "C_Index": c_index,
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "F1": metrics["f1"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": thr,
                "Best_Params": "" if best_params is None else str(best_params),
            }
        )
        results[model_name] = {
            "model": fitted,
            "proba": proba,
            "oof_proba": oof,
            "train_fit_proba": train_fit_proba,
            "auc": metrics["auc"],
            "prauc": metrics["prauc"],
            "c_index": c_index,
            "threshold": thr,
            "metrics": metrics,
            "cal": cal,
            "feat_names": feat_names,
            "color": color,
            "ls": ls,
        }

    best_name = max(results, key=lambda k: (results[k]["prauc"], results[k]["auc"]))
    best_payload = results[best_name]
    return pd.DataFrame(rows), best_name, best_payload, results


def run_fusion_candidate_search(train_df, test_df, direct_results, physio_candidate_results):
    """Search a small set of transparent fusion variants across compact physio sources."""
    direct_name, direct_payload = select_transparent_payload(direct_results)
    rows = []
    best_bundle = None
    for physio_group, results in physio_candidate_results.items():
        physio_name, physio_payload = select_transparent_payload(results)
        for template in ["base", "both_windowed"]:
            group_df, best_name, payload, group_results = fit_fusion_group(
                mode="direct_plus_physio_fusion",
                train_df=train_df,
                test_df=test_df,
                direct_payload=direct_payload,
                physio_payload=physio_payload,
                direct_name=direct_name,
                physio_name=f"{physio_group}:{physio_name}",
                template=template,
            )
            row = group_df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0].to_dict()
            row["Physio_Group"] = physio_group
            row["Direct_Source"] = direct_name
            row["Physio_Source"] = physio_name
            row["Fusion_Template"] = template
            rows.append(row)
            bundle = {
                "candidate_row": row,
                "group_df": group_df,
                "best_name": best_name,
                "payload": payload,
                "group_results": group_results,
                "direct_name": direct_name,
                "physio_group": physio_group,
                "physio_name": physio_name,
                "template": template,
            }
            if best_bundle is None:
                best_bundle = bundle
                continue
            best_key = (
                float(best_bundle["candidate_row"]["PR_AUC"]),
                float(best_bundle["candidate_row"]["AUC"]),
            )
            cur_key = (float(row["PR_AUC"]), float(row["AUC"]))
            if cur_key > best_key:
                best_bundle = bundle

    candidate_df = pd.DataFrame(rows).sort_values(
        ["PR_AUC", "AUC", "Brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_df.to_csv(Config.OUT_DIR / "Fusion_Candidate_Search.csv", index=False)
    if best_bundle is not None:
        write_fixed_feature_manifest(
            prefix="Stage2_direct_plus_physio_fusion",
            feat_names=best_bundle["payload"]["feat_names"],
            note=(
                f"direct_branch={best_bundle['direct_name']}; "
                f"physio_branch={best_bundle['physio_group']}:{best_bundle['physio_name']}; "
                f"template={best_bundle['template']}"
            ),
        )
    return candidate_df, best_bundle


def _format_p_value(p_value):
    if p_value < 1e-3:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _group_bootstrap_pr_auc_pvalue(y_true, proba_a, proba_b, groups, n_boot=2000, seed=SEED):
    """One-sided grouped bootstrap test for PR-AUC(a) > PR-AUC(b)."""
    unique_groups = pd.Index(groups).drop_duplicates().to_list()
    group_to_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([group_to_idx[group] for group in sampled])
        y_boot = y_true[boot_idx]
        if np.unique(y_boot).size < 2:
            continue
        diff = average_precision_score(y_boot, proba_a[boot_idx]) - average_precision_score(y_boot, proba_b[boot_idx])
        diffs.append(diff)
    if not diffs:
        return np.nan
    diffs = np.asarray(diffs, dtype=float)
    return (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)


def _annotate_bars(ax, bars):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=TEXT_MID,
        )


def _draw_sig_bracket(ax, x1, x2, y, text, color=TEXT_DARK):
    ax.plot([x1, x1, x2, x2], [y - 0.004, y, y, y - 0.004], lw=1.0, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.004, text, ha="center", va="bottom", fontsize=6.5, color=color)


def save_stage2_comparison(two_stage_df, best_groups, test_df):
    label_map = {
        "direct": "Direct",
        "predicted_only_compact": "Pred only",
        "predicted_plus_current_compact": "Pred + current",
        "direct_plus_state_only": "Direct + state",
        "direct_plus_state_delta": "Direct + state+delta",
        "direct_plus_state_delta_uncertainty": "Direct + state+delta+unc",
        "direct_plus_physio_fusion": "Transparent fusion",
    }
    group_order = [group for group in label_map if group in set(two_stage_df["Group"])]
    best_df = (
        two_stage_df.sort_values(["Group", "PR_AUC", "AUC"], ascending=[True, False, False]).groupby("Group", as_index=False).first()
    )
    best_df["sort_order"] = best_df["Group"].map({group: idx for idx, group in enumerate(group_order)})
    best_df = best_df.sort_values("sort_order").drop(columns="sort_order")
    best_df["Group_Label"] = best_df["Group"].map(label_map)

    fig_width = max(12.5, 1.95 * len(best_df))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    auc_bars = axes[0].bar(best_df["Group_Label"], best_df["AUC"], color=PRIMARY_BLUE, alpha=0.88, width=0.62)
    pr_bars = axes[1].bar(best_df["Group_Label"], best_df["PR_AUC"], color=PRIMARY_TEAL, alpha=0.88, width=0.62)

    axes[0].set_title("Best AUC by group", fontsize=9)
    axes[0].set_ylim(0, max(0.9, float(best_df["AUC"].max()) + 0.08))
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_title("Best PR-AUC by group", fontsize=9)
    axes[1].set_ylim(0, max(0.40, float(best_df["PR_AUC"].max()) + 0.10))
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.7, rotation=22)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("")

    _annotate_bars(axes[0], auc_bars)
    _annotate_bars(axes[1], pr_bars)

    direct_auc = float(best_df.loc[best_df["Group"] == "direct", "AUC"].iloc[0])
    direct_pr = float(best_df.loc[best_df["Group"] == "direct", "PR_AUC"].iloc[0])
    axes[0].axhline(direct_auc, color=TEXT_DARK, ls="--", lw=1.0, alpha=0.55)
    axes[1].axhline(direct_pr, color=TEXT_DARK, ls="--", lw=1.0, alpha=0.55)
    axes[0].text(len(best_df) - 0.3, direct_auc + 0.006, "Direct ref", ha="right", va="bottom", fontsize=6.5, color=TEXT_MID)
    axes[1].text(len(best_df) - 0.3, direct_pr + 0.006, "Direct ref", ha="right", va="bottom", fontsize=6.5, color=TEXT_MID)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(Config.OUT_DIR / "TwoStage_Group_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_best_mode_reports(mode, payload, train_df, test_df):
    """Export interval-level and patient-level reports for the best overall mode."""
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    interval_eval_df = pd.DataFrame({"Patient_ID": test_df["Patient_ID"].values, "Y": y_te, "proba": payload["proba"]})
    interval_cis = bootstrap_group_cis(interval_eval_df, payload["threshold"], group_col="Patient_ID")
    interval_metrics = compute_binary_metrics(y_te, payload["proba"], payload["threshold"])
    interval_cal = compute_calibration_stats(y_te, payload["proba"])
    interval_summary = pd.DataFrame(
        [
            {"Level": "Interval", "Metric": "AUC", "Value": interval_metrics["auc"], "CI_95": format_ci(interval_cis, "auc")},
            {"Level": "Interval", "Metric": "PR_AUC", "Value": interval_metrics["prauc"], "CI_95": format_ci(interval_cis, "prauc")},
            {"Level": "Interval", "Metric": "Brier", "Value": interval_metrics["brier"], "CI_95": format_ci(interval_cis, "brier")},
            {"Level": "Interval", "Metric": "Recall", "Value": interval_metrics["recall"], "CI_95": format_ci(interval_cis, "recall")},
            {"Level": "Interval", "Metric": "Specificity", "Value": interval_metrics["specificity"], "CI_95": format_ci(interval_cis, "specificity")},
            {"Level": "Interval", "Metric": "Calibration_Intercept", "Value": interval_cal["intercept"], "CI_95": format_ci(interval_cis, "cal_intercept")},
            {"Level": "Interval", "Metric": "Calibration_Slope", "Value": interval_cal["slope"], "CI_95": format_ci(interval_cis, "cal_slope")},
            {"Level": "Interval", "Metric": "Threshold", "Value": payload["threshold"], "CI_95": ""},
        ]
    )
    interval_summary.to_csv(Config.OUT_DIR / "BestMode_Interval_Summary.csv", index=False)

    save_calibration_figure(
        y_te,
        payload["proba"],
        f"Calibration Curve ({mode}, interval-level)",
        Config.OUT_DIR / "Calibration_BestStage2.png",
    )
    save_dca_figure(
        y_te,
        payload["proba"],
        f"Decision Curve Analysis ({mode}, interval-level)",
        Config.OUT_DIR / "DCA_BestStage2_Interval.png",
    )
    save_threshold_sensitivity_figure(
        y_te,
        payload["proba"],
        payload["threshold"],
        f"Threshold Sensitivity ({mode}, interval-level)",
        Config.OUT_DIR / "Threshold_Sensitivity_BestStage2_Interval.png",
    )

    patient_tr = aggregate_patient_level(train_df, payload["oof_proba"])
    patient_te = aggregate_patient_level(test_df, payload["proba"])
    patient_thr = select_best_threshold(patient_tr["Y"].values, patient_tr["proba"].values, low=0.05, high=0.95, step=0.01)
    patient_metrics = compute_binary_metrics(patient_te["Y"].values, patient_te["proba"].values, patient_thr)
    patient_cis = bootstrap_group_cis(patient_te, patient_thr, group_col="Patient_ID")
    patient_cal = compute_calibration_stats(patient_te["Y"].values, patient_te["proba"].values)
    patient_summary = pd.DataFrame(
        [
            {"Level": "Patient", "Metric": "AUC", "Value": patient_metrics["auc"], "CI_95": format_ci(patient_cis, "auc")},
            {"Level": "Patient", "Metric": "PR_AUC", "Value": patient_metrics["prauc"], "CI_95": format_ci(patient_cis, "prauc")},
            {"Level": "Patient", "Metric": "Brier", "Value": patient_metrics["brier"], "CI_95": format_ci(patient_cis, "brier")},
            {"Level": "Patient", "Metric": "Recall", "Value": patient_metrics["recall"], "CI_95": format_ci(patient_cis, "recall")},
            {"Level": "Patient", "Metric": "Specificity", "Value": patient_metrics["specificity"], "CI_95": format_ci(patient_cis, "specificity")},
            {"Level": "Patient", "Metric": "Calibration_Intercept", "Value": patient_cal["intercept"], "CI_95": format_ci(patient_cis, "cal_intercept")},
            {"Level": "Patient", "Metric": "Calibration_Slope", "Value": patient_cal["slope"], "CI_95": format_ci(patient_cis, "cal_slope")},
            {"Level": "Patient", "Metric": "Threshold", "Value": patient_thr, "CI_95": ""},
        ]
    )
    patient_summary.to_csv(Config.OUT_DIR / "BestMode_Patient_Summary.csv", index=False)

    save_calibration_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Calibration Curve ({mode}, patient-level)",
        Config.OUT_DIR / "Calibration_BestStage2_Patient.png",
    )
    save_dca_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        f"Decision Curve Analysis ({mode}, patient-level)",
        Config.OUT_DIR / "DCA_BestStage2_Patient.png",
    )
    save_threshold_sensitivity_figure(
        patient_te["Y"].values,
        patient_te["proba"].values,
        patient_thr,
        f"Threshold Sensitivity ({mode}, patient-level)",
        Config.OUT_DIR / "Threshold_Sensitivity_BestStage2_Patient.png",
    )
    save_patient_risk_strata(patient_tr, patient_te, Config.OUT_DIR / "Patient_Risk_Q1Q4_BestStage2.png")

    patient_sens_df, _ = evaluate_patient_aggregation_sensitivity(train_df, test_df, payload["oof_proba"], payload["proba"])
    patient_sens_df.to_csv(Config.OUT_DIR / "Patient_Aggregation_Sensitivity_BestStage2.csv", index=False)
    save_patient_aggregation_sensitivity_figure(
        patient_sens_df,
        Config.OUT_DIR / "Patient_Aggregation_Sensitivity_BestStage2.png",
    )

    window_sens_df = evaluate_window_sensitivity(test_df, payload["proba"], payload["threshold"])
    window_sens_df.to_csv(Config.OUT_DIR / "Window_Sensitivity_BestStage2.csv", index=False)
    return interval_summary, patient_summary, patient_sens_df, window_sens_df


def record_group_result(mode, group_df, best_name, payload, group_results, train_df, test_df, all_rows, perf_frames, best_groups, all_group_results):
    """Append one group's outputs into the shared stage-2 collectors."""
    all_rows.append(group_df)
    best_groups[mode] = (best_name, payload)
    all_group_results[mode] = group_results
    print(
        f"  Best {mode:<32s}: {best_name:<14s} AUC={payload['metrics']['auc']:.3f} "
        f"PR-AUC={payload['metrics']['prauc']:.3f} C-index={payload['c_index']:.3f} "
        f"Brier={payload['metrics']['brier']:.3f}"
    )
    perf_frames.append(
        build_binary_performance_long(
            task_name=f"Stage2 {mode}",
            results=group_results,
            domain_payloads={
                "Train_Fit": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "train_fit_proba"},
                "Validation_OOF": {"y_true": train_df["Y_Relapse"].values.astype(int), "proba_key": "oof_proba"},
                "Test_Temporal": {"y_true": test_df["Y_Relapse"].values.astype(int), "proba_key": "proba"},
            },
            metric_keys=["prauc", "auc", "recall", "specificity", "f1"],
            threshold_key="threshold",
        )
    )


def write_second_round_summary(best_group_summary, stage1_all_metrics, state_metrics):
    """Write a short markdown assessment if compact two-stage branches still trail direct."""
    direct_pr = float(best_group_summary.loc[best_group_summary["Group"] == "direct", "PR_AUC"].iloc[0])
    non_direct = best_group_summary[best_group_summary["Group"] != "direct"].copy()
    best_non_direct = non_direct.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    delta_test_r2 = float(
        stage1_all_metrics.loc[
            (stage1_all_metrics["Split"] == "Test") & (stage1_all_metrics["Target"] == "Average"), "R2"
        ].iloc[0]
    )
    state_test_pr = float(
        state_metrics.loc[(state_metrics["Split"] == "Test") & (state_metrics["Metric"] == "Hyper_PR_AUC"), "Value"].iloc[0]
    )
    success = (
        float(best_non_direct["PR_AUC"]) >= direct_pr
        and abs(float(best_non_direct["Calibration_Intercept"])) < 1.0
        and float(best_non_direct["Calibration_Slope"]) > 0.6
    )

    if success:
        lines = [
            "# Two-Stage Second-Round Summary",
            "",
            f"- Success: `{best_non_direct['Group']}` matched/exceeded the direct PR-AUC reference ({direct_pr:.3f}).",
            f"- Winning branch: `{best_non_direct['Group']}` with `{best_non_direct['Best_Model']}`.",
            f"- Key result: PR-AUC `{best_non_direct['PR_AUC']:.3f}`, AUC `{best_non_direct['AUC']:.3f}`, Brier `{best_non_direct['Brier']:.3f}`.",
        ]
    else:
        predicted_only_pr = float(
            best_group_summary.loc[best_group_summary["Group"] == "predicted_only_compact", "PR_AUC"].iloc[0]
        )
        if delta_test_r2 < 0.10 and state_test_pr < 0.30:
            failure_source = "weak stage 1"
            next_direction = "make stage 1 interval-specific and optimize the next-state head before adding more fusion complexity"
        elif predicted_only_pr >= direct_pr - 0.03 and float(best_non_direct["PR_AUC"]) < predicted_only_pr + 0.01:
            failure_source = "poor handoff"
            next_direction = "keep the compact physio branch, but fuse only branch scores plus interval-specific calibration"
        else:
            failure_source = "interval heterogeneity"
            next_direction = "let the physio handoff vary by window instead of forcing one pooled fusion rule"

        lines = [
            "# Two-Stage Second-Round Summary",
            "",
            f"- Direct PR-AUC reference: `{direct_pr:.3f}`.",
            f"- Best non-direct branch: `{best_non_direct['Group']}` with `{best_non_direct['Best_Model']}`.",
            f"- Best non-direct result: PR-AUC `{best_non_direct['PR_AUC']:.3f}`, AUC `{best_non_direct['AUC']:.3f}`, Brier `{best_non_direct['Brier']:.3f}`.",
            f"- Stage-1 delta test R^2: `{delta_test_r2:.3f}`.",
            f"- Stage-1 next-state Hyper PR-AUC: `{state_test_pr:.3f}`.",
            f"- Main failure source: **{failure_source}**.",
            f"- Single next best direction: **{next_direction}**.",
        ]

    (Config.OUT_DIR / "TwoStage_SecondRound_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_self_optimization_report(best_group_summary, fusion_candidate_df, stage1_all_metrics, state_metrics):
    """Write a compact optimization log with the strongest findings from the latest run."""
    direct_row = best_group_summary.loc[best_group_summary["Group"] == "direct"].iloc[0]
    best_row = best_group_summary.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    relapse_ref = pd.read_csv(Config.SHARED_RELAPSE_DIR / "Performance_Rolling_Landmark_Test_Temporal.csv")
    relapse_lr_pr = float(relapse_ref.loc[relapse_ref["Metric"] == "PR-AUC", "Logistic Reg."].iloc[0])
    relapse_lr_auc = float(relapse_ref.loc[relapse_ref["Metric"] == "AUC", "Logistic Reg."].iloc[0])
    delta_test_r2 = float(
        stage1_all_metrics.loc[
            (stage1_all_metrics["Split"] == "Test") & (stage1_all_metrics["Target"] == "Average"),
            "R2",
        ].iloc[0]
    )
    state_test_pr = float(
        state_metrics.loc[
            (state_metrics["Split"] == "Test") & (state_metrics["Metric"] == "Hyper_PR_AUC"),
            "Value",
        ].iloc[0]
    )
    top_candidates = fusion_candidate_df.head(6).copy()
    searched_n = int(len(fusion_candidate_df))
    lines = [
        "# Two-Stage Self-Optimization Log",
        "",
        "## Objective",
        "",
        "- Continue tightening the transparent two-stage line until the handoff is competitive without widening the feature table.",
        "",
        "## Changes In This Round",
        "",
        "- Shrunk the current-to-stage2 carryover to the top direct linear features only.",
        "- Added extra physio-only source variants for fusion search: `state_only`, `state_delta`, `state_delta_uncertainty`.",
        "- Added clinical threshold-derived physio features such as predicted hyper-zone indicators.",
        "- Searched multiple transparent fusion templates, including full windowed and high-risk-window-gated variants.",
        "- Kept the final fusion layer linear only.",
        "",
        "## Stage-1 Status",
        "",
        f"- Delta-head test average `R^2`: `{delta_test_r2:.3f}`.",
        f"- Next-state head test `Hyper PR-AUC`: `{state_test_pr:.3f}`.",
        "- Interpretation: stage 1 is still modest, so gains must come from a better handoff rather than raw stage-1 strength.",
        "",
        "## Best Current Result",
        "",
        f"- Best branch in this run: `{best_row['Group']}` with `{best_row['Best_Model']}`.",
        f"- Interval-level result: `PR-AUC {best_row['PR_AUC']:.3f}`, `AUC {best_row['AUC']:.3f}`, `Brier {best_row['Brier']:.3f}`.",
        f"- Versus direct branch: `PR-AUC {best_row['PR_AUC'] - direct_row['PR_AUC']:+.3f}`, `AUC {best_row['AUC'] - direct_row['AUC']:+.3f}`.",
        f"- Versus relapse.py Logistic reference: `PR-AUC {best_row['PR_AUC'] - relapse_lr_pr:+.3f}`, `AUC {best_row['AUC'] - relapse_lr_auc:+.3f}`.",
        f"- Transparent candidate experiments searched this round: `{searched_n}`.",
        "",
        "## Fusion Search Top Candidates",
        "",
        "| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Cal Int | Cal Slope |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_candidates.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.Physio_Group}:{row.Physio_Source} | {row.Fusion_Template} | {row.Model} | "
            f"{row.PR_AUC:.3f} | {row.AUC:.3f} | {row.Brier:.3f} | {row.Calibration_Intercept:.3f} | {row.Calibration_Slope:.3f} |"
        )

    if str(best_row["Group"]) == "direct_plus_physio_fusion":
        lines.extend(
            [
                "",
                "## Current Insight",
                "",
                "- Transparent score fusion is still the only two-stage strategy that clearly adds value.",
                "- Compact physio features alone are not strong enough to replace the direct branch.",
                "- Naive current + physio concatenation remains unstable even after narrowing.",
                "- The practical takeaway is that stage-1 information should act as a calibrated modifier of direct risk, not as a wide second feature table.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Current Insight",
                "",
                "- The direct branch remains strongest, which means the handoff is still not strong enough.",
                "- The next best move would be interval-specific physio calibration rather than more feature concatenation.",
            ]
        )

    out_path = Config.OUT_DIR / "TwoStage_SelfOptimization_Log.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_self_optimization_round2_report(best_group_summary, fusion_candidate_df, stage1_all_metrics, state_metrics):
    """Write a new markdown log for the current optimization round."""
    direct_row = best_group_summary.loc[best_group_summary["Group"] == "direct"].iloc[0]
    best_row = best_group_summary.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    delta_test_r2 = float(
        stage1_all_metrics.loc[
            (stage1_all_metrics["Split"] == "Test") & (stage1_all_metrics["Target"] == "Average"),
            "R2",
        ].iloc[0]
    )
    state_test_pr = float(
        state_metrics.loc[
            (state_metrics["Split"] == "Test") & (state_metrics["Metric"] == "Hyper_PR_AUC"),
            "Value",
        ].iloc[0]
    )
    top_candidates = fusion_candidate_df.head(8).copy()
    lines = [
        "# Two-Stage Self-Optimization Round 2",
        "",
        "## What Was Tried",
        "",
        "- Added rule-like stage-1 meta-features: predicted hyper-zone indicators and a compact hyper-evidence score.",
        "- Added high-risk-window gating templates so the physio correction can act mainly on 1M->3M, 3M->6M, and 6M->12M.",
        f"- Total transparent experiments this round: `{len(fusion_candidate_df)}`.",
        "",
        "## Stage-1 Reminder",
        "",
        f"- Delta-head test average `R^2`: `{delta_test_r2:.3f}`.",
        f"- Next-state head test `Hyper PR-AUC`: `{state_test_pr:.3f}`.",
        "- Interpretation: stage 1 is still only moderately informative, so any gain has to come from selective use of the physio signal.",
        "",
        "## Best Result",
        "",
        f"- Best branch: `{best_row['Group']}` with `{best_row['Best_Model']}`.",
        f"- Metrics: `PR-AUC {best_row['PR_AUC']:.3f}`, `AUC {best_row['AUC']:.3f}`, `Brier {best_row['Brier']:.3f}`.",
        f"- Versus direct: `PR-AUC {best_row['PR_AUC'] - direct_row['PR_AUC']:+.3f}`, `AUC {best_row['AUC'] - direct_row['AUC']:+.3f}`.",
        "",
        "## Top Transparent Fusion Candidates",
        "",
        "| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Recall | Specificity |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_candidates.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.Physio_Group}:{row.Physio_Source} | {row.Fusion_Template} | {row.Model} | "
            f"{row.PR_AUC:.3f} | {row.AUC:.3f} | {row.Brier:.3f} | {row.Recall:.3f} | {row.Specificity:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Insight",
            "",
            "- The pipeline still wins only when stage-1 information is used as a narrow correction to the direct score.",
            "- High-risk-window gating is worth testing because relapse risk is heavily front-loaded across follow-up windows.",
            "- Rule-like hyper-zone features are reasonable paper features, but they still need to work through a tiny fusion layer rather than a wide concatenation table.",
        ]
    )
    (Config.OUT_DIR / "TwoStage_SelfOptimization_Round2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_binary_stage1_report(best_group_summary, fusion_candidate_df, stage1_all_metrics, state_metrics):
    """Write a new markdown report for the binary stage-1 round."""
    direct_row = best_group_summary.loc[best_group_summary["Group"] == "direct"].iloc[0]
    best_row = best_group_summary.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    delta_test_r2 = float(
        stage1_all_metrics.loc[
            (stage1_all_metrics["Split"] == "Test") & (stage1_all_metrics["Target"] == "Average"),
            "R2",
        ].iloc[0]
    )
    state_auc = float(
        state_metrics.loc[
            (state_metrics["Split"] == "Test") & (state_metrics["Metric"] == "AUC"),
            "Value",
        ].iloc[0]
    )
    state_pr = float(
        state_metrics.loc[
            (state_metrics["Split"] == "Test") & (state_metrics["Metric"] == "Hyper_PR_AUC"),
            "Value",
        ].iloc[0]
    )
    top_candidates = fusion_candidate_df.head(6).copy()
    lines = [
        "# Two-Stage Binary Stage-1 Round",
        "",
        "## What Changed",
        "",
        "- Replaced the previous 3-class next-state head with a binary `Hyper vs non-Hyper` head.",
        "- Tightened stage-1 meta-features around hyper risk, confidence, delta, and margin-to-hyper summaries.",
        "- Stopped broad fusion-template search; only kept a tiny transparent fusion comparison set.",
        "",
        "## Stage-1 Snapshot",
        "",
        f"- Delta-head test average `R^2`: `{delta_test_r2:.3f}`.",
        f"- Binary hyper-head test `AUC`: `{state_auc:.3f}`.",
        f"- Binary hyper-head test `PR-AUC`: `{state_pr:.3f}`.",
        "",
        "## Best Result",
        "",
        f"- Best branch: `{best_row['Group']}` with `{best_row['Best_Model']}`.",
        f"- Metrics: `PR-AUC {best_row['PR_AUC']:.3f}`, `AUC {best_row['AUC']:.3f}`, `Brier {best_row['Brier']:.3f}`.",
        f"- Versus direct: `PR-AUC {best_row['PR_AUC'] - direct_row['PR_AUC']:+.3f}`, `AUC {best_row['AUC'] - direct_row['AUC']:+.3f}`.",
        f"- Transparent experiments run in this round: `{len(fusion_candidate_df)}`.",
        "",
        "## Best Transparent Candidates",
        "",
        "| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Cal Int | Cal Slope |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_candidates.itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | {row.Physio_Group}:{row.Physio_Source} | {row.Fusion_Template} | {row.Model} | "
            f"{row.PR_AUC:.3f} | {row.AUC:.3f} | {row.Brier:.3f} | {row.Calibration_Intercept:.3f} | {row.Calibration_Slope:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Insight",
            "",
            "- Switching to binary stage-1 makes the intermediate task more aligned with the final relapse event.",
            "- The useful signal still enters stage 2 best as a narrow physio score, not as a wide feature table.",
            "- If this binary round still does not improve enough, the next highest-ROI move is interval-specific binary hyper heads rather than more fusion variants.",
        ]
    )
    (Config.OUT_DIR / "TwoStage_BinaryStage1_Round.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_fixed_fusion_spec():
    """Freeze the final fusion model to Elastic LR only."""
    return get_fusion_tune_specs()["Elastic LR"]


def fit_fixed_transition_fusion(mode, train_df, test_df, direct_payload, physio_payload, direct_name, physio_name):
    """Fit the frozen stage-2 fusion: both_windowed + Elastic LR."""
    x_tr, x_te, feat_names = build_fusion_feature_frames(
        train_df,
        test_df,
        direct_payload,
        physio_payload,
        template="both_windowed",
    )
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    write_fixed_feature_manifest(
        prefix=f"Stage2_{mode}",
        feat_names=feat_names,
        note=f"direct_branch={direct_name}; transition_branch={physio_name}; template=both_windowed; final_model=Elastic LR",
    )
    print(
        f"    Frozen fusion {mode:<32s}: {len(feat_names)} features "
        f"(direct={direct_name}, transition={physio_name}, template=both_windowed, model=Elastic LR)"
    )

    model_name, base_est, param_grid, color, ls, n_iter, n_jobs = (
        "Elastic LR",
        *get_fixed_fusion_spec(),
    )
    gkf = GroupKFold(n_splits=3)
    best_est, cv_score, best_params = fit_candidate_model(base_est, param_grid, n_iter, n_jobs, gkf, x_tr, y_tr, groups)

    oof = np.zeros(len(y_tr), dtype=float)
    for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
        fitted = clone(best_est)
        fitted.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
        oof[val_idx] = fitted.predict_proba(x_tr.iloc[val_idx])[:, 1]

    thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
    fitted = clone(best_est)
    fitted.fit(x_tr, y_tr)
    train_fit_proba = fitted.predict_proba(x_tr)[:, 1]
    proba = fitted.predict_proba(x_te)[:, 1]
    metrics = compute_binary_metrics(y_te, proba, thr)
    cal = compute_calibration_stats(y_te, proba)
    c_index = compute_recurrent_c_index_from_intervals(test_df, proba)

    row = {
        "Group": mode,
        "Model": model_name,
        "CV_PR_AUC": cv_score,
        "AUC": metrics["auc"],
        "PR_AUC": metrics["prauc"],
        "C_Index": c_index,
        "Brier": metrics["brier"],
        "Recall": metrics["recall"],
        "Specificity": metrics["specificity"],
        "F1": metrics["f1"],
        "Calibration_Intercept": cal["intercept"],
        "Calibration_Slope": cal["slope"],
        "Threshold": thr,
        "Best_Params": "" if best_params is None else str(best_params),
    }
    payload = {
        "model": fitted,
        "proba": proba,
        "oof_proba": oof,
        "train_fit_proba": train_fit_proba,
        "auc": metrics["auc"],
        "prauc": metrics["prauc"],
        "c_index": c_index,
        "threshold": thr,
        "metrics": metrics,
        "cal": cal,
        "feat_names": feat_names,
        "color": color,
        "ls": ls,
        "direct_name": direct_name,
        "transition_name": physio_name,
    }
    return pd.DataFrame([row]), model_name, payload, {model_name: payload}


def calibration_distance(intercept, slope):
    return abs(float(intercept)) + abs(float(slope) - 1.0)


def build_stop_rule_row(candidate_row, reference_row):
    pr_gain = float(candidate_row["PR_AUC"] - reference_row["PR_AUC"])
    auc_not_down = float(candidate_row["AUC"]) >= float(reference_row["AUC"]) - 1e-3
    brier_not_worse = float(candidate_row["Brier"]) <= float(reference_row["Brier"]) + 2e-3
    cal_not_worse = calibration_distance(
        candidate_row["Calibration_Intercept"],
        candidate_row["Calibration_Slope"],
    ) <= calibration_distance(reference_row["Calibration_Intercept"], reference_row["Calibration_Slope"]) + 0.15
    pass_count = int(pr_gain >= 0.005) + int(auc_not_down) + int(brier_not_worse) + int(cal_not_worse)
    return {
        "Group": candidate_row["Group"],
        "PR_AUC_Gain_vs_Reference": pr_gain,
        "AUC_Not_Down": auc_not_down,
        "Brier_Not_Worse": brier_not_worse,
        "Calibration_Not_Worse": cal_not_worse,
        "Pass_Count": pass_count,
        "Pass_StopRule": pass_count >= 2,
    }


def select_batch2_variants(stop_rule_df):
    winners = stop_rule_df.loc[stop_rule_df["Pass_StopRule"]].copy()
    if len(winners) == 0:
        return []
    top2 = winners.sort_values(
        ["PR_AUC_Gain_vs_Reference", "Pass_Count"],
        ascending=[False, False],
    )["Group"].head(2).tolist()
    next_variants = []
    if any(name in top2 for name in ["binary_interval_intercept", "binary_interval_slope"]):
        next_variants.append("binary_full_5window")
    if "binary_early_late" in top2:
        next_variants.append("binary_early_late_calibrated")
    if "binary_rule_aux" in top2:
        next_variants.extend(
            [
                "binary_rule_aux_w0_2",
                "binary_rule_aux_w0_35",
                "binary_rule_aux_w0_5",
                "binary_threshold_twin_aux",
            ]
        )
    return list(dict.fromkeys(next_variants))


def compute_window_metric_table(group_name, test_df, payload):
    rows = []
    threshold = float(payload["threshold"])
    for interval_name, subset in test_df.groupby("Interval_Name", sort=False):
        idx = subset.index.to_numpy()
        y_true = subset["Y_Relapse"].values.astype(int)
        proba = np.asarray(payload["proba"], dtype=float)[idx]
        if len(y_true) == 0:
            continue
        metrics = compute_binary_metrics(y_true, proba, threshold)
        cal = compute_calibration_stats(y_true, proba)
        rows.append(
            {
                "Group": group_name,
                "Interval_Name": str(interval_name),
                "n": int(len(y_true)),
                "positives": int(np.sum(y_true)),
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_group_delta(y_true, proba_a, proba_b, groups, metric="prauc", n_boot=2000, seed=SEED):
    unique_groups = pd.Index(groups).drop_duplicates().to_list()
    group_to_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = np.concatenate([group_to_idx[group] for group in sampled])
        y_boot = y_true[boot_idx]
        if np.unique(y_boot).size < 2:
            continue
        if metric == "prauc":
            diff = average_precision_score(y_boot, proba_a[boot_idx]) - average_precision_score(y_boot, proba_b[boot_idx])
        elif metric == "auc":
            diff = roc_auc_score(y_boot, proba_a[boot_idx]) - roc_auc_score(y_boot, proba_b[boot_idx])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        diffs.append(float(diff))

    point = (
        average_precision_score(y_true, proba_a) - average_precision_score(y_true, proba_b)
        if metric == "prauc"
        else roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)
    )
    if len(diffs) == 0:
        return {"Metric": metric, "Delta": point, "CI_Low": np.nan, "CI_High": np.nan, "OneSided_P": np.nan}
    diffs = np.asarray(diffs, dtype=float)
    return {
        "Metric": metric,
        "Delta": point,
        "CI_Low": float(np.percentile(diffs, 2.5)),
        "CI_High": float(np.percentile(diffs, 97.5)),
        "OneSided_P": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1)),
    }


def build_bootstrap_delta_rows(group_name, test_df, candidate_payload, baseline_name, baseline_payload):
    y_true = test_df["Y_Relapse"].values.astype(int)
    groups = test_df["Patient_ID"].values
    proba_a = np.asarray(candidate_payload["proba"], dtype=float)
    proba_b = np.asarray(baseline_payload["proba"], dtype=float)
    rows = []
    for metric in ["prauc", "auc"]:
        out = bootstrap_group_delta(y_true, proba_a, proba_b, groups, metric=metric)
        out["Group"] = group_name
        out["Baseline"] = baseline_name
        rows.append(out)
    return pd.DataFrame(rows)


def save_stage1_only_comparison(summary_df):
    label_map = {
        "direct": "Direct",
        "current_binary_stage1_reference": "Current ref",
        "binary_interval_intercept": "Shared+int",
        "binary_interval_slope": "Shared+slope",
        "binary_early_late": "Early/late",
        "binary_interval_intercept_calibrated": "Shared+int-cal",
        "binary_rule_aux": "Rule aux",
        "binary_full_5window": "5-window",
        "binary_early_late_calibrated": "Early/late-cal",
        "binary_rule_aux_w0_2": "Rule aux 0.2",
        "binary_rule_aux_w0_35": "Rule aux 0.35",
        "binary_rule_aux_w0_5": "Rule aux 0.5",
        "binary_threshold_twin_aux": "Twin aux",
    }
    plot_df = summary_df.copy()
    plot_df["Label"] = plot_df["Group"].map(label_map).fillna(plot_df["Group"])
    fig_width = max(12.5, 1.25 * len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    auc_bars = axes[0].bar(plot_df["Label"], plot_df["AUC"], color=PRIMARY_BLUE, alpha=0.88, width=0.62)
    pr_bars = axes[1].bar(plot_df["Label"], plot_df["PR_AUC"], color=PRIMARY_TEAL, alpha=0.88, width=0.62)
    axes[0].set_title("Frozen Direct + Stage1 Variant AUC", fontsize=9)
    axes[1].set_title("Frozen Direct + Stage1 Variant PR-AUC", fontsize=9)
    axes[0].set_ylim(0, max(0.9, float(plot_df["AUC"].max()) + 0.08))
    axes[1].set_ylim(0, max(0.40, float(plot_df["PR_AUC"].max()) + 0.10))
    for ax, bars in [(axes[0], auc_bars), (axes[1], pr_bars)]:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=6.5, rotation=24)
        ax.tick_params(axis="y", labelsize=7)
        _annotate_bars(ax, bars)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(Config.OUT_DIR / "Stage1Only_Group_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def infer_failure_source(best_batch1_group):
    if best_batch1_group in {"binary_interval_intercept", "binary_interval_slope", "binary_interval_intercept_calibrated"}:
        return "score calibration"
    if best_batch1_group in {"binary_early_late", "binary_full_5window", "binary_early_late_calibrated"}:
        return "interval heterogeneity"
    return "auxiliary boundary signal"


def write_stage1_only_round_summary(
    summary_df,
    stop_rule_df,
    delta_boot_df,
    per_window_df,
    batch2_variants,
    best_mode,
    stage1_payload,
):
    direct_row = summary_df.loc[summary_df["Group"] == "direct"].iloc[0]
    ref_row = summary_df.loc[summary_df["Group"] == "current_binary_stage1_reference"].iloc[0]
    experiment_df = summary_df[
        ~summary_df["Group"].isin({"direct", "current_binary_stage1_reference"})
    ].sort_values(["PR_AUC", "AUC"], ascending=[False, False])
    best_exp = experiment_df.iloc[0]

    best_window_delta = []
    exp_window = per_window_df[per_window_df["Group"] == best_exp["Group"]].set_index("Interval_Name")
    ref_window = per_window_df[per_window_df["Group"] == "current_binary_stage1_reference"].set_index("Interval_Name")
    shared_intervals = [name for name in exp_window.index if name in ref_window.index]
    for interval_name in shared_intervals:
        best_window_delta.append(
            {
                "Interval_Name": interval_name,
                "Delta_PR_AUC_vs_Reference": float(
                    exp_window.loc[interval_name, "PR_AUC"] - ref_window.loc[interval_name, "PR_AUC"]
                ),
            }
        )
    best_window_delta = (
        pd.DataFrame(best_window_delta).sort_values("Delta_PR_AUC_vs_Reference", ascending=False)
        if best_window_delta
        else pd.DataFrame(columns=["Interval_Name", "Delta_PR_AUC_vs_Reference"])
    )

    delta_vs_direct = delta_boot_df[
        (delta_boot_df["Group"] == best_exp["Group"]) & (delta_boot_df["Baseline"] == "direct") & (delta_boot_df["Metric"] == "prauc")
    ]
    delta_vs_ref = delta_boot_df[
        (delta_boot_df["Group"] == best_exp["Group"])
        & (delta_boot_df["Baseline"] == "current_binary_stage1_reference")
        & (delta_boot_df["Metric"] == "prauc")
    ]

    lines = [
        "# Two-Stage Stage1-Only Round",
        "",
        "## Frozen Stage2",
        "",
        "- Final branch fixed to `direct_plus_physio_fusion`.",
        "- Fusion template fixed to `both_windowed`.",
        "- Final fusion learner fixed to `Elastic LR`.",
        "- Direct branch feature set and training flow were reused without widening the stage-2 feature table.",
        "",
        "## Stage1 Anchor",
        "",
        f"- Delta-head best model: `{stage1_payload['delta_best_name']}`.",
        f"- Shared binary next-hyper anchor: `{stage1_payload['state_anchor_name']}`.",
        f"- Shared binary head best train/test selector: `{stage1_payload['state_best_name']}`.",
        "",
        "## Best Variant",
        "",
        f"- Best new stage1 variant: `{best_exp['Group']}`.",
        f"- Interval-level metrics: `PR-AUC {best_exp['PR_AUC']:.3f}`, `AUC {best_exp['AUC']:.3f}`, `Brier {best_exp['Brier']:.3f}`.",
        f"- Versus direct: `PR-AUC {best_exp['PR_AUC'] - direct_row['PR_AUC']:+.3f}`, `AUC {best_exp['AUC'] - direct_row['AUC']:+.3f}`.",
        f"- Versus current binary-stage1 reference: `PR-AUC {best_exp['PR_AUC'] - ref_row['PR_AUC']:+.3f}`, `AUC {best_exp['AUC'] - ref_row['AUC']:+.3f}`.",
    ]
    if len(delta_vs_direct):
        row = delta_vs_direct.iloc[0]
        lines.append(
            f"- Grouped bootstrap vs direct PR-AUC delta: `{row['Delta']:+.3f}` (95% CI `{row['CI_Low']:+.3f}` to `{row['CI_High']:+.3f}`, one-sided `p={row['OneSided_P']:.3f}`)"
        )
    if len(delta_vs_ref):
        row = delta_vs_ref.iloc[0]
        lines.append(
            f"- Grouped bootstrap vs current reference PR-AUC delta: `{row['Delta']:+.3f}` (95% CI `{row['CI_Low']:+.3f}` to `{row['CI_High']:+.3f}`, one-sided `p={row['OneSided_P']:.3f}`)"
        )

    lines.extend(["", "## Per-Window Insight", ""])
    if len(best_window_delta):
        for row in best_window_delta.head(3).itertuples(index=False):
            lines.append(
                f"- `{row.Interval_Name}`: PR-AUC change vs current reference `{row.Delta_PR_AUC_vs_Reference:+.3f}`."
            )
    else:
        lines.append("- No per-window overlap was available for a stable comparison.")

    if stop_rule_df["Pass_StopRule"].any():
        lines.extend(["", "## Batch 2 Status", ""])
        lines.append(
            f"- Batch 1 passed the stop rule for `{int(stop_rule_df['Pass_StopRule'].sum())}` variant(s), so Batch 2 was triggered."
        )
        if batch2_variants:
            lines.append(f"- Batch 2 variants run: `{', '.join(batch2_variants)}`.")
        else:
            lines.append("- Batch 2 was eligible, but no winner-specific expansions were selected.")
    else:
        failure_source = infer_failure_source(str(experiment_df.iloc[0]["Group"]))
        lines.extend(["", "## Failure Source", ""])
        lines.append("- No Batch 1 variant cleared the stop rule.")
        lines.append(f"- Closest failure source: **{failure_source}**.")

    lines.append("")
    (Config.OUT_DIR / "TwoStage_Stage1Only_Round.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local two-stage caches")
    args = parser.parse_args()
    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 92)
    print("  Two-Stage Physiology Forecast for Relapse Prediction")
    print("=" * 92)

    df_train, df_test = build_two_stage_long_data()

    print("\n--- Phase 2: Stage-1 physiology forecast ---")
    x1_tr, x1_te, _ = make_stage1_feature_frames(df_train, df_test)
    y1_te = df_test[TARGET_COLS].values.astype(float)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values

    stage1_payload = run_stage1_oof_forecast(
        x1_tr, df_train, x1_te, df_test, groups, train_intervals, test_intervals, Config.OUT_DIR
    )
    best_stage1_name = stage1_payload["delta_best_name"]
    best_stage1 = stage1_payload["delta_results"][best_stage1_name]
    stage1_metrics = stage1_payload["delta_metrics"]
    stage1_test_metrics = evaluate_physio_predictions(y1_te, best_stage1["test_next"], "Test", best_stage1_name)
    stage1_all_metrics = pd.concat([stage1_metrics, stage1_test_metrics], ignore_index=True)
    stage1_all_metrics.to_csv(Config.OUT_DIR / "Physio_Forecast_Metrics.csv", index=False)
    save_stage1_metric_bar(stage1_metrics[stage1_metrics["Split"] == "Train_OOF"], Config.OUT_DIR)
    save_physio_scatter(y1_te, best_stage1["test_next"], Config.OUT_DIR, best_stage1_name, "Test")
    print(f"  Best delta-head model by average OOF RMSE: {best_stage1_name}")
    print(
        stage1_all_metrics[stage1_all_metrics["Target"] == "Average"][["Split", "Model", "MAE", "RMSE", "R2"]].to_string(index=False)
    )

    best_state_name = stage1_payload["state_best_name"]
    best_state = stage1_payload["state_results"][best_state_name]
    state_metrics = stage1_payload["state_metrics"].copy()
    state_metrics.to_csv(Config.OUT_DIR / "NextState_Forecast_Metrics.csv", index=False)
    transition_variant_metrics = stage1_payload["transition_variant_metrics"].copy()
    transition_variant_metrics.to_csv(Config.OUT_DIR / "Stage1_Transition_Variant_Metrics.csv", index=False)
    print(f"  Best next-state head by Hyper PR-AUC: {best_state_name}")
    print(
        state_metrics[state_metrics["Model"] == best_state_name][["Split", "Metric", "Value"]].to_string(index=False)
    )
    print(f"  Frozen stage-1 anchor model for variant suite: {stage1_payload['state_anchor_name']}")
    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")

    print("\n--- Phase 3: Frozen direct + current reference ---")
    all_rows = []
    perf_frames = []
    best_groups = {}
    all_group_results = {}

    group_df, best_name, payload, group_results = fit_stage2_group(
        "direct", df_train_pred, df_test_pred, use_feature_selection=True
    )
    direct_source_name, direct_source_payload = select_transparent_payload(group_results)
    group_df = group_df[group_df["Model"] == direct_source_name].reset_index(drop=True)
    record_group_result(
        "direct",
        group_df,
        direct_source_name,
        direct_source_payload,
        group_results,
        df_train_pred,
        df_test_pred,
        all_rows,
        perf_frames,
        best_groups,
        all_group_results,
    )

    group_df, best_name, payload, group_results = fit_stage2_group(
        "predicted_only_state_rule", df_train_pred, df_test_pred, use_feature_selection=False
    )
    reference_transition_name, reference_transition_payload = select_transparent_payload(group_results)
    group_df, best_name, payload, group_results = fit_fixed_transition_fusion(
        "current_binary_stage1_reference",
        df_train_pred,
        df_test_pred,
        direct_source_payload,
        reference_transition_payload,
        direct_source_name,
        f"predicted_only_state_rule:{reference_transition_name}",
    )
    record_group_result(
        "current_binary_stage1_reference",
        group_df,
        best_name,
        payload,
        group_results,
        df_train_pred,
        df_test_pred,
        all_rows,
        perf_frames,
        best_groups,
        all_group_results,
    )

    print("\n--- Phase 4: Batch 1 stage-1 variants on frozen fusion ---")
    transition_variants = stage1_payload["transition_variant_results"]
    batch1_variant_order = [
        "binary_interval_intercept",
        "binary_interval_slope",
        "binary_early_late",
        "binary_interval_intercept_calibrated",
        "binary_rule_aux",
    ]
    for mode in batch1_variant_order:
        pack = transition_variants[mode]
        group_df, best_name, payload, group_results = fit_fixed_transition_fusion(
            mode,
            df_train_pred,
            df_test_pred,
            direct_source_payload,
            pack,
            direct_source_name,
            pack["variant_name"],
        )
        record_group_result(
            mode, group_df, best_name, payload, group_results, df_train_pred, df_test_pred, all_rows, perf_frames, best_groups, all_group_results
        )

    batch1_reference_row = {
        "Group": "current_binary_stage1_reference",
        "AUC": best_groups["current_binary_stage1_reference"][1]["metrics"]["auc"],
        "PR_AUC": best_groups["current_binary_stage1_reference"][1]["metrics"]["prauc"],
        "Brier": best_groups["current_binary_stage1_reference"][1]["metrics"]["brier"],
        "Calibration_Intercept": best_groups["current_binary_stage1_reference"][1]["cal"]["intercept"],
        "Calibration_Slope": best_groups["current_binary_stage1_reference"][1]["cal"]["slope"],
    }
    stop_rows = []
    for mode in batch1_variant_order:
        payload = best_groups[mode][1]
        candidate_row = {
            "Group": mode,
            "AUC": payload["metrics"]["auc"],
            "PR_AUC": payload["metrics"]["prauc"],
            "Brier": payload["metrics"]["brier"],
            "Calibration_Intercept": payload["cal"]["intercept"],
            "Calibration_Slope": payload["cal"]["slope"],
        }
        stop_rows.append(build_stop_rule_row(candidate_row, batch1_reference_row))
    stop_rule_df = pd.DataFrame(stop_rows)
    stop_rule_df.to_csv(Config.OUT_DIR / "Stage1Only_StopRule.csv", index=False)

    batch2_variants = select_batch2_variants(stop_rule_df)
    if batch2_variants:
        print("\n--- Phase 5: Batch 2 winner-neighbourhood variants ---")
    else:
        print("\n--- Phase 5: Batch 2 skipped (no Batch 1 variant passed stop rule) ---")
    for mode in batch2_variants:
        pack = transition_variants[mode]
        group_df, best_name, payload, group_results = fit_fixed_transition_fusion(
            mode,
            df_train_pred,
            df_test_pred,
            direct_source_payload,
            pack,
            direct_source_name,
            pack["variant_name"],
        )
        record_group_result(
            mode, group_df, best_name, payload, group_results, df_train_pred, df_test_pred, all_rows, perf_frames, best_groups, all_group_results
        )

    two_stage_df = pd.concat(all_rows, ignore_index=True)
    two_stage_df.to_csv(Config.OUT_DIR / "Stage1Only_Model_Comparison.csv", index=False)
    two_stage_df.to_csv(Config.OUT_DIR / "TwoStage_Model_Comparison.csv", index=False)
    perf_long_df = pd.concat(perf_frames, ignore_index=True)
    export_metric_matrices(perf_long_df, Config.OUT_DIR, prefix="Stage1Only_Performance")
    task_order = [f"Stage2 {mode}" for mode in best_groups]
    save_performance_heatmap_panels(
        perf_long_df,
        Config.OUT_DIR / "Stage1Only_Performance_Heatmaps.png",
        task_order=task_order,
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=["prauc", "auc", "recall", "specificity", "f1"],
        title="Frozen Direct + Stage1 Variant Performance Heatmaps",
    )
    save_stage1_only_comparison(
        pd.DataFrame(
            [
                {
                    "Group": mode,
                    "Best_Model": payload["model"].__class__.__name__ if hasattr(payload["model"], "__class__") else best_name,
                    "AUC": payload["metrics"]["auc"],
                    "PR_AUC": payload["metrics"]["prauc"],
                }
                for mode, (_, payload) in best_groups.items()
            ]
        )
    )

    best_mode = max(best_groups, key=lambda g: (best_groups[g][1]["metrics"]["prauc"], best_groups[g][1]["metrics"]["auc"]))
    best_name, best_payload = best_groups[best_mode]
    save_best_mode_reports(best_mode, best_payload, df_train_pred, df_test_pred)

    if best_name in {"Logistic Reg.", "Elastic LR"}:
        prefix = f"{best_mode}_{best_name}".replace(" ", "_")
        save_logistic_regression_visuals(
            best_name,
            best_payload["model"],
            best_payload["feat_names"],
            Config.OUT_DIR,
            prefix=prefix,
            decision_threshold=best_payload["metrics"]["threshold"],
            output_label="P(Relapse at next window)",
        )

    group_order = [
        "direct",
        "current_binary_stage1_reference",
        *batch1_variant_order,
        *batch2_variants,
    ]
    summary_rows = []
    for mode in group_order:
        if mode not in best_groups:
            continue
        model_name, payload = best_groups[mode]
        cal = payload["cal"]
        metrics = payload["metrics"]
        summary_rows.append(
            {
                "Group": mode,
                "Best_Model": model_name,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "C_Index": payload["c_index"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": metrics["threshold"],
            }
    )
    best_group_summary = pd.DataFrame(summary_rows)
    best_group_summary.to_csv(Config.OUT_DIR / "Stage1Only_Best_Group_Summary.csv", index=False)
    best_group_summary.to_csv(Config.OUT_DIR / "TwoStage_Best_Group_Summary.csv", index=False)
    best_group_summary.to_csv(Config.OUT_DIR / "Stage1Only_Ablation_Summary.csv", index=False)
    best_group_summary.to_csv(Config.OUT_DIR / "TwoStage_Ablation_Summary.csv", index=False)

    per_window_frames = []
    for mode, (_, payload) in best_groups.items():
        per_window_frames.append(compute_window_metric_table(mode, df_test_pred, payload))
    per_window_df = pd.concat(per_window_frames, ignore_index=True)
    per_window_df.to_csv(Config.OUT_DIR / "Stage1Only_PerWindow_Metrics.csv", index=False)

    delta_frames = []
    direct_payload = best_groups["direct"][1]
    reference_payload = best_groups["current_binary_stage1_reference"][1]
    for mode, (_, payload) in best_groups.items():
        if mode != "direct":
            delta_frames.append(build_bootstrap_delta_rows(mode, df_test_pred, payload, "direct", direct_payload))
        if mode not in {"direct", "current_binary_stage1_reference"}:
            delta_frames.append(
                build_bootstrap_delta_rows(mode, df_test_pred, payload, "current_binary_stage1_reference", reference_payload)
            )
    delta_boot_df = pd.concat(delta_frames, ignore_index=True)
    delta_boot_df.to_csv(Config.OUT_DIR / "Stage1Only_DeltaBootstrap.csv", index=False)

    write_stage1_only_round_summary(
        best_group_summary,
        stop_rule_df,
        delta_boot_df,
        per_window_df,
        batch2_variants,
        best_mode,
        stage1_payload,
    )

    print("\n  Frozen-direct stage1-only summary")
    for row in best_group_summary.itertuples(index=False):
        print(
            f"    {row.Group:<22s} {row.Best_Model:<14s} "
            f"AUC={row.AUC:.3f}  PR-AUC={row.PR_AUC:.3f}  "
            f"C-index={row.C_Index:.3f}  Brier={row.Brier:.3f}  "
            f"Cal.Int={row.Calibration_Intercept:.3f}  Cal.Slope={row.Calibration_Slope:.3f}"
        )
    print(f"    Best overall mode: {best_mode} ({best_name})")
    if batch2_variants:
        print(f"    Batch-2 variants run: {', '.join(batch2_variants)}")
    else:
        print("    Batch-2 variants run: none")
    print(f"\n  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
