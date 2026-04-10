"""Export deployable artifacts for the current Chinese Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from scripts import fixed_landmark_binary as fixed_mod
from scripts import relapse_threehead_landmark as threehead_mod
from thyroid_app.training import (
    _build_sample_cases,
    _coerce_case_value,
    _fit_or_refit_imputer,
    ensure_export_dirs,
    export_relapse_artifact,
    get_export_paths,
)
from utils.config import STATIC_NAMES, TIME_STAMPS
from utils.data import load_data, split_imputed, temporal_split
from utils.evaluation import compute_binary_metrics, compute_calibration_stats


def _serialize_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _extract_feature_block_metadata(df_fit: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], list[str], dict]:
    interval_categories = sorted(df_fit["Interval_Name"].unique())
    prev_state_categories = sorted(df_fit["Prev_State"].unique())
    static_df, local_df, global_df = threehead_mod.build_feature_blocks(df_fit, interval_categories, prev_state_categories)
    scalers = threehead_mod.fit_block_scalers(static_df, local_df, global_df)
    return (
        list(static_df.columns),
        list(local_df.columns),
        list(global_df.columns),
        interval_categories,
        prev_state_categories,
        scalers,
    )


def _compute_projected_quartiles(pred_df: pd.DataFrame) -> list[float]:
    start_landmarks = pred_df["Interval_Name"].astype(str).str.split("->").str[0]
    remaining = start_landmarks.map(lambda x: max(1, len(TIME_STAMPS) - 1 - TIME_STAMPS.index(x))).astype(float)
    projected = 1.0 - np.power(1.0 - pred_df["FuseBest_Prob"].astype(float).values, remaining.values)
    quantiles = np.quantile(projected, [0.0, 0.25, 0.50, 0.75, 1.0]).tolist()
    return [float(x) for x in quantiles]


def _extract_fuse_drivers(formal_bundle: dict) -> list[dict]:
    model = formal_bundle["model"]
    feature_names = formal_bundle["feature_names"]
    clf = model.named_steps["clf"]
    coef = clf.coef_.ravel()
    ranked = sorted(zip(feature_names, coef), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "coefficient": float(value)} for name, value in ranked[:10]]


def export_teacher_frozen_relapse_artifact(paths) -> dict:
    bundle_path = paths.relapse_dir / "bundle.joblib"
    if bundle_path.exists():
        try:
            existing = joblib.load(bundle_path)
            if existing.get("task") == "teacher_frozen_relapse":
                return existing
        except Exception:
            pass

    formal_path = Path("results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_FormalWinner.pkl")
    meta_path = Path("results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_FormalWinner_Metadata.json")
    summary_path = Path("results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_Summary.csv")
    pred_path = Path("results/relapse_teacher_frozen_fuse/TemporalTest_Predictions.csv")

    formal_bundle = joblib.load(formal_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(metadata["backbone_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    df_tr, df_te, _, _, unique_pids = threehead_mod.build_longitudinal_tables()
    train_patient_order = unique_pids[: int(len(unique_pids) * 0.8)]
    df_fit, df_val = threehead_mod.split_train_validation(df_tr, train_patient_order)
    (
        static_features,
        local_features,
        global_features,
        interval_categories,
        prev_state_categories,
        block_scalers,
    ) = _extract_feature_block_metadata(df_fit)

    summary_df = pd.read_csv(summary_path)
    discrim_df = pd.read_csv("results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_Recomputed_Discrimination.csv")
    fit_row = discrim_df.loc[discrim_df["Split"] == "Fit"].iloc[0].to_dict()
    val_row = discrim_df.loc[discrim_df["Split"] == "Validation"].iloc[0].to_dict()
    te_row = discrim_df.loc[discrim_df["Split"] == "TemporalTest"].iloc[0].to_dict()
    pred_df = pd.read_csv(pred_path)
    patient_quartiles = _compute_projected_quartiles(pred_df)

    test_summary = summary_df.loc[summary_df["Split"] == "TemporalTest"].set_index("Metric")["Value"]
    val_summary = summary_df.loc[summary_df["Split"] == "Validation"].set_index("Metric")["Value"]

    bundle = {
        "task": "teacher_frozen_relapse",
        "model_name": "Landmark-guided Local-Global Multi-Head Dynamic Hazard Network",
        "backbone_state_dict": checkpoint["state_dict"],
        "embed_dim": int(checkpoint["embed_dim"]),
        "gate_variant": checkpoint["gate_variant"],
        "block_scalers": block_scalers,
        "static_features": static_features,
        "local_features": local_features,
        "global_features": global_features,
        "interval_categories": interval_categories,
        "prev_state_categories": prev_state_categories,
        "fuse_model": formal_bundle["model"],
        "fuse_feature_names": list(formal_bundle["feature_names"]),
        "decision_threshold": float(formal_bundle["threshold"]),
        "fit_metrics": {
            "auc": float(fit_row["AUC"]),
            "prauc": float(fit_row["PR_AUC"]),
        },
        "validation_metrics": {
            "auc": float(val_row["AUC"]),
            "prauc": float(val_row["PR_AUC"]),
            "f1": float(val_summary["F1"]),
            "threshold": float(val_summary["Threshold"]),
        },
        "test_metrics": {
            "auc": float(te_row["AUC"]),
            "prauc": float(te_row["PR_AUC"]),
            "f1": float(test_summary["F1"]),
            "recall": float(test_summary["Recall"]),
            "specificity": float(test_summary["Specificity"]),
            "threshold": float(formal_bundle["threshold"]),
        },
        "patient_quartiles": patient_quartiles,
        "global_drivers": _extract_fuse_drivers(formal_bundle),
        "metadata": metadata,
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def _load_existing_fixed_3m_bundle(paths) -> dict | None:
    bundle_path = paths.fixed_3m_dir / "bundle.joblib"
    if not bundle_path.exists():
        return None
    try:
        bundle = joblib.load(bundle_path)
    except Exception:
        return None
    return bundle if bundle.get("task") in {"fixed_landmark", "fixed_landmark_routed"} else None


def _load_existing_sample_cases(paths) -> dict | None:
    if not paths.manifest.exists():
        return None
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    sample_cases = manifest.get("sample_cases")
    if not isinstance(sample_cases, dict):
        return None
    return sample_cases


def _fit_selected_fixed_models(X_tr_df, X_te_df, y_tr, y_te, groups_tr):
    gkf = GroupKFold(n_splits=3)
    results = {}
    tune_specs = fixed_mod.get_tune_specs()
    selected_names = ["Elastic LR", "Random Forest"]
    if "LightGBM" in tune_specs:
        selected_names.append("LightGBM")

    for name in selected_names:
        base, grid, color, ls, n_iter, n_jobs = tune_specs[name]
        best_est, best_score, best_params = fixed_mod.fit_candidate_model(
            base, grid, n_iter, n_jobs, gkf, X_tr_df, y_tr, groups_tr
        )
        oof = np.zeros(len(y_tr), dtype=float)
        for f_tr, f_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            fitted = clone(best_est)
            fitted.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
            oof[f_val] = fitted.predict_proba(X_tr_df.iloc[f_val])[:, 1]

        thr = fixed_mod.search_best_threshold(y_tr, oof, objective="f1")
        fitted = clone(best_est)
        fitted.fit(X_tr_df, y_tr)
        train_fit_proba = fitted.predict_proba(X_tr_df)[:, 1]
        test_proba = fitted.predict_proba(X_te_df)[:, 1]
        metrics = compute_binary_metrics(y_te, test_proba, thr)
        tn, fp, fn, tp = confusion_matrix(y_te, (test_proba >= thr).astype(int)).ravel()
        results[name] = {
            "model": fitted,
            "oof_proba": oof,
            "train_fit_proba": train_fit_proba,
            "proba": test_proba,
            "thr": float(thr),
            "auc": float(metrics["auc"]),
            "prauc": float(metrics["prauc"]),
            "acc": float(metrics["acc"]),
            "bacc": float(metrics["bacc"]),
            "recall": float(metrics["recall"]),
            "specificity": float(metrics["specificity"]),
            "f1": float(metrics["f1"]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "best_params": best_params,
            "cv_prauc": float(best_score),
        }
    return results


def export_fixed_3m_routed_artifact(paths) -> dict:
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, _eval_raw, y_raw, pids = load_data()
    seq_len = 3
    n_static = X_s_raw.shape[1]
    train_idx, test_idx = temporal_split(pids, ratio=0.8)

    raw_all = np.hstack([X_s_raw, ft3_raw[:, :seq_len], ft4_raw[:, :seq_len], tsh_raw[:, :seq_len]])
    imputer_path = paths.cache / f"fixed_missforest_seqlen{seq_len}.pkl"
    imputer = _fit_or_refit_imputer(raw_all[train_idx], imputer_path)
    filled_tr = imputer.transform(raw_all[train_idx])
    filled_te = imputer.transform(raw_all[test_idx])
    xs_tr, f3_tr, f4_tr, ts_tr = split_imputed(filled_tr, n_static, seq_len)
    xs_te, f3_te, f4_te, ts_te = split_imputed(filled_te, n_static, seq_len)

    Xd_tr = np.stack([f3_tr, f4_tr, ts_tr], axis=-1)
    Xd_te = np.stack([f3_te, f4_te, ts_te], axis=-1)
    X_tr_feat, feat_names = fixed_mod.build_landmark_features(xs_tr, Xd_tr, seq_len)
    X_te_feat, _ = fixed_mod.build_landmark_features(xs_te, Xd_te, seq_len)
    X_tr_df = pd.DataFrame(X_tr_feat, columns=feat_names)
    X_te_df = pd.DataFrame(X_te_feat, columns=feat_names)

    y_tr = (y_raw[train_idx] == 1).astype(int)
    y_te = (y_raw[test_idx] == 1).astype(int)
    groups_tr = pids[train_idx]

    results = _fit_selected_fixed_models(X_tr_df, X_te_df, y_tr, y_te, groups_tr)
    if "Elastic LR" not in results or "LightGBM" not in results:
        raise RuntimeError("Current 3M deployment artifact requires both Elastic LR and LightGBM.")

    blend = fixed_mod.build_weighted_blend(results["Elastic LR"], results["LightGBM"], y_tr)
    blend_metrics = compute_binary_metrics(y_te, blend["proba"], blend["thr"])
    results["Elastic+LGBM Blend"] = {
        "model": None,
        "oof_proba": blend["oof_proba"],
        "train_fit_proba": blend["train_fit_proba"],
        "proba": blend["proba"],
        "thr": float(blend["thr"]),
        "auc": float(blend_metrics["auc"]),
        "prauc": float(blend_metrics["prauc"]),
        "acc": float(blend_metrics["acc"]),
        "bacc": float(blend_metrics["bacc"]),
        "recall": float(blend_metrics["recall"]),
        "specificity": float(blend_metrics["specificity"]),
        "f1": float(blend_metrics["f1"]),
        "blend_weights": tuple(float(x) for x in blend["weights"]),
    }

    routed, routed_search_df, routed_feature_df = fixed_mod.build_routed_specialist_model(
        "Elastic+LGBM Blend",
        results["Elastic+LGBM Blend"],
        results,
        X_tr_df,
        y_tr,
        groups_tr,
        X_te_df,
        GroupKFold(n_splits=3),
        objective="acc",
    )
    if routed is None:
        raise RuntimeError("Failed to rebuild the 3M routed deployment model.")

    routed_metrics = compute_binary_metrics(y_te, routed["proba"], routed["thr"])
    calibration = compute_calibration_stats(y_te, routed["proba"])
    global_drivers = []
    if not routed_feature_df.empty:
        global_drivers.extend(
            {"feature": row["feature"], "importance": float(row["error_shift"])}
            for _, row in routed_feature_df.head(8).iterrows()
        )

    bundle = {
        "task": "fixed_landmark_routed",
        "landmark": "3M",
        "seq_len": 3,
        "model_name": "Elastic+LGBM Blend Routed",
        "decision_threshold": float(routed["thr"]),
        "feature_names": feat_names,
        "imputer": imputer,
        "anchor_elastic_model": results["Elastic LR"]["model"],
        "anchor_lgbm_model": results["LightGBM"]["model"],
        "anchor_blend_weights": tuple(float(x) for x in blend["weights"]),
        "anchor_threshold": float(blend["thr"]),
        "specialist_model": routed["specialist_model"],
        "specialist_name": routed["specialist_name"],
        "route_feature": routed["gate_feature"],
        "route_cutoff": float(routed["gate_cutoff"]),
        "route_quantile": float(routed["gate_quantile"]),
        "route_gate_rate": float(routed["gate_rate"]),
        "feature_medians": X_tr_df.median().to_dict(),
        "candidate_summary": [
            {
                "model": name,
                "auc": payload["auc"],
                "prauc": payload["prauc"],
                "acc": payload["acc"],
                "f1": payload["f1"],
                "threshold": payload["thr"],
            }
            for name, payload in results.items()
            if name not in {"Elastic+LGBM Blend"}
        ],
        "test_metrics": _serialize_metrics(routed_metrics),
        "test_calibration": _serialize_metrics(calibration),
        "global_drivers": global_drivers,
    }
    bundle_path = paths.fixed_3m_dir / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    return bundle


def export_all_artifacts(base_dir: str | Path = "artifacts") -> dict:
    paths = get_export_paths(base_dir)
    ensure_export_dirs(paths)

    relapse_direct_bundle = export_relapse_artifact(paths)
    relapse_bundle = export_teacher_frozen_relapse_artifact(paths)
    fixed_3m_bundle = _load_existing_fixed_3m_bundle(paths) or export_fixed_3m_routed_artifact(paths)
    sample_cases = _load_existing_sample_cases(paths) or _build_sample_cases(paths, relapse_bundle)

    manifest = {
        "version": "v3",
        "ui_language": "zh",
        "artifacts_root": str(paths.root),
        "tasks": {
            "relapse": {
                "bundle_path": str(paths.relapse_dir / "bundle.joblib"),
                "current_landmarks": ["1M", "3M", "6M", "12M", "18M"],
                "required_static_fields": STATIC_NAMES,
                "required_state_semantics": "动态复发预测要求当前状态为 Normal。",
                "fit_metrics": relapse_bundle["fit_metrics"],
                "validation_metrics": relapse_bundle["validation_metrics"],
                "test_metrics": relapse_bundle["test_metrics"],
                "decision_threshold": relapse_bundle["decision_threshold"],
                "global_drivers": relapse_bundle["global_drivers"],
                "model_name": relapse_bundle["model_name"],
                "deployment_label": "动态复发预警",
                "baseline_task": "relapse_direct",
                "baseline_metrics": relapse_direct_bundle["test_metrics"],
                "patient_quartiles": relapse_bundle["patient_quartiles"],
            },
            "relapse_direct": {
                "bundle_path": str(paths.relapse_direct_dir / "bundle.joblib"),
                "current_landmarks": ["1M", "3M", "6M", "12M", "18M"],
                "required_static_fields": STATIC_NAMES,
                "required_state_semantics": "动态复发预测要求当前状态为 Normal。",
                "test_metrics": relapse_direct_bundle["test_metrics"],
                "patient_test_metrics": relapse_direct_bundle["patient_test_metrics"],
                "decision_threshold": relapse_direct_bundle["decision_threshold"],
                "global_drivers": relapse_direct_bundle["global_drivers"],
                "model_name": relapse_direct_bundle["model_name"],
            },
            "fixed_3m": {
                "bundle_path": str(paths.fixed_3m_dir / "bundle.joblib"),
                "landmark": "3M",
                "test_metrics": fixed_3m_bundle["test_metrics"],
                "decision_threshold": fixed_3m_bundle["decision_threshold"],
                "model_name": fixed_3m_bundle["model_name"],
                "global_drivers": fixed_3m_bundle["global_drivers"],
            },
        },
        "sample_cases": sample_cases,
    }

    with paths.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=_coerce_case_value)

    from thyroid_app.inference import predict_relapse

    relapse_case = manifest["sample_cases"]["relapse"]
    relapse_case["expected_probability"] = float(predict_relapse(relapse_case, paths.root)["predicted_probability"])
    with paths.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=_coerce_case_value)
    return manifest
