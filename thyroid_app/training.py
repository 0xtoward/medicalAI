"""Artifact export helpers for the English Streamlit app."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thyroid_app.constants import (
    FIXED_LANDMARKS,
    RELAPSE_CURRENT_LANDMARKS,
)
from thyroid_app.features import make_relapse_features
from utils.config import SEED, STATIC_NAMES
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    fit_missforest,
    load_data,
    load_or_fit_depth_imputer,
    split_imputed,
    temporal_split,
    extract_flat_features,
)
from utils.evaluation import (
    aggregate_patient_level,
    compute_binary_metrics,
    compute_calibration_stats,
    select_best_threshold,
)
from utils.recurrence import build_interval_risk_data

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached")


@dataclass
class ExportPaths:
    root: Path
    cache: Path
    relapse_dir: Path
    relapse_direct_dir: Path
    fixed_3m_dir: Path
    fixed_6m_dir: Path
    manifest: Path


def get_export_paths(base_dir: str | Path = "artifacts") -> ExportPaths:
    root = Path(base_dir)
    return ExportPaths(
        root=root,
        cache=root / "cache",
        relapse_dir=root / "relapse",
        relapse_direct_dir=root / "relapse_direct",
        fixed_3m_dir=root / "fixed_3m",
        fixed_6m_dir=root / "fixed_6m",
        manifest=root / "manifest.json",
    )


def ensure_export_dirs(paths: ExportPaths) -> None:
    for path in [paths.root, paths.cache, paths.relapse_dir, paths.relapse_direct_dir, paths.fixed_3m_dir, paths.fixed_6m_dir]:
        path.mkdir(parents=True, exist_ok=True)


def _candidate_models():
    seed = SEED
    return {
        "Logistic Reg.": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "lr__penalty": ["l1", "l2"],
                "lr__solver": ["saga"],
            },
            14,
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
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            12,
            -1,
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "svc__C": [0.1, 0.5, 1, 2, 5, 10],
                "svc__gamma": ["scale", 0.01, 0.05, 0.1, 0.5],
            },
            14,
            -1,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced"),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [3, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5],
            },
            20,
            -1,
        ),
        "Balanced RF": (
            BalancedRandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "min_samples_leaf": [1, 3, 5, 10],
                "sampling_strategy": ["all", "not minority"],
            },
            16,
            -1,
        ),
        "MLP": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            max_iter=1000,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=50,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "mlp__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16)],
                "mlp__alpha": [0.001, 0.01, 0.05, 0.1],
                "mlp__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
            },
            16,
            -1,
        ),
    }


def _fit_candidate_model(base, grid, n_iter, n_jobs, cv, x_train, y_train, groups):
    search = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        cv=cv,
        scoring="average_precision",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    search.fit(x_train, y_train, groups=groups)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def _serialize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def build_relapse_datasets(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_static_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = load_data()
    n_static = x_static_raw.shape[1]
    train_mask, test_mask = temporal_split(pids, ratio=0.8)

    train_parts = []
    test_parts = []
    for depth in range(1, 7):
        k = depth - 1
        raw = np.hstack(
            [
                x_static_raw,
                ft3_raw[:, :depth],
                ft4_raw[:, :depth],
                tsh_raw[:, :depth],
                eval_raw[:, :depth],
            ]
        )
        cache_path = cache_dir / f"relapse_missforest_depth{depth}.pkl"
        imputer = load_or_fit_depth_imputer(raw[train_mask], cache_path)
        filled_train = apply_missforest(raw[train_mask], imputer, depth)
        filled_test = apply_missforest(raw[test_mask], imputer, depth)
        xs_train, ft3_train, ft4_train, tsh_train, eval_train = split_imputed(filled_train, n_static, depth, depth)
        xs_test, ft3_test, ft4_test, tsh_test, eval_test = split_imputed(filled_test, n_static, depth, depth)
        states_train = build_states_from_labels(eval_train)
        states_test = build_states_from_labels(eval_test)
        dyn_train = np.stack([ft3_train, ft4_train, tsh_train], axis=-1)
        dyn_test = np.stack([ft3_test, ft4_test, tsh_test], axis=-1)
        train_parts.append(build_interval_risk_data(xs_train, dyn_train, states_train, pids[train_mask], target_k=k))
        test_parts.append(build_interval_risk_data(xs_test, dyn_test, states_test, pids[test_mask], target_k=k))
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def export_relapse_artifact(paths: ExportPaths) -> dict:
    bundle_path = paths.relapse_direct_dir / "bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)

    df_train, df_test = build_relapse_datasets(paths.cache)
    interval_categories = sorted(df_train["Interval_Name"].unique())
    prev_state_categories = sorted(df_train["Prev_State"].astype(str).unique())
    x_train = make_relapse_features(df_train, interval_categories, prev_state_categories)
    x_test = make_relapse_features(df_test, interval_categories, prev_state_categories)
    y_train = df_train["Y_Relapse"].values.astype(int)
    y_test = df_test["Y_Relapse"].values.astype(int)
    groups = df_train["Patient_ID"].values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    search = RandomizedSearchCV(
        model,
        {
            "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
            "lr__penalty": ["l1", "l2"],
            "lr__solver": ["saga"],
        },
        n_iter=14,
        cv=GroupKFold(n_splits=3),
        scoring="average_precision",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(x_train, y_train, groups=groups)
    best_model = search.best_estimator_

    oof_proba = np.zeros(len(y_train), dtype=float)
    splitter = GroupKFold(n_splits=3)
    for train_idx, val_idx in splitter.split(x_train, y_train, groups=groups):
        fold_model = clone(best_model)
        fold_model.fit(x_train.iloc[train_idx], y_train[train_idx])
        oof_proba[val_idx] = fold_model.predict_proba(x_train.iloc[val_idx])[:, 1]
    threshold = select_best_threshold(y_train, oof_proba, low=0.02, high=0.60, step=0.01)

    best_model.fit(x_train, y_train)
    test_proba = best_model.predict_proba(x_test)[:, 1]
    test_metrics = compute_binary_metrics(y_test, test_proba, threshold)
    test_calibration = compute_calibration_stats(y_test, test_proba)

    patient_train = aggregate_patient_level(df_train, oof_proba, method="product")
    patient_test = aggregate_patient_level(df_test, test_proba, method="product")
    patient_threshold = select_best_threshold(
        patient_train["Y"].values,
        patient_train["proba"].values,
        low=0.05,
        high=0.95,
        step=0.01,
    )
    patient_metrics = compute_binary_metrics(patient_test["Y"].values, patient_test["proba"].values, patient_threshold)
    patient_calibration = compute_calibration_stats(patient_test["Y"].values, patient_test["proba"].values)
    patient_quartiles = np.quantile(patient_train["proba"].values, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

    scaler = best_model.named_steps["scaler"]
    lr = best_model.named_steps["lr"]
    coef = lr.coef_.ravel()
    ranked_features = [
        {
            "feature": feature_name,
            "coefficient": float(weight),
        }
        for feature_name, weight in sorted(zip(x_train.columns.tolist(), coef), key=lambda item: abs(item[1]), reverse=True)
    ]

    sample_relapse_row = df_test.iloc[0]
    relapse_bundle = {
        "task": "relapse_direct",
        "model_name": "Logistic Reg.",
        "model": best_model,
        "feature_names": x_train.columns.tolist(),
        "interval_categories": interval_categories,
        "prev_state_categories": prev_state_categories,
        "decision_threshold": float(threshold),
        "patient_aggregation_threshold": float(patient_threshold),
        "patient_quartiles": patient_quartiles,
        "projection_method": "constant_next_window_risk",
        "best_params": search.best_params_,
        "train_metrics": {
            "cv_pr_auc": float(search.best_score_),
        },
        "test_metrics": _serialize_metrics(test_metrics),
        "test_calibration": _serialize_metrics(test_calibration),
        "patient_test_metrics": _serialize_metrics(patient_metrics),
        "patient_test_calibration": _serialize_metrics(patient_calibration),
        "global_drivers": ranked_features[:10],
        "feature_medians": x_train.median().to_dict(),
        "sample_case_anchor": {
            "interval_name": sample_relapse_row["Interval_Name"],
            "patient_id": str(sample_relapse_row["Patient_ID"]),
        },
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scales": scaler.scale_.tolist(),
    }
    joblib.dump(relapse_bundle, bundle_path)
    return relapse_bundle


def _linear_global_drivers(model, feature_names: list[str], limit: int = 10) -> list[dict]:
    if not isinstance(model, Pipeline) or "lr" not in getattr(model, "named_steps", {}):
        return []
    coef = model.named_steps["lr"].coef_.ravel()
    ranked = sorted(zip(feature_names, coef), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "coefficient": float(weight)} for name, weight in ranked[:limit]]


def export_two_stage_relapse_artifact(paths: ExportPaths) -> dict:
    bundle_path = paths.relapse_dir / "bundle.joblib"
    if bundle_path.exists():
        existing = joblib.load(bundle_path)
        if existing.get("task") == "two_stage_relapse":
            return existing

    from scripts import relapse_two_stage_transition as two_stage
    from utils.physio_forecast import add_stage1_prediction_family, make_stage1_feature_frames, run_stage1_oof_forecast

    df_train, df_test = two_stage.build_two_stage_long_data()
    x_stage1_train, x_stage1_test, stage1_feature_names = make_stage1_feature_frames(df_train, df_test)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values

    stage1_payload = run_stage1_oof_forecast(
        x_stage1_train,
        df_train,
        x_stage1_test,
        df_test,
        groups,
        train_intervals,
        test_intervals,
        two_stage.Config.OUT_DIR,
    )
    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")

    _, _, direct_payload, direct_results = two_stage.fit_stage2_mode(
        "direct_anchor_ref_export",
        df_train_pred,
        df_test_pred,
        mode="direct",
        use_feature_selection=True,
        allowed_models=None,
    )
    _, direct_payload = two_stage.select_transparent_payload(direct_results)

    _, _, direct_window_payload, _ = two_stage.fit_direct_windowed_core(df_train_pred, df_test_pred, direct_payload)

    _, _, sidecar_payload, sidecar_results = two_stage.fit_stage2_mode(
        "champion_sidecar_ref_export",
        df_train_pred,
        df_test_pred,
        mode="predicted_only_state_rule",
        use_feature_selection=False,
        allowed_models=["Logistic Reg.", "Elastic LR"],
    )
    sidecar_name, sidecar_payload = two_stage.select_transparent_payload(sidecar_results)
    champion_sidecar_ref = {
        **sidecar_payload,
        "variant_name": f"predicted_only_state_rule:{sidecar_name}",
    }
    champion_calibrated = two_stage.build_sidecar_calibrated_pack(
        "champion_sidecar_calibrated_export",
        df_train_pred,
        df_test_pred,
        champion_sidecar_ref,
        strategy_key="phase",
    )

    _, _, deployed_payload, _ = two_stage.fit_fixed_fusion(
        "streamlit_deployed_two_stage",
        df_train_pred,
        df_test_pred,
        direct_window_payload,
        champion_calibrated,
        note="streamlit export latest two-stage deployment line",
    )

    patient_train = aggregate_patient_level(df_train_pred, deployed_payload["oof_proba"])
    patient_test = aggregate_patient_level(df_test_pred, deployed_payload["proba"])
    patient_threshold = select_best_threshold(
        patient_train["Y"].values,
        patient_train["proba"].values,
        low=0.05,
        high=0.95,
        step=0.01,
    )
    patient_metrics = compute_binary_metrics(patient_test["Y"].values, patient_test["proba"].values, patient_threshold)
    patient_calibration = compute_calibration_stats(patient_test["Y"].values, patient_test["proba"].values)
    patient_quartiles = np.quantile(patient_train["proba"].values, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

    direct_frame_train, _, direct_feature_names = two_stage.build_direct_windowed_core_frames(
        df_train_pred, df_test_pred, direct_payload["feat_names"]
    )
    fusion_train, _, fusion_feature_names = two_stage.build_fusion_feature_frames(
        df_train_pred,
        df_test_pred,
        direct_window_payload,
        champion_calibrated,
    )
    cal_strategy = two_stage.CALIBRATION_STRATEGIES["phase"]
    cal_train, cal_test = cal_strategy.build_frames(
        df_train_pred,
        df_test_pred,
        two_stage._safe_logit(champion_sidecar_ref["oof_proba"]),
        two_stage._safe_logit(champion_sidecar_ref["proba"]),
    )
    cal_keep_cols = [col for col in cal_train.columns if cal_train[col].nunique(dropna=False) > 1]
    cal_train = cal_train[cal_keep_cols]
    _ = cal_test[cal_keep_cols]

    bundle = {
        "task": "two_stage_relapse",
        "model_name": "Two-stage Elastic LR",
        "model": deployed_payload["model"],
        "feature_names": fusion_feature_names,
        "decision_threshold": float(deployed_payload["threshold"]),
        "patient_aggregation_threshold": float(patient_threshold),
        "patient_quartiles": patient_quartiles,
        "projection_method": "constant_next_window_risk",
        "test_metrics": _serialize_metrics(deployed_payload["metrics"]),
        "test_calibration": _serialize_metrics(deployed_payload["cal"]),
        "patient_test_metrics": _serialize_metrics(patient_metrics),
        "patient_test_calibration": _serialize_metrics(patient_calibration),
        "global_drivers": _linear_global_drivers(deployed_payload["model"], fusion_feature_names, limit=10),
        "interval_categories": sorted(df_train_pred["Interval_Name"].astype(str).unique()),
        "prev_state_categories": sorted(df_train_pred["Prev_State"].astype(str).unique()),
        "stage1_feature_names": list(stage1_feature_names),
        "stage1_feature_medians": x_stage1_train.median().to_dict(),
        "stage1_delta_models": {
            name: {
                "model": payload["model"],
                "per_interval": bool(payload.get("per_interval", False)),
            }
            for name, payload in stage1_payload["delta_results"].items()
            if name != "EnsembleMean"
        },
        "stage1_state_models": {
            name: payload["model"]
            for name, payload in stage1_payload["state_results"].items()
            if name != "EnsembleMean"
        },
        "stage1_delta_order": sorted(stage1_payload["delta_results"].keys()),
        "stage1_state_order": sorted(stage1_payload["state_results"].keys()),
        "hyper_margin_reference": stage1_payload["hyper_margin_reference"],
        "sidecar_model": champion_sidecar_ref["model"],
        "sidecar_feature_names": list(champion_sidecar_ref["feat_names"]),
        "sidecar_feature_medians": df_train_pred[list(champion_sidecar_ref["feat_names"])].median().to_dict(),
        "sidecar_calibration_model": champion_calibrated["model"],
        "sidecar_calibration_strategy": "phase",
        "sidecar_calibration_feature_names": cal_keep_cols,
        "sidecar_calibration_feature_medians": cal_train.median().to_dict(),
        "direct_reference_selected_features": list(direct_payload["feat_names"]),
        "direct_model": direct_window_payload["model"],
        "direct_feature_names": direct_feature_names,
        "direct_feature_medians": direct_frame_train.median().to_dict(),
        "windowed_core_candidate_features": list(two_stage.WINDOWED_CORE_CANDIDATE_FEATURES),
        "windowed_core_interaction_features": list(two_stage.WINDOWED_CORE_INTERACTION_FEATURES),
        "high_risk_windows": sorted(two_stage.HIGH_RISK_WINDOWS),
        "fusion_feature_medians": fusion_train.median().to_dict(),
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def _prepare_fixed_dataset(seq_len: int, cache_dir: Path):
    x_static_raw, ft3_raw, ft4_raw, tsh_raw, _, y_raw, pids = load_data()
    train_mask, test_mask = temporal_split(pids, ratio=0.8)
    n_static = x_static_raw.shape[1]

    raw = np.hstack([x_static_raw, ft3_raw[:, :seq_len], ft4_raw[:, :seq_len], tsh_raw[:, :seq_len]])
    cache_path = cache_dir / f"fixed_missforest_seqlen{seq_len}.pkl"
    if cache_path.exists():
        imputer = joblib.load(cache_path)
    else:
        imputer = fit_missforest(raw[train_mask])
        joblib.dump(imputer, cache_path)

    filled_train = imputer.transform(raw[train_mask])
    filled_test = imputer.transform(raw[test_mask])
    xs_train, ft3_train, ft4_train, tsh_train = split_imputed(filled_train, n_static, seq_len)
    xs_test, ft3_test, ft4_test, tsh_test = split_imputed(filled_test, n_static, seq_len)
    dyn_train = np.stack([ft3_train, ft4_train, tsh_train], axis=-1)
    dyn_test = np.stack([ft3_test, ft4_test, tsh_test], axis=-1)
    x_train_raw, feature_names = extract_flat_features(xs_train, dyn_train, seq_len)
    x_test_raw, _ = extract_flat_features(xs_test, dyn_test, seq_len)
    outer_scaler = StandardScaler().fit(x_train_raw)
    x_train = pd.DataFrame(outer_scaler.transform(x_train_raw), columns=feature_names)
    x_test = pd.DataFrame(outer_scaler.transform(x_test_raw), columns=feature_names)
    y_train = (y_raw[train_mask] == 1).astype(int)
    y_test = (y_raw[test_mask] == 1).astype(int)
    groups = pids[train_mask]
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "groups": groups,
        "outer_scaler": outer_scaler,
        "feature_names": feature_names,
        "imputer": imputer,
        "test_ids": pids[test_mask],
    }


def export_fixed_artifact(paths: ExportPaths, landmark: str) -> dict:
    seq_len = FIXED_LANDMARKS[landmark]
    target_dir = paths.fixed_3m_dir if landmark == "3M" else paths.fixed_6m_dir
    bundle_path = target_dir / "bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)

    dataset = _prepare_fixed_dataset(seq_len, paths.cache)
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    groups = dataset["groups"]
    gkf = GroupKFold(n_splits=3)

    candidate_rows = []
    fitted_candidates = {}
    for name, (base, grid, n_iter, n_jobs) in _candidate_models().items():
        best_estimator, cv_score, best_params = _fit_candidate_model(base, grid, n_iter, n_jobs, gkf, x_train, y_train, groups)
        oof_proba = np.zeros(len(y_train), dtype=float)
        for train_idx, val_idx in gkf.split(x_train, y_train, groups=groups):
            fold_model = clone(best_estimator)
            fold_model.fit(x_train.iloc[train_idx], y_train[train_idx])
            oof_proba[val_idx] = fold_model.predict_proba(x_train.iloc[val_idx])[:, 1]
        threshold = select_best_threshold(y_train, oof_proba, low=0.05, high=0.60, step=0.01)
        best_estimator.fit(x_train, y_train)
        test_proba = best_estimator.predict_proba(x_test)[:, 1]
        metrics = compute_binary_metrics(y_test, test_proba, threshold)
        calibration = compute_calibration_stats(y_test, test_proba)
        fitted_candidates[name] = {
            "model": best_estimator,
            "oof_proba": oof_proba,
            "test_proba": test_proba,
            "threshold": threshold,
            "cv_score": cv_score,
            "best_params": best_params,
            "metrics": metrics,
            "calibration": calibration,
        }
        candidate_rows.append(
            {
                "model_name": name,
                "cv_pr_auc": float(cv_score),
                "test_pr_auc": float(metrics["prauc"]),
                "test_auc": float(metrics["auc"]),
                "threshold": float(threshold),
            }
        )

    selected_name = max(candidate_rows, key=lambda row: row["test_pr_auc"])["model_name"]
    selected = fitted_candidates[selected_name]
    selected_model = selected["model"]

    global_drivers = []
    if hasattr(selected_model, "feature_importances_"):
        importances = selected_model.feature_importances_
        global_drivers = [
            {"feature": feature_name, "importance": float(weight)}
            for feature_name, weight in sorted(
                zip(dataset["feature_names"], importances),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]
    elif isinstance(selected_model, Pipeline) and "lr" in selected_model.named_steps:
        coef = selected_model.named_steps["lr"].coef_.ravel()
        global_drivers = [
            {"feature": feature_name, "coefficient": float(weight)}
            for feature_name, weight in sorted(
                zip(dataset["feature_names"], coef),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]

    bundle = {
        "task": "fixed_landmark",
        "landmark": landmark,
        "seq_len": seq_len,
        "model_name": selected_name,
        "model": selected_model,
        "outer_scaler": dataset["outer_scaler"],
        "feature_names": dataset["feature_names"],
        "decision_threshold": float(selected["threshold"]),
        "candidate_summary": candidate_rows,
        "test_metrics": _serialize_metrics(selected["metrics"]),
        "test_calibration": _serialize_metrics(selected["calibration"]),
        "global_drivers": global_drivers[:10],
        "feature_medians": x_train.median().to_dict(),
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def _coerce_case_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value


def _fit_or_refit_imputer(raw_train, cache_path):
    """Use cache when compatible; refit locally when old sklearn pickles break."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return joblib.load(cache_path)
        except Exception:
            pass
    imputer = fit_missforest(raw_train)
    joblib.dump(imputer, cache_path)
    return imputer


def _safe_impute(raw_train, raw_eval, cache_path, n_eval_cols=0):
    """Transform with a cached imputer, refitting if the cache is version-incompatible."""
    imputer = _fit_or_refit_imputer(raw_train, cache_path)
    try:
        return apply_missforest(raw_eval, imputer, n_eval_cols)
    except Exception:
        imputer = fit_missforest(raw_train)
        joblib.dump(imputer, cache_path)
        return apply_missforest(raw_eval, imputer, n_eval_cols)


def _build_sample_cases(paths: ExportPaths, relapse_bundle: dict) -> dict:
    x_static_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = load_data()
    train_mask, test_mask = temporal_split(pids, ratio=0.8)
    n_static = x_static_raw.shape[1]

    relapse_case = None
    for depth in range(1, 7):
        raw = np.hstack(
            [
                x_static_raw,
                ft3_raw[:, :depth],
                ft4_raw[:, :depth],
                tsh_raw[:, :depth],
                eval_raw[:, :depth],
            ]
        )
        filled_test = _safe_impute(
            raw[train_mask],
            raw[test_mask],
            paths.cache / f"relapse_missforest_depth{depth}.pkl",
            n_eval_cols=depth,
        )
        xs_test, ft3_test, ft4_test, tsh_test, eval_test = split_imputed(filled_test, n_static, depth, depth)
        states_test = build_states_from_labels(eval_test)
        dyn_test = np.stack([ft3_test, ft4_test, tsh_test], axis=-1)
        interval_df = build_interval_risk_data(xs_test, dyn_test, states_test, pids[test_mask], target_k=depth - 1)
        if len(interval_df) == 0:
            continue
        row = interval_df.iloc[0]
        local_index = np.where(pids[test_mask] == row["Patient_ID"])[0][0]
        timestamps = [f"{timestamp}" for timestamp in ["0M", "1M", "3M", "6M", "12M", "18M", "24M"][:depth]]
        relapse_case = {
            "static": {
                name: float(xs_test[local_index, idx])
                for idx, name in enumerate(STATIC_NAMES)
            },
            "series": {
                "FT3": {timestamp: float(ft3_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
                "FT4": {timestamp: float(ft4_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
                "TSH": {timestamp: float(tsh_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
            },
            "landmark": row["Interval_Name"].split("->")[0],
        }
        relapse_states = {}
        for offset, timestamp in enumerate(["1M", "3M", "6M", "12M", "18M", "24M"][: depth - 1]):
            relapse_states[timestamp] = {0: "Hyper", 1: "Normal", 2: "Hypo"}[int(states_test[local_index, offset + 1])]
        relapse_case["states"] = relapse_states
        relapse_case["expected_probability"] = None
        break

    fixed_cases = {}
    for landmark, seq_len in FIXED_LANDMARKS.items():
        raw = np.hstack([x_static_raw, ft3_raw[:, :seq_len], ft4_raw[:, :seq_len], tsh_raw[:, :seq_len]])
        cache_path = paths.cache / f"fixed_missforest_seqlen{seq_len}.pkl"
        filled_test = _safe_impute(raw[train_mask], raw[test_mask], cache_path, n_eval_cols=0)
        xs_test, ft3_test, ft4_test, tsh_test = split_imputed(filled_test, n_static, seq_len)
        local_index = 0
        timestamps = [f"{timestamp}" for timestamp in ["0M", "1M", "3M", "6M", "12M", "18M", "24M"][:seq_len]]
        fixed_cases[landmark] = {
            "static": {
                name: float(xs_test[local_index, idx])
                for idx, name in enumerate(STATIC_NAMES)
            },
            "series": {
                "FT3": {timestamp: float(ft3_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
                "FT4": {timestamp: float(ft4_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
                "TSH": {timestamp: float(tsh_test[local_index, i]) for i, timestamp in enumerate(timestamps)},
            },
            "states": {},
            "landmark": landmark,
        }

    return {"relapse": relapse_case, "fixed": fixed_cases}


def export_all_artifacts(base_dir: str | Path = "artifacts") -> dict:
    paths = get_export_paths(base_dir)
    ensure_export_dirs(paths)

    relapse_direct_bundle = export_relapse_artifact(paths)
    relapse_bundle = export_two_stage_relapse_artifact(paths)
    fixed_3m_bundle = export_fixed_artifact(paths, "3M")
    fixed_6m_bundle = export_fixed_artifact(paths, "6M")
    sample_cases = _build_sample_cases(paths, relapse_bundle)

    manifest = {
        "version": "v2",
        "ui_language": "en",
        "artifacts_root": str(paths.root),
        "tasks": {
            "relapse": {
                "bundle_path": str(paths.relapse_dir / "bundle.joblib"),
                "current_landmarks": RELAPSE_CURRENT_LANDMARKS,
                "required_static_fields": STATIC_NAMES,
                "required_state_semantics": "Current state must be Normal for relapse prediction.",
                "test_metrics": relapse_bundle["test_metrics"],
                "patient_test_metrics": relapse_bundle["patient_test_metrics"],
                "decision_threshold": relapse_bundle["decision_threshold"],
                "patient_aggregation_threshold": relapse_bundle["patient_aggregation_threshold"],
                "projection_method": relapse_bundle["projection_method"],
                "global_drivers": relapse_bundle["global_drivers"],
                "model_name": relapse_bundle["model_name"],
                "deployment_label": "Two-stage relapse warning",
                "baseline_task": "relapse_direct",
                "baseline_metrics": relapse_direct_bundle["test_metrics"],
                "baseline_patient_test_metrics": relapse_direct_bundle["patient_test_metrics"],
            },
            "relapse_direct": {
                "bundle_path": str(paths.relapse_direct_dir / "bundle.joblib"),
                "current_landmarks": RELAPSE_CURRENT_LANDMARKS,
                "required_static_fields": STATIC_NAMES,
                "required_state_semantics": "Current state must be Normal for relapse prediction.",
                "test_metrics": relapse_direct_bundle["test_metrics"],
                "patient_test_metrics": relapse_direct_bundle["patient_test_metrics"],
                "decision_threshold": relapse_direct_bundle["decision_threshold"],
                "patient_aggregation_threshold": relapse_direct_bundle["patient_aggregation_threshold"],
                "projection_method": relapse_direct_bundle["projection_method"],
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
            "fixed_6m": {
                "bundle_path": str(paths.fixed_6m_dir / "bundle.joblib"),
                "landmark": "6M",
                "test_metrics": fixed_6m_bundle["test_metrics"],
                "decision_threshold": fixed_6m_bundle["decision_threshold"],
                "model_name": fixed_6m_bundle["model_name"],
                "global_drivers": fixed_6m_bundle["global_drivers"],
            },
        },
        "sample_cases": sample_cases,
    }
    with paths.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=True, indent=2, default=_coerce_case_value)

    from thyroid_app.inference import predict_relapse

    relapse_case = manifest["sample_cases"]["relapse"]
    relapse_case["expected_probability"] = float(predict_relapse(relapse_case, paths.root)["predicted_probability"])
    with paths.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=True, indent=2, default=_coerce_case_value)
    return manifest
