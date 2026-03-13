import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

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
    compute_binary_metrics,
    compute_calibration_stats,
    save_calibration_figure,
    select_best_threshold,
)
from utils.model_viz import save_logistic_regression_visuals
from utils.physio_forecast import (
    TARGET_COLS,
    add_predicted_physio_columns,
    build_next_physio_targets,
    evaluate_physio_predictions,
    get_stage2_models,
    make_stage1_feature_frames,
    make_stage2_feature_frames,
    run_stage1_oof_forecast,
    save_physio_scatter,
    save_stage1_metric_bar,
)
from utils.recurrence import build_interval_risk_data

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "DejaVu Sans"
np.random.seed(SEED)


class Config:
    OUT_DIR = Path("./results/repluse_two_stage_physio/")
    SHARED_REPLUSE_DIR = Path("./results/repluse/")
    LEGACY_REPLUSE_DIR = Path("./multistate_result/")


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

        # Current feature / outcome imputation: same as repluse.py
        raw_repluse = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        repluse_cache = Config.SHARED_REPLUSE_DIR / f"missforest_depth{depth}.pkl"
        if not repluse_cache.exists():
            repluse_cache = Config.LEGACY_REPLUSE_DIR / f"missforest_depth{depth}.pkl"
        local_repluse_cache = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer_cur = load_or_fit_with_message(
            raw_repluse[tr_mask],
            local_repluse_cache,
            fallback_cache_path=repluse_cache,
            label=f"current-depth-{depth} ({interval_name})",
        )
        filled_tr = apply_missforest(raw_repluse[tr_mask], imputer_cur, depth)
        filled_te = apply_missforest(raw_repluse[te_mask], imputer_cur, depth)
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

        # Next physiology labels: impute labs up to t+1 without eval labels.
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

        df_tr = df_cur_tr.merge(df_next_tr, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_te = df_cur_te.merge(df_next_te, on=["Patient_ID", "Source_Row", "Interval_ID"], how="left")
        df_tr_parts.append(df_tr)
        df_te_parts.append(df_te)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr)}  test {len(df_te)} rows")

    df_train = pd.concat(df_tr_parts, ignore_index=True)
    df_test = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_train)}  test {len(df_test)} rows")
    df_train.to_csv(Config.OUT_DIR / "two_stage_train.csv", index=False)
    df_test.to_csv(Config.OUT_DIR / "two_stage_test.csv", index=False)
    return df_train, df_test


def fit_stage2_group(mode, train_df, test_df):
    x_tr, x_te, feat_names = make_stage2_feature_frames(train_df, test_df, mode=mode)
    y_tr = train_df["Y_Relapse"].values.astype(int)
    y_te = test_df["Y_Relapse"].values.astype(int)
    groups = train_df["Patient_ID"].values
    gkf = GroupKFold(n_splits=3)

    rows = []
    best_name = None
    best_payload = None
    best_pr = -np.inf

    for model_name, base_model in get_stage2_models().items():
        oof = np.zeros(len(y_tr), dtype=float)
        for tr_idx, val_idx in gkf.split(x_tr, y_tr, groups=groups):
            m = clone(base_model)
            m.fit(x_tr.iloc[tr_idx], y_tr[tr_idx])
            oof[val_idx] = m.predict_proba(x_tr.iloc[val_idx])[:, 1]
        thr = select_best_threshold(y_tr, oof, low=0.02, high=0.60, step=0.01)
        model = clone(base_model)
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_te)[:, 1]
        metrics = compute_binary_metrics(y_te, proba, thr)
        cal = compute_calibration_stats(y_te, proba)
        rows.append(
            {
                "Group": mode,
                "Model": model_name,
                "AUC": metrics["auc"],
                "PR_AUC": metrics["prauc"],
                "Brier": metrics["brier"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": thr,
            }
        )
        if metrics["prauc"] > best_pr:
            best_pr = metrics["prauc"]
            best_name = model_name
            best_payload = {
                "model": model,
                "proba": proba,
                "feat_names": feat_names,
                "metrics": metrics,
                "cal": cal,
            }
    return pd.DataFrame(rows), best_name, best_payload


def save_stage2_comparison(two_stage_df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    best_df = two_stage_df.sort_values(["Group", "PR_AUC"], ascending=[True, False]).groupby("Group", as_index=False).first()
    axes[0].bar(best_df["Group"], best_df["AUC"], color="steelblue")
    axes[0].set_title("Best AUC by group")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(best_df["Group"], best_df["PR_AUC"], color="coral")
    axes[1].set_title("Best PR-AUC by group")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "TwoStage_Group_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    y1_tr = df_train[TARGET_COLS].values.astype(float)
    y1_te = df_test[TARGET_COLS].values.astype(float)
    groups = df_train["Patient_ID"].values

    best_stage1_name, stage1_results, stage1_metrics = run_stage1_oof_forecast(
        x1_tr, y1_tr, x1_te, groups, Config.OUT_DIR
    )
    best_stage1 = stage1_results[best_stage1_name]
    stage1_test_metrics = evaluate_physio_predictions(y1_te, best_stage1["test_pred"], "Test", best_stage1_name)
    stage1_all_metrics = pd.concat([stage1_metrics, stage1_test_metrics], ignore_index=True)
    stage1_all_metrics.to_csv(Config.OUT_DIR / "Physio_Forecast_Metrics.csv", index=False)
    save_stage1_metric_bar(stage1_metrics[stage1_metrics["Split"] == "Train_OOF"], Config.OUT_DIR)
    save_physio_scatter(y1_te, best_stage1["test_pred"], Config.OUT_DIR, best_stage1_name, "Test")
    print(f"  Best stage-1 model by average OOF RMSE: {best_stage1_name}")
    print(stage1_all_metrics[stage1_all_metrics["Target"] == "Average"][["Split", "Model", "MAE", "RMSE", "R2"]].to_string(index=False))

    df_train_pred = add_predicted_physio_columns(df_train, best_stage1["oof_pred"])
    df_test_pred = add_predicted_physio_columns(df_test, best_stage1["test_pred"])

    print("\n--- Phase 3: Stage-2 relapse prediction ---")
    all_rows = []
    best_groups = {}
    for mode in ["direct", "predicted_only", "predicted_plus_current"]:
        group_df, best_name, payload = fit_stage2_group(mode, df_train_pred, df_test_pred)
        all_rows.append(group_df)
        best_groups[mode] = (best_name, payload)
        print(f"  Best {mode:<22s}: {best_name}  AUC={payload['metrics']['auc']:.3f}  PR-AUC={payload['metrics']['prauc']:.3f}  Brier={payload['metrics']['brier']:.3f}")

    two_stage_df = pd.concat(all_rows, ignore_index=True)
    two_stage_df.to_csv(Config.OUT_DIR / "TwoStage_Model_Comparison.csv", index=False)
    save_stage2_comparison(two_stage_df)

    best_mode = max(best_groups, key=lambda g: best_groups[g][1]["metrics"]["prauc"])
    best_name, best_payload = best_groups[best_mode]
    save_calibration_figure(
        df_test_pred["Y_Relapse"].values,
        best_payload["proba"],
        f"Calibration Curve ({best_mode}, {best_name})",
        Config.OUT_DIR / "Calibration_BestStage2.png",
    )
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

    print("\n  Best stage-2 group summary")
    for mode in ["direct", "predicted_only", "predicted_plus_current"]:
        model_name, payload = best_groups[mode]
        cal = payload["cal"]
        m = payload["metrics"]
        print(
            f"    {mode:<22s} {model_name:<14s} "
            f"AUC={m['auc']:.3f}  PR-AUC={m['prauc']:.3f}  "
            f"Brier={m['brier']:.3f}  Cal.Int={cal['intercept']:.3f}  Cal.Slope={cal['slope']:.3f}"
        )

    print(f"\n  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
