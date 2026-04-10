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

import numpy as np
import pandas as pd

from scripts import relapse_two_stage_transition as base
from scripts import relapse_two_stage_transition_intermediate as exp1
from utils.physio_forecast import add_stage1_prediction_family, make_stage1_feature_frames


OUT_DIR = Path("./results/relapse_two_stage_transition_intermediate_tuned/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
base.Config.OUT_DIR = OUT_DIR
exp1.OUT_DIR = OUT_DIR

np.random.seed(base.SEED)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    description: str


VARIANTS = [
    VariantSpec("g2_pca_manifold", "compressed low-dimensional manifold of continuous stage-1 forecasts"),
    VariantSpec("g3_cluster_soft_state", "soft archetype memberships from predicted future physiology"),
    VariantSpec("g4_residual_pca", "residual manifold after removing current-window context"),
    VariantSpec("g5_residual_cluster_hybrid", "residual manifold plus residual archetypes"),
]
CALIBRATION_KEYS = ["global", "phase", "interval"]
SCOPE_KEYS = ["normal", "nonhyper"]


def _align_array(reference_df, scope_df, values):
    keys = ["Patient_ID", "Source_Row", "Interval_ID"]
    arr = np.asarray(values)
    if arr.ndim == 1:
        tmp = scope_df[keys].copy()
        tmp["pred"] = arr
        merged = reference_df[keys].merge(tmp, on=keys, how="left")
        return merged["pred"].values.astype(float)

    cols = [f"pred_{idx}" for idx in range(arr.shape[1])]
    tmp = scope_df[keys].copy()
    for idx, col in enumerate(cols):
        tmp[col] = arr[:, idx]
    merged = reference_df[keys].merge(tmp, on=keys, how="left")
    return merged[cols].values.astype(float)


def align_scope_delta_payload(reference_train, reference_test, scope_train, scope_test, stage1_payload_scope, base_payload):
    aligned_delta_results = {}
    for name, pack in stage1_payload_scope["delta_results"].items():
        aligned_delta_results[name] = {
            **pack,
            "oof_delta": _align_array(reference_train, scope_train, pack["oof_delta"]),
            "test_delta": _align_array(reference_test, scope_test, pack["test_delta"]),
        }
    return {
        **base_payload,
        "delta_results": aligned_delta_results,
        "delta_best_name": stage1_payload_scope["delta_best_name"],
    }


def build_latent_frame_family(train_df, test_df, stage1_payload, scope_key):
    scope_dir = OUT_DIR / scope_key
    scope_dir.mkdir(parents=True, exist_ok=True)
    prev_out_dir = exp1.OUT_DIR
    exp1.OUT_DIR = scope_dir
    try:
        g1_tr, g1_te, g1_cols = exp1.build_delta_latent_frames(train_df, test_df, stage1_payload)
        g2_tr, g2_te, g2_cols = exp1.build_pca_manifold(g1_tr, g1_te)
        g3_tr, g3_te, g3_cols = exp1.build_cluster_soft_state(g1_tr, g1_te)
        g4_tr, g4_te, g4_cols = exp1.build_residual_pca(train_df, test_df, g1_tr, g1_te)
        g5_tr, g5_te, g5_cols = exp1.build_residual_cluster_hybrid(train_df, test_df, g1_tr, g1_te)
    finally:
        exp1.OUT_DIR = prev_out_dir

    frame_map = {
        "g1_delta_latent": (g1_tr, g1_te, g1_cols),
        "g2_pca_manifold": (g2_tr, g2_te, g2_cols),
        "g3_cluster_soft_state": (g3_tr, g3_te, g3_cols),
        "g4_residual_pca": (g4_tr, g4_te, g4_cols),
        "g5_residual_cluster_hybrid": (g5_tr, g5_te, g5_cols),
    }
    for key, (x_tr, x_te, _) in frame_map.items():
        x_tr.to_csv(OUT_DIR / f"{scope_key}_{key}_Train.csv", index=False)
        x_te.to_csv(OUT_DIR / f"{scope_key}_{key}_Test.csv", index=False)
    return frame_map


def score_payload(payload, train_df, test_df):
    return exp1.attach_eval_fields(payload, train_df, test_df)


def fit_sidecar_candidates(group_prefix, x_tr, x_te, train_df, test_df, note):
    candidate_rows = []
    candidate_payloads = {}

    for use_fs in [False, True]:
        fs_tag = "fs" if use_fs else "nofs"
        group_name = f"{group_prefix}_{fs_tag}"
        group_df, best_name, best_payload, results = exp1.fit_custom_classifier(
            group_name,
            x_tr,
            x_te,
            train_df,
            test_df,
            note=f"{note}; feature_selection={use_fs}",
            use_feature_selection=use_fs,
        )
        best_name, best_payload = base.select_transparent_payload(results)
        key = f"{group_name}_raw"
        candidate_payloads[key] = best_payload
        candidate_rows.append(
            {
                "Candidate": key,
                "Base_Type": "raw",
                "Feature_Selection": use_fs,
                "Calibration": "none",
                "Best_Model": best_name,
                "PR_AUC": best_payload["metrics"]["prauc"],
                "AUC": best_payload["metrics"]["auc"],
                "Brier": best_payload["metrics"]["brier"],
                "Threshold": best_payload["threshold"],
                "Note": note,
            }
        )

        for strategy_key in CALIBRATION_KEYS:
            cal_payload = base.build_sidecar_calibrated_pack(
                f"{group_name}_{strategy_key}",
                train_df,
                test_df,
                best_payload,
                strategy_key=strategy_key,
            )
            cal_payload = score_payload(cal_payload, train_df, test_df)
            cal_key = f"{group_name}_cal_{strategy_key}"
            candidate_payloads[cal_key] = cal_payload
            candidate_rows.append(
                {
                    "Candidate": cal_key,
                    "Base_Type": "calibrated",
                    "Feature_Selection": use_fs,
                    "Calibration": strategy_key,
                    "Best_Model": "Calibrated",
                    "PR_AUC": cal_payload["metrics"]["prauc"],
                    "AUC": cal_payload["metrics"]["auc"],
                    "Brier": cal_payload["metrics"]["brier"],
                    "Threshold": cal_payload["threshold"],
                    "Note": note,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    best_key = str(candidate_df.iloc[0]["Candidate"])
    return candidate_df, best_key, candidate_payloads[best_key], candidate_payloads


def append_summary(rows, group, family, scope, source_variant, best_name, payload, note):
    rows.append(
        {
            "Group": group,
            "Family": family,
            "Scope": scope,
            "Source_Variant": source_variant,
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


def compare_against_previous(summary_df):
    prev_intermediate = pd.read_csv("./results/relapse_two_stage_transition_intermediate/Intermediate_Best_Group_Summary.csv")
    prev_transition = pd.read_csv("./results/relapse_two_stage_transition/TwoStageTransition_Best_Group_Summary.csv")
    prev_best_intermediate = prev_intermediate.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    prev_best_transition = prev_transition.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    best_now = summary_df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    return pd.DataFrame(
        [
            {
                "Reference": "previous_intermediate_best",
                "Group": prev_best_intermediate["Group"],
                "PR_AUC": prev_best_intermediate["PR_AUC"],
                "AUC": prev_best_intermediate["AUC"],
                "Delta_PR_AUC_vs_Current": best_now["PR_AUC"] - prev_best_intermediate["PR_AUC"],
                "Delta_AUC_vs_Current": best_now["AUC"] - prev_best_intermediate["AUC"],
            },
            {
                "Reference": "transition_best",
                "Group": prev_best_transition["Group"],
                "PR_AUC": prev_best_transition["PR_AUC"],
                "AUC": prev_best_transition["AUC"],
                "Delta_PR_AUC_vs_Current": best_now["PR_AUC"] - prev_best_transition["PR_AUC"],
                "Delta_AUC_vs_Current": best_now["AUC"] - prev_best_transition["AUC"],
            },
        ]
    )


def write_markdown(summary_df, sidecar_best_df, fusion_df, comparison_df, optimization_audit_df):
    best_now = summary_df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    lines = [
        "# Tuned Intermediate Transition Sweep",
        "",
        "## Optimization Audit",
        "",
    ]
    for row in optimization_audit_df.itertuples(index=False):
        lines.append(f"- `{row.Optimization}`: used = `{row.Used}`; detail = {row.Detail}")
    lines.extend(
        [
            "",
            "## Best Overall",
            "",
            f"- `{best_now['Group']}` with `{best_now['Best_Model']}`.",
            f"- PR-AUC `{best_now['PR_AUC']:.3f}`, AUC `{best_now['AUC']:.3f}`, Brier `{best_now['Brier']:.3f}`.",
            "",
            "## Best Sidecars",
            "",
        ]
    )
    for row in sidecar_best_df.head(10).itertuples(index=False):
        lines.append(
            f"- `{row.Group}`: `{row.Best_Model}` with `PR-AUC {row.PR_AUC:.3f}`, `AUC {row.AUC:.3f}`, `Brier {row.Brier:.3f}`. {row.Note}"
        )
    lines.extend(["", "## Best Fusions", ""])
    for row in fusion_df.head(12).itertuples(index=False):
        lines.append(
            f"- `{row.Group}`: `{row.Best_Model}` with `PR-AUC {row.PR_AUC:.3f}`, `AUC {row.AUC:.3f}`, `Brier {row.Brier:.3f}`. {row.Note}"
        )
    lines.extend(["", "## Comparison", ""])
    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"- vs `{row.Reference}` / `{row.Group}`: `ΔPR-AUC {row.Delta_PR_AUC_vs_Current:+.3f}`, `ΔAUC {row.Delta_AUC_vs_Current:+.3f}`."
        )
    (OUT_DIR / "Tuned_Intermediate_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local caches in the tuned experiment directory")
    args = parser.parse_args()
    if args.clear_cache:
        base.clear_pkl_cache(OUT_DIR)

    print("=" * 92)
    print("  Tuned Two-Stage Transition Intermediate Sweep")
    print("=" * 92)

    df_train, df_test = base.build_two_stage_long_data()

    print("\n--- Normal-scope stage-1 payload ---")
    x1_tr, x1_te, _ = make_stage1_feature_frames(df_train, df_test)
    groups = df_train["Patient_ID"].values
    train_intervals = df_train["Interval_Name"].values
    test_intervals = df_test["Interval_Name"].values
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
    normal_frames = build_latent_frame_family(df_train, df_test, stage1_payload, "normal")

    print("\n--- Nonhyper-scope stage-1 payload ---")
    scope_train, scope_test, scope_count_df = base.build_stage1_transition_scope_data(current_scope="nonhyper")
    x_scope_tr, x_scope_te, _ = base.make_stage1_scope_feature_frames(scope_train, scope_test)
    scope_groups = scope_train["Patient_ID"].values
    scope_train_intervals = scope_train["Interval_Name"].values
    scope_test_intervals = scope_test["Interval_Name"].values
    stage1_scope_payload = base.run_stage1_oof_forecast(
        x_scope_tr,
        scope_train,
        x_scope_te,
        scope_test,
        scope_groups,
        scope_train_intervals,
        scope_test_intervals,
        OUT_DIR,
    )
    stage1_scope_aligned = align_scope_delta_payload(df_train, df_test, scope_train, scope_test, stage1_scope_payload, stage1_payload)
    nonhyper_frames = build_latent_frame_family(df_train, df_test, stage1_scope_aligned, "nonhyper")
    scope_count_df.to_csv(OUT_DIR / "Stage1_Scope_Counts.csv", index=False)

    print("\n--- Direct anchors and champion auxiliary sidecar ---")
    direct_df, direct_name, direct_payload, direct_results = base.fit_stage2_mode(
        "tuned_direct_anchor",
        df_train,
        df_test,
        mode="direct",
        use_feature_selection=True,
        allowed_models=None,
    )
    direct_name, direct_payload = base.select_transparent_payload(direct_results)
    direct_payload["c_index"] = base.compute_recurrent_c_index_from_intervals(df_test, direct_payload["proba"])

    _, window_name, window_payload, _ = base.fit_direct_windowed_core(df_train, df_test, direct_payload)
    _, mono_name, mono_payload, _ = base.fit_monotonic_tree_direct(df_train, df_test, direct_payload)
    direct_anchors = {
        "direct_ref": direct_payload,
        "direct_windowed": window_payload,
        "direct_monotonic": mono_payload,
    }

    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")
    _, champion_name, champion_payload, champion_results = base.fit_stage2_mode(
        "tuned_champion_sidecar_ref",
        df_train_pred,
        df_test_pred,
        mode="predicted_only_state_rule",
        use_feature_selection=False,
        allowed_models=["Logistic Reg.", "Elastic LR"],
    )
    champion_name, champion_payload = base.select_transparent_payload(champion_results)
    champion_candidates = []
    champion_candidate_payloads = {"champion_raw": champion_payload}
    champion_candidates.append(
        {
            "Candidate": "champion_raw",
            "PR_AUC": champion_payload["metrics"]["prauc"],
            "AUC": champion_payload["metrics"]["auc"],
            "Brier": champion_payload["metrics"]["brier"],
            "Calibration": "none",
        }
    )
    for strategy_key in CALIBRATION_KEYS:
        payload = base.build_sidecar_calibrated_pack(
            f"tuned_champion_sidecar_{strategy_key}",
            df_train,
            df_test,
            champion_payload,
            strategy_key=strategy_key,
        )
        payload = score_payload(payload, df_train, df_test)
        champion_candidate_payloads[f"champion_cal_{strategy_key}"] = payload
        champion_candidates.append(
            {
                "Candidate": f"champion_cal_{strategy_key}",
                "PR_AUC": payload["metrics"]["prauc"],
                "AUC": payload["metrics"]["auc"],
                "Brier": payload["metrics"]["brier"],
                "Calibration": strategy_key,
            }
        )
    champion_candidate_df = pd.DataFrame(champion_candidates).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    champion_best_key = str(champion_candidate_df.iloc[0]["Candidate"])
    champion_best_payload = champion_candidate_payloads[champion_best_key]
    champion_candidate_df.to_csv(OUT_DIR / "Champion_Calibration_Sweep.csv", index=False)

    print("\n--- Sidecar tuning sweep ---")
    sidecar_rows = []
    sidecar_candidate_tables = []
    best_sidecars = {}
    frame_map_by_scope = {"normal": normal_frames, "nonhyper": nonhyper_frames}
    for scope_key in SCOPE_KEYS:
        for variant in VARIANTS:
            x_tr, x_te, _ = frame_map_by_scope[scope_key][variant.key]
            candidate_df, best_key, best_payload, _ = fit_sidecar_candidates(
                f"{scope_key}_{variant.key}",
                x_tr,
                x_te,
                df_train,
                df_test,
                note=f"{variant.description}; scope={scope_key}",
            )
            candidate_df.insert(0, "Scope", scope_key)
            candidate_df.insert(1, "Variant", variant.key)
            sidecar_candidate_tables.append(candidate_df)
            candidate_df.to_csv(OUT_DIR / f"{scope_key}_{variant.key}_CandidateSweep.csv", index=False)
            best_sidecars[(scope_key, variant.key)] = best_payload
            append_summary(
                sidecar_rows,
                f"{scope_key}_{variant.key}_best_sidecar",
                "sidecar",
                scope_key,
                variant.key,
                "Calibrated" if "cal" in best_key else "Linear",
                best_payload,
                f"best sidecar after fs/calibration sweep; candidate={best_key}",
            )

    sidecar_best_df = pd.DataFrame(sidecar_rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    pd.concat(sidecar_candidate_tables, ignore_index=True).to_csv(OUT_DIR / "All_Sidecar_Candidates.csv", index=False)

    print("\n--- Fusion tuning sweep ---")
    fusion_rows = []
    top_sidecar_keys = sidecar_best_df.head(6)[["Scope", "Source_Variant"]].itertuples(index=False, name=None)
    top_sidecar_keys = list(dict.fromkeys(top_sidecar_keys))
    for scope_key, variant_key in top_sidecar_keys:
        sidecar_payload = best_sidecars[(scope_key, variant_key)]
        for direct_key, direct_pack in direct_anchors.items():
            _, best_name, best_payload, _ = base.fit_fixed_fusion(
                f"{scope_key}_{variant_key}_{direct_key}",
                df_train,
                df_test,
                direct_pack,
                sidecar_payload,
                note=f"scope={scope_key}; variant={variant_key}; direct={direct_key}",
            )
            append_summary(
                fusion_rows,
                f"{scope_key}_{variant_key}_{direct_key}_fusion",
                "fusion",
                scope_key,
                variant_key,
                best_name,
                best_payload,
                f"direct={direct_key}; latent_sidecar={variant_key}; scope={scope_key}",
            )

        _, best_name, best_payload, _ = base.fit_fixed_fusion(
            f"{scope_key}_{variant_key}_windowed_mainLatent_auxChampion",
            df_train,
            df_test,
            direct_anchors["direct_windowed"],
            sidecar_payload,
            aux_rule_payload=champion_best_payload,
            note=f"scope={scope_key}; variant={variant_key}; direct=windowed; aux_sidecar=best_champion",
        )
        append_summary(
            fusion_rows,
            f"{scope_key}_{variant_key}_windowed_mainLatent_auxChampion",
            "fusion",
            scope_key,
            variant_key,
            best_name,
            best_payload,
            "windowed direct + latent main sidecar + champion auxiliary sidecar",
        )

        _, best_name, best_payload, _ = base.fit_fixed_fusion(
            f"{scope_key}_{variant_key}_windowed_mainChampion_auxLatent",
            df_train,
            df_test,
            direct_anchors["direct_windowed"],
            champion_best_payload,
            aux_rule_payload=sidecar_payload,
            note=f"scope={scope_key}; variant={variant_key}; direct=windowed; main=champion; aux=latent",
        )
        append_summary(
            fusion_rows,
            f"{scope_key}_{variant_key}_windowed_mainChampion_auxLatent",
            "fusion",
            scope_key,
            variant_key,
            best_name,
            best_payload,
            "windowed direct + champion main sidecar + latent auxiliary sidecar",
        )

    fusion_df = pd.DataFrame(fusion_rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)

    baseline_rows = []
    append_summary(
        baseline_rows,
        "baseline_direct_ref",
        "direct",
        "normal",
        "direct",
        direct_name,
        direct_payload,
        "transparent direct anchor",
    )
    append_summary(
        baseline_rows,
        "baseline_direct_windowed",
        "direct",
        "normal",
        "direct_windowed",
        window_name,
        window_payload,
        "windowed direct anchor",
    )
    append_summary(
        baseline_rows,
        "baseline_direct_monotonic",
        "direct",
        "normal",
        "direct_monotonic",
        mono_name,
        mono_payload,
        "monotonic direct anchor",
    )
    append_summary(
        baseline_rows,
        "baseline_champion_best",
        "sidecar",
        "normal",
        "champion",
        "Calibrated" if "cal" in champion_best_key else champion_name,
        champion_best_payload,
        f"best champion after calibration sweep; candidate={champion_best_key}",
    )

    summary_df = pd.concat([pd.DataFrame(baseline_rows), sidecar_best_df, fusion_df], ignore_index=True)
    summary_df = summary_df.sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "Tuned_Intermediate_Best_Group_Summary.csv", index=False)
    sidecar_best_df.to_csv(OUT_DIR / "Tuned_Intermediate_Best_Sidecars.csv", index=False)
    fusion_df.to_csv(OUT_DIR / "Tuned_Intermediate_Fusion_Summary.csv", index=False)

    comparison_df = compare_against_previous(summary_df)
    comparison_df.to_csv(OUT_DIR / "Tuned_Intermediate_Comparison.csv", index=False)

    optimization_audit_df = pd.DataFrame(
        [
            {"Optimization": "direct_anchor_sweep", "Used": True, "Detail": "direct_ref, direct_windowed, direct_monotonic"},
            {"Optimization": "sidecar_feature_selection", "Used": True, "Detail": "nofs and L1-selected sidecar branches"},
            {"Optimization": "calibration_sweep", "Used": True, "Detail": "global, phase, interval intercept strategies"},
            {"Optimization": "stage1_scope_sweep", "Used": True, "Detail": "normal and nonhyper stage-1 training scopes"},
            {"Optimization": "auxiliary_sidecar_fusion", "Used": True, "Detail": "main-latent+aux-champion and main-champion+aux-latent"},
        ]
    )
    optimization_audit_df.to_csv(OUT_DIR / "Optimization_Audit.csv", index=False)

    exp1.save_result_barplot(summary_df[["Group", "AUC", "PR_AUC"]].assign(Family="all"))
    write_markdown(summary_df, sidecar_best_df, fusion_df, comparison_df, optimization_audit_df)

    print("\n  Top tuned results")
    for row in summary_df.head(15).itertuples(index=False):
        print(
            f"    {row.Group:<52s} {row.Best_Model:<14s} "
            f"PR-AUC={row.PR_AUC:.3f} AUC={row.AUC:.3f} Brier={row.Brier:.3f}"
        )
    print(f"\n  Outputs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
