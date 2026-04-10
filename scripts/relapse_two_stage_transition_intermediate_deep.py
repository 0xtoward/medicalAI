import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd

from scripts import relapse_two_stage_transition as base
from scripts import relapse_two_stage_transition_intermediate as exp1
from scripts import relapse_two_stage_transition_intermediate_tuned as tuned
from utils.physio_forecast import add_stage1_prediction_family, make_stage1_feature_frames


OUT_DIR = Path("./results/relapse_two_stage_transition_intermediate_deep/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
base.Config.OUT_DIR = OUT_DIR
exp1.OUT_DIR = OUT_DIR
tuned.OUT_DIR = OUT_DIR

TUNED_DIR = Path("./results/relapse_two_stage_transition_intermediate_tuned/")
np.random.seed(base.SEED)


LATENT_COMPONENTS = {
    "g2_nh": ("nonhyper", "g2_pca_manifold"),
    "g3_n": ("normal", "g3_cluster_soft_state"),
    "g4_nh": ("nonhyper", "g4_residual_pca"),
    "g4_n": ("normal", "g4_residual_pca"),
}

COMMITTEES = {
    "committee_g23": ["g2_nh", "g3_n"],
    "committee_g34": ["g3_n", "g4_nh"],
    "committee_g234": ["g2_nh", "g3_n", "g4_nh"],
    "committee_g2344": ["g2_nh", "g3_n", "g4_n", "g4_nh"],
}


def _load_latent_frame(scope_key, variant_key, split):
    path = TUNED_DIR / f"{scope_key}_{variant_key}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_logit(proba):
    return base._safe_logit(np.asarray(proba, dtype=float))


def fit_payload_sweep(group_prefix, x_tr, x_te, train_df, test_df, note):
    candidate_df, best_key, best_payload, all_payloads = tuned.fit_sidecar_candidates(
        group_prefix,
        x_tr,
        x_te,
        train_df,
        test_df,
        note=note,
    )
    candidate_df.to_csv(OUT_DIR / f"{group_prefix}_CandidateSweep.csv", index=False)
    return candidate_df, best_key, best_payload, all_payloads


def build_committee_frame(train_df, test_df, payload_map, component_keys, prefix):
    tr = pd.DataFrame(index=np.arange(len(train_df)))
    te = pd.DataFrame(index=np.arange(len(test_df)))

    for key in component_keys:
        payload = payload_map[key]
        tr[f"{key}_Logit"] = _safe_logit(payload["oof_proba"])
        te[f"{key}_Logit"] = _safe_logit(payload["proba"])
        tr[f"{key}_Prob"] = np.asarray(payload["oof_proba"], dtype=float)
        te[f"{key}_Prob"] = np.asarray(payload["proba"], dtype=float)

    logit_cols = [c for c in tr.columns if c.endswith("_Logit")]
    prob_cols = [c for c in tr.columns if c.endswith("_Prob")]
    tr[f"{prefix}_MeanLogit"] = tr[logit_cols].mean(axis=1)
    te[f"{prefix}_MeanLogit"] = te[logit_cols].mean(axis=1)
    tr[f"{prefix}_StdLogit"] = tr[logit_cols].std(axis=1)
    te[f"{prefix}_StdLogit"] = te[logit_cols].std(axis=1)
    tr[f"{prefix}_MeanProb"] = tr[prob_cols].mean(axis=1)
    te[f"{prefix}_MeanProb"] = te[prob_cols].mean(axis=1)
    tr[f"{prefix}_MaxProb"] = tr[prob_cols].max(axis=1)
    te[f"{prefix}_MaxProb"] = te[prob_cols].max(axis=1)
    tr[f"{prefix}_MinProb"] = tr[prob_cols].min(axis=1)
    te[f"{prefix}_MinProb"] = te[prob_cols].min(axis=1)
    tr[f"{prefix}_SpreadProb"] = tr[f"{prefix}_MaxProb"] - tr[f"{prefix}_MinProb"]
    te[f"{prefix}_SpreadProb"] = te[f"{prefix}_MaxProb"] - te[f"{prefix}_MinProb"]

    for a, b in itertools.combinations(component_keys, 2):
        tr[f"{a}_x_{b}"] = tr[f"{a}_Logit"].values * tr[f"{b}_Logit"].values
        te[f"{a}_x_{b}"] = te[f"{a}_Logit"].values * te[f"{b}_Logit"].values
        tr[f"{a}_minus_{b}"] = tr[f"{a}_Logit"].values - tr[f"{b}_Logit"].values
        te[f"{a}_minus_{b}"] = te[f"{a}_Logit"].values - te[f"{b}_Logit"].values

    for interval_name in sorted(train_df["Interval_Name"].astype(str).unique()):
        dummy = f"Window_{interval_name}"
        tr[dummy] = (train_df["Interval_Name"].astype(str).values == interval_name).astype(float)
        te[dummy] = (test_df["Interval_Name"].astype(str).values == interval_name).astype(float)
        for key in component_keys:
            inter = f"{dummy}_x_{key}_Logit"
            tr[inter] = tr[dummy].values * tr[f"{key}_Logit"].values
            te[inter] = te[dummy].values * te[f"{key}_Logit"].values

    return exp1._fill_and_keep(tr, te)


def build_multilogit_fusion_frame(train_df, test_df, payloads, prefix):
    tr = pd.DataFrame(index=np.arange(len(train_df)))
    te = pd.DataFrame(index=np.arange(len(test_df)))

    for name, payload in payloads.items():
        tr[f"{name}_Logit"] = _safe_logit(payload["oof_proba"])
        te[f"{name}_Logit"] = _safe_logit(payload["proba"])
        tr[f"{name}_Prob"] = np.asarray(payload["oof_proba"], dtype=float)
        te[f"{name}_Prob"] = np.asarray(payload["proba"], dtype=float)

    base_names = list(payloads.keys())
    logit_cols = [f"{name}_Logit" for name in base_names]
    prob_cols = [f"{name}_Prob" for name in base_names]

    tr[f"{prefix}_MeanLogit"] = tr[logit_cols].mean(axis=1)
    te[f"{prefix}_MeanLogit"] = te[logit_cols].mean(axis=1)
    tr[f"{prefix}_StdLogit"] = tr[logit_cols].std(axis=1)
    te[f"{prefix}_StdLogit"] = te[logit_cols].std(axis=1)
    tr[f"{prefix}_MeanProb"] = tr[prob_cols].mean(axis=1)
    te[f"{prefix}_MeanProb"] = te[prob_cols].mean(axis=1)
    tr[f"{prefix}_MaxProb"] = tr[prob_cols].max(axis=1)
    te[f"{prefix}_MaxProb"] = te[prob_cols].max(axis=1)

    for a, b in itertools.combinations(base_names, 2):
        tr[f"{a}_x_{b}"] = tr[f"{a}_Logit"].values * tr[f"{b}_Logit"].values
        te[f"{a}_x_{b}"] = te[f"{a}_Logit"].values * te[f"{b}_Logit"].values
        tr[f"{a}_minus_{b}"] = tr[f"{a}_Logit"].values - tr[f"{b}_Logit"].values
        te[f"{a}_minus_{b}"] = te[f"{a}_Logit"].values - te[f"{b}_Logit"].values

    for interval_name in sorted(train_df["Interval_Name"].astype(str).unique()):
        dummy = f"Window_{interval_name}"
        tr[dummy] = (train_df["Interval_Name"].astype(str).values == interval_name).astype(float)
        te[dummy] = (test_df["Interval_Name"].astype(str).values == interval_name).astype(float)
        for name in base_names:
            inter = f"{dummy}_x_{name}_Logit"
            tr[inter] = tr[dummy].values * tr[f"{name}_Logit"].values
            te[inter] = te[dummy].values * te[f"{name}_Logit"].values

    tr[f"{prefix}_Champion_x_LatentMean"] = tr["Champion_Logit"].values * tr[f"{prefix}_MeanLogit"].values
    te[f"{prefix}_Champion_x_LatentMean"] = te["Champion_Logit"].values * te[f"{prefix}_MeanLogit"].values
    tr[f"{prefix}_Direct_x_LatentMean"] = tr["Direct_Logit"].values * tr[f"{prefix}_MeanLogit"].values
    te[f"{prefix}_Direct_x_LatentMean"] = te["Direct_Logit"].values * te[f"{prefix}_MeanLogit"].values
    return exp1._fill_and_keep(tr, te)


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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear local caches in the deep experiment directory")
    args = parser.parse_args()
    if args.clear_cache:
        base.clear_pkl_cache(OUT_DIR)

    print("=" * 92)
    print("  Deep Intermediate Transition Sweep")
    print("=" * 92)

    df_train, df_test = base.build_two_stage_long_data()

    print("\n--- Rebuild normal champion + direct anchors ---")
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
    df_train_pred = add_stage1_prediction_family(df_train, stage1_payload, "oof")
    df_test_pred = add_stage1_prediction_family(df_test, stage1_payload, "test")

    _, direct_name, direct_payload, direct_results = base.fit_stage2_mode(
        "deep_direct_anchor",
        df_train,
        df_test,
        mode="direct",
        use_feature_selection=True,
        allowed_models=None,
    )
    direct_name, direct_payload = base.select_transparent_payload(direct_results)
    _, window_name, window_payload, _ = base.fit_direct_windowed_core(df_train, df_test, direct_payload)

    _, champion_name, champion_payload, champion_results = base.fit_stage2_mode(
        "deep_champion_sidecar_ref",
        df_train_pred,
        df_test_pred,
        mode="predicted_only_state_rule",
        use_feature_selection=False,
        allowed_models=["Logistic Reg.", "Elastic LR"],
    )
    champion_name, champion_payload = base.select_transparent_payload(champion_results)
    champion_candidates = []
    champion_payloads = {"raw": champion_payload}
    champion_candidates.append(
        {
            "Candidate": "raw",
            "PR_AUC": champion_payload["metrics"]["prauc"],
            "AUC": champion_payload["metrics"]["auc"],
            "Brier": champion_payload["metrics"]["brier"],
        }
    )
    for strategy_key in ["global", "phase", "interval"]:
        payload = base.build_sidecar_calibrated_pack(
            f"deep_champion_{strategy_key}",
            df_train,
            df_test,
            champion_payload,
            strategy_key=strategy_key,
        )
        payload = exp1.attach_eval_fields(payload, df_train, df_test)
        champion_payloads[strategy_key] = payload
        champion_candidates.append(
            {
                "Candidate": strategy_key,
                "PR_AUC": payload["metrics"]["prauc"],
                "AUC": payload["metrics"]["auc"],
                "Brier": payload["metrics"]["brier"],
            }
        )
    champion_df = pd.DataFrame(champion_candidates).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    champion_df.to_csv(OUT_DIR / "Champion_CandidateSweep.csv", index=False)
    champion_best_key = str(champion_df.iloc[0]["Candidate"])
    champion_best_payload = champion_payloads[champion_best_key]

    print("\n--- Refit targeted latent sidecars from tuned latent feature caches ---")
    component_payloads = {}
    component_rows = []
    for short_key, (scope_key, variant_key) in LATENT_COMPONENTS.items():
        x_tr = _load_latent_frame(scope_key, variant_key, "Train")
        x_te = _load_latent_frame(scope_key, variant_key, "Test")
        candidate_df, best_key, best_payload, _ = fit_payload_sweep(
            f"deep_{short_key}",
            x_tr,
            x_te,
            df_train,
            df_test,
            note=f"targeted refit for {short_key}",
        )
        component_payloads[short_key] = best_payload
        component_rows.append(
            {
                "Component": short_key,
                "Scope": scope_key,
                "Variant": variant_key,
                "Best_Candidate": best_key,
                "PR_AUC": best_payload["metrics"]["prauc"],
                "AUC": best_payload["metrics"]["auc"],
                "Brier": best_payload["metrics"]["brier"],
            }
        )
    pd.DataFrame(component_rows).to_csv(OUT_DIR / "Targeted_Latent_Components.csv", index=False)

    print("\n--- Committee sidecars ---")
    summary_rows = []
    committee_payloads = {}
    for committee_name, members in COMMITTEES.items():
        x_tr, x_te, _ = build_committee_frame(df_train, df_test, component_payloads, members, prefix=committee_name)
        x_tr.to_csv(OUT_DIR / f"{committee_name}_Train.csv", index=False)
        x_te.to_csv(OUT_DIR / f"{committee_name}_Test.csv", index=False)
        candidate_df, best_key, best_payload, _ = fit_payload_sweep(
            f"deep_{committee_name}",
            x_tr,
            x_te,
            df_train,
            df_test,
            note=f"committee sidecar over {','.join(members)}",
        )
        committee_payloads[committee_name] = best_payload
        append_summary(summary_rows, committee_name, "committee_sidecar", "Calibrated" if "cal" in best_key else "Linear", best_payload, f"members={','.join(members)}; best={best_key}")

    print("\n--- Multi-latent fusion sweeps ---")
    fusion_specs = {
        "fusion_sep_g34": ["Champion", "Direct", "g3_n", "g4_nh"],
        "fusion_sep_g234": ["Champion", "Direct", "g2_nh", "g3_n", "g4_nh"],
        "fusion_sep_g2344": ["Champion", "Direct", "g2_nh", "g3_n", "g4_n", "g4_nh"],
        "fusion_committee_g34": ["Champion", "Direct", "committee_g34"],
        "fusion_committee_g234": ["Champion", "Direct", "committee_g234"],
        "fusion_committee_g2344": ["Champion", "Direct", "committee_g2344"],
    }

    payload_registry = {
        "Champion": champion_best_payload,
        "Direct": window_payload,
        **component_payloads,
        **committee_payloads,
    }
    for fusion_name, members in fusion_specs.items():
        payloads = {name: payload_registry[name] for name in members}
        x_tr, x_te, _ = build_multilogit_fusion_frame(df_train, df_test, payloads, prefix=fusion_name)
        x_tr.to_csv(OUT_DIR / f"{fusion_name}_Train.csv", index=False)
        x_te.to_csv(OUT_DIR / f"{fusion_name}_Test.csv", index=False)
        candidate_df, best_key, best_payload, _ = fit_payload_sweep(
            f"deep_{fusion_name}",
            x_tr,
            x_te,
            df_train,
            df_test,
            note=f"multilogit fusion over {','.join(members)}",
        )
        append_summary(summary_rows, fusion_name, "deep_fusion", "Calibrated" if "cal" in best_key else "Linear", best_payload, f"members={','.join(members)}; best={best_key}")

    prev_tuned = pd.read_csv(TUNED_DIR / "Tuned_Intermediate_Best_Group_Summary.csv").sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]
    prev_transition = pd.read_csv("./results/relapse_two_stage_transition/TwoStageTransition_Best_Group_Summary.csv").sort_values(["PR_AUC", "AUC"], ascending=[False, False]).iloc[0]

    append_summary(summary_rows, "baseline_windowed_direct", "baseline", window_name, window_payload, "windowed direct anchor baseline")
    append_summary(summary_rows, "baseline_champion_best", "baseline", champion_best_key, champion_best_payload, "best calibrated champion baseline")

    summary_df = pd.DataFrame(summary_rows).sort_values(["PR_AUC", "AUC"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "Deep_Intermediate_Best_Group_Summary.csv", index=False)

    best_now = summary_df.iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "Reference": "tuned_intermediate_best",
                "Group": prev_tuned["Group"],
                "PR_AUC": prev_tuned["PR_AUC"],
                "AUC": prev_tuned["AUC"],
                "Delta_PR_AUC_vs_Current": best_now["PR_AUC"] - prev_tuned["PR_AUC"],
                "Delta_AUC_vs_Current": best_now["AUC"] - prev_tuned["AUC"],
            },
            {
                "Reference": "transition_best",
                "Group": prev_transition["Group"],
                "PR_AUC": prev_transition["PR_AUC"],
                "AUC": prev_transition["AUC"],
                "Delta_PR_AUC_vs_Current": best_now["PR_AUC"] - prev_transition["PR_AUC"],
                "Delta_AUC_vs_Current": best_now["AUC"] - prev_transition["AUC"],
            },
        ]
    )
    comparison_df.to_csv(OUT_DIR / "Deep_Intermediate_Comparison.csv", index=False)

    lines = [
        "# Deep Intermediate Transition Sweep",
        "",
        "## Best Overall",
        "",
        f"- `{best_now['Group']}` with `{best_now['Best_Model']}`.",
        f"- PR-AUC `{best_now['PR_AUC']:.3f}`, AUC `{best_now['AUC']:.3f}`, Brier `{best_now['Brier']:.3f}`.",
        "",
        "## Top Results",
        "",
    ]
    for row in summary_df.head(15).itertuples(index=False):
        lines.append(f"- `{row.Group}`: `PR-AUC {row.PR_AUC:.3f}`, `AUC {row.AUC:.3f}`, `Brier {row.Brier:.3f}`. {row.Note}")
    lines.extend(["", "## Comparison", ""])
    for row in comparison_df.itertuples(index=False):
        lines.append(f"- vs `{row.Reference}` / `{row.Group}`: `ΔPR-AUC {row.Delta_PR_AUC_vs_Current:+.3f}`, `ΔAUC {row.Delta_AUC_vs_Current:+.3f}`.")
    (OUT_DIR / "Deep_Intermediate_Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n  Top deep results")
    for row in summary_df.head(15).itertuples(index=False):
        print(
            f"    {row.Group:<34s} {row.Best_Model:<14s} "
            f"PR-AUC={row.PR_AUC:.3f} AUC={row.AUC:.3f} Brier={row.Brier:.3f}"
        )
    print(f"\n  Outputs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
