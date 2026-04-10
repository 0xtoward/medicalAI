# Two-Stage Transition Round

## Frozen Baselines

- Direct anchor reference: `PR-AUC 0.331`, `AUC 0.842`.
- Current champion fused reference: `PR-AUC 0.349`, `AUC 0.850`.
- Champion calibration protocol in this round: `shared slope + early/late intercepts`.
- Direct window core interaction kernel in this round: `Time_In_Normal, FT3_Current, Delta_TSH_k0`.
- Previous best-new motif benchmark: `binary_interval_intercept_calibrated` with `PR-AUC 0.345`, `AUC 0.850`.

## Best Overall

- Best experiment: `direct_windowed_core_plus_champion_calibrated` with `Elastic LR`.
- Interval-level metrics: `PR-AUC 0.351`, `AUC 0.849`, `Brier 0.063`.
- Versus current champion: `PR-AUC +0.003`, `AUC -0.001`.
- Versus direct anchor: `PR-AUC +0.021`, `AUC +0.007`.
- Grouped bootstrap vs direct: `ΔPR-AUC +0.021` with 95% CI `+0.005` to `+0.040`.
- Grouped bootstrap vs current champion: `ΔPR-AUC +0.003` with 95% CI `-0.003` to `+0.009`.

## Per-Window Where It Improved

- `3M->6M`: PR-AUC change vs current champion `+0.007`.
- `6M->12M`: PR-AUC change vs current champion `+0.005`.
- `18M->24M`: PR-AUC change vs current champion `+0.003`.
- `1M->3M`: PR-AUC change vs current champion `+0.002`.
- `12M->18M`: PR-AUC change vs current champion `+0.000`.

## Gain Attribution

- Main gain source: **anchor + sidecar** together.

## Stop Rule

- Experiments passing the 6-of-3 stop rule: `4`.
- Strongest stop-rule row: `direct_windowed_core` with `Pass_Count=4` and `ΔPR-AUC=-0.003`.

## Stage1 Scope Counts

- `Test`: current_normal `514`, current_nonhyper `656`, next_hyper_from_normal `41`, next_hyper_from_nonhyper `58`.
- `Train`: current_normal `2096`, current_nonhyper `2603`, next_hyper_from_normal `169`, next_hyper_from_nonhyper `207`.
