# Two-Stage Self-Optimization Log

## Objective

- Continue tightening the transparent two-stage line until the handoff is competitive without widening the feature table.

## Changes In This Round

- Shrunk the current-to-stage2 carryover to the top direct linear features only.
- Added extra physio-only source variants for fusion search: `state_only`, `state_delta`, `state_delta_uncertainty`.
- Added clinical threshold-derived physio features such as predicted hyper-zone indicators.
- Searched multiple transparent fusion templates, including full windowed and high-risk-window-gated variants.
- Kept the final fusion layer linear only.

## Stage-1 Status

- Delta-head test average `R^2`: `0.072`.
- Next-state head test `Hyper PR-AUC`: `0.323`.
- Interpretation: stage 1 is still modest, so gains must come from a better handoff rather than raw stage-1 strength.

## Best Current Result

- Best branch in this run: `direct_plus_physio_fusion` with `Elastic LR`.
- Interval-level result: `PR-AUC 0.350`, `AUC 0.851`, `Brier 0.063`.
- Versus direct branch: `PR-AUC +0.020`, `AUC +0.009`.
- Versus relapse.py Logistic reference: `PR-AUC +0.016`, `AUC +0.010`.
- Transparent candidate experiments searched this round: `6`.

## Fusion Search Top Candidates

| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Cal Int | Cal Slope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | predicted_only_state_rule:Logistic Reg. | both_windowed | Elastic LR | 0.350 | 0.851 | 0.063 | 0.745 | 1.468 |
| 2 | predicted_only_state_only:Elastic LR | both_windowed | Elastic LR | 0.346 | 0.848 | 0.063 | 0.993 | 1.600 |
| 3 | predicted_only_state_rule:Logistic Reg. | base | Logistic Reg. | 0.332 | 0.843 | 0.068 | 4.938 | 3.293 |
| 4 | predicted_only_state_delta:Elastic LR | both_windowed | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.590 | 3.107 |
| 5 | predicted_only_state_only:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.592 | 3.107 |
| 6 | predicted_only_state_delta:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.592 | 3.107 |

## Current Insight

- Transparent score fusion is still the only two-stage strategy that clearly adds value.
- Compact physio features alone are not strong enough to replace the direct branch.
- Naive current + physio concatenation remains unstable even after narrowing.
- The practical takeaway is that stage-1 information should act as a calibrated modifier of direct risk, not as a wide second feature table.
