# Two-Stage Self-Optimization Log

## Objective

- Continue tightening the transparent two-stage line until the handoff is competitive without widening the feature table.

## Changes In This Round

- Shrunk the current-to-stage2 carryover to the top direct linear features only.
- Added extra physio-only source variants for fusion search: `state_only`, `state_delta`, `state_delta_uncertainty`.
- Searched multiple transparent fusion templates: `base`, `physio_windowed`, `both_windowed`, `gap_windowed`.
- Kept the final fusion layer linear only.

## Stage-1 Status

- Delta-head test average `R^2`: `0.072`.
- Next-state head test `Hyper PR-AUC`: `0.285`.
- Interpretation: stage 1 is still modest, so gains must come from a better handoff rather than raw stage-1 strength.

## Best Current Result

- Best branch in this run: `direct_plus_physio_fusion` with `Elastic LR`.
- Interval-level result: `PR-AUC 0.347`, `AUC 0.849`, `Brier 0.063`.
- Versus direct branch: `PR-AUC +0.017`, `AUC +0.007`.
- Versus relapse.py Logistic reference: `PR-AUC +0.014`, `AUC +0.007`.

## Fusion Search Top Candidates

| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Cal Int | Cal Slope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | predicted_only_state_only:Logistic Reg. | both_windowed | Elastic LR | 0.347 | 0.849 | 0.063 | 0.794 | 1.511 |
| 2 | predicted_only_compact:Elastic LR | both_windowed | Elastic LR | 0.346 | 0.846 | 0.063 | 0.017 | 1.188 |
| 3 | predicted_only_state_delta:Logistic Reg. | both_windowed | Logistic Reg. | 0.345 | 0.847 | 0.063 | 0.898 | 1.548 |
| 4 | predicted_only_state_only:Logistic Reg. | gap_windowed | Elastic LR | 0.336 | 0.848 | 0.063 | 0.723 | 1.474 |
| 5 | predicted_only_compact:Elastic LR | gap_windowed | Elastic LR | 0.336 | 0.844 | 0.064 | -0.048 | 1.155 |
| 6 | predicted_only_state_delta:Logistic Reg. | gap_windowed | Elastic LR | 0.333 | 0.845 | 0.063 | 0.780 | 1.494 |

## Current Insight

- Transparent score fusion is still the only two-stage strategy that clearly adds value.
- Compact physio features alone are not strong enough to replace the direct branch.
- Naive current + physio concatenation remains unstable even after narrowing.
- The practical takeaway is that stage-1 information should act as a calibrated modifier of direct risk, not as a wide second feature table.
