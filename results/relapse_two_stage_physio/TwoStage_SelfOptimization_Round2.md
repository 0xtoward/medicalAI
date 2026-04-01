# Two-Stage Self-Optimization Round 2

## What Was Tried

- Added rule-like stage-1 meta-features: predicted hyper-zone indicators and a compact hyper-evidence score.
- Added high-risk-window gating templates so the physio correction can act mainly on 1M->3M, 3M->6M, and 6M->12M.
- Total transparent experiments this round: `6`.

## Stage-1 Reminder

- Delta-head test average `R^2`: `0.072`.
- Next-state head test `Hyper PR-AUC`: `0.323`.
- Interpretation: stage 1 is still only moderately informative, so any gain has to come from selective use of the physio signal.

## Best Result

- Best branch: `direct_plus_physio_fusion` with `Elastic LR`.
- Metrics: `PR-AUC 0.350`, `AUC 0.851`, `Brier 0.063`.
- Versus direct: `PR-AUC +0.020`, `AUC +0.009`.

## Top Transparent Fusion Candidates

| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Recall | Specificity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | predicted_only_state_rule:Logistic Reg. | both_windowed | Elastic LR | 0.350 | 0.851 | 0.063 | 0.683 | 0.822 |
| 2 | predicted_only_state_only:Elastic LR | both_windowed | Elastic LR | 0.346 | 0.848 | 0.063 | 0.707 | 0.825 |
| 3 | predicted_only_state_rule:Logistic Reg. | base | Logistic Reg. | 0.332 | 0.843 | 0.068 | 0.854 | 0.679 |
| 4 | predicted_only_state_delta:Elastic LR | both_windowed | Logistic Reg. | 0.331 | 0.842 | 0.068 | 0.829 | 0.698 |
| 5 | predicted_only_state_only:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 0.829 | 0.698 |
| 6 | predicted_only_state_delta:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 0.829 | 0.698 |

## Insight

- The pipeline still wins only when stage-1 information is used as a narrow correction to the direct score.
- High-risk-window gating is worth testing because relapse risk is heavily front-loaded across follow-up windows.
- Rule-like hyper-zone features are reasonable paper features, but they still need to work through a tiny fusion layer rather than a wide concatenation table.
