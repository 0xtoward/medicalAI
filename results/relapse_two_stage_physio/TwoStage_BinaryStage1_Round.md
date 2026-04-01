# Two-Stage Binary Stage-1 Round

## What Changed

- Replaced the previous 3-class next-state head with a binary `Hyper vs non-Hyper` head.
- Tightened stage-1 meta-features around hyper risk, confidence, delta, and margin-to-hyper summaries.
- Stopped broad fusion-template search; only kept a tiny transparent fusion comparison set.

## Stage-1 Snapshot

- Delta-head test average `R^2`: `0.072`.
- Binary hyper-head test `AUC`: `0.826`.
- Binary hyper-head test `PR-AUC`: `0.323`.

## Best Result

- Best branch: `direct_plus_physio_fusion` with `Elastic LR`.
- Metrics: `PR-AUC 0.350`, `AUC 0.851`, `Brier 0.063`.
- Versus direct: `PR-AUC +0.020`, `AUC +0.009`.
- Transparent experiments run in this round: `6`.

## Best Transparent Candidates

| Rank | Physio source | Template | Model | PR-AUC | AUC | Brier | Cal Int | Cal Slope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | predicted_only_state_rule:Logistic Reg. | both_windowed | Elastic LR | 0.350 | 0.851 | 0.063 | 0.745 | 1.468 |
| 2 | predicted_only_state_only:Elastic LR | both_windowed | Elastic LR | 0.346 | 0.848 | 0.063 | 0.993 | 1.600 |
| 3 | predicted_only_state_rule:Logistic Reg. | base | Logistic Reg. | 0.332 | 0.843 | 0.068 | 4.938 | 3.293 |
| 4 | predicted_only_state_delta:Elastic LR | both_windowed | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.590 | 3.107 |
| 5 | predicted_only_state_only:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.592 | 3.107 |
| 6 | predicted_only_state_delta:Elastic LR | base | Logistic Reg. | 0.331 | 0.842 | 0.068 | 4.592 | 3.107 |

## Insight

- Switching to binary stage-1 makes the intermediate task more aligned with the final relapse event.
- The useful signal still enters stage 2 best as a narrow physio score, not as a wide feature table.
- If this binary round still does not improve enough, the next highest-ROI move is interval-specific binary hyper heads rather than more fusion variants.
