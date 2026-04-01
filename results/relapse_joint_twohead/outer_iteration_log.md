# Joint Two-Head Outer Iteration Log

## Iteration 1

- Best variant: `c2_interval_slope`.
- Metrics: `PR-AUC 0.322`, `AUC 0.843`, `Brier 0.108`.
- Interpretation: anchor+residual fixed the Brier blow-up, but discrimination still trailed the direct anchor.

## Iteration 2

- Best variant: `c2_interval_slope_ranked`.
- Metrics: `PR-AUC 0.330`, `AUC 0.851`, `Brier 0.095`.
- Delta vs Iteration 1: `PR-AUC +0.008`, `AUC +0.008`, `Brier -0.013`.
- Interpretation: adding light ranking pressure helped the shared transition scorer recover AUC and nearly match direct PR-AUC, but calibration remained the bottleneck.

## Positioning

- Direct anchor: `PR-AUC 0.331`, `AUC 0.842`, `Brier 0.064`.
- Current strongest two-stage reference: `current_binary_stage1_reference / Elastic LR` with `PR-AUC 0.350`, `AUC 0.851`.
- Practical conclusion: the best joint two-head route is now competitive on AUC but still behind on PR-AUC and clearly behind on calibration/Brier.
