# Two-Stage Stage1-Only Round

## Frozen Stage2

- Final branch fixed to `direct_plus_physio_fusion`.
- Fusion template fixed to `both_windowed`.
- Final fusion learner fixed to `Elastic LR`.
- Direct branch feature set and training flow were reused without widening the stage-2 feature table.

## Stage1 Anchor

- Delta-head best model: `EnsembleMean`.
- Shared binary next-hyper anchor: `Binary LR`.
- Shared binary head best train/test selector: `Binary LR`.

## Best Variant

- Best new stage1 variant: `binary_interval_intercept_calibrated`.
- Interval-level metrics: `PR-AUC 0.345`, `AUC 0.850`, `Brier 0.063`.
- Versus direct: `PR-AUC +0.014`, `AUC +0.009`.
- Versus current binary-stage1 reference: `PR-AUC -0.005`, `AUC -0.001`.
- Grouped bootstrap vs direct PR-AUC delta: `+0.014` (95% CI `-0.005` to `+0.035`, one-sided `p=0.087`)
- Grouped bootstrap vs current reference PR-AUC delta: `-0.005` (95% CI `-0.018` to `+0.005`, one-sided `p=0.846`)

## Per-Window Insight

- `6M->12M`: PR-AUC change vs current reference `+0.010`.
- `12M->18M`: PR-AUC change vs current reference `-0.001`.
- `18M->24M`: PR-AUC change vs current reference `-0.004`.

## Batch 2 Status

- Batch 1 passed the stop rule for `3` variant(s), so Batch 2 was triggered.
- Batch 2 variants run: `binary_early_late_calibrated`.

