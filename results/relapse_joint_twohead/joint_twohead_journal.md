# Joint Two-Head Journal

## Setup
- Data source: current pooled landmark long-format tables under `results/relapse_two_stage_physio/`.
- Direct anchor selected features: `36`.
- Current static best two-stage reference: `current_binary_stage1_reference` with PR-AUC `0.350`.
- Allowed joint families: `C1 rule auxiliary`, `C2 interval correction`.

## Round 1
- Winner: `c2_interval_slope` (c2).
- Metrics: PR-AUC `0.322`, AUC `0.843`, Brier `0.108`.
- Stop-rule hits: `2`.
- Decision: refine the winning family locally in Round 2 instead of reopening wide architecture search.

## Round 2
- Winner: `c2_interval_slope_ranked` (c2).
- Metrics: PR-AUC `0.330`, AUC `0.851`, Brier `0.095`.
- Stop-rule hits: `2`.
- Interpretation: keep only the refined family winner as the final joint two-head candidate for this turn.

## Round 3
- Winner: `c2_anchor_slope_ranked_tight` (c2).
- Metrics: PR-AUC `0.328`, AUC `0.847`, Brier `0.070`.
- Stop-rule hits: `2`.
- Interpretation: this round anchors the joint scorer to the direct landmark risk and only learns small residual/correction heads.

