# Joint Two-Head Tradeoff Summary

| Candidate | AUC | PR-AUC | Brier | Cal Int | Cal Slope |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct_Anchor | 0.842 | 0.331 | 0.064 | -0.120 | 1.028 |
| c2_interval_slope_ranked | 0.851 | 0.330 | 0.095 | -1.421 | 1.262 |
| c2_anchor_slope_ranked_tight | 0.847 | 0.328 | 0.070 | -0.783 | 0.918 |
| c2_anchor_slope_mid | 0.844 | 0.325 | 0.072 | -0.832 | 0.892 |
| Current_Best_TwoStage | 0.851 | 0.350 | 0.063 | 0.745 | 1.468 |

- `c2_interval_slope_ranked` is still the best pure discrimination-oriented joint 2-head candidate.
- `c2_anchor_slope_ranked_tight` is the best anchor-conditioned compromise: much better Brier/calibration than the ranked variant, but still slightly worse PR-AUC.
- `c2_anchor_slope_mid` did not improve beyond `anchor_tight` in the fourth refinement round.
