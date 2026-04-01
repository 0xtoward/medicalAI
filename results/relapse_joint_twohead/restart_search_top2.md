# Restart Search Top-2 Variants

| Variant | Restart | OOF PR-AUC | Test PR-AUC | Test AUC | Test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| c2_anchor_slope_ranked_tight | 0 | 0.206 | 0.314 | 0.846 | 0.073 |
| c2_anchor_slope_ranked_tight | 1 | 0.199 | 0.301 | 0.838 | 0.075 |
| c2_anchor_slope_ranked_tight | 2 | 0.207 | 0.325 | 0.848 | 0.072 |
| c2_anchor_slope_ranked_tight | 3 | 0.205 | 0.314 | 0.843 | 0.072 |
| c2_interval_slope_ranked | 0 | 0.204 | 0.312 | 0.840 | 0.094 |
| c2_interval_slope_ranked | 1 | 0.233 | 0.305 | 0.848 | 0.095 |
| c2_interval_slope_ranked | 2 | 0.174 | 0.334 | 0.846 | 0.095 |
| c2_interval_slope_ranked | 3 | 0.240 | 0.296 | 0.843 | 0.095 |

- Best-by-OOF restart for `c2_anchor_slope_ranked_tight` kept the good calibration/Brier profile, but did not exceed the best point-estimate frontier from the earlier focused run.
- Best-by-OOF restart for `c2_interval_slope_ranked` was not the best test-time restart, so the current model is still restart-sensitive.
- Practical implication: the next improvement should target training stability or loss shaping, not more blind restart search.
