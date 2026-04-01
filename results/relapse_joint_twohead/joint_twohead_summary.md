# Joint Two-Head Summary

## 设计

- 这轮不再做 hard two-stage handoff，而是把 pooled landmark 数据直接送进一个共享 trunk。
- 主头始终预测 `Y_Relapse / Y_next_hyper`。
- 只试两类辅助头：`binary hyper + rule aux` 和 `binary hyper + interval correction`。

## 当前最好 joint two-head

- 最终赢家：`c2_interval_slope_ranked`。
- Interval-level：`PR-AUC 0.330`，`AUC 0.851`，`Brier 0.095`。
- Calibration：`intercept -1.421`，`slope 1.262`。

## 对比

- 本轮 direct anchor：`PR-AUC 0.331`，`AUC 0.842`，`Brier 0.064`。
- 相对 direct 的 grouped bootstrap delta PR-AUC：`0.005`，95% CI `[-0.051, 0.057]`，one-sided p=`0.425`。
- 相对 direct 的 grouped bootstrap delta AUC：`0.010`，95% CI `[-0.022, 0.035]`，one-sided p=`0.227`。
- 当前已存在的最佳 two-stage 参考：`current_binary_stage1_reference / Elastic LR`，`PR-AUC 0.350`，`AUC 0.851`。
- 本 joint 路线相对该参考的静态差值：`PR-AUC -0.021`，`AUC -0.000`。

## 逐轮优化结论

- Round 1 冠军：`c2_interval_slope`，`PR-AUC 0.322`，`AUC 0.843`。
- Round 2 冠军：`c2_interval_slope_ranked`，`PR-AUC 0.330`，`AUC 0.851`。
- Round 3 冠军：`c2_anchor_slope_ranked_tight`，`PR-AUC 0.328`，`AUC 0.847`。
- 说明：赢家 family 是 `c2`，说明最有效的信息共享方式是 `interval-specific correction`。

## Window-wise

- 最强窗口：`6M->12M`，`PR-AUC 0.537`，`AUC 0.876`。
- 最弱窗口：`12M->18M`，`PR-AUC 0.027`，`AUC 0.508`。

## 判断

- 如果 joint two-head 继续落后于当前最佳 two-stage，主要原因不是 landmark 主线有问题，而是共享表示还没有把‘窗口差异 + 机制阈值’压缩成比当前透明 fusion 更强的风险修正。
- 如果 joint two-head 接近或超过 direct，但还没超过当前最佳 two-stage，那么它依然是一个更论文友好的联合学习备选，因为它避免了 hard handoff 和 error propagation。
