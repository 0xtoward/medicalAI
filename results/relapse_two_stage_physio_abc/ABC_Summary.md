# ABC No-Stacking Summary

## 最强分支
`BranchB / fusion_basic / Logistic Reg.`，AUC=0.840，PR-AUC=0.334，Brier=0.066。

## 是否超过 direct logistic
有。最强的是 `BranchB / fusion_basic / Logistic Reg.`，interval-level AUC=0.840，PR-AUC=0.334，超过 baseline direct logistic 的 AUC=0.842，PR-AUC=0.331。

## 最值得写论文的分支
`BranchB / fusion_basic / Logistic Reg.`。它已经在 no-stacking 条件下超过 direct logistic，而且仍然保持了非常小的融合结构，最适合直接写成主要结果。

## 可能的剩余瓶颈
当前最可能的瓶颈仍然是：当前窗口的 FT4 / TSH 与既往短期轨迹已经包含了大部分即刻 relapse 信号，而 stage-1 预测出来的是一个带噪声的中间代理量。换句话说，forecast head 还没有把“未来生理变化”预测得足够稳定，导致 sidecar / fusion 更多是在补充而不是替代强直接信号。
