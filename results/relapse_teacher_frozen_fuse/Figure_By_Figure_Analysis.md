# Teacher Frozen Fuse Figure-by-Figure Analysis

Formal winner:
- Backbone: `global_floor_040`
- Fuse: `teacher_landmark_gate_raw__lr`
- Validation PR-AUC: `0.3721`
- Temporal test PR-AUC: `0.3490`
- Temporal test AUC: `0.7941`
- Temporal threshold used for discrete metrics: `0.20`

## 1. Architecture
File: `TeacherFrozenFuse_Architecture.png`

Interpretation:
- 这张图把当前 deployed pipeline 讲清楚了: `static/local/global` 三个输入块先各自编码，再经过 `landmark_head + gate_net + teacher_head + next_hyper_head` 预训练。
- 真正上线的最终风险不是神经网络 head 直接输出，而是冻结 backbone 后，用 `Teacher_Logit + Landmark_Signal + Gate_Static + Gate_Local` 过一个最简单的 LR fuse。
- 这张图也解释了为什么“fuse-level SHAP”和“end-to-end SHAP”是两层不同解释: 前者解释最终 LR，后者解释整个冻结管线。

## 2. Backbone Comparison
File: `TeacherFrozenFuse_Backbone_Comparison.png`

Interpretation:
- 左图说明验证集上 `global_floor_040` 的 best screen fuse 最高，达到大约 `0.372`，高于原 teacher head 的 `0.349` 左右。
- `side_cap_035` 也很强，接近 `0.364`，说明不是所有收益都来自“强行给 global 更多”，而是 soft gate 搜索本身有效。
- `reuse_current_hard_cap_025` 的 validation 表现并不最强，说明当前正式 winner 仍然主要由 validation 选型驱动。
- 右图显示“teacher head 本身”到“selected fuse”之间确实存在明显增益，尤其 `global_floor_040`、`side_cap_035`、`softmax_free` 这些 backbone，说明大头搜索不是空转。

## 3. Gate Comparison
File: `TeacherFrozenFuse_Gate_Comparison.png`

Interpretation:
- `global_floor_040` 的 gate 平均分配大致是 `static 0.14 / local 0.46 / global 0.40`，说明它不是“全部压给 global”，而是给 global 一个硬下限，同时让 local 仍然占最大份额。
- `softmax_free` 明显偏向 `static + local`，global 很低，说明如果完全不约束，模型会弱化全病程分支。
- `reuse_current_hard_cap_025` 和 `global_floor_050` 则更像“强 global”路线。
- 这张图的关键信息是: validation winner 并不是最极端的 global-dominant，而是“local 主导 + global 保底”的折中结构。

## 4. Top Experiment Comparison
File: `TeacherFrozenFuse_Top_Experiment_Comparison.png`

Interpretation:
- 左图按 validation PR-AUC 排名，前排实验的 validation 分数集中在 `0.364-0.372`，说明当前候选 fuse 之间差距不大。
- 右图按 temporal test PR-AUC 排名，能看出一些 pack 在 test 侧更强，但整体仍围绕同一类 teacher-driven 低维融合结构波动。
- 这张图更适合说明“结构搜索带来的是有限但真实的增益”，而不是说明模型族之间存在完全不同的机制。

## 5. Formal Winner Discrimination
File: `TeacherFrozenFuse_FormalWinner_Discrimination.png`

Interpretation:
- AUC 从 `Fit 0.781` 到 `Validation 0.869` 再到 `Temporal 0.794`，说明模型在内部验证集最好，在时间外推集上略有回落，但仍保持稳定判别。
- PR-AUC 从 `Fit 0.392` 到 `Validation 0.372` 再到 `Temporal 0.349`，三段之间相对接近，说明当前版本的排序能力在不同数据段上是一致的。
- 这张图现在支持的结论是：formal winner 既可以作为内部验证最优模型，也可以作为时间外推可接受的正式结果模型。

## 6. Formal Winner Coefficients
File: `TeacherFrozenFuse_FormalWinner_Coefficients.png`

Interpretation:
- `Teacher_Logit` 系数最大，约 `+0.551`，说明最终 LR 主要还是在吃 old `0.351` teacher 的风险排序。
- `Gate_Static` 为正、`Gate_Local` 为负，幅度几乎对称，说明这条 winner 在用“静态权重高一点更危险、本地近期权重高一点更保守”的方式修正 teacher risk。
- `Landmark_Signal` 为正但幅度较小，说明 3M landmark 在补充信息，但不是主导项。
- 这张图支持一个直接结论: formal winner 的本质不是复杂非线性融合，而是“teacher risk + gate pattern + 一点 landmark”。

## 7. Calibration
File: `Calibration_Temporal.png`

Interpretation:
- 模型输出大多集中在低概率区，绝大多数预测值低于 `0.25`，这和低复发率场景一致。
- 在当前被访问到的概率区间里，校准曲线整体还算贴近对角线，没有明显系统性高估或低估。
- 但要注意，这张图没有覆盖高概率端，因为模型几乎不给高概率，所以它说明的是“低风险区校准尚可”，不是“全概率范围都校准优秀”。

## 8. DCA
File: `DCA_Temporal.png`

Interpretation:
- 在低阈值区间，模型净获益始终高于 `treat none`，也明显高于 `treat all`。
- 到大约 `0.25` 之后，模型净获益接近 0，说明这条模型的临床可用区间偏低阈值。
- 换句话说，这个模型更适合“早筛/宁可多看一些”的使用方式，不适合拿很高阈值做强确认。

## 9. Threshold Sensitivity
File: `Threshold_Sensitivity_Temporal.png`

Interpretation:
- 随着阈值升高，`recall` 快速下降，`specificity` 快速上升，这是正常现象。
- `F1` 的峰值大致出现在 `0.16-0.20` 一带，选用的 `0.20` 基本贴近这一区域，不算拍脑袋。
- `0.20` 这个阈值对应的是“偏高特异度、偏低召回”的 operating point，所以它更偏保守，而不是激进抓阳性。

## 10. Confusion Matrix
File: `Confusion_Temporal.png`

Interpretation:
- 矩阵是 `TN=439, FP=34, FN=27, TP=14`。
- 这对应 `specificity = 0.928`，但 `recall = 0.341`，说明模型当前主要擅长少报假阳性，不擅长把真阳性尽量多抓出来。
- 这意味着当前模型的 operating point 偏保守；尽管离散召回不高，但整体排序层面的 `PR-AUC = 0.349` 仍说明其对高风险样本具有可用的优先级区分能力。

## 11. Patient-Level Risk Stratification
File: `TeacherFrozenFuse_Patient_Risk_Q1Q4.png`

Interpretation:
- 总体上 Q1 到 Q4 的预测风险和观测复发率都在上升，说明 patient-level stratification 是有分层能力的。
- 但 Q2 和 Q3 出现了局部反转: Q2 的观测复发率高于 Q3。
- 这说明模型能较稳定地区分高风险尾部人群，但中间风险层级仍存在一定波动；总体上，这与 `Temporal PR-AUC = 0.349` 的中上等排序能力是相容的。

## 12. End-to-End SHAP Summary
File: `TeacherFrozenFuse_EndToEnd_SHAP.png`

Interpretation:
- 这张图是更有用的解释图，因为它解释的是最终冻结管线对工程化输入的依赖，而不是只解释 LR fuse 的 4 个输入。
- 前排重要特征几乎都来自 `Global` 块，尤其是 `Time_In_Normal`、`FT3_Current_x_CoreWindow_1M->3M`、`FT4_Current`、`Interval_Width`、`FT3_Current`。
- `Local::Window_1M->3M` 也进入前列，说明早期窗口信息确实在驱动风险估计。
- 这张图说明 current winner 的判别逻辑更接近“病程全局状态 + 早期窗口交互”，而不是简单依赖某个单点指标。

## 13. End-to-End SHAP Mean |SHAP|
File: `TeacherFrozenFuse_EndToEnd_SHAP_Bar.png`

Interpretation:
- 全局前 10 名里，`Global` 特征占绝对主导，`Local` 只在少数窗口和近期实验室指标上进入前排，`Static` 基本不在前列。
- 这和 backbone 结构也一致: formal winner 最终学到的有效信号主要来自病程块，而不是静态背景块。
- 如果后面要继续做“减法”，这张图也提示优先不要先砍 `Global`，反而该先审视那些贡献长期偏小的静态变量。

## 14. End-to-End SHAP High-risk Example
File: `TeacherFrozenFuse_EndToEnd_SHAP_Local_HighRisk.png`

Interpretation:
- 这个高风险个体预测值约 `0.336`，明显高于总体基线 `E[f(x)] = 0.085`。
- 抬高风险的主因包括: `Time_In_Normal = 0`、`LandmarkObsValue = 1`、较高的 `FT3_Current`、以及部分 global interaction。
- 这说明模型把“长期未稳定正常 + 已观察到 3M 高风险信号 + 当前甲功相关异常”组合成了高风险判定。

## 15. End-to-End SHAP Low-risk Example
File: `TeacherFrozenFuse_EndToEnd_SHAP_Local_LowRisk.png`

Interpretation:
- 这个低风险个体预测值约 `0.012`，远低于基线 `0.085`。
- 降低风险的关键因素包括: `Time_In_Normal = 3`、更晚窗口 `Window_18M->24M`、更高的 `Start_Time`，以及若干较平稳的 global lab 特征。
- 这说明模型把“已稳定较长时间、位于更晚随访窗口、整体状态平稳”解释为低风险，这在临床语义上是合理的。

## 16. Fuse-space SHAP Summary
File: `TeacherFrozenFuse_SHAP.png`

Interpretation:
- 这张图只是最终 LR fuse 层的解释，不是整套管线的解释。
- 它显示 `Teacher_Logit` 影响最大，`Landmark_Signal` 次之，`Gate_Static` 和 `Gate_Local` 也在修正结果。
- 作为 sanity check 它有用，但不能替代 end-to-end SHAP，因为它解释不到工程化输入层。

## 17. Fuse-space SHAP Mean |SHAP|
File: `TeacherFrozenFuse_SHAP_Bar.png`

Interpretation:
- 重要性排序是 `Teacher_Logit > Landmark_Signal ≈ Gate_Static ≈ Gate_Local`。
- 这再次说明 formal winner 的最终 LR 很简单，核心仍是 teacher score。
- 这张图更多是在证明“最终 fuse 没有乱学”，而不是在提供可发表的临床解释。

## 18. Fuse-space SHAP High-risk Example
File: `TeacherFrozenFuse_SHAP_Local_HighRisk.png`

Interpretation:
- 这个高风险例子里，4 个 fuse 输入全部在推高风险，尤其 `Gate_Static`、`Gate_Local` 和 `Teacher_Logit` 的增幅接近。
- 这说明在某些样本上，最终 LR 不是只看 teacher，而是会把 gate pattern 也当成强证据。
- 但这个解释仍然太靠后，只能说明“哪个二级输入在推分”，不能告诉我们底层是哪类工程化特征导致的。

## 19. Fuse-space SHAP Low-risk Example
File: `TeacherFrozenFuse_SHAP_Local_LowRisk.png`

Interpretation:
- 低风险例子里，最主要的降分因素是更低的 `Teacher_Logit`，其次是 `Gate_Static` 和 `Gate_Local`。
- `Landmark_Signal = 0` 的影响相对更小，说明没有 3M 高风险信号本身不足以强力拉低风险，还是要靠 teacher score 主导。
- 所以 fuse-space 层面的主要结论仍然是: 这个 formal winner 对 teacher risk 的依赖非常强。

## Overall Takeaway
- 这套 formal winner 的强项是: validation 与 temporal test 排序表现接近、结构极简、低阈值下有净获益、patient-level Q4 能明显拉开。
- 它的弱项仍然是: 当前固定阈值下召回偏低，中间风险层级仍有一定不稳定。
- 从 end-to-end SHAP 看，当前有效信号主要来自 `Global` 病程特征和早期窗口交互；从 fuse-space SHAP 看，最终 LR 主要靠 `Teacher_Logit` 加上 gate 修正。
- 因而这条线最准确的定位是:
  - 它是一个“validation 最优、temporal test 也保持可接受表现、解释已补齐”的正式模型。
