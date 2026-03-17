# 运行结果整理

## 说明

以下结果均来自 `conda activate med` 环境下的实际运行，绘图缓存目录设为 `MPLCONFIGDIR=/tmp/mpl`。  
本次共完成 5 个主脚本的运行，均成功结束，未出现中断或未完成输出的情况。

对应脚本如下：

- `fixed_landmark_binary.py`
- `fixed_landmark_multiclass.py`
- `relapse.py`
- `relapse_recurrent_survival.py`
- `relapse_two_stage_physio.py`

---

## 1. 固定 Landmark 二分类结果

固定 landmark 二分类用于回答一个更偏“当前状态识别”的问题：患者在 `3M` 或 `6M` 复诊时，当前是否仍处于甲亢状态。

### 3 个月结果

在 `3M` 时点，不同模型均能达到较稳定的甲亢截获能力，其中：

- 最佳 `AUC` 为 `Random Forest = 0.849`
- 最佳 `PR-AUC` 为 `Balanced RF = 0.772`

若按照稀少事件更敏感的 `PR-AUC` 选择主报告模型，则 `Balanced RF` 在测试集上的表现为：

- `AUC = 0.845`，`95% CI [0.786, 0.893]`
- `PR-AUC = 0.772`，`95% CI [0.667, 0.857]`
- `Brier = 0.164`
- `Recall = 0.688`
- `Specificity = 0.852`
- `Calibration intercept = -0.534`
- `Calibration slope = 1.334`
- `Threshold = 0.59`

这说明在治疗后早期，固定时点信息已经足以较好地区分“仍为甲亢”与“非甲亢”，但概率刻度仍存在一定偏移，模型更适合作为状态识别工具，而非直接替代动态复发预警。

### 6 个月结果

到 `6M` 时点后，整体判别能力进一步提高：

- 最佳 `AUC` 为 `Random Forest = 0.904`
- 最佳 `PR-AUC` 亦为 `Random Forest = 0.852`

该模型在测试集上的具体表现为：

- `AUC = 0.904`，`95% CI [0.858, 0.947]`
- `PR-AUC = 0.852`，`95% CI [0.758, 0.927]`
- `Brier = 0.114`
- `Recall = 0.887`
- `Specificity = 0.789`
- `Calibration intercept = -0.191`
- `Calibration slope = 0.913`
- `Threshold = 0.42`

与 `3M` 相比，`6M` 的结果更稳健，提示随着治疗后实验室信息的累积，“当前是否仍为甲亢”这一问题变得更容易判断。

### 新增图形输出

本次补充生成了固定 landmark 二分类的概率校准与临床可用性图：

- `results/fixed_landmark_binary/Calibration_3-Mo.png`
- `results/fixed_landmark_binary/Calibration_6-Mo.png`
- `results/fixed_landmark_binary/DCA_3-Mo.png`
- `results/fixed_landmark_binary/DCA_6-Mo.png`
- `results/fixed_landmark_binary/Threshold_Sensitivity_3-Mo.png`
- `results/fixed_landmark_binary/Threshold_Sensitivity_6-Mo.png`

---

## 2. 固定 Landmark 三分类结果

固定 landmark 三分类用于回答另一个更细化的门诊问题：在固定随访节点，患者当前更接近 `Hyper / Normal / Hypo` 中的哪一类。

### 3 个月结果

在 `3M` 时点，`Cascade RF` 的整体表现最好：

- `3-class ACC = 0.649`，`95% CI [0.582, 0.711]`
- `Macro ROC-AUC = 0.801`，`95% CI [0.746, 0.846]`
- `Macro PR-AUC = 0.665`，`95% CI [0.600, 0.740]`
- `Multiclass Brier = 0.488`

按临床最关心的两个端点状态拆分：

- `Hyper ROC-AUC = 0.845`
- `Hypo ROC-AUC = 0.730`

若进一步做 one-vs-rest 分析：

- `Hyper vs rest`：AUC `0.845`，PR-AUC `0.759`，Threshold `0.29`
- `Hypo vs rest`：AUC `0.730`，PR-AUC `0.570`，Threshold `0.31`

这表明在 `3M` 时点，模型对“是否仍甲亢”的识别明显优于“是否进入甲减”的识别，符合早期治疗后状态仍较不稳定的临床直觉。

### 6 个月结果

在 `6M` 时点，整体分流性能进一步改善，最佳宏观表现来自 `Cascade GBDT`：

- `3-class ACC = 0.702`，`95% CI [0.632, 0.764]`
- `Macro ROC-AUC = 0.847`，`95% CI [0.799, 0.887]`
- `Macro PR-AUC = 0.722`，`95% CI [0.654, 0.800]`
- `Multiclass Brier = 0.433`

类别层面：

- `Hyper ROC-AUC = 0.906`
- `Hypo ROC-AUC = 0.762`

对应 one-vs-rest 结果为：

- `Hyper vs rest`：AUC `0.906`，PR-AUC `0.849`，Threshold `0.41`
- `Hypo vs rest`：AUC `0.762`，PR-AUC `0.667`，Threshold `0.15`

因此，`6M` 时点的三状态分流已经具备较好的应用潜力，但它依然回答的是“当前状态归类”，而不是“下一窗口是否复发”。

### 新增图形输出

本次新增的三分类相关图形主要围绕 `Hyper` 与 `Hypo` 两个临床重点类别：

- `results/fixed_landmark_multiclass/CM_3-Mo.png`
- `results/fixed_landmark_multiclass/CM_6-Mo.png`
- `results/fixed_landmark_multiclass/Calibration_3-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/Calibration_3-Mo_Hypo.png`
- `results/fixed_landmark_multiclass/Calibration_6-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/Calibration_6-Mo_Hypo.png`
- `results/fixed_landmark_multiclass/DCA_3-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/DCA_3-Mo_Hypo.png`
- `results/fixed_landmark_multiclass/DCA_6-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/DCA_6-Mo_Hypo.png`
- `results/fixed_landmark_multiclass/Threshold_Sensitivity_3-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/Threshold_Sensitivity_3-Mo_Hypo.png`
- `results/fixed_landmark_multiclass/Threshold_Sensitivity_6-Mo_Hyper.png`
- `results/fixed_landmark_multiclass/Threshold_Sensitivity_6-Mo_Hypo.png`

---

## 3. Rolling Landmark 直接复发预警结果

`relapse.py` 是当前最应作为主文主线的分析。其任务定义为：仅在当前已恢复 `Normal` 的 at-risk interval 上，预测患者下一窗口是否重新回到 `Hyper`。

### 数据与事件分布

本次运行得到：

- 总状态转移数 `6018`
- 稳定转移数 `4167`
- `Normal -> Hyper` 复发转移数 `210`
- 训练集 interval 数 `2096`
- 测试集 interval 数 `514`
- 测试集复发事件数 `41`

### interval-level 主结果

不同模型比较后：

- 最佳 `AUC`：`Logistic Reg. = 0.841`
- 最佳 `PR-AUC`：`Logistic Reg. = 0.334`
- 最佳 `C-index`：`SVM = 0.766`

按主线模型 `Logistic Reg.` 报告，其 interval-level 表现为：

- `AUC = 0.841`，`95% CI [0.793, 0.890]`
- `PR-AUC = 0.334`，`95% CI [0.214, 0.479]`
- `Brier = 0.064`，`95% CI [0.046, 0.081]`
- `Recall = 0.585`，`95% CI [0.437, 0.750]`
- `Specificity = 0.839`，`95% CI [0.809, 0.872]`
- `Calibration intercept = -0.053`
- `Calibration slope = 1.061`
- `Threshold = 0.15`

这提示模型在稀少事件场景下仍具有较好的区分度，且概率刻度总体可接受。

### patient-level 聚合结果

若将 interval-level 风险聚合为“整个后续观察期内是否至少复发一次”的患者级风险，则：

- `AUC = 0.749`，`95% CI [0.657, 0.835]`
- `PR-AUC = 0.453`，`95% CI [0.324, 0.625]`
- `Brier = 0.154`
- `Recall = 0.778`
- `Specificity = 0.585`
- `Calibration intercept = -0.275`
- `Calibration slope = 0.929`
- `Threshold = 0.22`

不同聚合方式下，`Mean_Interval_Risk` 获得了最高的患者级 `AUC = 0.801` 与 `PR-AUC = 0.595`，但主文仍更适合优先报告默认的 `Product_AnyRisk`，因为它与“至少一次复发”的临床语义更一致。

### 窗口敏感性

窗口敏感性分析显示：

- 全窗口：AUC `0.841`，PR-AUC `0.334`
- 去除 `1M->3M` 后：AUC `0.864`，PR-AUC `0.389`
- 仅保留 `3M` 及以后：AUC `0.864`，PR-AUC `0.389`
- 去除 `12M+` 后：AUC `0.739`，PR-AUC `0.351`

这说明模型并不是单纯依赖最早期窗口“带飞”，而是对早中期风险识别确有较强能力。

---

## 4. Recurrent-Survival Benchmark 结果

`relapse_recurrent_survival.py` 的目的不是替代主线，而是回答：如果改用 recurrent-event 语言，结论方向是否仍然一致。

### 数据概况

- 训练集 interval rows：`2096`
- 测试集 interval rows：`514`
- 训练集 recurrent spells：`842`
- 测试集 recurrent spells：`213`
- 训练集 recurrent events：`168`
- 测试集 recurrent events：`39`

### 模型比较

主要结果如下：

- `AG Cox [markov_lite]`：C-index `0.782`，next-window AUC `0.729`，PR-AUC `0.342`
- `PWP Gap Cox [markov_lite]`：C-index `0.780`，next-window AUC `0.517`，PR-AUC `0.128`
- `AG Cox [history_augmented]`：C-index `0.779`，next-window AUC `0.731`，PR-AUC `0.359`
- `PWP Gap Cox [history_augmented]`：C-index `0.775`，next-window AUC `0.504`，PR-AUC `0.125`
- `Event-Specific RSF [markov_lite]`：C-index `0.791`，next-window AUC `0.793`，PR-AUC `0.347`
- `Event-Specific RSF [history_augmented]`：C-index `0.779`，next-window AUC `0.799`，PR-AUC `0.335`

若按不同指标选择最佳模型：

- 最佳 `C-index`：`Event-Specific RSF [markov_lite] = 0.791`
- 最佳 next-window `AUC`：`Event-Specific RSF [history_augmented] = 0.799`
- 最佳 next-window `PR-AUC`：`AG Cox [history_augmented] = 0.359`

以 `AG Cox [history_augmented]` 作为校准参考模型时：

- `Next-window Brier = 0.103`
- `Calibration intercept = 0.292`
- `Calibration slope = 1.028`

总体上，recurrent-survival 视角支持了主线结论：早中期窗口依然是最关键的风险识别阶段，但在当前任务定义下，它并未明显优于直接 rolling-landmark 复发分类。

---

## 5. 两阶段生理预测复发结果

`relapse_two_stage_physio.py` 评估的是一条更复杂的分析路径：先预测下一窗口的生理值，再利用预测的未来生理值判断是否复发。

### Stage 1：未来生理值预测

最佳 Stage-1 模型为 `RandomForest`。其平均表现为：

- 训练集 OOF：`MAE = 1.285`，`RMSE = 2.320`，`R² = 0.067`
- 测试集：`MAE = 1.397`，`RMSE = 2.729`，`R² = 0.068`

这表明未来实验室值并非完全不可预测，但拟合强度仍然有限。

### Stage 2：复发分类

三种方案中，最佳结果如下：

- `direct`：`Elastic LR`，AUC `0.828`，PR-AUC `0.319`，Brier `0.067`
- `predicted_only`：`Logistic Reg.`，AUC `0.840`，PR-AUC `0.265`，Brier `0.067`
- `predicted_plus_current`：`Logistic Reg.`，AUC `0.834`，PR-AUC `0.311`，Brier `0.070`

校准方面：

- `direct`：intercept `-0.589`，slope `0.778`
- `predicted_only`：intercept `-0.320`，slope `0.918`
- `predicted_plus_current`：intercept `-0.729`，slope `0.734`

从结果上看，虽然 `predicted_only` 的 AUC 略高，但最关键的稀少事件指标 `PR-AUC` 仍以 `direct` 方案最佳，因此两阶段路线目前更适合作为支持性分析，而非主文主轴。

---

## 6. 总体结论

综合五个脚本的结果，可以形成较清晰的论文式结论：

1. 固定 landmark 二分类说明，到 `3M` 和 `6M` 时点，当前甲亢状态已经可以被较准确地识别，且 `6M` 明显优于 `3M`。
2. 固定 landmark 三分类表明，到 `6M` 时点，`Hyper / Normal / Hypo` 三状态分流具备一定可行性，但其本质仍是“当前状态识别”。
3. rolling landmark 直接复发预警依然是最有临床价值的主线，因为它直接回答“当前恢复正常后，下一窗口是否复发”这一真实随访问题。
4. recurrent-survival benchmark 说明，用 recurrent-event 语言重述问题后，方向性结论保持一致，但并未明显优于主线 direct 模型。
5. 两阶段生理预测提示未来生理值包含一定信号，但误差传播会削弱下游复发识别，因此当前尚不足以替代直接基于当前状态的复发预警模型。
