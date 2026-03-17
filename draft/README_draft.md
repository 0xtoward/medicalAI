# 基于时序 Landmark 机器学习的 Graves 甲亢 RAI 治疗后早期状态识别与动态复发风险分层

本仓库当前最适合整合成一篇论文的主线，不是“做了几个模型实验”，而是：

> 在同一真实世界纵向队列中，建立一个统一的时序 Landmark 机器学习框架，同时完成 RAI 治疗后的早期状态识别与后续动态复发监测。

这条主线对应两个连续的临床决策问题：

1. 患者在 `3M` 或 `6M` 复诊时，当前是否仍处于甲亢状态？
2. 对已经恢复到正常状态的患者，下一次复诊是否会再次复发，以及整个后续观察期内谁属于高风险人群？

为避免文章发散，建议将整篇工作组织为：

- 主分析 1：固定 Landmark 的早期二分类状态识别，基于 `hyper_detect.py`
- 主分析 2：滚动 Landmark 的动态复发预警，基于 `relapse.py`
- 扩展分析：固定 Landmark 的三分类状态分流，基于 `all2.py`

README 以下内容按论文写作逻辑组织，尽量让医学合作者不依赖代码也能理解研究问题、方法、结果和图示含义。

---

## 1 论文主线与推荐定位

### 1.1 一句话定位

这是一个围绕 RAI 治疗后随访管理的时序风险分层框架，而不是单纯追求更高 AUC 的模型竞赛。

### 1.2 推荐题目方向

- 基于纵向 Landmark 机器学习的 Graves 甲亢 RAI 治疗后早期状态识别与动态复发风险预测
- A temporal landmark-based machine learning framework for early post-RAI state assessment and dynamic relapse surveillance in Graves' hyperthyroidism

### 1.3 为什么三块工作可以统一成一篇 paper

三块分析对应同一条临床路径上的三个层级：

```text
RAI 治疗后纵向随访
  ├── 3M / 6M 时点：当前是否仍为甲亢？         -> fixed Landmark binary
  ├── 3M / 6M 时点：当前属于甲亢/正常/甲减哪类？ -> fixed Landmark multiclass
  └── 恢复正常之后：下一次会不会复发？         -> rolling Landmark relapse
```

因此更自然的论文叙事不是“3 个脚本”，而是“1 个统一框架，覆盖 2 个主临床决策点，外加 1 个可扩展分析”。

### 1.4 与现有文献应如何比较

我们检索到的相关文献多聚焦于：

- 基线或固定时点预测缓解/未缓解
- 单一终点预测
- 可解释模型用于治疗决策优化

例如部分文献的核心问题更接近“固定特征预测远期疗效”。而本项目的优势在于：

- 使用同一纵向队列
- 同时处理早期状态识别与后续复发监测
- 不仅报告区分度，还报告校准、DCA、patient-level 风险分层与 SHAP

需要特别注意的是：

- 不建议在稿件里直接写“完爆某篇论文”
- 因为任务定义不完全一致，不能简单视为严格 head-to-head
- 更合适的写法是：在当前 Landmark 设定与内部验证框架下，我们的模型在区分度和临床可用性方面显示出较强表现

---

## 2 数据来源与临床问题

当前数据为 Graves 甲亢患者接受 RAI 治疗后的纵向真实世界随访数据。按照现有清洗逻辑：

| 项目 | 内容 |
|---|---|
| 治疗记录数 | `1003` |
| 唯一患者数 | `889` |
| 随访时间点 | `0M`、`1M`、`3M`、`6M`、`12M`、`18M`、`24M` |
| 核心实验室指标 | `FT3`、`FT4`、`TSH` |
| 临床状态标签 | 医生评价：甲亢 / 正常 / 甲减 |
| 主要基线变量 | 性别、年龄、身高、体重、BMI、突眼、甲状腺重量、RAI 3 日摄取率、TRAb、TPOAb、TGAb、剂量等 |

状态编码为：

- `Hyper`：甲亢
- `Normal`：正常
- `Hypo`：甲减

其中 `0M` 为治疗前，默认全部患者处于 `Hyper`。

这套数据支持三个互相关联的问题：

1. 固定 Landmark 二分类：`Hyper vs Non-Hyper`
2. 固定 Landmark 三分类：`Hyper / Normal / Hypo`
3. 滚动 Landmark 复发预测：`Normal -> Hyper`

---

## 3 研究终点与两种分析层级

### 3.1 Fixed Landmark 是什么

固定 Landmark 指在某个固定随访时点，例如 `3M` 或 `6M`，仅使用截至该时点可获得的数据做预测。

这对应的临床问题是：

- 到 `3M` 时，患者当前是否仍为甲亢？
- 到 `6M` 时，患者当前属于甲亢、正常还是甲减？

它适合回答“现在控制得怎么样”。

### 3.2 Rolling Landmark 是什么

滚动 Landmark 则是在每个可用随访节点都建立一次预测，把每个相邻区间都变成一个样本。

这对应的临床问题是：

- 患者这次已经正常，下次会不会复发？

它适合回答“下一步会不会出问题”。

### 3.3 Interval-level 和 patient-level 的区别

这部分主要属于 `relapse.py` 的结果表达方式。

#### Interval-level

`interval-level` 指“区间层面”的预测。每个样本对应：

- 某位患者
- 某个相邻随访区间，如 `1M->3M`、`3M->6M`
- 且该患者在区间起点时为 `Normal`

模型预测的是：

$$
P(\text{Relapse at } k+1 \mid S_k = \text{Normal}, X_{\le k})
$$

可以通俗理解成：

“患者本次复查正常，到下一次复查时会不会重新回到甲亢？”

#### Patient-level

`patient-level` 指“患者整体层面”的风险汇总。它不再盯着单个区间，而是把一个患者在所有可评估区间上的风险综合起来。

本项目 patient-level 终点定义为：

> 患者在其观察期内是否至少出现过一次 `Normal -> Hyper` 复发。

脚本中采用的汇总方式为：

$$
P(\text{any relapse during follow-up}) = 1 - \prod_{j=1}^{k}(1-p_j)
$$

可理解为：

“这个患者在后续整个观察期内，至少复发一次的总体概率有多大？”

#### Q1-Q4 是什么

`Q1~Q4` 不是终点，而是 patient-level 风险分层方式：

- 终点：观察期内是否至少复发一次
- 分层：按照训练集预测风险四分位数，将测试集患者分为 `Q1-Q4`

它的临床意义是“能否把患者分成低、中、高风险组以安排随访强度”，而不是“重新定义结局”。

---

## 4 统一方法学框架

### 4.1 总体建模策略

整篇 paper 可以被理解为“一套统一时序框架 + 两个主分析 + 一个扩展分析”：

| 分析层级 | 脚本 | 目标 | 论文角色 |
|---|---|---|---|
| 固定 Landmark 二分类 | `hyper_detect.py` | `3M/6M` 时点识别 `Hyper vs Non-Hyper` | 主分析 1 |
| 滚动 Landmark 复发预测 | `relapse.py` | 对正常患者预测下一次是否复发 | 主分析 2 |
| 固定 Landmark 三分类 | `all2.py` | `3M/6M` 时点分流到 `Hyper / Normal / Hypo` | 扩展分析 / 补充材料 |

### 4.2 防泄漏设计

纵向数据最重要的是防止未来信息泄漏。本项目的主要防护包括：

- 训练/测试切分先于预处理
- MissForest 仅在训练集上拟合
- Landmark 输入矩阵按时间截断，只保留当前时点及之前的列
- `GroupKFold` 或按患者分组切分，避免同一患者跨折泄漏
- 阈值由训练集 OOF 预测决定，测试集只用于最终评估

### 4.3 缺失值处理

使用 MissForest 风格的迭代填充：

- 实现方式：`IterativeImputer + RandomForestRegressor`
- 固定 Landmark：按 `seq_len` 截断输入列
- 滚动 Landmark：按 `depth` 截断输入列

这样在预测某个 Landmark 时，只使用当时“理论上可见”的信息。

### 4.4 特征设计

#### 固定 Landmark 模块

`hyper_detect.py` 和 `all2.py` 的特征主要包括：

- 静态基线变量
- 截至 Landmark 的 `FT3 / FT4 / TSH` 序列
- 相对基线的变化量
- 在 `6M` 时还加入 `6M-3M` 的短期变化量

#### 滚动 Landmark 模块

`relapse.py` 除上述信息外，还进一步加入：

- 当前实验室值：`FT3_Current`、`FT4_Current`、`logTSH_Current`
- 历史状态记忆：`Ever_Hyper_Before`、`Ever_Hypo_Before`、`Time_In_Normal`
- 动量特征：`Delta_FT4_k0`、`Delta_TSH_k0`、`Delta_FT4_1step`、`Delta_TSH_1step`
- 时间窗口变量：`Window_*`
- 既往状态变量：`PrevState_*`

因此，`relapse.py` 更适合承担文章中的方法学亮点。

---

## 5 主分析 1：固定 Landmark 的早期甲亢状态识别

这一部分建议作为主文的第一个结果板块，用于回答：

> 在 `3M` 和 `6M` 复诊时，患者当前是否仍处于甲亢状态？

### 5.1 3 个月 Landmark 二分类结果

终端结果显示，在 `3M` 二分类识别中：

- 最佳 `AUC`：`LightGBM = 0.869`
- 最佳 `PR-AUC`：`LightGBM = 0.824`

其他模型也表现较强，说明在 `3M` 时点，利用截至 `3M` 的信息识别当前是否仍为甲亢是可行的。

若以 `Logistic Reg.` 为参考：

- `AUC = 0.860`
- `PR-AUC = 0.814`
- `Accuracy = 0.797`
- `F1 = 0.731`

说明线性模型已经能取得较强表现，而树模型略有进一步提升。

### 5.2 6 个月 Landmark 二分类结果

`6M` 结果进一步增强，是整篇文章中非常有竞争力的一部分：

- 最佳 `AUC`：`GradientBoosting = 0.928`
- 最佳 `PR-AUC`：`GradientBoosting = 0.894`

同时其他模型也处于较高水平：

- `Random Forest`: `AUC = 0.915`, `PR-AUC = 0.878`
- `AdaBoost`: `AUC = 0.916`, `PR-AUC = 0.889`
- `Stacking`: `AUC = 0.921`, `PR-AUC = 0.886`
- `Logistic Reg.`: `AUC = 0.907`, `PR-AUC = 0.874`

这说明：

- 到 `6M` 时点，利用纵向随访信息进行甲亢状态截获具有很强的区分能力
- 你们的数据和特征工程在固定 Landmark 任务上已经达到很高水平
- 这一块完全可以作为论文主文中的第一主分析，而不是只放在附录里

### 5.3 这一部分在论文里应该怎么讲

推荐表述为：

> 在 `3M` 和 `6M` 的固定 Landmark 评估中，多种机器学习模型均显示出良好的甲亢状态识别能力，其中 `6M` 时表现最佳，提示早期随访数据足以支持对当前疾病控制状态的精准判别。

这部分的临床意义是：

- 回答“患者当前控制得如何”
- 可用于辅助判断是否仍需强化治疗或更密集复查

### 5.4 图示建议

如 `hyper_detect_result/` 中的图文件可用，推荐优先使用：

- `ROC_3-Mo.png`
- `ROC_6-Mo.png`
- 对应 SHAP 图

这两张 ROC 图非常适合做主文 Figure 2。

---

## 6 主分析 2：滚动 Landmark 的动态复发预警

这部分建议作为整篇文章的方法学亮点和第二主结果，用于回答：

> 对已经恢复到正常状态的患者，下一次是否会再次复发？以及谁是未来整体复发高风险患者？

### 6.1 状态转移的描述性结果

状态转移热图显示，在全部相邻随访转移中：

- 总转移次数：`6018`
- 稳定转移：`4167`
- `Normal -> Hyper` 复发：`210`
- `Hypo -> Normal` 恢复：`358`

从时间窗口看，复发主要集中在：

- `1M->3M`
- `3M->6M`
- `6M->12M`

提示治疗后早期至中期是复发监测的关键时段。

### 6.2 Interval-level 结果

在 `interval-level` 测试集上：

- 测试区间数：`514`
- 复发事件数：`41`

最佳模型为 `Logistic Regression`：

| 指标 | 数值 |
|---|---|
| ROC-AUC | `0.841` |
| PR-AUC | `0.334` |
| Accuracy | `0.772` |
| Balanced Accuracy | `0.732` |
| Recall | `0.683` |
| Specificity | `0.780` |
| Brier score | `0.064` |
| 阈值 | `0.12` |

95% CI：

- AUC：`0.793–0.890`
- PR-AUC：`0.214–0.479`
- Brier：`0.046–0.081`
- Recall：`0.545–0.822`
- Specificity：`0.747–0.812`

这部分说明：

- 模型对“下一次是否复发”的判别能力较好
- 虽然 `ACC` 不算高，但在低事件率场景下不应把 `ACC` 作为主卖点
- 对复发预警任务，更应关注 `PR-AUC`、`Recall`、`Brier`、`Calibration` 和 `DCA`

### 6.3 Patient-level 结果

在 patient-level 汇总后：

- 训练患者数：`626`
- 测试患者数：`159`

终点定义为“观察期内是否至少复发一次”，结果如下：

| 指标 | 数值 |
|---|---|
| ROC-AUC | `0.749` |
| PR-AUC | `0.453` |
| Recall | `0.750` |
| Specificity | `0.593` |
| Brier score | `0.154` |
| 阈值 | `0.23` |

95% CI：

- AUC：`0.657–0.835`
- PR-AUC：`0.324–0.625`
- Brier：`0.119–0.185`
- Recall：`0.600–0.889`
- Specificity：`0.512–0.686`

这说明：

- patient-level 汇总后，模型仍能实现中等到良好的风险区分
- patient-level 结果更接近临床“谁该重点盯”的实际使用场景
- 这部分非常适合写进 Discussion 作为临床转化价值

### 6.4 为什么这部分是方法学亮点

相较于固定终点模型，滚动 Landmark 复发分析具有三点优势：

- 真正利用了纵向中间状态，而不是只预测一个固定远期结局
- 能输出不同随访窗口下的条件风险
- 能自然扩展到 patient-level 风险分层和 DCA

因此整篇 paper 中，`relapse.py` 更适合作为“方法学新意”和“临床随访管理价值”的核心支撑。

---

## 7 扩展分析：固定 Landmark 三分类状态分流

`all2.py` 的定位建议明确为：

> 对统一框架的可扩展性验证，而不是与两个主分析并列的第三条主线。

### 7.1 为什么不建议把 `all2.py` 再升成第三主分析

虽然 `all2.py` 很有方法学特色，尤其包括：

- `CascadeClassifier`
- `Cascade RF`
- `Cascade GBDT`
- `Hybrid(MLP+GBDT)`
- `Ensemble(Soft)`

但如果把它也放进主文与 `hyper_detect.py`、`relapse.py` 并列，会带来两个问题：

- 文章目标显得过于分散
- 医学读者容易看不清主问题到底是“当前状态识别”还是“未来复发预测”

### 7.2 更合适的角色

建议将 `all2.py` 作为：

- 扩展分析
- 方法学补充
- Supplementary Results

它可以证明：

- 这套固定 Landmark 框架不仅能做二分类
- 还能进一步扩展到更贴近临床状态分流的多分类任务

### 7.3 论文中的推荐写法

在主文中可以用一小段概括：

> 除二分类状态识别外，我们还在 `3M` 和 `6M` Landmark 上进行了三分类扩展分析，将患者分流为甲亢、正常和甲减。该结果提示本框架除支持当前甲亢状态截获外，亦具备向更细粒度状态管理扩展的潜力。

详细图表与 SHAP 结果放入补充材料即可。

---

## 8 结果图的推荐解读方式

### 8.1 固定 Landmark 二分类图

`hyper_detect.py` 的 ROC 图应承担主文中“当前状态识别能力”的展示作用。

推荐强调：

- `3M` 已有较强表现
- `6M` 明显更优
- 多模型结果一致较强，说明并非单一模型偶然获胜

### 8.2 动态复发风险曲线：`multistate_result/Hazard_Curve_Strict.png`

该图比较真实复发风险与各模型预测风险。

推荐解读：

- 真实风险总体呈“前高后低”
- 模型能够追踪这一动态趋势
- 后期低事件率区间存在一定高估，但总体时间模式一致

### 8.3 Interval-level 校准图：`multistate_result/Calibration_Interval.png`

推荐解读：

- 低到中等风险区间校准较好
- 大多数预测概率集中在低风险区域，符合低事件率背景
- 高风险尾部样本少，解释需谨慎

### 8.4 Patient-level 校准图：`multistate_result/Calibration_Patient.png`

推荐解读：

- 随预测风险增加，观察到的复发率总体升高
- 高风险段存在波动，提示仍需更大样本进一步验证

### 8.5 DCA：`multistate_result/DCA_Interval.png` 与 `multistate_result/DCA_Patient.png`

推荐解读：

- interval-level DCA 显示模型在较低至中等阈值范围内具有净获益
- patient-level DCA 在更宽阈值区间保持正净获益
- 因此 patient-level 结果比 interval-level 更容易转化为临床随访策略

### 8.6 Patient-level 风险分层：`multistate_result/Patient_Risk_Q1Q4.png`

当前结果显示：

- `Q1` 观察复发率约 `8%`
- `Q2` 约 `16%`
- `Q3` 约 `21%`
- `Q4` 约 `43%`

说明模型具有较清晰的患者分层能力，适合服务于“谁应更密切随访”的临床场景。

### 8.7 SHAP：`multistate_result/Hazard_SHAP_Relapse.png`

当前最值得关注的特征包括：

- `Time_In_Normal`
- `Ever_Hypo_Before`
- `ThyroidW`
- `FT3_Current`
- `Delta_TSH_k0`
- `logTSH_Current`

推荐临床解释：

- 维持正常越久，复发风险越低
- 曾经出现甲减者，再次回到甲亢的风险可能更低
- 甲状腺重量较大、当前 FT3 偏高提示更高复发风险
- TSH 恢复不足相关特征可能提示复发倾向

需要强调的是，SHAP 用于解释重要性与方向，不等同于因果结论。

---

## 9 论文结构建议

### 9.1 Introduction

建议聚焦两个临床痛点：

1. RAI 治疗后早期如何识别当前控制不佳或持续甲亢患者？
2. 对已经恢复正常的患者，如何判断谁更可能再次复发？

由此引出本文目标：

> 建立一个统一的时序 Landmark 机器学习框架，同时覆盖早期状态识别与后续动态复发监测。

### 9.2 Methods

建议按如下顺序写：

1. 队列来源、纳入排除标准、随访时间点
2. 状态定义：`Hyper / Normal / Hypo`
3. 两个主终点与一个扩展终点
4. 固定 Landmark 建模：`hyper_detect.py`
5. 滚动 Landmark 建模：`relapse.py`
6. 三分类扩展：`all2.py`
7. 防泄漏设计
8. 评估指标：`AUC`、`PR-AUC`、`Recall`、`Specificity`、`Brier`、`Calibration`、`DCA`

### 9.3 Results

主文推荐写成三个结果板块：

1. `3M/6M` 固定 Landmark 二分类状态识别
2. 正常后动态复发预警的 interval-level 与 patient-level 结果
3. 三分类扩展分析作为补充说明

### 9.4 Discussion

建议按以下逻辑讨论：

1. 同一时序框架覆盖了“当前状态识别”与“未来复发监测”
2. 固定 Landmark 结果强，说明早期随访信息足以识别当前甲亢状态
3. 滚动 Landmark 结果补足了传统固定终点模型缺少的动态监测能力
4. patient-level 风险分层和 DCA 提供了潜在临床落地价值
5. 局限性包括单中心、内部验证、事件数有限和外部验证缺失

---

## 10 图表配置建议

### 10.1 主文图

- Figure 1：研究设计示意图
- Figure 2：`hyper_detect.py` 的 `3M/6M` ROC 或性能对比图
- Figure 3：`multistate_result/Hazard_Curve_Strict.png`
- Figure 4：`multistate_result/Calibration_Interval.png` + `multistate_result/DCA_Interval.png`
- Figure 5：`multistate_result/Calibration_Patient.png` + `multistate_result/Patient_Risk_Q1Q4.png`
- Figure 6：`multistate_result/Hazard_SHAP_Relapse.png`

### 10.2 主文表

- Table 1：基线特征
- Table 2：`3M/6M` 二分类主要性能
- Table 3：动态复发分析的 interval-level 与 patient-level 结果

### 10.3 补充材料

- `all2.py` 的三分类结果
- 额外模型比较图
- 混淆矩阵
- 额外 SHAP 图

---

## 11 对医学合作者的核心解释

### 11.1 这篇文章最该强调什么

最该强调的是：

- 统一的时序 Landmark 框架
- 两阶段临床决策价值
- 不仅有区分度，还有校准、DCA 和患者层面的风险分层

### 11.2 不该怎么写

不建议：

- 把 3 个脚本写成 3 个互不相干的小研究
- 把“比某篇论文高多少 AUC”作为唯一卖点
- 让三分类分析与动态复发分析并列争夺主轴

### 11.3 最推荐的临床叙事

如果用一句话总结全文，最推荐的表述是：

> 我们构建了一个基于纵向随访数据的 Landmark 机器学习框架，可在 `3M/6M` 时点准确识别当前甲亢状态，并在恢复正常后进一步对短期复发风险和长期患者级复发风险进行动态分层，从而为 RAI 治疗后的个体化随访管理提供依据。

---

## 12 当前阶段的论文成熟度判断

从内部验证角度看，目前这套工作已经具备较完整的主文支撑材料：

- 固定 Landmark 二分类结果很强
- 动态复发分析已补齐 `CI`
- 已有 `Calibration`
- 已有 `DCA`
- 已有 `patient-level` 风险分层
- 已有 SHAP 解释

因此目前更适合的定位是：

> 一篇具有较强方法学完整性和临床解释性的探索性内部验证研究。

但正式投稿时仍需补充或在局限性中说明：

1. 单中心/回顾性设计
2. 外部验证缺失
3. 测试集事件数有限
4. patient-level 高风险段校准仍有波动
5. 纳入排除标准、流程图和 EPV 论证需在正式手稿中完善

---

## 13 仓库中各脚本的论文角色

| 脚本 | 当前定位 | 建议论文角色 |
|---|---|---|
| `hyper_detect.py` | 固定 Landmark 二分类甲亢识别 | 主分析 1 |
| `relapse.py` | 滚动 Landmark 动态复发预警 | 主分析 2 / 方法学亮点 |
| `all2.py` | 固定 Landmark 三分类分流 | 扩展分析 / 补充材料 |
| `causal.py` | 剂量因果异质性分析 | 独立后续工作或另一篇论文 |
| `cluster.py` | 轨迹聚类 | 机制探索或另一篇论文 |

这也是当前最推荐的“一篇 paper 内部组织方式”。

---

## 14 技术术语简明解释

### 14.1 Landmark

在某个随访时间点上，仅使用该时点及之前的信息做预测，避免未来信息泄漏。

### 14.2 Long format

将“一个患者一行”的宽表，展开为“一个患者多个区间多行”的长表，以便研究状态随时间变化。

### 14.3 PR-AUC

适合阳性事件较少的不平衡问题。对于复发预警，比单纯 `ACC` 更有意义。

### 14.4 Brier score

衡量预测概率与真实结局之间偏差的指标，越低越好。

### 14.5 Calibration

衡量模型给出的风险概率是否接近真实发生概率。

### 14.6 DCA

Decision Curve Analysis，用于判断模型在不同阈值下是否真正带来临床净获益。

### 14.7 SHAP

用于解释特征如何影响模型输出，适合解释重要性与方向，不等于因果推断。

---

## 15 环境与运行

推荐环境：

```bash
conda activate med
```

主要依赖：

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `shap`
- `matplotlib`
- `seaborn`
- `lifelines`
- `econml`
- `openpyxl`
- `torch`
- `joblib`

常用运行命令：

```bash
python hyper_detect.py
python relapse.py
python all2.py
python causal.py
python cluster.py
```

主要输出目录：

- `hyper_detect_result/`：固定 Landmark 二分类结果
- `multistate_result/`：动态复发风险主结果
- `all_result/`：固定 Landmark 三分类扩展结果
- `causal_result/`：因果分析结果
- `Clustering_Results/`：轨迹聚类结果

---

## 16 从 README 到正式手稿的映射

如果后续要把本 README 继续压缩成论文初稿，可按如下映射：

| README 部分 | 论文部分 |
|---|---|
| 第 1-3 节 | Introduction + Clinical questions + Endpoints |
| 第 4 节 | Methods |
| 第 5-8 节 | Results |
| 第 9-12 节 | Discussion |
| 第 14 节 | 合作者术语说明，可不直接入文 |

当前版本的目标不是完整成文，而是把论文主线、分析角色、图表组织和医学叙事先统一下来。
