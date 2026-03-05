# 5篇paper读后感

# 0x01  

https://www.sciencedirect.com/science/article/pii/S2666636724005876

> 学这个图

![图 1](https://ars.els-cdn.com/content/image/1-s2.0-S2666636724005876-gr1.jpg)

可以。你贴的这种 **“从临床问题→数据→建模→验证→落地工具”** 的大长图，在医学信息学/移植/肿瘤随访类论文里很常见，通常有两种放法：

1. **正文 Figure 1（多分面 A–E）**：作为“研究流程图/方法总览图”。只要排版后字体仍可读、信息不重复、期刊允许单图多 panel，一般没问题。
2. **Graphical Abstract（图形摘要）**：很多期刊（尤其 Elsevier 系）会单独收图形摘要；正文再放精简版方法图。该论文在 ScienceDirect 页面也明确有“Graphical abstract”。 

------

### 这张图具体说了什么（逐 panel 精读）

> 这张图本质是在回答：**怎么做到“随访过程中任意一天都能更新复发风险”**（dynamic prediction），以及他们用什么变量、什么验证方式证明它更好。

### A) Dynamic prediction 的目标

- **预测时点**：随访中的任意时刻（“time point of prediction”）。
- **预测目标**：从该时点起的 **1 年复发概率**（不是“终身风险”）。
- 关键点：动态模型允许把随访中新出现的信息（如生物标志物）纳入风险评估，而不是只用移植前的基线信息。

> 用一句人话：**今天复查完（有最新 WT1mRNA），立刻更新“未来 1 年复发概率”。**

### B) 数据准备（把变量分三类）

1. **移植前风险因子**：年龄、rDRI、预处理强度、移植次数、移植前 WT1mRNA 等。
2. **移植后当下的 WT1mRNA 值**（latest value）。
3. **移植后 WT1mRNA 的“动力学/趋势”**：把轨迹粗粒度成“上升/下降/不变”这种类别特征（kinetics）。

> 这一步很像你在 Graves 里把“当期 FT4/TSH + 变化量 ΔFT4/ΔTSH + 是否出现过某状态”做成特征。

### C) 模型开发（multiple landmarking + supermodel）

- 他们用 **landmarking supermodel**：在许多 landmark time 上反复建模/堆叠，把“到某时刻仍未复发的人”作为风险集，用截至该时刻可得的信息预测未来。
- 图里提到的 “ipl* model” 属于他们实现 landmark-supermodel 的技术细节（把不同 landmark 的信息整合成统一动态模型）。

并且他们做了 **三种模型对照**（非常关键）：

- **Pseudo-dynamic model**：只用移植前信息（不含移植后 WT1 的值/趋势）。
- **Dynamic model 1**：加入移植后 WT1mRNA 的值（但不加入动力学）。
- **Dynamic model 2**：同时加入移植后 WT1mRNA 的值 + 动力学（趋势）。

### D) 模型评估

- 按时间把队列分训练/测试（图中写“before and after 2017”，并提 TRIPOD 的思路）。
- 评估用 **time-dependent ROC-AUC**、校准等。

### E) 落地：交互式 web app

- 用动态模型做一个可交互工具，让临床输入最新 WT1mRNA 等信息，输出个体化 1 年复发概率曲线。

------

### “伪动态模型（pseudo-dynamic）”到底是啥？为什么不如“真动态”？

### 它是什么

**伪动态 ≈ 形式上在每个时点都能给一个风险，但“信息来源仍然是基线的”。**

更严格地说：

- 它在每个 landmark 时点会“重新计算”风险（因为你已经活到这个时点且未复发，风险集变了），
- 但协变量只有移植前（baseline）那一套，所以模型无法响应病人在随访中出现的新变化。

### 为什么通常会更差

1. **缺失最强信号**：复发往往在发生前一段时间，生物标志物（WT1mRNA）会出现上升或趋势改变；只用基线信息捕捉不到这种“临近事件的征兆”。
2. **区分度随时间衰减**：很多基线风险因子对“早期风险”有效，但对“已经平稳度过一段时间后的未来风险”解释力下降；而动态 biomarker 恰恰在后期更有判别力。这也是他们报告中“pseudo-dynamic 的 AUC 随时间下降、真动态更稳定”的核心原因。
3. **动态模型避免把“未来才会出现的信息”硬塞进基线**：landmarking 的框架天然保证“只用预测时点之前的信息”，避免所谓 immortal time / look-ahead 的结构性偏差（至少在设计层面）。

> 对照到你 Graves：
>  **伪动态** = 只用基线 TRAb、甲状腺重量、剂量等做“随时间更新风险”；
>  **真动态** = 在每次随访把最新 FT4/TSH 及其变化趋势加进来，预测下一窗口 E→H（复发转移）概率。

------

### 你写自己 paper 时这张“大长图”怎么借鉴？

你完全可以画一张同构图（A–E），把：

- A：预测目标改成 “预测下一随访窗口 Normal→Hyper 的转移概率 / 或未来 6 个月复发概率”
- B：三类变量（基线 + 当期甲功 + 动量/历史状态）
- C：你的离散时间 hazard / multi-state（以及 LR vs Tree 的对照）
- D：严格时间外推 + patient-level CV + 校准
- E：一个简单 demo（哪怕不是 web app，也可以是 risk score + 曲线）

如果你把你当前模型的**预测窗口定义**（例如 3→6M 是预测目标还是所有窗口统一建模）说一句，我可以把你这张图的每个 panel 该写哪一句、该放哪三个变量，直接给你一个可照抄的“Figure 1 文案草稿”。











# 0x02  

> 学这个画像(修改我们的聚类)

## 1) paper2 的“患者画像”到底是什么？——不是聚类、不是 data-driven 分层
paper2 在 Methods 里写得很明确：他们构造了两个“better / worse” profile 来可视化 RSF 预测的累计发生风险（CIF），**这些 profile 不是通过数据驱动分层（not derived through data-driven stratification）**，而是**手工根据模型最重要变量组合出来的对照病例**。

- 他们先用 RSF 的 **minimal depth** 找到“最重要的变量”（更早出现在树根的变量更重要）。
- 然后人为设定两套患者特征：一套低风险（better），一套高风险（worse），并在 landmark=1/3/5 年时分别画出三类事件（复发、癌死、他因死）的 CIF 曲线。

**所以你看到的“画像”本质是：解释图（illustrative profiles），不是新的研究分组方法。**

------

## 2) 你要不要像他一样做“画像研究”？

### 建议：做，但把它定位成 **解释与展示**，不要当成核心方法学贡献

理由很简单：你现在的工作读者最难理解的是——

> “模型输出的区间复发概率/ hazard 怎么转化成临床可读的随访策略？”

paper2 用“better/worse profile”把模型输出变成临床直觉能读懂的曲线，这一步非常值钱。

对你来说，你可以做**两种更适配 Graves 的展示**（比聚类更稳）：

### A) 画像（手工极端病例）——最像 paper2

- 选 2–3 个 landmark（比如 1M/3M/6M）
- 构造 **低风险画像** vs **高风险画像**
  - 低风险：小腺体、TRAb 低、FT3/FT4 低正常、TSH 回升、Time_in_normal 较长、无 Ever_hypo/或稳定等
  - 高风险：大腺体、TRAb 高、FT3_current 偏高（仍在“正常态风险集”里但靠边界）、TSH 偏低/变化不利、Time_in_normal 短、既往出现过高波动等
- 画：在每个 landmark 后“**下一个区间**复发概率”或“未来 6 个月复发概率”的曲线/柱状图（你现在 hazard curve 已经有雏形）

> 这跟 paper2 的 CIF 图功能一致：展示“同一时间点下不同画像的风险差异”。

### B) 风险分层（data-driven，但不需要聚类）——比聚类更可控

- 在 test 队列里，用模型预测概率把区间记录分成 **低/中/高风险三分位或四分位**
- 对每一组计算真实 E→H 的经验发生率（empirical hazard）
- 画：风险组 vs 实际复发率（这一步对审稿人非常有说服力，因为它直接检验“模型排序是否有临床意义”）

这比聚类更干净，因为你分组**直接围绕你的预测目标**，而不是额外引入一个“聚类标签”。

你现在要做的“风险分层”这一步，目标不是再做一个模型，而是把模型的输出变成**临床可读、可解释、可行动（actionable）**的结果页。亮眼的关键是：**分层要和“决策/随访策略”挂钩，并且要证明分层后的风险在时间窗上真的拉开**（discrimination）且概率不要太离谱（calibration）。

下面给你一套“论文级”的做法，按**从易到硬、从展示到统计**组织。

------

## 1) 先把你分层的对象说清楚：你是“区间级”风险分层

你模型预测的是
$$
\hat p_{i,k}=P(\text{在区间 }(t_k,t_{k+1}] \text{发生 }E\to H \mid S_{i,k}=E, X_{i,\le k})
$$
也就是**下一随访窗口复发转移概率**（one-step hazard）。

所以风险分层有两种合理单位：

- **区间级分层（推荐主文）**：每条区间记录一个风险，直接对应“下次复诊前要不要加密随访”。
- **患者级分层（补充材料）**：把同一患者在多个区间的风险聚合（比如最大值/最近一次/加权平均），用于“个体随访计划”。

------

## 2) 最亮眼的主图：Risk group vs Observed hazard（带 CI）

### 做法

在 **test 队列**里，把所有区间记录按预测概率 $\hat p$ 分成 3 或 4 组（推荐 4 组：Q1–Q4）：

- Q1：最低风险
- Q4：最高风险

对每组、每个随访窗口（1→3、3→6、6→12、…）分别算：

- **经验 hazard（真实复发率）**：
  $$
  \hat h = \frac{\#(E\to H)}{\#(E\text{ at start})}
  $$

- **95% CI**：二项分布 CI（Wilson 更稳）

### 图长什么样

两种画法都行，推荐更“审稿人爱看”的：

- **分面图（facet by interval）**：每个 interval 一张小图，横轴风险组 Q1–Q4，纵轴经验 hazard，画点+误差棒
- 或者 **热力图**：行=interval，列=risk group，格子颜色=经验 hazard，格子里写事件数/在险数

### 为什么亮眼

它直接回答三个审稿人问题：

1. 你说“高风险”真的更高吗？（单调性）
2. 这个分层对早期窗口/后期窗口都成立吗？（时间窗异质性）
3. 高风险组的绝对风险是多少？（能不能行动）

------

## 3) 校准：别只画 hazard curve 平均值，做“grouped calibration”

你现在有 hazard curve（按 interval 平均），但更强的是：

### Grouped calibration（推荐）

在 test 集，把 $\hat p$ 分成 10 组（deciles）或 5 组（quintiles）：

- x 轴：每组的平均预测概率 $\bar p_g$
- y 轴：每组的观测概率 $\hat h_g$（同样用二项CI）
- 画 45° 线

这就是最标准的“predicted vs observed”校准图思路。

> 你要强调一句：模型可能 discrimination 好（AUC/AP高）但 calibration 仍可能偏，校准是必须报告的。

------

## 4) “可行动”：Decision Curve Analysis（DCA）把风险阈值变成随访策略

风险分层如果只停留在“高/低”，还不够亮。要把它变成“临床决策”：

- 设一个阈值 $p_t$：如果 $\hat p \ge p_t$，就触发**更密随访/复查甲功/提前干预评估**。
- DCA 计算不同 $p_t$ 下的 **net benefit**，并和 “treat-all / treat-none” 比较。

DCA 的优势是：它能回答

> “用模型来触发干预，整体是赚还是亏？”
>  这比 AUC 更贴临床。

------

## 5) 你该怎么写这一段 Results（可直接套用的句式）

- “We stratified test-set intervals into quartiles based on predicted relapse transition risk.”
- “Observed relapse hazard increased monotonically across risk strata, with the highest quartile showing X-fold higher hazard than the lowest.”
- “Calibration plots comparing predicted vs observed relapse risk demonstrated [good/acceptable] agreement.”
- “Decision curve analysis suggested clinical utility for threshold probabilities between a and b.”

------

## 6) 具体到你数据/代码：你已经有的东西怎么改最省事

你现在已经输出了：

- test 队列每条区间的 `proba`
- 每条区间有 `Interval_Name`, `Y_Relapse`

最小增量就是：

1. 在 test 集加一列 `risk_bin = pd.qcut(proba, q=4, labels=["Q1","Q2","Q3","Q4"])`
2. `groupby(["Interval_Name","risk_bin"]).agg(N, Events, Hazard=mean)`
3. 画 facet/heatmap + Wilson CI
4. 再做 decile calibration plot
5. （可选）DCA：遍历阈值 $p_t$ 算 net benefit

------

## 7) 你要“最亮眼”的一页图表组合（建议作为 Figure 2）

**Figure 2A**：各 interval 的经验 hazard（你已有）
 **Figure 2B**：test 集按 Q1–Q4 风险分层后的经验 hazard（带 CI，分 interval）
 **Figure 2C（可选）**：grouped calibration（deciles predicted vs observed）
 **Figure 2D（可选）**：DCA（net benefit vs threshold）

这样审稿人一眼就懂：
 “你不仅能预测，还能把它变成可用的随访阈值工具。”

------

如果你把你当前 test 集里每个 interval 的样本量/事件数（你图上已有）发我一行（例如 `1→3: 67/16` 这种），我可以直接帮你定一个**临床阈值区间**（比如 0.10–0.20 最合理），并告诉你 DCA 该重点展示哪些阈值段。

## 4) 你现在 DCA 的“决策窗”应该选什么？

建议把阈值范围限定在一个临床可辩护的区间，比如：

- **0.05–0.30**（5% 到 30%）

理由：

- 你的区间事件率大约 8–10%，在这个附近做阈值分析最有意义；
- 太小（<2%）会变成几乎全干预；
- 太大（>50%）基本没人触发，曲线没有解读价值。

这是 DCA 教程里常见的做法：只展示“临床可能采用”的阈值区间。

------

## 5) 最务实的建议

- **主文 DCA：用“加强随访/复查”作为干预**（最稳、最好辩护）。
- “加大剂量/再治疗策略”放到 Discussion 里作为未来工作：需要 target trial / 因果设计支持，不在本文直接做推荐。








# 0x03  

>  要不要探索多状态

https://pmc.ncbi.nlm.nih.gov/articles/PMC11441995/

要做也能做  

现在我们只重点做了1）跟师妹那篇一样的用三个月的数据预测最终的结果：治好（包括甲减）or依然甲亢，结果比她的好 79%>74%；      2） “landmark预测甲亢复发”，在这个任务中1）的模型被碾压

如果不做的理由：后面不太好写，一篇文章放了太多东西；结果一定不会太好看

# 0x04

> 我记得还有个 TRIPOD ？ 也一定要引用

这篇文章的核心不是“又一个模型”，而是**把动态预测里一个常见但隐蔽的评估/训练错误正式命名并证明危害**：**relaxed landmarking**。 

### 1) 两个概念（用人话）

- **Strict landmarking**：
   在某个 landmark time $s$，**训练集和测试集都要先“landmark 化”**：只保留在 $s$ 时刻仍然“在风险集”的人（例如尚未发生事件），并且训练时只能用 **$\le s$** 的纵向信息。 
- **Relaxed landmarking**：
   **只把测试集截断到 $s$**；但训练模型时仍然用训练集中个体的**完整纵向轨迹**（包含 $>s$ 的未来观测）。这意味着“建模时使用的信息 > 预测时能拿到的信息”。 

> 你之前在 MissForest/MICE 里“用全时间点一起补全”本质上也会制造这种“未来信息流入当前”的问题（即使 train/test 不泄漏）。

### 2) 为什么 relaxed 会出事

他们明确指出 relaxed 的训练集会变得“不像你真正要解决的预测问题”：

- 会把**已经在 $s$ 前发生事件**的人也纳入训练（strict 会剔除），导致训练人群不代表“在 $s$ 仍在风险集的人”。 
- 更关键：训练时会出现“用未来观测去帮助估计/总结过去轨迹”（他们在讨论里点名这种机制）。 

### 3) 他们展示的经验规律（你可以直接借来写）

- 在**小 landmark（早期）**时 strict/relaxed 可能差不多；
- **landmark 越往后**，strict 的优势越明显，因为 relaxed 的训练集与真实风险集差距越来越大。 
- 他们用 **time-dependent AUC、Brier/MSE** 等随 landmark/预测窗变化的曲线展示这一点。



# 0x05

## 2) 你能直接照抄的校准方法：**predicted vs observed risk curve（分箱校准）**

Paper E 给的 BLR-IPCW（binary logistic recalibration + 权重）在你这里的“最小版本”就是：

1. 在 test 风险集里拿到每条区间记录的预测概率 $\hat p$
2. 按 $\hat p$ 分成 5–10 个分位组（quintile/decile）
3. 对每组算：
   - 组内平均预测：$\bar p_g$
   - 组内真实发生率：$\hat h_g=\frac{\sum Y}{n}$
   - 置信区间（Wilson 比较稳）
4. 画图：横轴 $\bar p_g$，纵轴 $\hat h_g$，加 45° 线

这就是最实用的“校准曲线”。Paper E 的贡献在于给你一个“为什么要这么做”的方法学背书：它们把这个框架推广到 multi-state，但二元情况就是它的子集。

> 你现在的 hazard curve（按 interval 平均）更多是在看“时间窗非平稳性”；
>  分箱校准是在看“概率值本身可信不可信”。

------

## 3) 如果你担心缺失/失访：借鉴 Paper E 的 **IPCW 思路**

你如果满足以下任何一个条件，就建议加 IPCW（至少做敏感性分析）：

- 有不少区间的 $S_{k+1}$ 缺失（你现在用 MissForest 填标签其实是回避这个问题）
- 怀疑缺失并非随机（例如复发/再治疗后不回访）

Paper E 的套路是：用权重修正删失，让校准不被“只剩下好随访的人”偏倚。你的二元版是：

- 定义删失指示：$C_{i,k}=1$ 表示该区间终点 $k+1$ 有观测（不缺失）
- 拟合一个模型估计 $P(C_{i,k}=1 \mid \text{可用协变量})$
- 权重 $w_{i,k}=1/\hat P(C_{i,k}=1|\cdot)$
- 用这些权重去算每个分箱的观测率（或拟合“观察率 vs $\hat p$”的加权逻辑回归）

这就是 BLR-IPCW 的二元化版本。Paper E 对“权重模型很关键”的提醒也可以直接借来写局限/敏感性分析。
