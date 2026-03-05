# 甲亢 RAI 治疗后复发预测与因果分析

基于 1003 例 Graves 甲亢患者放射性碘（RAI）治疗后的纵向随访数据，利用机器学习和因果推断方法，预测治疗后复发风险并识别关键影响因素。

## 数据概况

- **样本量**：889 名患者，1003 条治疗记录（含复诊）
- **随访时间点**：治疗前(0M)、1M、3M、6M、12M、18M、24M
- **核心指标**：FT3、FT4、TSH、医生诊断评价（甲亢/正常/甲减）
- **基线特征**：性别、年龄、身高体重、突眼、甲状腺重量、RAI 3 日摄取率、TGAb、TPOAb、TRAb、剂量等

## 项目结构

```
├── multi-state.py          # 主分析：多状态转移 + 9 模型对比 + SHAP 解释
├── all2.py                 # 3/6 月疗效预测（RF/XGBoost/LightGBM + SHAP）
├── causal.py               # 因果推断：CausalForestDML 估计 CATE/ATE
├── cluster.py              # 时间序列聚类（K-Means on TSH/FT3/FT4 轨迹）
├── draft.md                # 文献阅读笔记
├── draft/                  # 早期探索脚本（RF、TabPFN、MLP、t-SNE 等）
│   ├── probe.py            # 患者轨迹分析（复发/甲减/稳定比例）
│   ├── state1.py           # 状态转移热力图（已合并入 multi-state.py）
│   └── ...
├── multistate_result/      # multi-state.py 输出图表
├── all_result/             # all2.py 输出图表
├── causal_result/          # causal.py 输出图表
├── Clustering_Results/     # cluster.py 输出图表
└── .gitignore
```

## 核心分析模块

### 1. 多状态复发预测 (`multi-state.py`)

将随访过程建模为 **甲亢 → 正常 → 甲减** 三状态转移系统，预测"当期正常的患者下一期是否复发为甲亢"。

**方法要点**：
- **时序验证**：按患者入组顺序 80/20 划分训练/测试集，杜绝未来信息泄漏
- **MissForest 填充**：使用 `IterativeImputer`（随机森林）联合填充缺失的实验室值和医生评价标签，仅在训练集上拟合
- **GroupKFold 调参**：`RandomizedSearchCV` 以患者 ID 分组，防止同一患者跨折泄漏
- **OOF 阈值选择**：在训练集内部通过 Out-of-Fold 预测最大化 F1 来确定分类阈值

**9 模型对比**：

| 模型 | 类型 |
|---|---|
| Logistic Regression | 线性（Pipeline + StandardScaler） |
| Random Forest | 集成-Bagging |
| AdaBoost | 集成-Boosting |
| GradientBoosting | 集成-Boosting（调参） |
| XGBoost | 集成-Boosting（调参） |
| LightGBM | 集成-Boosting |
| MLP | 神经网络（Pipeline + StandardScaler） |
| Cox PH | 生存分析（lifelines） |
| Stacking Ensemble | 元学习（LR+RF+GBC → LR） |

**输出**：状态转移热力图、多模型 Hazard 曲线、AUC/PR-AUC 柱状图、SHAP 特征重要性图

### 2. 疗效预测 (`all2.py`)

对 3 个月和 6 个月时间点分别建立二分类模型，预测患者是否仍为甲亢状态。使用 RF、XGBoost、LightGBM 三模型对比，输出 ROC 曲线和 SHAP 分析。

### 3. 因果推断 (`causal.py`)

使用 `econml.CausalForestDML` 估计不同剂量对治疗效果的条件平均处理效应（CATE），识别哪些患者亚群对剂量调整更敏感。输出倾向性得分分布、CATE 象限图、特征重要性图。

### 4. 轨迹聚类 (`cluster.py`)

对 TSH/FT3/FT4 时间序列轨迹进行 K-Means 聚类，发现不同恢复模式的患者亚型。输出 3D 散点图、TSH 轨迹面板、疗效堆叠柱状图。

## 环境

```bash
conda activate med  # Python 3.10
# 主要依赖：pandas, numpy, scikit-learn, xgboost, lightgbm, shap,
#           matplotlib, seaborn, econml, lifelines, openpyxl
```

## 运行

```bash
python multi-state.py       # 多状态复发预测（结果 → multistate_result/）
python all2.py              # 3/6 月疗效预测（结果 → all_result/）
python causal.py            # 因果推断分析（结果 → causal_result/）
python cluster.py           # 轨迹聚类（结果 → Clustering_Results/）
```
