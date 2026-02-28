import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# Output directory for all results
OUT_DIR = "./causal_result"
os.makedirs(OUT_DIR, exist_ok=True)

# Use default (English) font for matplotlib
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. Config & 数据加载 (仅限 Baseline 数据)
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    SEED = 42
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24], # 移除了Dose(25)，作为干预变量独立处理
    }
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos", 
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb", 
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife"]

np.random.seed(Config.SEED)

def parse_dose_string(val):
    """处理剂量字符串，如 '（15+12）' 或 '26+14'"""
    if pd.isna(val):
        return np.nan
    
    # 如果已经是数字，直接返回
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    
    # 尝试解析加法表达式
    s = str(val).strip()
    # 移除括号（中英文）
    s = s.replace('（', '').replace('）', '').replace('(', '').replace(')', '')
    
    # 尝试用 + 分割并求和
    if '+' in s:
        try:
            parts = [float(x.strip()) for x in s.split('+')]
            return sum(parts)
        except:
            pass
    
    return np.nan

def robust_base_impute(seq_df):
    """仅提取第 0 个月(Base)的数据并插补"""
    base_col = pd.to_numeric(seq_df.iloc[:, 0], errors='coerce').astype(float)
    return base_col.fillna(base_col.median() if not np.isnan(base_col.median()) else 0).values

def load_causal_data():
    print(f"📦 加载数据并构建因果框架...")
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    
    # 1. 协变量 X (混杂因素：基线静态 + 基线实验室指标)
    X_s = df.iloc[:, Config.COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    gt = df.iloc[:, 3].astype(str).str.strip()
    X_s[:, 0] = np.where((gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1), 1.0, 0.0)
    
    ft3_0 = robust_base_impute(df.iloc[:, [16]])
    ft4_0 = robust_base_impute(df.iloc[:, [17]])
    tsh_0 = np.log1p(np.clip(robust_base_impute(df.iloc[:, [18]]), 0, None))
    
    X = np.column_stack([X_s, ft3_0, ft4_0, tsh_0])
    feature_names = Config.STATIC_NAMES + ["FT3_0M", "FT4_0M", "log(TSH_0M)"]
    
    # 2. 干预变量 T (Treatment: Dose_Int 二值化)
    # 取出 Dose (25) 和 ThyroidW (9)
    dose_raw = df.iloc[:, 25]
    # 使用自定义解析函数处理字符串
    dose_parsed = dose_raw.apply(parse_dose_string)
    dose_median = dose_parsed.median()
    dose = dose_parsed.fillna(dose_median).values
    
    thyroid_w = X_s[:, Config.STATIC_NAMES.index("ThyroidW")] + 1e-5
    dose_int = dose / thyroid_w
    
    # 按照中位数切分：0 为低比活度(保守)，1 为高比活度(激进)
    T = (dose_int >= np.median(dose_int)).astype(int)
    
    # 3. 结局变量 Y (双目标)
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    valid_idx = y_raw.dropna().index
    y_raw = y_raw.loc[valid_idx].values
    
    # 过滤掉缺失结局的行
    take = df.index.get_indexer(valid_idx)
    X, T = X[take], T[take]
    
    Y_fail = (y_raw == 1).astype(int) # 是否持续甲亢
    Y_hypo = (y_raw == 3).astype(int) # 是否发生甲减
    
    return X, T, Y_fail, Y_hypo, feature_names

# ==========================================
# 2. 因果森林训练与 CATE 估计
# ==========================================
def train_causal_forest(X, T, Y, target_name):
    print(f"\n🌲 正在训练因果森林 (CausalForestDML) 目标: {target_name}...")
    
    # Y是二值的(0/1)，但DML期望连续的残差，所以用Regressor
    # T是二值的且明确标记为discrete_treatment
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=Config.SEED),  # 改为Regressor
        model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=Config.SEED),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=10,
        random_state=Config.SEED
    )
    
    # Y转为float以确保被视为连续值（虽然只有0和1）
    Y_continuous = Y.astype(float)
    
    # 拟合因果效应
    est.fit(Y_continuous, T, X=X, W=None)
    
    # 计算 CATE
    cate = est.effect(X) 
    
    ate = np.mean(cate)
    print(f"  👉 平均处理效应 (ATE): {ate:.4f}")
    if target_name == "Fail":
        print(f"     (解读: 总体而言，采用高比活度剂量可使治疗失败率平均变化 {ate*100:.1f}%)")
    else:
        print(f"     (解读: 总体而言，采用高比活度剂量可使过度甲减率平均变化 {ate*100:.1f}%)")
        
    return est, cate

# ==========================================
# 3. 临床决策矩阵 (四象限散点图)
# ==========================================
def plot_causal_quadrants(cate_fail, cate_hypo):
    print("\n📊 绘制基于反事实推演的个体化决策矩阵...")
    plt.figure(figsize=(10, 8))
    
    # 绘制散点
    plt.scatter(cate_fail, cate_hypo, alpha=0.6, c='royalblue', edgecolors='white', s=50)
    
    # 划分象限 (假设阈值：失败率下降至少 5% 为显著收益，甲减率上升超 10% 为高代价)
    # 注意：cate_fail 是负数才代表降低失败风险，越负越好！
    x_thresh = -0.05 
    y_thresh = 0.10  
    
    plt.axvline(x=x_thresh, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=y_thresh, color='gray', linestyle='--', alpha=0.7)
    
    # 象限标注 (English)
    plt.text(x_thresh - 0.02, y_thresh - 0.02, 'Q1: High benefit / Low cost\n(Strongly recommend high dose)', 
             ha='right', va='top', fontsize=12, color='green', fontweight='bold')
    
    plt.text(x_thresh + 0.02, y_thresh + 0.02, 'Q2: Low benefit / High cost\n(Caution, recommend conservative)', 
             ha='left', va='bottom', fontsize=12, color='red', fontweight='bold')
             
    plt.text(x_thresh - 0.02, y_thresh + 0.02, 'Q3: High benefit / High cost\n(Trade-off / Consider fractionation)', 
             ha='right', va='bottom', fontsize=12, color='orange', fontweight='bold')
             
    plt.text(x_thresh + 0.02, y_thresh - 0.02, 'Q4: Low benefit / Low cost\n(Conservative conventional dose)', 
             ha='left', va='top', fontsize=12, color='gray', fontweight='bold')

    plt.xlabel(r"$\Delta$ Failure probability (CATE_fail)" + "\n<-- Less failure with higher dose (good)", fontsize=12)
    plt.ylabel(r"$\Delta$ Hypothyroidism probability (CATE_hypo)" + "\nHigher dose increases hypothyroidism risk -->", fontsize=12)
    plt.title("Causal Patient Stratification: Counterfactual Dose Response", fontsize=14, pad=15)
    plt.grid(alpha=0.2)
    
    out_path = os.path.join(OUT_DIR, "Causal_Quadrant_Policy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

# ==========================================
# 4. 因果特征重要性 (谁决定了你对药敏不敏感？)
# ==========================================
def plot_causal_feature_importance(est_fail, X, feature_names):
    print("\n🔍 挖掘剂量异质性驱动因素 (Causal Feature Importance)...")
    # EconML 支持直接提取 Causal Forest 内部树的特征重要性
    importances = est_fail.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(8, 6))
    plt.title("What Drives the Heterogeneous Response to High Dose? (Failure Prevention)", fontsize=12)
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Causal Feature Importance")
    plt.grid(axis='x', alpha=0.3)
    out_path = os.path.join(OUT_DIR, "Causal_Feature_Importance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def main():
    print("="*80)
    print("🔮 启动平行宇宙推演：基于反事实框架的个体化剂量决策系统")
    print("="*80)
    
    X, T, Y_fail, Y_hypo, feat_names = load_causal_data()
    
    # 标准化特征 (因果推断中，特征的尺度对倾向性得分匹配有影响)
    X_scaled = StandardScaler().fit_transform(X)
    
    # 1. 计算加大剂量对“阻击失败”的因果效应
    est_fail, cate_fail = train_causal_forest(X_scaled, T, Y_fail, "Fail (持续甲亢)")
    
    # 2. 计算加大剂量对“引发甲减”的因果效应
    est_hypo, cate_hypo = train_causal_forest(X_scaled, T, Y_hypo, "Hypo (过度甲减)")
    
    # 3. 绘制临床决策象限图
    plot_causal_quadrants(cate_fail, cate_hypo)
    
    # 4. 看看是哪些特征在影响病人的“药敏性”
    plot_causal_feature_importance(est_fail, X_scaled, feat_names)
    
    print("\n🎉 运行完成！这是你论文中最具临床颠覆性的部分。")

if __name__ == "__main__":
    main()