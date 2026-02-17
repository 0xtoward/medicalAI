import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor 
import time

# ==========================================
# 1. 加载数据
# ==========================================
file_path = "700.xlsx"
print(f"🔄 正在加载数据: {file_path} ...")

if file_path.endswith('xlsx'):
    df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
else:
    df_raw = pd.read_csv(file_path, header=None)

df = df_raw.iloc[2:].copy()

# 映射与清洗
col_map = {
    4: 'Age', 
    6: 'Weight', 
    9: 'Thyroid_Weight', 
    14: 'Outcome', 
    16: 'FT3_Base', 
    21: 'TRAb', 
    25: 'Dose'
}
df = df.rename(columns=col_map)[list(col_map.values())]
df = df.apply(pd.to_numeric, errors='coerce').dropna()

print(f"📊 数据清洗完毕，共 {len(df)} 条样本。")

# ==========================================
# 2. 定义变量
# ==========================================
# Y (Outcome): 1=成功(缓解/甲减), 0=失败(甲亢)
# 注意：我们要把 2(甲减) 和 3(正常) 都视为 1，把 1(甲亢) 视为 0
df['Y'] = df['Outcome'].apply(lambda x: 0 if x == 1 else 1) 

# T (Treatment): 医生给的药 (连续变量)
T = df['Dose']

# X (Covariates): 我们感兴趣的异质性特征 (用来画图分析的)
X = df[['Age', 'Weight', 'Thyroid_Weight', 'FT3_Base', 'TRAb']]

# W (Confounders): 所有的混杂因子 (用来去噪的)
W = X.copy()

# ==========================================
# 3. 训练 Causal Forest (因果森林)
# ==========================================
print("🌲 正在训练 Causal Forest (Double Machine Learning)...")
start_time = time.time()

# ⚠️ 关键修改：model_y 使用 Regressor 而不是 Classifier
# 原理：我们希望它预测的是 Y=1 的“概率”，而不是直接分类。
cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5),
    model_t=RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5),
    discrete_treatment=False, # 剂量是连续的
    n_estimators=300,        # 树越多越准
    min_samples_leaf=5,      # 保证每个叶子节点有足够样本
    random_state=42
)

cf.fit(df['Y'], T, X=X, W=W)

print(f"✅ 训练完成！耗时 {time.time() - start_time:.1f} 秒")

# ==========================================
# 4. 探索性分析：CATE (条件平均治疗效应)
# ==========================================
# 计算每个人的 CATE: 
# 含义: "如果再多给 1 mCi 药，这个人的治愈概率会增加/减少多少？"
treatment_effects = cf.effect(X)

# 统计描述
avg_effect = np.mean(treatment_effects)
print(f"\n📏 平均治疗效应 (ATE): {avg_effect:.4f}")
print("   (如果 > 0，说明整体来看加药是有好处的；如果接近 0，说明总体平衡)")

# --- 图1: 效应分布直方图 ---
plt.figure(figsize=(10, 6))
plt.hist(treatment_effects, bins=50, color='#1f77b4', edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='No Effect')
plt.axvline(avg_effect, color='orange', linestyle='-', linewidth=2, label=f'Avg Effect ({avg_effect:.3f})')
plt.title("Distribution of Individual Treatment Effects (CATE)", fontsize=14)
plt.xlabel("Effect of +1 mCi Dose on Success Probability", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("cate_distribution.png")
print("🖼️  效应分布图已保存为 cate_distribution.png")

# --- 图2: 异质性分析 (TRAb vs Effect) ---
# 这是你论文最关键的一张图：看 TRAb 是否决定了药物敏感度
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['TRAb'], treatment_effects, alpha=0.7, c=treatment_effects, cmap='coolwarm', edgecolors='k', linewidth=0.5)
plt.colorbar(sc, label='Treatment Effect')
plt.axhline(0, color='gray', linestyle='--')

# 添加趋势线
z = np.polyfit(df['TRAb'], treatment_effects, 1)
p = np.poly1d(z)
plt.plot(df['TRAb'], p(df['TRAb']), "r--", linewidth=2, label='Trend Line')

plt.xlabel("TRAb Level", fontsize=12)
plt.ylabel("Treatment Effect (Benefit of +Dose)", fontsize=12)
plt.title("Heterogeneity Analysis: TRAb vs. Dose Efficacy", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("trab_vs_effect.png")
print("🖼️  TRAb异质性图已保存为 trab_vs_effect.png")

# ==========================================
# 5. 生成“最佳剂量建议”
# ==========================================
print("\n🔍 重点关注：高 TRAb 患者的因果效应")
# 找出 TRAb 最高的 5 个病人
top_trab_df = df.nlargest(5, 'TRAb')
top_indices = top_trab_df.index

for idx in top_indices:
    patient = df.loc[idx]
    # 找到对应行的效应值 (需要用 iloc 索引)
    loc_idx = df.index.get_loc(idx)
    effect = treatment_effects[loc_idx]
    
    status = "需要加药 ⬆️" if effect > 0.01 else ("需要减药 ⬇️" if effect < -0.01 else "剂量合适 ✅")
    
    print(f"病人ID: {int(patient.name)} | TRAb: {patient['TRAb']:.1f} | 实际Dose: {patient['Dose']} | 效应: {effect:.4f} -> {status}")