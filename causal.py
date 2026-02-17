import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import StandardScaler

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

# 映射列名 (根据你的数据结构)
col_map = {
    4: 'Age', 
    6: 'Weight', 
    9: 'Thyroid_Weight', 
    14: 'Outcome', 
    16: 'FT3_Base', 
    21: 'TRAb', 
    22: 'Uptake', 
    25: 'Dose'
}

# 提取并重命名
df = df.rename(columns=col_map)[list(col_map.values())]
# 转数字并丢弃缺失值
df = df.apply(pd.to_numeric, errors='coerce').dropna()

print(f"📊 数据准备完毕: {len(df)} 条记录")

# ==========================================
# 2. 预处理：标准化 (重要!)
# ==========================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# ==========================================
# 3. 设置“禁忌边” (Tabu Edges) - 注入医学常识
# ==========================================
# 禁止：结果 -> 原因
# 禁止：后续变量 -> 前置变量
tabu_edges = [
    ("Outcome", "Age"), ("Outcome", "Weight"), ("Outcome", "Dose"), ("Outcome", "TRAb"), ("Outcome", "Thyroid_Weight"),
    ("Dose", "Age"), ("Dose", "FT3_Base"), ("Dose", "TRAb"), ("Dose", "Weight"),
    ("FT3_Base", "Age"), ("FT3_Base", "Dose"),
    ("TRAb", "Dose") # 暂时假设抗体不直接由剂量决定（剂量是果，抗体是因）
]

print("🧠 正在运行 NOTEARS 算法挖掘因果结构 (这可能需要几秒钟)...")

# ==========================================
# 4. 核心算法：NOTEARS
# ==========================================
# w_threshold: 权重阈值，小于 0.3 的弱连接会被剪断
sm = from_pandas(df_scaled, tabu_edges=tabu_edges, w_threshold=0.3) 

# ==========================================
# 5. 可视化 (使用 NetworkX 原生绘图，无需 graphviz)
# ==========================================
print("🖼️ 正在绘制因果图...")

plt.figure(figsize=(14, 10))

# 生成布局 (Spring Layout 会把强连接的节点拉近)
pos = nx.spring_layout(sm, k=0.6, iterations=50, seed=42)

# 1. 画节点
nx.draw_networkx_nodes(sm, pos, node_size=3000, node_color='lightblue', alpha=0.9, edgecolors='black')

# 2. 画标签
nx.draw_networkx_labels(sm, pos, font_size=11, font_weight='bold', font_family='sans-serif')

# 3. 画边 (根据权重调整粗细)
edges = sm.edges(data=True)
weights = [abs(d['weight']) * 5 for u, v, d in edges] # 边越粗代表关系越强
nx.draw_networkx_edges(sm, pos, arrowstyle='-|>', arrowsize=25, edge_color='gray', width=weights, node_size=3000)

# 4. 标注边的权重数值
edge_labels = {
    (u, v): f"{d['weight']:.2f}" 
    for u, v, d in edges 
}
nx.draw_networkx_edge_labels(sm, pos, edge_labels=edge_labels, font_color='red', font_size=10)

plt.title("Causal DAG Discovery (NOTEARS Algorithm)", fontsize=16)
plt.axis('off') # 关闭坐标轴
plt.tight_layout()

# 保存
save_name = "causal_dag.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"✅ 因果图已保存为 {save_name}")

# ==========================================
# 6. 打印文字版结果 (Top Edges)
# ==========================================
print("\n🔍 算法发现的强因果连接 (按强度排序):")
sorted_edges = sorted(sm.edges(data='weight'), key=lambda x: abs(x[2]), reverse=True)

for u, v, w in sorted_edges:
    direction = "正相关 (+)" if w > 0 else "负相关 (-)"
    print(f"  📌 {u} --> {v} : {w:.4f} [{direction}]")