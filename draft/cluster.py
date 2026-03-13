import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from all import Config # 依赖上面的Config

warnings.filterwarnings("ignore")

TIME_POINTS = [0, 1, 3, 6, 12, 18, 24] 
TIME_NAMES = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
CLUSTER_COLORS = ["#E63946", "#2A9D8F", "#F4A261", "#264653", "#8AB17D"]

def robust_impute_cluster(seq_df: pd.DataFrame) -> np.ndarray:
    """🌟 严谨插补：处理文本报错，且禁止用 y 或未来填补过去"""
    seq = seq_df.apply(pd.to_numeric, errors='coerce').astype(float)
    base_med = seq.iloc[:, 0].median()
    seq.iloc[:, 0] = seq.iloc[:, 0].fillna(base_med if not np.isnan(base_med) else 0)
    seq = seq.ffill(axis=1) # 只能向前
    seq = seq.fillna(seq.median(axis=0)).fillna(0)
    return seq.values

def load_data():
    print(f"📦 正在加载并清洗数据 (无泄漏模式)...")
    df = pd.read_excel(Config.FILE_PATH, header=None, engine="openpyxl").iloc[2:]
    df["Patient_ID"] = df.iloc[:, Config.COL_IDX["ID"]].ffill()

    X_s = df.iloc[:, Config.COL_IDX["Static_Feats"]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    gt = df.iloc[:, 3].astype(str).str.strip()
    gender_vec = np.zeros(len(df), dtype=np.float32)
    gender_vec[(gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1)] = 1.0
    X_s[:, 0] = gender_vec

    y_raw = df.iloc[:, Config.COL_IDX["Outcome"]].apply(pd.to_numeric, errors="coerce")

    # 完全不碰 y，单纯填补
    ft3 = robust_impute_cluster(df.iloc[:, Config.COL_IDX["FT3_Sequence"]])
    ft4 = robust_impute_cluster(df.iloc[:, Config.COL_IDX["FT4_Sequence"]])
    tsh = robust_impute_cluster(df.iloc[:, Config.COL_IDX["TSH_Sequence"]])
    X_d = np.stack([ft3, ft4, tsh], axis=-1)

    valid_idx = y_raw.dropna().index
    take_idx = df.index.get_indexer(valid_idx)
    return X_s[take_idx], X_d[take_idx], y_raw.loc[valid_idx].to_numpy(dtype=int)

def extract_shape_features(X_s, X_d, mode):
    X_d_proc = X_d.copy()
    X_d_proc[:, :, 2] = np.log1p(np.clip(X_d_proc[:, :, 2], 0, None)) # TSH 压缩长尾
    
    feats = []
    if mode == "all3_shape":
        channels = [0, 1, 2]
        # 先验特征：TRAb 和 腺体重量
        feats.extend([X_s[:, 11].reshape(-1, 1), X_s[:, 6].reshape(-1, 1)])
    else:
        channels = [2] # tsh_shape

    for c in channels:
        seq = X_d_proc[:, :, c]
        base = seq[:, 0]
        d_early = seq[:, 2] - seq[:, 0]  # 0~3M
        d_mid = seq[:, 3] - seq[:, 2]    # 3~6M
        auc = np.trapz(seq, x=TIME_POINTS, axis=1)
        instab = np.max(seq, axis=1) - np.min(seq, axis=1)
        feats.append(np.column_stack([base, d_early, d_mid, auc, instab]))

    feat_matrix = np.concatenate(feats, axis=1)
    return StandardScaler().fit_transform(feat_matrix)

def auto_cluster(X):
    best = None
    for method in ["kmeans", "gmm"]:
        for k in [3, 4, 5]:
            if method == "kmeans":
                labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
            else:
                labels = GaussianMixture(n_components=k, random_state=42).fit_predict(X)
            
            # 约束：拒绝小于 5% 的无意义碎簇
            if np.min(np.bincount(labels)) / len(labels) < 0.05:
                continue

            sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            if best is None or sil > best['sil']:
                best = {'method': method, 'k': k, 'labels': labels, 'sil': sil}
    return best

def plot_3d_pca(out_file, X_feat, labels, title):
    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X_feat)
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    for c in sorted(np.unique(labels)):
        idx = (labels == c)
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], 
                   c=color, label=f"Subtype {c} (n={idx.sum()})", s=40, alpha=0.75, edgecolors='white')
                   
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} Var)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} Var)")
    ax.view_init(elev=20, azim=45) 
    plt.title(title, fontsize=14, pad=10)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(out_file, dpi=250, bbox_inches='tight')
    plt.close()

def main():
    outdir = Path("Clustering_Results")
    outdir.mkdir(exist_ok=True)
    X_s, X_d, y = load_data()
    
    for mode in ["tsh_shape", "all3_shape"]:
        print(f"\n▶ 正在聚类: {mode}")
        X_feat = extract_shape_features(X_s, X_d, mode)
        best = auto_cluster(X_feat)
        
        if not best:
            print("未能找到满足条件的聚类方案。")
            continue
            
        labels = best['labels']
        print(f"  ✅ 方案: {best['method'].upper()} (K={best['k']})")
        
        # 1. 绘制 3D 轨迹特征空间图
        plot_3d_pca(outdir / f"{mode}_3D_Scatter.png", X_feat, labels, f"{mode.upper()} Feature Space")
        
        # 2. 输出总结报表 (挂载结局)
        summary = []
        for c in sorted(np.unique(labels)):
            idx = labels == c
            summary.append({
                "Subtype": c, "Count": int(idx.sum()),
                "Fail_Rate(Y=1)": f"{np.mean(y[idx] == 1):.1%}",
                "Normal_Rate(Y=2)": f"{np.mean(y[idx] == 2):.1%}",
                "Cure_Rate(Y=3)": f"{np.mean(y[idx] == 3):.1%}",
            })
        pd.DataFrame(summary).to_csv(outdir / f"{mode}_Summary.csv", index=False)
        print(f"  📁 结果已保存至 {outdir.resolve()}")

if __name__ == "__main__":
    main()