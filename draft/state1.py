import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础配置 (复用你现有的逻辑)
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    OUT_DIR = Path("./multistate_result/")
    
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    }
    
    # 🔴 核心生化阈值 (用于定义状态)
    FT3_UPPER = 6.01   
    FT4_UPPER = 19.05  
    FT4_LOWER = 9.01   
    TSH_UPPER = 4.94   
    
    TIME_STAMPS = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
    STATE_NAMES = ["Hyper(甲亢)", "Normal(正常)", "Hypo(甲减)"]

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

def robust_impute(seq_df):
    seq = seq_df.apply(pd.to_numeric, errors='coerce').astype(float)
    base_med = seq.iloc[:, 0].median()
    seq.iloc[:, 0] = seq.iloc[:, 0].fillna(base_med if not np.isnan(base_med) else 0)
    seq = seq.ffill(axis=1)
    seq = seq.fillna(seq.median(axis=0)).fillna(0)
    return seq.values

def load_data():
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    # 1003.xlsx 含 100 万+ 空行，必须过滤：只保留有结局的有效记录
    outcome = pd.to_numeric(df.iloc[:, Config.COL_IDX['Outcome']], errors='coerce')
    df = df.loc[outcome.notna()].reset_index(drop=True)

    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()

    ft3 = robust_impute(df.iloc[:, Config.COL_IDX['FT3_Sequence']])
    ft4 = robust_impute(df.iloc[:, Config.COL_IDX['FT4_Sequence']])
    tsh = robust_impute(df.iloc[:, Config.COL_IDX['TSH_Sequence']])
    X_d = np.stack([ft3, ft4, tsh], axis=-1)

    n_unique = df['Patient_ID'].nunique()
    print(f"  有效治疗记录: {len(df)} (独立患者: {n_unique})")
    return X_d, df['Patient_ID'].values

# ==========================================
# 2. 状态推导引擎 (State Derivation Engine)
# ==========================================
def derive_states_matrix(X_d):
    """
    根据每个时间点的生化指标，推导出病人的离散状态 S(t)
    X_d shape: (N, 7, 3) -> FT3, FT4, TSH
    返回: S_matrix shape: (N, 7)
    0: Hyper(甲亢), 1: Normal(正常), 2: Hypo(甲减)
    """
    N, T, _ = X_d.shape
    S_matrix = np.ones((N, T), dtype=int) * 1 # 默认全为正常(1)
    
    for t in range(T):
        ft3_t = X_d[:, t, 0]
        ft4_t = X_d[:, t, 1]
        tsh_t = X_d[:, t, 2]
        
        # 规则1: 甲亢 (Hyper = 0) - FT3或FT4超上限
        hyper_mask = (ft3_t > Config.FT3_UPPER) | (ft4_t > Config.FT4_UPPER)
        S_matrix[hyper_mask, t] = 0
        
        # 规则2: 甲减 (Hypo = 2) - FT4低于下限 或 TSH超标 (优先覆盖甲亢误判)
        hypo_mask = (ft4_t < Config.FT4_LOWER) | (tsh_t > Config.TSH_UPPER)
        S_matrix[hypo_mask, t] = 2
        
    return S_matrix

# ==========================================
# 3. 转移矩阵计算与可视化 (Transition Matrix)
# ==========================================
def plot_transition_heatmaps(S_matrix):
    """绘制所有区间的 3x3 状态转移热力图"""
    N, T = S_matrix.shape
    
    # 汇总全局的转移情况
    total_transitions = np.zeros((3, 3), dtype=int)
    
    # 建立多子图画板
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i in range(T - 1):
        t_start, t_end = i, i + 1
        interval_name = f"{Config.TIME_STAMPS[t_start]} -> {Config.TIME_STAMPS[t_end]}"
        
        # 计算当前区间的 3x3 转移频次
        trans_mat = np.zeros((3, 3), dtype=int)
        for s_from in [0, 1, 2]:
            for s_to in [0, 1, 2]:
                count = np.sum((S_matrix[:, t_start] == s_from) & (S_matrix[:, t_end] == s_to))
                trans_mat[s_from, s_to] = count
                total_transitions[s_from, s_to] += count
        
        # 绘图
        ax = axes[i]
        sns.heatmap(trans_mat, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=Config.STATE_NAMES, yticklabels=Config.STATE_NAMES, ax=ax)
        ax.set_title(interval_name, fontsize=12)
        ax.set_xlabel("State To (t+1)")
        ax.set_ylabel("State From (t)")
        
        # 统计跳变与复发
        stable = np.trace(trans_mat)
        total = np.sum(trans_mat)
        relapse_count = trans_mat[1, 0] # Normal -> Hyper
        print(f"[{interval_name}] 总跳变率: {(total-stable)/total:.1%} | ⚠️ 复发(正常->甲亢)次数: {relapse_count}")

    plt.tight_layout()
    plt.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close()
    
    # 打印全局关键统计
    print("\n" + "="*50)
    print("🌍 全局核心发现 (Overall Findings)")
    print("="*50)
    print(f"总计观察到的状态转移次数: {np.sum(total_transitions)}")
    print(f"其中，【保持稳定】次数: {np.trace(total_transitions)}")
    print(f"其中，【复发 (Normal -> Hyper)】总次数: {total_transitions[1, 0]}")
    print(f"其中，【甲减治愈 (Hypo -> Normal)】总次数: {total_transitions[2, 1]}")
    print("="*50)

def main():
    print("="*60)
    print("🚀 阶段 1: 临床状态重构与多状态转移 (Multi-state) 分析")
    print("="*60)
    
    X_d, pids = load_data()
    print(f"✅ 成功加载动态序列数据，共 {len(pids)} 例患者。")
    
    # 1. 规则推导：把连续的激素指标变成 [0,1,2] 的离散状态
    S_matrix = derive_states_matrix(X_d)
    
    # 2. 分析各个时间段的跳变，并画图
    plot_transition_heatmaps(S_matrix)
    print(f"\n🎉 转移矩阵热力图已生成: {Config.OUT_DIR / 'Transition_Heatmaps.png'}")

if __name__ == "__main__":
    main()