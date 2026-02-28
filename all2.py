import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import shap
import copy
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Config & 目录设置
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    SEED = 42
    DEVICE = torch.device("cpu") 
    OUT_DIR = Path("./all_result/")
    
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    }
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos", 
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb", 
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose"]

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. 数据处理与特征工程
# ==========================================
def robust_impute(seq_df):
    seq = seq_df.apply(pd.to_numeric, errors='coerce').astype(float)
    base_med = seq.iloc[:, 0].median()
    seq.iloc[:, 0] = seq.iloc[:, 0].fillna(base_med if not np.isnan(base_med) else 0)
    seq = seq.ffill(axis=1)
    seq = seq.fillna(seq.median(axis=0)).fillna(0)
    return seq.values

def load_data():
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()
            
    X_s = df.iloc[:, Config.COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    gt = df.iloc[:, 3].astype(str).str.strip()
    gender_vec = np.zeros(len(df), dtype=np.float32)
    gender_vec[(gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1)] = 1.0
    X_s[:, 0] = gender_vec 
    
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    
    ft3 = robust_impute(df.iloc[:, Config.COL_IDX['FT3_Sequence']])
    ft4 = robust_impute(df.iloc[:, Config.COL_IDX['FT4_Sequence']])
    tsh = robust_impute(df.iloc[:, Config.COL_IDX['TSH_Sequence']])
    X_d = np.stack([ft3, ft4, tsh], axis=-1) 

    valid_idx = y_raw.dropna().index
    take = df.index.get_indexer(valid_idx)
    y = y_raw.loc[valid_idx].map({1: 0, 2: 1, 3: 2}).values.astype(int)
    return X_s[take], X_d[take], y, df.loc[valid_idx, 'Patient_ID'].values 

def extract_flat_features(X_s, X_d, seq_len):
    X_d_trunc = X_d[:, :seq_len, :] 
    X_d_flat = X_d_trunc.reshape(X_d_trunc.shape[0], -1)
    deltas = []
    feature_names = list(Config.STATIC_NAMES)
    
    times = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
    for t in range(seq_len):
        for h in ["FT3", "FT4", "TSH"]: feature_names.append(f"{h}_{times[t]}")
            
    if seq_len >= 3:
        deltas.append(X_d_trunc[:, 2, :] - X_d_trunc[:, 0, :])
        for h in ["FT3", "FT4", "TSH"]: feature_names.append(f"Δ{h}_3M-0M")
    if seq_len >= 4:
        deltas.append(X_d_trunc[:, 3, :] - X_d_trunc[:, 0, :])
        for h in ["FT3", "FT4", "TSH"]: feature_names.append(f"Δ{h}_6M-0M")
        deltas.append(X_d_trunc[:, 3, :] - X_d_trunc[:, 2, :])
        for h in ["FT3", "FT4", "TSH"]: feature_names.append(f"Δ{h}_6M-3M")
        
    X_feat = np.concatenate([X_s, X_d_flat, np.concatenate(deltas, axis=1)] if deltas else [X_s, X_d_flat], axis=1)
    return X_feat, feature_names

# ==========================================
# 3. 级联分类器 (Cascade Framework)
# ==========================================
class CascadeClassifier:
    """万能级联包装器：支持包入任何 sklearn 分类器"""
    def __init__(self, stage1_model, stage2_model):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        
    def fit(self, X, y):
        # Stage 1: 0(甲亢) vs 非0(正常+甲减)
        y_stage1 = (y != 0).astype(int)
        self.stage1.fit(X, y_stage1)
        
        # Stage 2: 在非甲亢样本中, 1(正常) vs 2(甲减)
        mask = (y != 0)
        X_s2, y_s2 = X[mask], y[mask]
        y_stage2 = (y_s2 == 2).astype(int) 
        if len(np.unique(y_stage2)) > 1:
            self.stage2.fit(X_s2, y_stage2)
        
    def predict_proba(self, X):
        p_non_hyper = self.stage1.predict_proba(X)[:, 1]
        p_hyper = 1 - p_non_hyper
        
        try: p_hypo_given_non = self.stage2.predict_proba(X)[:, 1]
        except: p_hypo_given_non = np.zeros(len(X))
        
        prob_normal = p_non_hyper * (1 - p_hypo_given_non)
        prob_hypo = p_non_hyper * p_hypo_given_non
        return np.column_stack([p_hyper, prob_normal, prob_hypo])

# ==========================================
# 4. MLP (支持隐层特征提取)
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hid=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.BatchNorm1d(hid // 2), nn.ReLU()
        )
        self.classifier = nn.Linear(hid // 2, 3)
        
    def forward(self, x): return self.classifier(self.feature_extractor(x))
    def get_features(self, x): return self.feature_extractor(x)

def train_mlp(X_tr, y_tr):
    model = FeatureMLP(X_tr.shape[1]).to(Config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.002)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(ThyroidDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    for _ in range(120):
        model.train()
        for bx, by in dl:
            opt.zero_grad()
            crit(model(bx.to(Config.DEVICE)), by.to(Config.DEVICE)).backward()
            opt.step()
    return model

# ==========================================
# 5. 评估与可视化
# ==========================================
def evaluate_probs(probs, labels, model_name):
    preds = probs.argmax(axis=1)
    acc3 = accuracy_score(labels, preds)
    
    y_hyper = (labels == 0).astype(int)
    pred_hyper = (preds == 0).astype(int)
    acc_hyper = accuracy_score(y_hyper, pred_hyper)
    auc_hyper = roc_auc_score(y_hyper, probs[:, 0]) if len(np.unique(y_hyper))>1 else 0.5
    
    y_hypo = (labels == 2).astype(int)
    pred_hypo = (preds == 2).astype(int)
    acc_hypo = accuracy_score(y_hypo, pred_hypo)
    auc_hypo = roc_auc_score(y_hypo, probs[:, 2]) if len(np.unique(y_hypo))>1 else 0.5
    
    print(f"[{model_name:<18}] 3分类ACC: {acc3:.3f} | 🔴 甲亢截获(AUC:{auc_hyper:.3f}/ACC:{acc_hyper:.3f}) | 🔵 甲减风险(AUC:{auc_hypo:.3f}/ACC:{acc_hypo:.3f})")
    return probs[:, 0]

def plot_roc_curves(y_te, roc_data_dict, landmark_name):
    y_hyper = (y_te == 0).astype(int)
    plt.figure(figsize=(8, 6))
    for name, probs in roc_data_dict.items():
        fpr, tpr, _ = roc_curve(y_hyper, probs)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_score(y_hyper, probs):.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Hyperthyroidism Interception ROC ({landmark_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.savefig(Config.OUT_DIR / f'ROC_{landmark_name[:4]}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap(cascade_model, X_tr, feat_names, title, filename):
    print(f"🔍 正在生成 SHAP 解释图: {title}...")
    explainer = shap.TreeExplainer(cascade_model.stage1)
    shap_values = explainer.shap_values(X_tr)
    # 兼容不同版本 sklearn/shap 输出
    if isinstance(shap_values, list): shap_values = shap_values[1] 
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_tr, feature_names=feat_names, show=False)
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(Config.OUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 6. 主执行
# ==========================================
def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*105}\n🚀 启动实验: {landmark_name}\n{'='*105}")
    X_s, X_d, y, pids = load_data()
    X_all, feat_names = extract_flat_features(X_s, X_d, seq_len)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    train_idx, test_idx = next(gss.split(X_all, y, groups=pids))
    
    scaler = StandardScaler().fit(X_all[train_idx])
    X_tr, X_te = scaler.transform(X_all[train_idx]), scaler.transform(X_all[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # --- 模型 1 & 2: 纯树模型级联 ---
    rf_cas = CascadeClassifier(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=Config.SEED),
                               RandomForestClassifier(n_estimators=100, max_depth=6, random_state=Config.SEED))
    rf_cas.fit(X_tr, y_tr)
    
    gbdt_cas = CascadeClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=Config.SEED),
                                 GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=Config.SEED))
    gbdt_cas.fit(X_tr, y_tr)
    
    # --- 模型 3: MLP (自带隐层输出功能) ---
    mlp = train_mlp(X_tr, y_tr)
    mlp.eval()
    with torch.no_grad():
        p_mlp = torch.softmax(mlp(torch.FloatTensor(X_te).to(Config.DEVICE)), 1).cpu().numpy()
        # 🔥 提取隐藏层特征 (Size: 32)
        H_tr = mlp.get_features(torch.FloatTensor(X_tr).to(Config.DEVICE)).cpu().numpy()
        H_te = mlp.get_features(torch.FloatTensor(X_te).to(Config.DEVICE)).cpu().numpy()
    
    # --- 模型 4: Hybrid Cascade (原始特征 + MLP隐层特征 -> GBDT级联) ---
    X_tr_hybrid = np.concatenate([X_tr, H_tr], axis=1)
    X_te_hybrid = np.concatenate([X_te, H_te], axis=1)
    hybrid_names = feat_names + [f"MLP_Node_{i}" for i in range(H_tr.shape[1])]
    
    hybrid_cas = CascadeClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=Config.SEED),
                                   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=Config.SEED))
    hybrid_cas.fit(X_tr_hybrid, y_tr)
    
    # --- 收集所有预测结果 ---
    p_rf = rf_cas.predict_proba(X_te)
    p_gbdt = gbdt_cas.predict_proba(X_te)
    p_hybrid = hybrid_cas.predict_proba(X_te_hybrid)
    p_ens_soft = (p_gbdt + p_mlp) / 2.0  # 传统软投票
    
    roc_data = {
        'Cascade RF': evaluate_probs(p_rf, y_te, "Cascade RF"),
        'Cascade GBDT': evaluate_probs(p_gbdt, y_te, "Cascade GBDT"),
        'Pure MLP': evaluate_probs(p_mlp, y_te, "Pure MLP"),
        '-'*10: np.zeros(len(y_te)), # 分隔符，不画ROC
        'Hybrid(MLP+GBDT)': evaluate_probs(p_hybrid, y_te, "Hybrid(Feat+GBDT)"),
        'Ensemble(Soft)': evaluate_probs(p_ens_soft, y_te, "Ensemble(Soft)")
    }
    
    # 清理画图用数据
    del roc_data['-'*10]
    plot_roc_curves(y_te, roc_data, landmark_name)
    
    # 为 3M 和 6M 均绘制核心模型 (Cascade GBDT) 的 SHAP 机制图
    plot_shap(gbdt_cas, X_tr, feat_names, 
              title=f"SHAP: Hyperthyroidism Interception ({landmark_name[:4]})", 
              filename=f"SHAP_{landmark_name[:4]}_Hyper.png")

if __name__ == "__main__":
    run_experiment("3-Month Assessment", seq_len=3)
    run_experiment("6-Month Assessment", seq_len=4)
    print(f"\n🎉 运行完毕！所有结果图表已保存至: {Config.OUT_DIR.resolve()}")