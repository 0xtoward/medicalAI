import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import copy
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Config (全局配置)
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    SEED = 42
    DEVICE = torch.device("cpu") 
    
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    }

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. 数据处理与特征工程
# ==========================================
def robust_impute(seq_df):
    """严谨插补：处理文本，拒绝未来穿越"""
    seq = seq_df.apply(pd.to_numeric, errors='coerce').astype(float)
    base_med = seq.iloc[:, 0].median()
    seq.iloc[:, 0] = seq.iloc[:, 0].fillna(base_med if not np.isnan(base_med) else 0)
    seq = seq.ffill(axis=1) # 只能向前填
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

def extract_flat_features_with_delta(X_s, X_d, seq_len):
    """
    🌟 核心：为树模型和 MLP 提供拍扁的特征，并【显式计算变化率(Delta)】
    """
    X_d_trunc = X_d[:, :seq_len, :] # 截断未来数据
    X_d_flat = X_d_trunc.reshape(X_d_trunc.shape[0], -1)
    
    deltas = []
    if seq_len >= 3: # 有 3个月数据 (0M, 1M, 3M)
        # 3个月 - 基线 (FT3, FT4, TSH)
        deltas.append(X_d_trunc[:, 2, :] - X_d_trunc[:, 0, :]) 
    if seq_len >= 4: # 有 6个月数据 (0M, 1M, 3M, 6M)
        # 6个月 - 基线
        deltas.append(X_d_trunc[:, 3, :] - X_d_trunc[:, 0, :])
        # 6个月 - 3个月
        deltas.append(X_d_trunc[:, 3, :] - X_d_trunc[:, 2, :])
        
    if deltas:
        delta_feat = np.concatenate(deltas, axis=1)
        X_feat = np.concatenate([X_s, X_d_flat, delta_feat], axis=1)
    else:
        X_feat = np.concatenate([X_s, X_d_flat], axis=1)
        
    return X_feat

# ==========================================
# 3. PyTorch MLP 定义及训练器
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class PureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
    def forward(self, x): return self.net(x)

def train_mlp_grid_search(X_tr, y_tr, X_val, y_val, weights):
    grid = [{'bs': 16, 'hid': 64}, {'bs': 32, 'hid': 128}]
    best_loss = float('inf')
    best_model = None
    
    ds_train = ThyroidDataset(X_tr, y_tr)
    X_val_t, y_val_t = torch.FloatTensor(X_val).to(Config.DEVICE), torch.LongTensor(y_val).to(Config.DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(Config.DEVICE))
    
    for p in grid:
        model = PureMLP(X_tr.shape[1], p['hid']).to(Config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        dl_train = DataLoader(ds_train, batch_size=p['bs'], shuffle=True, drop_last=True)
        
        for epoch in range(150): # MLP epochs
            model.train()
            for bx, by in dl_train:
                bx, by = bx.to(Config.DEVICE), by.to(Config.DEVICE)
                optimizer.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                v_loss = crit(model(X_val_t), y_val_t).item()
                if v_loss < best_loss:
                    best_loss = v_loss
                    best_model = copy.deepcopy(model)
                    
    return best_model

def predict_mlp(model, X_te):
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_te).to(Config.DEVICE))
        probs = torch.softmax(out, 1).cpu().numpy()
    return probs

# ==========================================
# 4. 树模型 Grid Search 训练器
# ==========================================
def train_tree_grid_search(model_class, param_grid, X_tr, y_tr, X_val, y_val, sample_weights):
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import log_loss
    
    best_loss = float('inf')
    best_model = None
    
    for params in ParameterGrid(param_grid):
        model = model_class(**params, random_state=Config.SEED)
        # RF 和 GBDT 支持 sample_weight 缓解类别不平衡
        sw = np.array([sample_weights[i] for i in y_tr])
        model.fit(X_tr, y_tr, sample_weight=sw)
        
        # 使用 Val 验证集上的 LogLoss 选择最佳超参，防止过拟合
        probs_val = model.predict_proba(X_val)
        v_loss = log_loss(y_val, probs_val)
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_model = model
            
    return best_model

# ==========================================
# 5. 评估器 (双头输出)
# ==========================================
def evaluate_probs(probs, labels, model_name):
    preds = probs.argmax(axis=1)
    acc3 = accuracy_score(labels, preds)
    
    # 🔴 甲亢(Fail=0) vs 非甲亢(1,2)
    y_hyper = (labels == 0).astype(int)
    pred_hyper = (preds == 0).astype(int)
    acc_hyper = accuracy_score(y_hyper, pred_hyper)
    try: auc_hyper = roc_auc_score(y_hyper, probs[:, 0])
    except: auc_hyper = 0.5

    # 🔵 甲减(Cure=2) vs 非甲减(0,1)
    y_hypo = (labels == 2).astype(int)
    pred_hypo = (preds == 2).astype(int)
    acc_hypo = accuracy_score(y_hypo, pred_hypo)
    try: auc_hypo = roc_auc_score(y_hypo, probs[:, 2])
    except: auc_hypo = 0.5
    
    print(f"[{model_name:<13}] 3分类Acc: {acc3:.3f} | 🔴 甲亢截获(AUC:{auc_hyper:.3f}/Acc:{acc_hyper:.3f}) | 🔵 甲减风险(AUC:{auc_hypo:.3f}/Acc:{acc_hypo:.3f})")

# ==========================================
# 6. 主执行实验 (Arena)
# ==========================================
def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*90}\n🚀 启动临床 Landmark 实验: {landmark_name}\n{'='*90}")
    X_s, X_d, y, pids = load_data()
    
    # 🌟 特征抽取：拍扁 + Delta 特征
    X_all = extract_flat_features_with_delta(X_s, X_d, seq_len)
    
    # 严格三段切分 (Train 60%, Val 20%, Test 20%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    tr_val_idx, test_idx = next(gss1.split(X_all, y, groups=pids))
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=Config.SEED)
    tr_idx_rel, val_idx_rel = next(gss2.split(X_all[tr_val_idx], y[tr_val_idx], groups=pids[tr_val_idx]))
    train_idx, val_idx = tr_val_idx[tr_idx_rel], tr_val_idx[val_idx_rel]
    
    # 🌟 防泄漏标准化
    scaler = StandardScaler().fit(X_all[train_idx])
    X_tr  = scaler.transform(X_all[train_idx])
    X_val = scaler.transform(X_all[val_idx])
    X_te  = scaler.transform(X_all[test_idx])
    
    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]
    
    # 类别权重
    counts = np.bincount(y_tr)
    weights = len(y_tr) / (3 * counts)
    
    # ---------------- 训练场 ----------------
    # 1. Random Forest (稳健基线)
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_leaf': [2, 4]}
    best_rf = train_tree_grid_search(RandomForestClassifier, rf_grid, X_tr, y_tr, X_val, y_val, weights)
    probs_rf = best_rf.predict_proba(X_te)
    
    # 2. GBDT (表格数据之王)
    gbdt_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    best_gbdt = train_tree_grid_search(GradientBoostingClassifier, gbdt_grid, X_tr, y_tr, X_val, y_val, weights)
    probs_gbdt = best_gbdt.predict_proba(X_te)
    
    # 3. Pure MLP (深度学习代表)
    best_mlp = train_mlp_grid_search(X_tr, y_tr, X_val, y_val, weights)
    probs_mlp = predict_mlp(best_mlp, X_te)
    
    # 4. Ensemble (MLP + GBDT 软投票)
    # 取概率均值，融合非线性与决策树优势
    probs_ensemble = (probs_gbdt + probs_mlp) / 2.0
    
    # ---------------- 评估打分 ----------------
    evaluate_probs(probs_rf, y_te, "Random Forest")
    evaluate_probs(probs_gbdt, y_te, "GBDT")
    evaluate_probs(probs_mlp, y_te, "Pure MLP")
    print("-" * 90)
    evaluate_probs(probs_ensemble, y_te, "Ensemble(G+M)")

if __name__ == "__main__":
    run_experiment("3-Month Assessment (仅使用 0, 1, 3 个月数据)", seq_len=3)
    run_experiment("6-Month Assessment (仅使用 0, 1, 3, 6 个月数据)", seq_len=4)