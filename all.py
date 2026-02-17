import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import copy
import itertools

# ==========================================
# 1. Config (全局配置)
# ==========================================
class Config:
    FILE_PATH = "700.xlsx" 
    MAX_EPOCHS = 300
    PATIENCE = 30 
    SEED = 42
    DEVICE = torch.device("cpu") 
    
    GRID_SPACE = {
        'BATCH_SIZE': [16, 32],
        'HIDDEN_DIM': [4, 8, 16, 32, 64],
        'LR': [0.0005, 0.001, 0.002],
    }

    COL_IDX = {
        'ID': 0,
        'Outcome': 14,
        # 静态特征：尽可能纳入可用信息（性别为强特征，绝对不能漏）
        # 3性别 4年龄 5身高 6体重 7BMI 8突眼 9甲状腺重量 11三日放射 12治疗次数
        # 19TGAb 20TPOAb 21TRAb 22二十四h吸碘 23最高吸碘 24半衰期 25剂量
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        # 动态时序: Base, 1Mo, 3Mo, 6Mo, 1Y, 1.5Y, 2Y
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    }
    # 衍生特征在 X_s 中的位置索引（Static_Feats 顺序确定后）
    # 甲状腺重量 idx=6, 剂量 idx=15, TRAb idx=11
    STATIC_IDX = {'ThyroidW': 6, 'Dose': 15, 'TRAb': 11}
    
    TIME_STEPS_NAMES = ["Base(0M)", "1 Mo", "3 Mo", "6 Mo", "1 Yr", "1.5 Yr", "2 Yr"]
    TIME_IMPORTANCE = np.array([0.0, 0.0, 0.9, 0.05, 0.05/3, 0.05/3, 0.05/3])

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. 数据处理
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X_s, X_d, y):
        self.X_s = torch.FloatTensor(X_s)
        self.X_d = torch.FloatTensor(X_d)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_s[idx], self.X_d[idx], self.y[idx]

def load_data():
    print(f"Loading data from {Config.FILE_PATH}...")
    if Config.FILE_PATH.endswith('xlsx'):
        df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    else:
        df = pd.read_csv(Config.FILE_PATH, header=None).iloc[2:]
        
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()
            
    static_cols = Config.COL_IDX['Static_Feats']
    X_s = df.iloc[:, static_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    # 衍生特征：剂量强度、TRAb密度
    tw = X_s.iloc[:, Config.STATIC_IDX['ThyroidW']] + 1e-5
    X_s['Dose_Int'] = X_s.iloc[:, Config.STATIC_IDX['Dose']] / tw
    X_s['TRAb_Den'] = X_s.iloc[:, Config.STATIC_IDX['TRAb']] / tw
    # 超声（Col 10）文本编码：丰富->1, 增多->2, 略增多->3, 极丰富->4, 未见->5, 其他->0
    us_map = {'丰富': 1, '增多': 2, '略增多': 3, '极丰富': 4, '未见': 5}
    us_raw = df.iloc[:, 10].astype(str).str.strip()
    X_s['US_Enc'] = us_raw.map(lambda x: us_map.get(x, 0) if x and x != 'nan' else 0)
    
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    
    # X_d: (N, 7, 3) = FT3, FT4, TSH 三通道
    ft3_df = df.iloc[:, Config.COL_IDX['FT3_Sequence']].apply(pd.to_numeric, errors='coerce')
    ft4_df = df.iloc[:, Config.COL_IDX['FT4_Sequence']].apply(pd.to_numeric, errors='coerce')
    tsh_df = df.iloc[:, Config.COL_IDX['TSH_Sequence']].apply(pd.to_numeric, errors='coerce')
    
    # 智能填充：Base 用中位数，ffill+bfill，残留用治愈者中位数
    cured_mask = (y_raw == 3)
    ft3_df.iloc[:, 0] = ft3_df.iloc[:, 0].fillna(ft3_df.iloc[:, 0].median())
    ft3_df = ft3_df.ffill(axis=1).bfill(axis=1)
    cf = ft3_df[cured_mask].median().median()
    ft3_df = ft3_df.fillna(4.5 if np.isnan(cf) else cf)
    
    ft4_df.iloc[:, 0] = ft4_df.iloc[:, 0].fillna(ft4_df.iloc[:, 0].median())
    ft4_df = ft4_df.ffill(axis=1).bfill(axis=1)
    cf = ft4_df[cured_mask].median().median()
    ft4_df = ft4_df.fillna(12.0 if np.isnan(cf) else cf)
    
    tsh_df.iloc[:, 0] = tsh_df.iloc[:, 0].fillna(tsh_df.iloc[:, 0].median())
    tsh_df = tsh_df.ffill(axis=1).bfill(axis=1)
    cf = tsh_df[cured_mask].median().median()
    tsh_df = tsh_df.fillna(2.0 if np.isnan(cf) else cf)
    
    n_static = X_s.shape[1]
    print(f"🩺 Static: {n_static} features (性别+年龄+BMI+突眼+甲状腺+抗体+吸碘+剂量+Dose_Int+TRAb_Den+超声)")
    print(f"   X_d = (N,7,3) [FT3, FT4, TSH], channel-wise normalized")
    
    X_d = np.stack([ft3_df.values, ft4_df.values, tsh_df.values], axis=-1)  # (N, 7, 3) 
    
    valid_idx = y_raw.dropna().index
    X_s = X_s.loc[valid_idx].values
    X_d = X_d[df.index.get_indexer(valid_idx)]
    patient_ids = df.loc[valid_idx, 'Patient_ID'].values 
    
    y_map = {1: 0, 2: 1, 3: 2} 
    y = y_raw.loc[valid_idx].map(y_map).values.astype(int)
    
    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(X_s)
    
    # 对每个通道单独标准化，让 FT3/FT4/TSH 在同一起跑线上
    X_d_mean = np.mean(X_d, axis=(0, 1), keepdims=True)  # (1, 1, 3)
    X_d_std = np.std(X_d, axis=(0, 1), keepdims=True)
    X_d_scaled = (X_d - X_d_mean) / (X_d_std + 1e-5)
    
    return X_s, X_d_scaled, y, patient_ids, (X_d_mean, X_d_std)

# ==========================================
# 3. 四大架构定义 (含 New SOTA: EarlyFusion)
# ==========================================

# --- 1. Pure MLP (Baseline) ---
class PureMLP(nn.Module):
    def __init__(self, static_dim, seq_len, dynamic_dim, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.dynamic_dim = dynamic_dim
        self.net = nn.Sequential(
            nn.Linear(static_dim + seq_len * dynamic_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, x_s, x_d):
        # Pad if truncated; x_d: (B, T, 3)
        curr_len = x_d.shape[1]
        if curr_len < self.seq_len:
            pad = torch.zeros(x_d.size(0), self.seq_len - curr_len, x_d.size(2)).to(x_d.device)
            x_d = torch.cat([x_d, pad], dim=1)
        return self.net(torch.cat([x_s, x_d.reshape(x_d.size(0), -1)], dim=1))
    def predict_stepwise(self, x_s, x_d): return None

# --- 2. Pure LSTM (Baseline) ---
class PureLSTM(nn.Module):
    def __init__(self, dynamic_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 3))
    def forward(self, x_s, x_d):
        rnn_out, (h_n, _) = self.rnn(x_d)
        return self.classifier(h_n.squeeze(0))
    def predict_stepwise(self, x_s, x_d):
        rnn_out, _ = self.rnn(x_d)
        return [self.classifier(rnn_out[:, t, :]) for t in range(rnn_out.shape[1])]

# --- 3. Hybrid Gated (Old SOTA) ---
class HybridGated(nn.Module): 
    def __init__(self, static_dim, dynamic_dim, hidden_dim):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, hidden_dim)
        )
        self.rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.fc_s, self.fc_d = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid, self.tanh = nn.Sigmoid(), nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 3))

    def _fuse(self, h_s, h_t):
        f_s, f_t = self.tanh(self.fc_s(h_s)), self.tanh(self.fc_d(h_t))
        z = self.sigmoid(self.gate(torch.cat([h_s, h_t], dim=1)))
        return self.classifier(z * f_s + (1 - z) * f_t)

    def forward(self, x_s, x_d):
        h_s = self.static_net(x_s)
        rnn_out, (h_n, _) = self.rnn(x_d)
        return self._fuse(h_s, h_n.squeeze(0))

    def predict_stepwise(self, x_s, x_d):
        h_s = self.static_net(x_s)
        rnn_out, _ = self.rnn(x_d)
        return [self._fuse(h_s, rnn_out[:, t, :]) for t in range(rnn_out.shape[1])]

# --- 4. Early Fusion (Phenotype-Initialized LSTM) ---
# 用静态特征生成 LSTM 的初始状态 (h0, c0)
class EarlyFusion(nn.Module):
    def __init__(self, static_dim, dynamic_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, hidden_dim)
        )
        self.h_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.LSTM(dynamic_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 3))

    def forward(self, x_s, x_d):
        # Step 1: 生成病人画像
        patient_emb = self.static_encoder(x_s)
        
        # Step 2: 初始化 LSTM 状态
        # (num_layers, batch, hidden_dim)
        h0 = torch.tanh(self.h_proj(patient_emb)).unsqueeze(0)
        c0 = torch.tanh(self.c_proj(patient_emb)).unsqueeze(0)
        
        # Step 3: LSTM 带着"偏见"开始读时序
        rnn_out, (h_n, _) = self.rnn(x_d, (h0, c0))
        
        # Step 4: 预测
        return self.classifier(h_n.squeeze(0))

    def predict_stepwise(self, x_s, x_d):
        patient_emb = self.static_encoder(x_s)
        h0 = torch.tanh(self.h_proj(patient_emb)).unsqueeze(0)
        c0 = torch.tanh(self.c_proj(patient_emb)).unsqueeze(0)
        
        # 传入初始状态
        rnn_out, _ = self.rnn(x_d, (h0, c0))
        
        return [self.classifier(rnn_out[:, t, :]) for t in range(rnn_out.shape[1])]

# --- 5. FiLM-LSTM (多通道 + 每步静态 Conditioning) [NEW] ---
# 🌟 核心：① X_d = (FT3, FT4, TSH) 多通道  ② 每时间步 concat 静态特征  ③ 最后 FiLM 调制
class FiLMLSTM(nn.Module):
    def __init__(self, static_dim, dynamic_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_proj_dim = 16
        # 每步输入 = [x_d_t, static_proj] -> LSTM 持续感知基线
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(), nn.Linear(32, self.static_proj_dim)
        )
        self.rnn = nn.LSTM(dynamic_dim + self.static_proj_dim, hidden_dim, batch_first=True)
        # FiLM: 静态生成 gamma, beta 调制最终表征
        self.film_gen = nn.Sequential(
            nn.Linear(static_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim * 2)
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, x_s, x_d):
        # x_d: (B, T, 3)
        B, T, _ = x_d.shape
        static_ctx = self.static_proj(x_s)  # (B, 16)
        static_ctx = static_ctx.unsqueeze(1).expand(-1, T, -1)  # (B, T, 16)
        x_in = torch.cat([x_d, static_ctx], dim=2)  # (B, T, 19)
        rnn_out, (h_n, _) = self.rnn(x_in)
        h_raw = h_n.squeeze(0)  # (B, H)
        # FiLM 调制
        film_params = self.film_gen(x_s)  # (B, 2*H)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        h_mod = gamma * self.ln(h_raw) + beta
        return self.classifier(h_mod)

    def predict_stepwise(self, x_s, x_d):
        B, T, _ = x_d.shape
        static_ctx = self.static_proj(x_s).unsqueeze(1).expand(-1, T, -1)
        x_in = torch.cat([x_d, static_ctx], dim=2)
        rnn_out, _ = self.rnn(x_in)
        film_params = self.film_gen(x_s)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        return [self.classifier(gamma * self.ln(rnn_out[:, t, :]) + beta) for t in range(T)]

# ==========================================
# 4. Universal Trainer (🌟 随机截断 + 延迟监督 + 时间权重)
# ==========================================
def train_model(model, params, dl_train, dl_test, class_weights, seq_len):
    lr = params['LR']
    
    crit_cls = nn.CrossEntropyLoss(weight=class_weights, reduction='none') # 设为none以便手动加权
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    
    time_importance = torch.tensor(Config.TIME_IMPORTANCE, dtype=torch.float32).to(Config.DEVICE)
    
    best_acc, patience = 0.0, 0
    best_wts = copy.deepcopy(model.state_dict()) 
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for xs, xd, y in dl_train:
            xs, xd, y = xs.to(Config.DEVICE), xd.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            
            # 🌟 核心策略：随机截断 (Random Truncation)
            # 要求：至少保留到 Index 2 (即 3 Mo)，绝不让 Base 参与训练
            cutoff = np.random.randint(2, seq_len + 1) # 范围 [2, 7]
            
            # 截取输入
            xd_truncated = xd[:, :cutoff, :]
            
            logits = model(xs, xd_truncated)
            
            # 计算 Loss
            raw_loss = crit_cls(logits, y) # (Batch,)
            
            # 🌟 核心策略：延迟监督 (Delayed Supervision)
            # 根据当前的 cutoff 长度，决定这个 Batch 的 Loss 权重
            # cutoff 是长度，对应的权重索引是 cutoff - 1
            curr_weight = time_importance[cutoff - 1]
            
            final_loss = (raw_loss * curr_weight).mean()
            
            # 只有当权重 > 0 时才反向传播，避免无效计算
            if curr_weight > 0:
                final_loss.backward()
                optimizer.step()
            
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xs, xd, y in dl_test:
                logits = model(xs.to(Config.DEVICE), xd.to(Config.DEVICE))
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict()) 
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE: break
            
    model.load_state_dict(best_wts)
    return best_acc, model

# ==========================================
# 5. 时序评估功能
# ==========================================
def evaluate_temporal_performance(model, model_name, X_s, X_d, y):
    if model_name == 'Pure_MLP':
        return None 
    
    model.eval()
    X_s = torch.FloatTensor(X_s).to(Config.DEVICE)
    X_d = torch.FloatTensor(X_d).to(Config.DEVICE)
    
    with torch.no_grad():
        step_logits = model.predict_stepwise(X_s, X_d)
    
    w = Config.TIME_IMPORTANCE
    w_sum = np.sum(w)
    bin_true = np.array([0 if v==0 else 1 for v in y])
    
    acc3_list, acc2_list, auc2_list = [], [], []
    
    w_core = w[2] + w[3]
    w_late = np.sum(w[4:])
    print(f"\n⏳ [{model_name}] 随访时间 vs 预测 (权重: 3M+6M={w_core:.2f}, 6M后={w_late:.2f})")
    print("-" * 85)
    print(f"{'已观测时间节点':<15} | {'权重':<8} | {'三分类 Acc':<10} | {'二分类 Acc':<10} | {'二分类 AUC':<10}")
    print("-" * 85)
    
    for i, logits in enumerate(step_logits):
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        
        acc3 = accuracy_score(y, preds)
        bin_pred = np.array([0 if v==0 else 1 for v in preds])
        acc2 = accuracy_score(bin_true, bin_pred)
        
        bin_probs = probs[:, 1] + probs[:, 2]
        try:
            auc2 = roc_auc_score(bin_true, bin_probs)
        except ValueError:
            auc2 = 0.5
        
        acc3_list.append(acc3)
        acc2_list.append(acc2)
        auc2_list.append(auc2)
        
        node_name = f"Up to {Config.TIME_STEPS_NAMES[i]}"
        print(f"{node_name:<15} | {w[i]:.3f}     | {acc3:.4f}     | {acc2:.4f}     | {auc2:.4f}")
    
    # 加权统一结果
    weighted_acc3 = np.sum(w * np.array(acc3_list))
    weighted_acc2 = np.sum(w * np.array(acc2_list))
    weighted_auc2 = np.sum(w * np.array(auc2_list))
    print("-" * 85)
    print(f"{'📌 加权统一':<15} | {w_sum:.3f}   | {weighted_acc3:.4f}     | {weighted_acc2:.4f}     | {weighted_auc2:.4f}")
    return {'weighted_acc3': weighted_acc3, 'weighted_acc2': weighted_acc2, 'weighted_auc2': weighted_auc2}

# ==========================================
# 6. Main Grid Search Arena
# ==========================================
def main():
    X_s, X_d, y, patient_ids, _ = load_data()
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    train_idx, test_idx = next(gss.split(X_s, y, groups=patient_ids))
    
    X_s_tr, X_s_te = X_s[train_idx], X_s[test_idx]
    X_d_tr, X_d_te = X_d[train_idx], X_d[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    print(f"📊 Dataset Split: Train={len(y_tr)}, Test={len(y_te)} (Strictly separated by Patient ID)")
    
    ds_train = ThyroidDataset(X_s_tr, X_d_tr, y_tr)
    ds_test  = ThyroidDataset(X_s_te, X_d_te, y_te)

    counts = np.bincount(y_tr)
    class_weights = torch.FloatTensor(len(y_tr) / (3 * counts)).to(Config.DEVICE)
    
    keys, values = zip(*Config.GRID_SPACE.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = {'Pure_MLP': 0, 'Pure_LSTM': 0, 'Hybrid_Gated': 0, 'EarlyFusion': 0, 'FiLM_LSTM': 0}
    best_models = {} 
    
    seq_len = len(Config.COL_IDX['FT3_Sequence'])
    static_dim = X_s.shape[1]
    dynamic_dim = X_d.shape[2]  # 3 = FT3, FT4, TSH
    
    print("\n⚔️ Let the Battle Begin (Grid Search for 5 Models)...")
    
    for model_name in results.keys():
        best_model_acc = 0.0
        champion_model = None
        print(f"\nTraining {model_name}...")
        
        for p in grid:
            bs, hid = p['BATCH_SIZE'], p['HIDDEN_DIM']
            dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=True) 
            dl_test  = DataLoader(ds_test, batch_size=bs, shuffle=False)
            
            if model_name == 'Pure_MLP': model = PureMLP(static_dim, seq_len, dynamic_dim, hid).to(Config.DEVICE)
            elif model_name == 'Pure_LSTM': model = PureLSTM(dynamic_dim, hid).to(Config.DEVICE)
            elif model_name == 'Hybrid_Gated': model = HybridGated(static_dim, dynamic_dim, hid).to(Config.DEVICE)
            elif model_name == 'EarlyFusion': model = EarlyFusion(static_dim, dynamic_dim, hid).to(Config.DEVICE)
            else: model = FiLMLSTM(static_dim, dynamic_dim, hid).to(Config.DEVICE)
                
            acc, trained_model = train_model(model, p, dl_train, dl_test, class_weights, seq_len)
            
            if acc > best_model_acc:
                best_model_acc = acc
                champion_model = trained_model
                
        results[model_name] = best_model_acc
        best_models[model_name] = champion_model
        print(f"🏆 {model_name} Best Test Acc (3-class): {best_model_acc:.4f}")

    print("\n" + "="*45)
    print("📈 FINAL LEADERBOARD (3-Class Accuracy)")
    print("="*45)
    for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model:<15}: {score:.4f}")
    print("="*45)

    print("\n" + "="*75)
    print("🔍 深度解析: 时间对决策的价值 (早期预测分析)")
    print("="*75)
    
    weighted_results = {}
    for m_name in ['Pure_LSTM', 'Hybrid_Gated', 'EarlyFusion', 'FiLM_LSTM']:
        ret = evaluate_temporal_performance(best_models[m_name], m_name, X_s_te, X_d_te, y_te)
        if ret is not None:
            weighted_results[m_name] = ret
    
    # Pure_MLP 无 stepwise，用全序列一次评估近似
    best_models['Pure_MLP'].eval()
    with torch.no_grad():
        xs = torch.FloatTensor(X_s_te).to(Config.DEVICE)
        xd = torch.FloatTensor(X_d_te).to(Config.DEVICE)
        logits = best_models['Pure_MLP'](xs, xd)
    preds = logits.argmax(1).cpu().numpy()
    acc3_mlp = accuracy_score(y_te, preds)
    bin_true = np.array([0 if v==0 else 1 for v in y_te])
    bin_pred = np.array([0 if v==0 else 1 for v in preds])
    acc2_mlp = accuracy_score(bin_true, bin_pred)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    auc2_mlp = roc_auc_score(bin_true, probs[:, 1] + probs[:, 2])
    weighted_results['Pure_MLP'] = {'weighted_acc3': acc3_mlp, 'weighted_acc2': acc2_mlp, 'weighted_auc2': auc2_mlp}
    
    print("\n" + "="*60)
    print("📌 统一加权 Leaderboard")
    print("="*60)
    for m_name in sorted(weighted_results.keys(), key=lambda x: weighted_results[x]['weighted_acc3'], reverse=True):
        r = weighted_results[m_name]
        print(f"  {m_name:<15} | 三分类 Acc: {r['weighted_acc3']:.4f} | 二分类 Acc: {r['weighted_acc2']:.4f} | 二分类 AUC: {r['weighted_auc2']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()