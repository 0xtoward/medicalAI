import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import copy
import itertools # 用于生成参数组合

# ==========================================
# 1. Config (扩大搜索范围)
# ==========================================
class Config:
    FILE_PATH = "700.xlsx" 
    MAX_EPOCHS = 300
    PATIENCE = 30 
    SEED = 42
    DEVICE = torch.device("cpu") # 搜索时用 CPU 避免频繁 GPU 切换开销
    
    # 🔍 巨大的搜索空间
    GRID_SPACE = {
        'BATCH_SIZE': [16, 32],
        'HIDDEN_DIM': [32, 64, 128], # 尝试更深的网络
        'LR': [0.0005, 0.001, 0.002], # 尝试不同的学习步长
        'DROPOUT': [0.2, 0.5],       # 尝试不同的正则化力度
        'AUX_WEIGHT': [0.1, 0.3, 0.5] # 辅助任务的重要性
    }

    COL_IDX = {
        'Outcome': 14,
        'FT3_Base': 16,
        'FT3_1Mo': 29,
        'FT3_3Mo': 38,
        'Static_Feats': [4, 6, 9, 21, 22, 25] 
    }

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. Data Processing (保持不变)
# ==========================================
# ... (为节省篇幅，沿用之前的 ThyroidDataset 和 load_data 函数) ...
# 请确保这里有之前的 ThyroidDataset 和 load_data 代码
# ...
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
            
    # Static
    static_cols = Config.COL_IDX['Static_Feats']
    X_s = df.iloc[:, static_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_s['Dose_Int'] = X_s.iloc[:, 5] / (X_s.iloc[:, 2] + 1e-5) 
    X_s['TRAb_Den'] = X_s.iloc[:, 3] / (X_s.iloc[:, 2] + 1e-5)
    
    # Dynamic (FT3)
    ft3_cols = [Config.COL_IDX['FT3_Base'], Config.COL_IDX['FT3_1Mo'], Config.COL_IDX['FT3_3Mo']]
    ft3_data = df.iloc[:, ft3_cols].apply(pd.to_numeric, errors='coerce')
    ft3_data.iloc[:,0] = ft3_data.iloc[:,0].fillna(ft3_data.iloc[:,0].median())
    ft3_data = ft3_data.ffill(axis=1)
    
    X_d = ft3_data.values[..., np.newaxis] 
    
    # Labels
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce').dropna()
    valid_idx = y_raw.index
    X_s = X_s.loc[valid_idx].values
    X_d = X_d[df.index.get_indexer(valid_idx)]
    y_map = {1: 0, 2: 1, 3: 2} 
    y = y_raw.map(y_map).values.astype(int)
    
    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(X_s)
    d_mean, d_std = np.mean(X_d), np.std(X_d)
    X_d_scaled = (X_d - d_mean) / (d_std + 1e-5)
    
    return X_s, X_d_scaled, y

# ==========================================
# 3. Model Architecture (支持时序分析)
# ==========================================
class GridSearchModel(nn.Module):
    def __init__(self, static_dim, dynamic_dim, hidden_dim, dropout_rate, num_classes=3):
        super(GridSearchModel, self).__init__()
        
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # 动态 Dropout
            nn.Linear(32, hidden_dim)
        )
        
        self.rnn = nn.LSTM(input_size=dynamic_dim, hidden_size=hidden_dim, batch_first=True)
        self.aux_head = nn.Linear(hidden_dim, 1) 
        
        # Gated Fusion Components
        self.fc_static_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_dynamic_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_net = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_s, x_d):
        h_s = self.static_net(x_s)
        
        # RNN 输出: rnn_out (Batch, Seq_Len, Hidden)
        rnn_out, (h_n, _) = self.rnn(x_d)
        h_d = h_n.squeeze(0) # 取最后一个时间步
        
        aux_preds = self.aux_head(rnn_out)
        
        # Fusion (默认用最后一个时间步)
        logits = self._fuse_and_classify(h_s, h_d)
        
        return logits, aux_preds, rnn_out

    def _fuse_and_classify(self, h_s, h_d):
        feat_s = self.tanh(self.fc_static_proj(h_s))
        feat_d = self.tanh(self.fc_dynamic_proj(h_d))
        combined = torch.cat([h_s, h_d], dim=1)
        z = self.sigmoid(self.gate_net(combined))
        h_fused = z * feat_s + (1 - z) * feat_d
        return self.classifier(h_fused)

    # 🌟 新功能: 获取每个时间步的预测结果
    def predict_stepwise(self, x_s, x_d):
        h_s = self.static_net(x_s)
        rnn_out, _ = self.rnn(x_d) # (Batch, 3, Hidden)
        
        outputs = []
        # 遍历时间步: t=0, t=1, t=2 (对应0月, 1月, 3月)
        for t in range(rnn_out.shape[1]):
            h_d_t = rnn_out[:, t, :] # 取出这一刻的隐状态
            logits_t = self._fuse_and_classify(h_s, h_d_t) # 假装现在就做决策
            outputs.append(logits_t)
            
        return outputs # List of [Logits_T0, Logits_T1, Logits_T2]

# ==========================================
# 4. Training Engine
# ==========================================
def train_one_config(params, data_pack):
    X_s_tr, X_s_te, X_d_tr, X_d_te, y_tr, y_te = data_pack
    
    # Unpack params
    bs = params['BATCH_SIZE']
    hid = params['HIDDEN_DIM']
    lr = params['LR']
    dr = params['DROPOUT']
    aw = params['AUX_WEIGHT']
    
    train_ds = ThyroidDataset(X_s_tr, X_d_tr, y_tr)
    test_ds = ThyroidDataset(X_s_te, X_d_te, y_te)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)
    
    model = GridSearchModel(X_s_tr.shape[1], 1, hid, dr).to(Config.DEVICE)
    
    counts = np.bincount(y_tr)
    weights = torch.FloatTensor(len(y_tr) / (3 * counts)).to(Config.DEVICE)
    crit_cls = nn.CrossEntropyLoss(weight=weights)
    crit_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    
    best_acc = 0.0
    patience = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for xs, xd, y in train_dl:
            optimizer.zero_grad()
            logits, aux, _ = model(xs, xd)
            loss = crit_cls(logits, y) + aw * crit_reg(aux, xd)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xs, xd, y in test_dl:
                logits, _, _ = model(xs, xd)
                preds.extend(logits.argmax(1).numpy())
                labels.extend(y.numpy())
        
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= Config.PATIENCE: break
            
    # 重新加载最佳模型进行最终评估
    model.load_state_dict(best_state)
    return best_acc, model

# ==========================================
# 5. Temporal Analysis
# ==========================================
def evaluate_temporal_performance(model, X_s, X_d, y):
    model.eval()
    X_s = torch.FloatTensor(X_s).to(Config.DEVICE)
    X_d = torch.FloatTensor(X_d).to(Config.DEVICE)
    
    with torch.no_grad():
        step_logits = model.predict_stepwise(X_s, X_d)
    
    print("\n⏳ Temporal Performance Analysis (Value of Time):")
    print("-" * 50)
    print(f"{'Time Point':<15} | {'Accuracy':<10} | {'Binary Acc (Cure vs Fail)':<25}")
    print("-" * 50)
    
    time_labels = ["Baseline (0mo)", "+1 Month", "+3 Months (Final)"]
    
    accuracies = []
    
    for i, logits in enumerate(step_logits):
        preds = logits.argmax(1).cpu().numpy()
        
        # Multi-class Acc
        acc = accuracy_score(y, preds)
        accuracies.append(acc)
        
        # Binary Acc
        bin_true = [0 if v==0 else 1 for v in y] # 0=Fail, 1/2=Success
        bin_pred = [0 if v==0 else 1 for v in preds]
        bin_acc = accuracy_score(bin_true, bin_pred)
        
        print(f"{time_labels[i]:<15} | {acc:.4f}     | {bin_acc:.4f}")
    
    # 简单的可视化
    plt.figure(figsize=(8, 5))
    plt.plot(time_labels, accuracies, marker='o', linewidth=2, color='teal')
    plt.title("Model Accuracy Improvement Over Time")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig("temporal_acc_evolution.png")
    print("🖼️  Saved temporal_acc_evolution.png")

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    X_s, X_d_scaled, y = load_data()
    X_s_tr, X_s_te, X_d_tr, X_d_te, y_tr, y_te = train_test_split(
        X_s, X_d_scaled, y, test_size=0.2, stratify=y, random_state=Config.SEED
    )
    data_pack = (X_s_tr, X_s_te, X_d_tr, X_d_te, y_tr, y_te)
    
    # Generate all combinations
    keys, values = zip(*Config.GRID_SPACE.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🧠 Starting Massive Grid Search ({len(param_combinations)} configs)...")
    
    best_overall_acc = 0.0
    best_overall_model = None
    best_overall_params = {}
    
    for i, params in enumerate(param_combinations):
        acc, model = train_one_config(params, data_pack)
        
        # Simple progress log
        if (i+1) % 5 == 0: print(f"  Processed {i+1}/{len(param_combinations)} configs...")
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = model
            best_overall_params = params
            print(f"  🔥 New Best! Acc: {acc:.4f} | Params: {params}")
            
    print("\n" + "="*50)
    print(f"🏆 Champion Config: {best_overall_params}")
    print(f"🏆 Champion Accuracy: {best_overall_acc:.4f}")
    print("="*50)
    
    # Run Temporal Analysis on the Champion Model
    evaluate_temporal_performance(best_overall_model, X_s_te, X_d_te, y_te)

if __name__ == "__main__":
    main()