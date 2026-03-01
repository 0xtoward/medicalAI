import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import copy

# ==========================================
# 1. Config
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    BATCH_SIZE = 16  # MLP 参数少，batch size 可以小一点
    MAX_EPOCHS = 300
    PATIENCE = 30 
    SEED = 42
    DEVICE = torch.device("cpu") 
    
    LR = 0.001
    HIDDEN_DIM = 64

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
# 2. Data Processing (拍扁数据)
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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
    
    # Labels
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce').dropna()
    valid_idx = y_raw.index
    
    X_s = X_s.loc[valid_idx].values
    X_d = ft3_data.loc[valid_idx].values # (N, 3) 已经是 2D 了，不需要 newaxis
    
    y_map = {1: 0, 2: 1, 3: 2} 
    y = y_raw.map(y_map).values.astype(int)
    
    # Scaling
    scaler_s = StandardScaler()
    X_s_scaled = scaler_s.fit_transform(X_s)
    
    scaler_d = StandardScaler()
    X_d_scaled = scaler_d.fit_transform(X_d) # 单独对拍扁的 FT3 做标准化
    
    # 🗜️ 核心操作：直接把静态和动态拼接成一个长向量
    X_combined = np.concatenate([X_s_scaled, X_d_scaled], axis=1) 
    
    return X_combined, y

# ==========================================
# 3. Model Architecture: Pure MLP
# ==========================================
class PureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3):
        super(PureMLP, self).__init__()
        
        # 简单粗暴的多层感知机
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Grid Search
# ==========================================
GRID_PARAMS = {
    "hidden_dim": [32, 64, 96],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-4, 1e-3],
}

def train_single(X_tr, y_tr, X_val, y_val, input_dim, params):
    """训练单组超参数，返回验证集最佳准确率"""
    hidden_dim = params["hidden_dim"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    
    train_ds = ThyroidDataset(X_tr, y_tr)
    val_ds = ThyroidDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = PureMLP(input_dim, hidden_dim).to(Config.DEVICE)
    counts = np.bincount(y_tr)
    weights = torch.FloatTensor(len(y_tr) / (3 * counts)).to(Config.DEVICE)
    crit_cls = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = crit_cls(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                logits = model(x_batch.to(Config.DEVICE))
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y_batch.numpy())
        val_acc = accuracy_score(labels, preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                break
    
    model.load_state_dict(best_wts)
    return model, best_val_acc

def run_grid_search(X, y):
    """网格搜索最佳超参数"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=Config.SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=Config.SEED
    )
    
    keys = list(GRID_PARAMS.keys())
    values = list(GRID_PARAMS.values())
    n_combos = np.prod([len(v) for v in values])
    
    print(f"🔍 Grid Search: {n_combos} combinations")
    
    best_acc = 0.0
    best_params = None
    best_model = None
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        model, val_acc = train_single(X_tr, y_tr, X_val, y_val, X.shape[1], params)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params.copy()
            best_model = model
        print(f"  {params} -> Val Acc: {val_acc:.4f}")
    
    print(f"\n✅ Best: {best_params} -> Val Acc: {best_acc:.4f}")
    
    test_ds = ThyroidDataset(X_te, y_te)
    test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    preds, labels = [], []
    best_model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            logits = best_model(x_batch.to(Config.DEVICE))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y_batch.numpy())
    test_acc = accuracy_score(labels, preds)
    print(f"📊 Test Acc (best model): {test_acc:.4f}")
    
    return best_model, best_params, test_dl, labels, preds

# ==========================================
# 5. Training Engine
# ==========================================
def main(use_grid_search=True):
    X, y = load_data()
    input_dim = X.shape[1]
    
    if use_grid_search:
        model, best_params, test_dl, labels, preds = run_grid_search(X, y)
        print(f"📌 Best params: {best_params}")
        Config.HIDDEN_DIM = best_params["hidden_dim"]
        Config.LR = best_params["lr"]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=Config.SEED
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=Config.SEED
        )
        params = {
            "hidden_dim": Config.HIDDEN_DIM,
            "lr": Config.LR,
            "weight_decay": 1e-3,
        }
        model, _ = train_single(X_tr, y_tr, X_val, y_val, input_dim, params)
        test_ds = ThyroidDataset(X_te, y_te)
        test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                logits = model(x_batch.to(Config.DEVICE))
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y_batch.numpy())
        print(f"\n✅ Best Test Acc: {accuracy_score(labels, preds):.4f}")
    
    bin_true = [0 if v == 0 else 1 for v in labels]
    bin_pred = [0 if v == 0 else 1 for v in preds]
    print(f"🎯 Final Binary Accuracy: {accuracy_score(bin_true, bin_pred):.4f}")
    
    target_names = ['Fail', 'Over', 'Cure']
    print(classification_report(labels, preds, target_names=target_names))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-grid", action="store_true", help="跳过网格搜索，快速单次训练")
    args = parser.parse_args()
    main(use_grid_search=not args.no_grid)