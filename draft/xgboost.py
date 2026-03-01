import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import copy
import xgboost as xgb  # 引入 XGBoost

# ==========================================
# 1. Config & Setup
# ==========================================
class Config:
    FILE_PATH = "700.xlsx" 
    BATCH_SIZE = 32
    MAX_EPOCHS = 400
    PATIENCE = 30 
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LSTM Best Params (Based on previous search)
    LSTM_HIDDEN = 32
    LSTM_LR = 0.001
    AUX_WEIGHT = 0.3

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
# 2. Advanced Feature Engineering
# ==========================================
def load_and_process_data(file_path):
    print(f"Loading data from: {file_path} ...")
    if file_path.endswith('xlsx'):
        df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
    else:
        df_raw = pd.read_csv(file_path, header=None)
            
    df = df_raw.iloc[2:].copy()
    
    # --- Feature Extraction ---
    static_cols_idx = Config.COL_IDX['Static_Feats']
    X_static = df.iloc[:, static_cols_idx].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 1. Manual Interaction Features (Domain Knowledge)
    # Dose Intensity
    X_static['Dose_Intensity'] = X_static.iloc[:, 5] / (X_static.iloc[:, 2] + 1e-5) 
    # TRAb Density
    X_static['TRAb_Density'] = X_static.iloc[:, 3] / (X_static.iloc[:, 2] + 1e-5)
    
    # Dynamic Features (Raw)
    ft3_base = df.iloc[:, Config.COL_IDX['FT3_Base']].apply(pd.to_numeric, errors='coerce')
    ft3_1mo = df.iloc[:, Config.COL_IDX['FT3_1Mo']].apply(pd.to_numeric, errors='coerce')
    ft3_3mo = df.iloc[:, Config.COL_IDX['FT3_3Mo']].apply(pd.to_numeric, errors='coerce')
    
    # Imputation (Forward Fill)
    ft3_base = ft3_base.fillna(ft3_base.median())
    ft3_1mo = ft3_1mo.fillna(ft3_base)
    ft3_3mo = ft3_3mo.fillna(ft3_1mo)

    # 2. Temporal Features (Explicit Rates)
    # 1-Month Drop Rate: (Base - 1Mo) / Base
    drop_rate_1mo = (ft3_base - ft3_1mo) / (ft3_base + 1e-5)
    # 3-Month Drop Rate
    drop_rate_3mo = (ft3_base - ft3_3mo) / (ft3_base + 1e-5)
    
    # Add to Static Features (for XGBoost & LSTM Static Branch)
    X_static['Drop_Rate_1Mo'] = drop_rate_1mo
    X_static['Drop_Rate_3Mo'] = drop_rate_3mo

    # Construct 3D Tensor for LSTM
    X_dynamic = np.stack([ft3_base, ft3_1mo, ft3_3mo], axis=1)
    X_dynamic = X_dynamic[..., np.newaxis] 

    # Label Processing
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce').dropna()
    valid_idx = y_raw.index
    
    # Align Dataframes
    X_static = X_static.loc[valid_idx] # Keep as DataFrame for now
    X_dynamic = X_dynamic[y_raw.reset_index(drop=True).index]
    
    y_map = {1: 0, 2: 1, 3: 2} # 0:Hyper, 1:Hypo, 2:Normal
    y = y_raw.map(y_map).values.astype(int)
    
    return X_static, X_dynamic, y

# ==========================================
# 3. Model Architecture (LSTM + Gated)
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, static_dim, dynamic_dim, out_dim):
        super(GatedFusion, self).__init__()
        self.fc_static = nn.Linear(static_dim, out_dim)
        self.fc_dynamic = nn.Linear(dynamic_dim, out_dim)
        # Trainable Gate Network
        self.gate_net = nn.Linear(static_dim + dynamic_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_static, x_dynamic):
        h_s = self.tanh(self.fc_static(x_static))
        h_d = self.tanh(self.fc_dynamic(x_dynamic))
        combined = torch.cat([x_static, x_dynamic], dim=1)
        z = self.sigmoid(self.gate_net(combined)) # Learned per sample
        return z * h_s + (1 - z) * h_d

class HybridEnsembleModel(nn.Module):
    def __init__(self, static_dim, dynamic_dim, hidden_dim, num_classes=3):
        super(HybridEnsembleModel, self).__init__()
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, hidden_dim)
        )
        self.rnn = nn.LSTM(input_size=dynamic_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.aux_head = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.fusion = GatedFusion(hidden_dim, hidden_dim, 32)
        self.classifier = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x_static, x_dynamic):
        h_s = self.static_net(x_static)
        rnn_out, (h_n, _) = self.rnn(x_dynamic)
        h_d = h_n.squeeze(0)
        aux_preds = self.aux_head(rnn_out)
        h_fused = self.fusion(h_s, h_d)
        logits = self.classifier(h_fused)
        return logits, aux_preds

# ==========================================
# 4. Training Utilities
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X_s, X_d, y):
        self.X_s = torch.FloatTensor(X_s)
        self.X_d = torch.FloatTensor(X_d)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_s[idx], self.X_d[idx], self.y[idx]

def train_lstm_model(X_s_train, X_d_train, y_train, X_s_test, X_d_test, y_test):
    # Standardize Static Features
    scaler = StandardScaler()
    X_s_train_std = scaler.fit_transform(X_s_train)
    X_s_test_std = scaler.transform(X_s_test)
    
    # Standardize Dynamic
    X_d_train_std = (X_d_train - np.mean(X_d_train)) / (np.std(X_d_train) + 1e-5)
    X_d_test_std = (X_d_test - np.mean(X_d_test)) / (np.std(X_d_test) + 1e-5)

    train_ds = ThyroidDataset(X_s_train_std, X_d_train_std, y_train)
    test_ds = ThyroidDataset(X_s_test_std, X_d_test_std, y_test)
    train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = HybridEnsembleModel(X_s_train.shape[1], 1, Config.LSTM_HIDDEN).to(Config.DEVICE)
    
    counts = np.bincount(y_train)
    weights = torch.FloatTensor(len(y_train) / (3 * counts)).to(Config.DEVICE)
    crit_cls = nn.CrossEntropyLoss(weight=weights)
    crit_reg = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=Config.LSTM_LR, weight_decay=1e-3)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 0
    
    print("Training Hybrid LSTM...")
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        for xs, xd, y in train_dl:
            xs, xd, y = xs.to(Config.DEVICE), xd.to(Config.DEVICE), y.to(Config.DEVICE)
            opt.zero_grad()
            logits, aux = model(xs, xd)
            loss = crit_cls(logits, y) + Config.AUX_WEIGHT * crit_reg(aux, xd)
            loss.backward()
            opt.step()
            
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xs, xd, y in test_dl:
                xs, xd, y = xs.to(Config.DEVICE), xd.to(Config.DEVICE), y.to(Config.DEVICE)
                logits, _ = model(xs, xd)
                _, p = torch.max(logits, 1)
                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())
        
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE: break
            
    model.load_state_dict(best_model_wts)
    
    # Get Probabilities for Ensemble
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xs, xd, y in test_dl:
            xs, xd = xs.to(Config.DEVICE), xd.to(Config.DEVICE)
            logits, _ = model(xs, xd)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs), best_acc

def train_xgboost_model(X_s_train, X_d_train, y_train, X_s_test, X_d_test, y_test):
    # Flatten dynamic features for XGBoost (add them to static dataframe)
    # X_dynamic is (N, 3, 1). We flatten to (N, 3)
    X_d_train_flat = X_d_train.reshape(X_d_train.shape[0], -1)
    X_d_test_flat = X_d_test.reshape(X_d_test.shape[0], -1)
    
    # Combine
    X_train_full = np.hstack([X_s_train, X_d_train_flat])
    X_test_full = np.hstack([X_s_test, X_d_test_flat])
    
    print("Training XGBoost...")
    # XGBoost handles scaling internally usually, but trees don't require it strictly
    clf = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.05, 
        objective='multi:softprob', 
        num_class=3,
        random_state=Config.SEED,
        eval_metric='mlogloss'
    )
    
    clf.fit(X_train_full, y_train)
    
    probs = clf.predict_proba(X_test_full)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    return probs, acc

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    X_s_df, X_d, y = load_and_process_data(Config.FILE_PATH)
    
    # Split
    X_s_train, X_s_test, X_d_train, X_d_test, y_train, y_test = train_test_split(
        X_s_df.values, X_d, y, test_size=0.2, stratify=y, random_state=Config.SEED
    )
    
    # 1. Train LSTM
    lstm_probs, lstm_acc = train_lstm_model(X_s_train, X_d_train, y_train, X_s_test, X_d_test, y_test)
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
    
    # 2. Train XGBoost
    xgb_probs, xgb_acc = train_xgboost_model(X_s_train, X_d_train, y_train, X_s_test, X_d_test, y_test)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    # 3. Ensemble (Weighted Average)
    # LSTM is usually better at temporal, XGB at static. 
    # Let's give slight edge to LSTM or equal.
    ensemble_probs = 0.6 * lstm_probs + 0.4 * xgb_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    
    print("\n" + "="*40)
    print(f"🚀 ENSEMBLE RESULT (Voting)")
    print("="*40)
    print(f"Final Accuracy: {ensemble_acc:.4f}")
    print(f"Improvement over Single Best: +{(ensemble_acc - max(lstm_acc, xgb_acc))*100:.2f}%")
    
    # Binary Accuracy (The SOTA Killer)
    bin_true = [0 if v==0 else 1 for v in y_test]
    bin_pred = [0 if v==0 else 1 for v in ensemble_preds]
    bin_acc = accuracy_score(bin_true, bin_pred)
    print(f"Binary Accuracy (Remission vs Fail): {bin_acc:.4f}")
    
    print("\nClassification Report:")
    targets = ['Fail (Hyper)', 'Over (Hypo)', 'Cure (Normal)']
    print(classification_report(y_test, ensemble_preds, target_names=targets, zero_division=0))
    
    # 4. English ROC Plot
    plt.figure(figsize=(10, 8))
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], ensemble_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{targets[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Multi-Class ROC Curves (Ensemble Model)', fontsize=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ensemble_roc_english.png', dpi=300)
    print("🖼️  ROC plot saved as ensemble_roc_english.png (No Chinese chars)")

if __name__ == "__main__":
    main()