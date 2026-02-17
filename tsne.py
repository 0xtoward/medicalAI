import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Config & Data (记忆恢复)
# ==========================================
class Config:
    FILE_PATH = "700.xlsx" 
    BATCH_SIZE = 32
    # 为了演示快速出图，我们训练 150 轮即可，足以让特征分层
    MAX_EPOCHS = 150 
    HIDDEN_DIM = 64
    LR = 0.002
    SEED = 42
    DEVICE = torch.device("cpu")
    
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [4, 6, 9, 21, 22, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 73]
    }

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. Data Processing
# ==========================================
class ThyroidDataset(Dataset):
    def __init__(self, X_s, X_d, y):
        self.X_s = torch.FloatTensor(X_s)
        self.X_d = torch.FloatTensor(X_d)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_s[idx], self.X_d[idx], self.y[idx]

def load_data():
    print(f"🔄 Loading data for visualization...")
    if Config.FILE_PATH.endswith('xlsx'):
        df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    else:
        df = pd.read_csv(Config.FILE_PATH, header=None).iloc[2:]
        
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()
    
    # Static
    static_cols = Config.COL_IDX['Static_Feats']
    X_s = df.iloc[:, static_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_s['Dose_Int'] = X_s.iloc[:, 5] / (X_s.iloc[:, 2] + 1e-5) 
    
    # Labels
    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    
    # Dynamic (Cured Median Imputation)
    ft3_cols = Config.COL_IDX['FT3_Sequence']
    temp_ft3 = df.iloc[:, ft3_cols].apply(pd.to_numeric, errors='coerce')
    cured_ft3_median = temp_ft3[y_raw == 3].median().median()
    if np.isnan(cured_ft3_median): cured_ft3_median = 4.5
    temp_ft3 = temp_ft3.fillna(cured_ft3_median)
    X_d = temp_ft3.values[..., np.newaxis] 
    
    # Filter valid
    valid_idx = y_raw.dropna().index
    X_s = X_s.loc[valid_idx].values
    X_d = X_d[df.index.get_indexer(valid_idx)]
    y = y_raw.loc[valid_idx].map({1: 0, 2: 1, 3: 2}).values.astype(int)
    patient_ids = df.loc[valid_idx, 'Patient_ID'].values
    
    # Scale
    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(X_s)
    X_d = (X_d - np.mean(X_d)) / (np.std(X_d) + 1e-5)
    
    return X_s, X_d, y, patient_ids

# ==========================================
# 3. Model with Feature Extraction Hook
# ==========================================
class HybridGated(nn.Module):
    def __init__(self, static_dim, hidden_dim):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, hidden_dim)
        )
        self.rnn = nn.LSTM(1, hidden_dim, batch_first=True)
        
        self.fc_s = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid, self.tanh = nn.Sigmoid(), nn.Tanh()
        
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 3))

    def forward(self, x_s, x_d, return_feats=False):
        h_s = self.static_net(x_s)
        rnn_out, (h_n, _) = self.rnn(x_d)
        h_d = h_n.squeeze(0)
        
        f_s, f_d = self.tanh(self.fc_s(h_s)), self.tanh(self.fc_d(h_d))
        z = self.sigmoid(self.gate(torch.cat([h_s, h_d], dim=1)))
        
        # 🌟 这就是我们要可视化的"特征向量" (Latent Vector)
        h_fused = z * f_s + (1 - z) * f_d
        
        logits = self.classifier(h_fused)
        
        if return_feats:
            return logits, h_fused # 返回特征供 t-SNE 使用
        return logits

# ==========================================
# 4. Training & Visualization Engine
# ==========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
# 必须引入这个用于3D绘图
from mpl_toolkits.mplot3d import Axes3D 

# ... (Config, Dataset, load_data, HybridGated 模型定义保持不变，此处省略以节省篇幅) ...
# 请直接复用上一段代码中的 Config, Dataset, load_data, HybridGated 类定义

def main():
    X_s, X_d, y, p_ids = load_data()
    
    # Train/Test Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    train_idx, test_idx = next(gss.split(X_s, y, groups=p_ids))
    X_s_te, X_d_te, y_te = X_s[test_idx], X_d[test_idx], y[test_idx]
    
    # 1. 快速训练 (复用之前的训练逻辑)
    print("🚀 Training model for feature extraction...")
    ds_train = ThyroidDataset(X_s[train_idx], X_d[train_idx], y[train_idx])
    dl_train = DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = HybridGated(X_s.shape[1], Config.HIDDEN_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(150): # 150轮足够看分布了
        for xs, xd, labels in dl_train:
            optimizer.zero_grad()
            logits = model(xs, xd)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    # 2. 提取特征
    print("✅ Extracting features...")
    model.eval()
    with torch.no_grad():
        X_s_tensor = torch.FloatTensor(X_s_te).to(Config.DEVICE)
        X_d_tensor = torch.FloatTensor(X_d_te).to(Config.DEVICE)
        _, h_latent = model(X_s_tensor, X_d_tensor, return_feats=True)
        learned_features = h_latent.cpu().numpy()
        
        # Raw features (Flattened)
        raw_features = np.hstack([X_s_te, X_d_te.reshape(len(X_d_te), -1)])

    # 3. 运行 3D t-SNE
    print("🎨 Running 3D t-SNE...")
    # 🌟 关键修改：n_components=3
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    
    tsne_raw = tsne.fit_transform(raw_features)
    tsne_learned = tsne.fit_transform(learned_features)
    
    # 4. 画 3D 图
    labels_map = {0: 'Fail (Hyper)', 1: 'Over (Hypo)', 2: 'Cure (Normal)'}
    y_str = [labels_map[i] for i in y_te]
    # 颜色映射
    color_dict = {'Fail (Hyper)': '#ff595e', 'Over (Hypo)': '#ffca3a', 'Cure (Normal)': '#1982c4'}
    c_list = [color_dict[val] for val in y_str]
    
    fig = plt.figure(figsize=(18, 8))
    
    # Plot A: Raw Data (3D)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(tsne_raw[:,0], tsne_raw[:,1], tsne_raw[:,2], c=c_list, s=40, alpha=0.6)
    ax1.set_title('(a) Raw Data Distribution (3D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2'); ax1.set_zlabel('Dim 3')
    
    # Plot B: Learned Features (3D)
    ax2 = fig.add_subplot(122, projection='3d')
    sc = ax2.scatter(tsne_learned[:,0], tsne_learned[:,1], tsne_learned[:,2], c=c_list, s=40, alpha=0.8)
    ax2.set_title('(b) Hybrid Latent Space (3D)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2'); ax2.set_zlabel('Dim 3')
    
    # 添加Legend (手动)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=10) for k, v in color_dict.items()]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('tsne_3d_comparison.png', dpi=300)
    print("🖼️  Saved 3D t-SNE to 'tsne_3d_comparison.png'")

if __name__ == "__main__":
    main()