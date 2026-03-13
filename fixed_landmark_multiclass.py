import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

from utils.config import SEED, STATIC_NAMES
from utils.data import (load_data, fit_missforest, split_imputed,
                        extract_flat_features)

# ==========================================
# 1. Config & 目录设置
# ==========================================
class Config:
    OUT_DIR = Path("./results/fixed_landmark_multiclass/")

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

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
# 4. 评估与可视化
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
    plt.title(f'Multi-class ROC — {landmark_name}')
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
# 5. 主执行
# ==========================================
def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*105}\n  Multi-class Classification — {landmark_name}\n{'='*105}")
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, _eval, y_outcome, pids = load_data()
    y = np.vectorize({1: 0, 2: 1, 3: 2}.get)(y_outcome)
    n_static = X_s_raw.shape[1]

    # --- Step 1: Temporal split FIRST (by enrollment order) ---
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    train_idx = np.where(np.array([pid in train_pids for pid in pids]))[0]
    test_idx = np.where(np.array([pid not in train_pids for pid in pids]))[0]
    n_train_records = len(train_idx)
    n_test_records = len(test_idx)
    print(f"  Train: {n_train_records} records")
    print(f"  Test:  {n_test_records} records")

    # --- Step 2: Temporal-safe MissForest: only columns ≤ seq_len ---
    ft3_cut = ft3_raw[:, :seq_len]
    ft4_cut = ft4_raw[:, :seq_len]
    tsh_cut = tsh_raw[:, :seq_len]
    raw_all = np.hstack([X_s_raw, ft3_cut, ft4_cut, tsh_cut])

    imputer_path = Config.OUT_DIR / f"missforest_seqlen{seq_len}.pkl"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        print(f"  MissForest (seq≤{seq_len}): loaded from cache")
    else:
        print(f"  MissForest (seq≤{seq_len}): fitting on train ({n_train_records} records)...")
        imputer = fit_missforest(raw_all[train_idx], seed=SEED)
        joblib.dump(imputer, imputer_path)
        print(f"  MissForest (seq≤{seq_len}): saved to {imputer_path}")

    filled_tr = imputer.transform(raw_all[train_idx])
    filled_te = imputer.transform(raw_all[test_idx])
    X_s_tr, ft3_tr, ft4_tr, tsh_tr = split_imputed(filled_tr, n_static, seq_len, n_eval=0)
    X_s_te, ft3_te, ft4_te, tsh_te = split_imputed(filled_te, n_static, seq_len, n_eval=0)

    X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
    X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

    # --- Step 3: Feature extraction ---
    X_tr_feat, feat_names = extract_flat_features(X_s_tr, X_d_tr, seq_len, static_names=STATIC_NAMES)
    X_te_feat, _          = extract_flat_features(X_s_te, X_d_te, seq_len, static_names=STATIC_NAMES)

    # --- Step 4: Scale on TRAIN ONLY ---
    scaler = StandardScaler().fit(X_tr_feat)
    X_tr, X_te = scaler.transform(X_tr_feat), scaler.transform(X_te_feat)
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # --- 补充分析: 仅保留级联多分类框架 ---
    rf_cas = CascadeClassifier(RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED),
                               RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED))
    rf_cas.fit(X_tr, y_tr)
    
    gbdt_cas = CascadeClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
                                 GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED))
    gbdt_cas.fit(X_tr, y_tr)
    
    # --- 收集所有预测结果 ---
    p_rf = rf_cas.predict_proba(X_te)
    p_gbdt = gbdt_cas.predict_proba(X_te)
    
    roc_data = {
        'Cascade RF': evaluate_probs(p_rf, y_te, "Cascade RF"),
        'Cascade GBDT': evaluate_probs(p_gbdt, y_te, "Cascade GBDT"),
    }
    plot_roc_curves(y_te, roc_data, landmark_name)
    
    # 为 3M 和 6M 均绘制核心模型 (Cascade GBDT) 的 SHAP 机制图
    plot_shap(gbdt_cas, X_tr, feat_names, 
              title=f"Multi-class SHAP — Cascade GBDT ({landmark_name[:4]})", 
              filename=f"SHAP_{landmark_name[:4]}_Hyper.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached MissForest .pkl files')
    args = parser.parse_args()
    if args.clear_cache:
        from utils.data import clear_pkl_cache
        clear_pkl_cache(Config.OUT_DIR)

    run_experiment("3-Month Assessment", seq_len=3)
    run_experiment("6-Month Assessment", seq_len=4)
    print(f"\n🎉 运行完毕！所有结果图表已保存至: {Config.OUT_DIR.resolve()}")