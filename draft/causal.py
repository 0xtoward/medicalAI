import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from econml.dml import CausalForestDML
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "./causal_result"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def parse_dose_string(val):
    if pd.isna(val): return np.nan
    try: return float(val)
    except (ValueError, TypeError): pass
    s = str(val).strip().replace('（', '').replace('）', '').replace('(', '').replace(')', '')
    if '+' in s:
        try: return sum(float(x.strip()) for x in s.split('+'))
        except: pass
    return np.nan

# ==========================================
# 1. Config
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx"
    SEED = 42
    STATIC_FEAT_COLS = [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24]
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos",
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb",
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife"]
    FT3_COLS = [16, 29, 38, 47, 56, 65, 74]
    FT4_COLS = [17, 30, 39, 48, 57, 66, 75]
    TSH_COLS = [18, 31, 40, 49, 58, 67, 76]

    FT3_UPPER = 6.01   
    FT4_UPPER = 19.05  
    FT4_LOWER = 9.01   
    TSH_UPPER = 4.94   

np.random.seed(Config.SEED)

# ==========================================
# 2. 倾向性得分裁剪 (Propensity Trimming)
# ==========================================
def compute_propensity_and_trim(X, T, label):
    print(f"  [Propensity] 正在计算 {label} 的倾向性得分并执行裁剪 (Overlap Trimming)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=Config.SEED)
    ps = cross_val_predict(clf, X, T, cv=5, method='predict_proba')[:, 1]
    
    # 画分布图
    plt.figure(figsize=(8, 6))
    plt.hist(ps[T==0], bins=30, alpha=0.5, label='T=0 (Conservative Dose)', color='gray')
    plt.hist(ps[T==1], bins=30, alpha=0.5, label='T=1 (Aggressive Dose)', color='crimson')
    plt.axvline(0.05, color='black', linestyle='--')
    plt.axvline(0.95, color='black', linestyle='--')
    plt.title(f'[{label}] Propensity Score Distribution & Trimming')
    plt.xlabel('P(T=1 | X)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f"Propensity_{label.replace('-','_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 严格裁剪
    trim_mask = (ps >= 0.05) & (ps <= 0.95)
    print(f"    -> 移除了 {len(T) - trim_mask.sum()} 例极端倾向性患者，保留 {trim_mask.sum()} 例进入反事实推演。")
    return trim_mask

# ==========================================
# 3. 数据加载 (严格掩码模式)
# ==========================================
def load_causal_data():
    print("Loading data ...")
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    df = df.dropna(subset=[0]).reset_index(drop=True)

    # --- Covariates X (baseline only) ---
    X_s = df.iloc[:, Config.STATIC_FEAT_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values
    gt = df.iloc[:, 3].astype(str).str.strip()
    X_s[:, 0] = np.where((gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1), 1.0, 0.0)

    ft3_base = pd.to_numeric(df.iloc[:, 16], errors='coerce').fillna(method='bfill').values
    ft4_base = pd.to_numeric(df.iloc[:, 17], errors='coerce').fillna(method='bfill').values
    tsh_base = np.log1p(np.clip(pd.to_numeric(df.iloc[:, 18], errors='coerce').fillna(method='bfill').values, 0, None))

    X = np.column_stack([X_s, ft3_base, ft4_base, tsh_base])
    feature_names = Config.STATIC_NAMES + ["FT3_0M", "FT4_0M", "log(TSH_0M)"]

    # --- Treatment T ---
    dose = df.iloc[:, 25].apply(parse_dose_string).fillna(method='bfill').values
    thyroid_w = X_s[:, Config.STATIC_NAMES.index("ThyroidW")] + 1e-5
    dose_int = dose / thyroid_w
    T = (dose_int >= np.median(dose_int)).astype(int)

    # --- Outcomes (严格判定模式) ---
    outcomes = {}
    for label, time_idx in [("3-Mo", 2), ("6-Mo", 3)]:
        ft3_raw = pd.to_numeric(df.iloc[:, Config.FT3_COLS[time_idx]], errors='coerce')
        ft4_raw = pd.to_numeric(df.iloc[:, Config.FT4_COLS[time_idx]], errors='coerce')
        tsh_raw = pd.to_numeric(df.iloc[:, Config.TSH_COLS[time_idx]], errors='coerce')

        # 🌟 核心修复 1：严格掩码 (Strict Mask)
        valid_fail_mask = ft3_raw.notna() | ft4_raw.notna()
        valid_hypo_mask = ft4_raw.notna() & tsh_raw.notna() # 甲减必须要有TSH和FT4！

        Y_fail = ((ft3_raw > Config.FT3_UPPER) | (ft4_raw > Config.FT4_UPPER)).astype(int).values
        Y_hypo = ((ft4_raw < Config.FT4_LOWER) | (tsh_raw > Config.TSH_UPPER)).astype(int).values

        outcomes[label] = {
            "X": X, "T": T,
            "Y_fail": Y_fail, "mask_fail": valid_fail_mask.values,
            "Y_hypo": Y_hypo, "mask_hypo": valid_hypo_mask.values
        }
    return outcomes, feature_names

# ==========================================
# 4. 因果森林训练与推演
# ==========================================
def train_causal_forest(X_train, T_train, Y_train, target_name):
    # 🌟 核心修复 2：激活置信区间 (Confidence Intervals)
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=Config.SEED),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=Config.SEED),
        discrete_treatment=True, 
        n_estimators=400, # 增加树的数量以稳定区间计算
        min_samples_leaf=10, 
        random_state=Config.SEED
    )
    est.fit(Y_train.astype(float), T_train, X=X_train, W=None)
    return est

# ==========================================
# 5. 高级置信区间象限图
# ==========================================
def plot_advanced_quadrant(cate_fail, cate_hypo, time_label):
    plt.figure(figsize=(12, 10))
    benefit = -cate_fail
    risk = cate_hypo
    
    plt.scatter(benefit, risk, alpha=0.7, c='teal', edgecolors='white', s=50)

    x_thresh, y_thresh = 0.02, 0.05
    plt.axvline(x=x_thresh, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=y_thresh, color='gray', linestyle='--', alpha=0.7)

    # 🌟 核心修复 3：动态象限统计
    q1_mask = (benefit > x_thresh) & (risk <= y_thresh)
    q2_mask = (benefit <= x_thresh) & (risk > y_thresh)
    q3_mask = (benefit > x_thresh) & (risk > y_thresh)
    q4_mask = (benefit <= x_thresh) & (risk <= y_thresh)
    
    total = len(benefit)
    stats = {
        'Q1': f"Q1 (High Benefit/Low Cost)\n{q1_mask.sum()} pts ({q1_mask.sum()/total:.1%})",
        'Q2': f"Q2 (Low Benefit/High Cost)\n{q2_mask.sum()} pts ({q2_mask.sum()/total:.1%})",
        'Q3': f"Q3 (High Benefit/High Cost)\n{q3_mask.sum()} pts ({q3_mask.sum()/total:.1%})",
        'Q4': f"Q4 (Low Benefit/Low Cost)\n{q4_mask.sum()} pts ({q4_mask.sum()/total:.1%})"
    }

    plt.text(x_thresh + 0.01, y_thresh - 0.01, stats['Q1'], ha='left', va='top', fontsize=11, color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.text(x_thresh - 0.01, y_thresh + 0.01, stats['Q2'], ha='right', va='bottom', fontsize=11, color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.text(x_thresh + 0.01, y_thresh + 0.01, stats['Q3'], ha='left', va='bottom', fontsize=11, color='darkorange', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.text(x_thresh - 0.01, y_thresh - 0.01, stats['Q4'], ha='right', va='top', fontsize=11, color='gray', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlabel("Absolute Benefit: Reduction in Hyperthyroidism Risk (+%)\n<-- Escalate Dose -->", fontsize=13)
    plt.ylabel("Absolute Cost: Increase in Hypothyroidism Risk (+%)\n<-- Hypo Risk -->", fontsize=13)
    plt.title(f"[{time_label}] Causal Patient Stratification (Trimmed Cohort n={total})", fontsize=15, pad=15)
    plt.grid(alpha=0.2)

    path = os.path.join(OUT_DIR, f"Causal_Quadrant_CI_{time_label.replace('-','_')}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 6. 主逻辑
# ==========================================
def main():
    print("=" * 80)
    print("Causal Inference (Strict Mode + Overlap Trimming + CI)")
    print("=" * 80)

    outcomes, feat_names = load_causal_data()

    for label in ["3-Mo", "6-Mo"]:
        print(f"\n{'='*60}\n  {label} Analysis\n{'='*60}")
        d = outcomes[label]
        
        # 1. 全局倾向性得分裁剪 (确保 X 空间是一致的)
        trim_mask = compute_propensity_and_trim(d["X"], d["T"], label)
        
        # 截取裁剪后的安全总人群用于最终推演
        X_eval = d["X"][trim_mask]
        X_eval_scaled = StandardScaler().fit_transform(X_eval)
        
        # 2. 分别用严格掩码提取 Fail 和 Hypo 的训练集
        # Fail 训练集
        train_mask_fail = trim_mask & d["mask_fail"]
        X_fail_tr = StandardScaler().fit_transform(d["X"][train_mask_fail])
        est_fail = train_causal_forest(X_fail_tr, d["T"][train_mask_fail], d["Y_fail"][train_mask_fail], f"{label}-Fail")
        
        # Hypo 训练集
        train_mask_hypo = trim_mask & d["mask_hypo"]
        X_hypo_tr = StandardScaler().fit_transform(d["X"][train_mask_hypo])
        est_hypo = train_causal_forest(X_hypo_tr, d["T"][train_mask_hypo], d["Y_hypo"][train_mask_hypo], f"{label}-Hypo")
        
        # 3. 对安全总人群 (X_eval) 执行全局反事实推演！
        cate_fail = est_fail.effect(X_eval_scaled)
        cate_hypo = est_hypo.effect(X_eval_scaled)
        
        # 提取置信区间
        fail_lower, fail_upper = est_fail.effect_interval(X_eval_scaled, alpha=0.05)
        hypo_lower, hypo_upper = est_hypo.effect_interval(X_eval_scaled, alpha=0.05)
        
        print(f"    [ATE 统计] Hyper(Fail) 风险平均变化: {np.mean(cate_fail)*100:+.2f}% (95% CI: {np.mean(fail_lower)*100:+.2f}% ~ {np.mean(fail_upper)*100:+.2f}%)")
        print(f"    [ATE 统计] Hypo(甲减)  风险平均变化: {np.mean(cate_hypo)*100:+.2f}% (95% CI: {np.mean(hypo_lower)*100:+.2f}% ~ {np.mean(hypo_upper)*100:+.2f}%)")

        # 4. 绘图
        plot_advanced_quadrant(cate_fail, cate_hypo, label)
        
        # 特征重要性
        importances = est_fail.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(9, 6))
        plt.title(f"[{label}] Top Drivers of Dose Heterogeneity (Strict Mode)", fontsize=13)
        plt.barh(range(10), importances[indices], color='crimson')
        plt.yticks(range(10), [feat_names[i] for i in indices], fontsize=11)
        plt.savefig(os.path.join(OUT_DIR, f"Causal_Feature_Imp_{label.replace('-','_')}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print("\n✅ Done. 所有严谨版图表已存入:", OUT_DIR)

if __name__ == "__main__":
    main()