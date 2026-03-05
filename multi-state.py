import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
from pathlib import Path
import scipy.integrate
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = scipy.integrate.trapezoid
from lifelines import CoxPHFitter
from scipy.special import expit

plt.rcParams["font.family"] = "DejaVu Sans"

# ==========================================
# 1. 基础配置
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx" 
    OUT_DIR = Path("./multistate_result/")
    SEED = 42
    
    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
        'Eval_Cols': [35, 44, 53, 62, 71, 80],  # 1M,3M,6M,1Y,1.5Y,2Y
    }
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos",
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb",
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose"]

    # 7 time points: 0M (all hyper) + 6 doctor evaluations
    TIME_STAMPS = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
    STATE_NAMES = ["Hyper", "Normal", "Hypo"]

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.SEED)

def load_data():
    """加载数据，返回原始特征（未填充）和医生评价标签"""
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:].reset_index(drop=True)
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()
    valid_idx = df.iloc[:, Config.COL_IDX['Outcome']].dropna().index
    df = df.loc[valid_idx].reset_index(drop=True)

    X_s = df.iloc[:, Config.COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').values
    gt = df.iloc[:, 3].astype(str).str.strip()
    X_s[:, 0] = np.where((gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1), 1.0, 0.0)

    ft3_raw = df.iloc[:, Config.COL_IDX['FT3_Sequence']].apply(pd.to_numeric, errors='coerce').values
    ft4_raw = df.iloc[:, Config.COL_IDX['FT4_Sequence']].apply(pd.to_numeric, errors='coerce').values
    tsh_raw = df.iloc[:, Config.COL_IDX['TSH_Sequence']].apply(pd.to_numeric, errors='coerce').values

    # Doctor evaluation labels: 1=甲亢, 2=甲减, 3=正常
    eval_raw = df.iloc[:, Config.COL_IDX['Eval_Cols']].apply(pd.to_numeric, errors='coerce').values

    return X_s, ft3_raw, ft4_raw, tsh_raw, eval_raw, df['Patient_ID'].values


def build_impute_matrix(X_s, ft3, ft4, tsh, evals):
    """Stack all raw features into one matrix for joint imputation."""
    return np.hstack([X_s, ft3, ft4, tsh, evals])


def split_imputed(arr, n_static, n_seq):
    """Split imputed matrix back into components."""
    i = 0
    X_s = arr[:, i:i+n_static];       i += n_static
    ft3 = arr[:, i:i+n_seq];          i += n_seq
    ft4 = arr[:, i:i+n_seq];          i += n_seq
    tsh = arr[:, i:i+n_seq];          i += n_seq
    evals = arr[:, i:]
    return X_s, ft3, ft4, tsh, evals


def fit_missforest(raw_matrix):
    """Fit IterativeImputer (MissForest) on training data."""
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                        random_state=Config.SEED, n_jobs=1),
        max_iter=10, random_state=Config.SEED
    )
    imputer.fit(raw_matrix)
    return imputer


def apply_missforest(raw_matrix, imputer, n_eval_cols):
    """Transform with fitted imputer, then round eval cols to {1,2,3}."""
    filled = imputer.transform(raw_matrix)
    eval_start = filled.shape[1] - n_eval_cols
    filled[:, eval_start:] = np.clip(np.round(filled[:, eval_start:]), 1, 3)
    return filled

def build_states_from_labels(eval_imputed):
    """
    Build state matrix from (imputed) doctor evaluation labels.
    0M: all Hyper (pre-treatment). 1M-24M: from labels.
    Mapping: 1=甲亢→0(Hyper), 3=正常→1(Normal), 2=甲减→2(Hypo).
    """
    N = eval_imputed.shape[0]
    T = len(Config.TIME_STAMPS)
    S = np.zeros((N, T), dtype=int)
    label_map = {1: 0, 3: 1, 2: 2}
    for t in range(eval_imputed.shape[1]):
        for i in range(N):
            v = int(round(eval_imputed[i, t]))
            S[i, t + 1] = label_map.get(v, 0)
    return S

def plot_transition_heatmaps(S_matrix):
    """Plot 3x3 state transition heatmaps for each consecutive interval."""
    N, T = S_matrix.shape
    n_intervals = T - 1
    ncols = min(n_intervals, 3)
    nrows = (n_intervals + ncols - 1) // ncols
    total = np.zeros((3, 3), dtype=int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    if n_intervals == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_intervals:
            ax.set_visible(False)
            continue
        label = f"{Config.TIME_STAMPS[i]} -> {Config.TIME_STAMPS[i+1]}"
        mat = np.zeros((3, 3), dtype=int)
        for sf in range(3):
            for st in range(3):
                mat[sf, st] = np.sum((S_matrix[:, i] == sf) & (S_matrix[:, i+1] == st))
        total += mat

        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=Config.STATE_NAMES, yticklabels=Config.STATE_NAMES, ax=ax)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("State at t+1")
        ax.set_ylabel("State at t")

        stable = np.trace(mat)
        relapse = mat[1, 0]
        print(f"  [{label}]  transition: {(mat.sum()-stable)/mat.sum():.1%}  "
              f"relapse(Normal->Hyper): {relapse}")

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close(fig)

    print(f"\n  Total transitions: {total.sum()}  Stable: {np.trace(total)}  "
          f"Relapse(N->H): {total[1,0]}  Hypo recovered(Hp->N): {total[2,1]}")


# ==========================================
# 2. Feature engineering: wide -> long (with history & momentum)
# ==========================================
def build_long_format_data(X_s, X_d, S_matrix, pids):
    records = []
    N, T = S_matrix.shape
    for i in range(N):
        pid = pids[i]
        xs = X_s[i]
        for k in range(T - 1):
            if S_matrix[i, k] == 1: # 当期处于正常态
                labs_k = X_d[i, k, :] 
                labs_0 = X_d[i, 0, :]
                y_relapse = 1 if S_matrix[i, k+1] == 0 else 0
                
                # --- 核心特征注入 1：历史状态记忆 ---
                hist_states = S_matrix[i, :k] # 取出 0 到 k-1 的历史状态
                prev_state = S_matrix[i, k-1] if k > 0 else -1 # 上一步状态 (-1代表没有历史)
                ever_hyper = 1 if 0 in hist_states else 0
                ever_hypo = 1 if 2 in hist_states else 0
                time_in_normal = np.sum(hist_states == 1) # 之前总共正常过几个随访点
                
                # --- 核心特征注入 2：动态变化率 (Delta) ---
                delta_ft4_k0 = labs_k[1] - labs_0[1] # 相比基线的变化
                delta_tsh_k0 = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(np.clip(labs_0[2], 0, None))
                
                if k > 0:
                    labs_prev = X_d[i, k-1, :]
                    delta_ft4_1step = labs_k[1] - labs_prev[1] # 相比上一次随访的变化(动量)
                    delta_tsh_1step = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(np.clip(labs_prev[2], 0, None))
                else:
                    delta_ft4_1step = 0.0
                    delta_tsh_1step = 0.0
                
                records.append({
                    "Patient_ID": pid,
                    "Interval_ID": k,
                    "Interval_Name": f"{Config.TIME_STAMPS[k]}->{Config.TIME_STAMPS[k+1]}",
                    "Y_Relapse": y_relapse,
                    **{name: val for name, val in zip(Config.STATIC_NAMES, xs)},
                    "FT3_Current": labs_k[0],
                    "FT4_Current": labs_k[1],
                    "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                    # 注入的新特征
                    "Prev_State": str(prev_state),
                    "Ever_Hyper_Before": ever_hyper,
                    "Ever_Hypo_Before": ever_hypo,
                    "Time_In_Normal": time_in_normal,
                    "Delta_FT4_k0": delta_ft4_k0,
                    "Delta_TSH_k0": delta_tsh_k0,
                    "Delta_FT4_1step": delta_ft4_1step,
                    "Delta_TSH_1step": delta_tsh_1step
                })
    return pd.DataFrame(records)

class CoxPHWrapper:
    """sklearn-compatible wrapper around lifelines CoxPHFitter."""
    def __init__(self, penalizer=1.0):
        self.penalizer = penalizer
        self.cph = None
        self._cols = None
        self._scaler = StandardScaler()
        self._fallback = None

    def fit(self, X, y):
        arr = X.values if isinstance(X, pd.DataFrame) else X
        self._cols = list(X.columns) if isinstance(X, pd.DataFrame) else [f'f{i}' for i in range(arr.shape[1])]
        arr_s = self._scaler.fit_transform(arr)
        df = pd.DataFrame(arr_s, columns=self._cols)
        df['_dur'] = 1
        df['_evt'] = y.astype(float)
        try:
            self.cph = CoxPHFitter(penalizer=self.penalizer)
            self.cph.fit(df, duration_col='_dur', event_col='_evt')
            self._fallback = None
        except Exception:
            self._fallback = LogisticRegression(max_iter=2000, C=0.1)
            self._fallback.fit(arr_s, y)
            self.cph = None
        return self

    def predict_proba(self, X):
        arr = X.values if isinstance(X, pd.DataFrame) else X
        arr_s = self._scaler.transform(arr)
        if self._fallback is not None:
            return self._fallback.predict_proba(arr_s)
        df = pd.DataFrame(arr_s, columns=self._cols)
        log_ph = self.cph.predict_log_partial_hazard(df).values.flatten()
        p = expit(log_ph)
        return np.column_stack([1 - p, p])

    def get_params(self, deep=True):
        return {'penalizer': self.penalizer}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ==========================================
# 3. 多算法对比 + 严格前瞻验证
# ==========================================
def get_model_zoo():
    """返回候选模型字典: {名称: (模型实例, 颜色, 线型)}"""
    S = Config.SEED
    return {
        "Logistic Regression": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=S))]),
            "#1f77b4", "-."
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=5,
                                   random_state=S, n_jobs=1),
            "#ff7f0e", "--"
        ),
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=S),
            "#9467bd", "--"
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                       subsample=0.8, random_state=S),
            "#2ca02c", "--"
        ),
        "XGBoost": (
            xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                              subsample=0.8, eval_metric='logloss',
                              random_state=S, n_jobs=1, verbosity=0),
            "#d62728", "--"
        ),
        "LightGBM": (
            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                               subsample=0.8, random_state=S,
                               n_jobs=1, verbosity=-1),
            "#e377c2", "--"
        ),
        "MLP (Neural Net)": (
            Pipeline([('scaler', StandardScaler()),
                      ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000,
                                            learning_rate_init=0.001, alpha=0.01,
                                            early_stopping=True, validation_fraction=0.15,
                                            n_iter_no_change=50, random_state=S))]),
            "#8c564b", ":"
        ),
        "Cox PH": (
            CoxPHWrapper(penalizer=0.1),
            "#bcbd22", "-."
        ),
        "Stacking Ensemble": (
            StackingClassifier(
                estimators=[
                    ('lr', Pipeline([('s', StandardScaler()),
                                     ('lr', LogisticRegression(max_iter=2000, C=0.1, random_state=S))])),
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=4,
                                                  random_state=S, n_jobs=1)),
                    ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                       max_depth=3, random_state=S)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, random_state=S),
                cv=3, n_jobs=1, passthrough=False
            ),
            "#17becf", "-"
        ),
    }


def train_and_evaluate_hazard_strict(df_long_train, df_long_test):
    feat_cols = Config.STATIC_NAMES + [
        "FT3_Current", "FT4_Current", "logTSH_Current",
        "Ever_Hyper_Before", "Ever_Hypo_Before", "Time_In_Normal",
        "Delta_FT4_k0", "Delta_TSH_k0", "Delta_FT4_1step", "Delta_TSH_1step"
    ]

    df_all = pd.concat([df_long_train, df_long_test], ignore_index=True)
    interval_dummies = pd.get_dummies(df_all['Interval_Name'], prefix='Window')
    prev_state_dummies = pd.get_dummies(df_all['Prev_State'], prefix='PrevState')
    X_df = pd.concat([df_all[feat_cols], interval_dummies, prev_state_dummies], axis=1)

    n_tr = len(df_long_train)
    feat_names = list(X_df.columns)
    X_tr_df, y_tr = X_df.iloc[:n_tr], df_long_train['Y_Relapse'].values
    X_te_df, y_te = X_df.iloc[n_tr:], df_long_test['Y_Relapse'].values
    X_tr, X_te = X_tr_df.values, X_te_df.values
    groups_tr = df_long_train['Patient_ID'].values

    n_tr_patients = df_long_train['Patient_ID'].nunique()
    n_te_patients = df_long_test['Patient_ID'].nunique()
    print(f"\n  Train: {n_tr_patients} patients, {len(X_tr)} intervals (relapse: {y_tr.sum()})")
    print(f"  Test : {n_te_patients} patients, {len(X_te)} intervals (relapse: {y_te.sum()})")
    print(f"  (No leakage: train-only imputation + GroupKFold tuning)")

    # ================= 自动调参 (GroupKFold 避免患者泄漏) =================
    print("\n  Tuning (GroupKFold by Patient_ID)...")
    S = Config.SEED
    gkf = GroupKFold(n_splits=3)
    tuned_models = {}

    tune_specs = {
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=S),
            {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [2, 3, 5], 'subsample': [0.7, 0.8, 1.0]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(eval_metric='logloss', random_state=S, n_jobs=1, verbosity=0),
            {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [2, 3, 5], 'subsample': [0.7, 0.8, 1.0]}
        ),
    }
    for tname, (base_est, param_dist) in tune_specs.items():
        rs = RandomizedSearchCV(base_est, param_dist, n_iter=10, cv=gkf,
                                scoring='average_precision', random_state=S, n_jobs=-1)
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        tuned_models[tname] = rs.best_estimator_
        print(f"    {tname}: {rs.best_params_}  (CV PR-AUC={rs.best_score_:.3f})")

    # ================= OOF threshold selection + refit =================
    from sklearn.base import clone

    model_zoo = get_model_zoo()
    for tname, tuned_est in tuned_models.items():
        _, color, ls = model_zoo[tname]
        model_zoo[tname] = (tuned_est, color, ls)

    print("\n  Selecting thresholds via OOF (GroupKFold, maximize F1)...")
    results = {}

    print(f"\n{'='*95}")
    print(f"  {'Model':<22s} {'AUC':>6} {'PR-AUC':>8} {'Thr':>5} {'Acc':>6} {'BalAcc':>8} {'F1':>6}  {'TP':>4} {'FP':>5} {'FN':>4} {'TN':>5}")
    print(f"{'='*95}")

    for name, (model, color, ls) in model_zoo.items():
        # Step 1: Collect OOF predictions to find best threshold
        oof_proba = np.zeros(len(y_tr))
        for fold_tr, fold_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[fold_tr], y_tr[fold_tr])
            oof_proba[fold_val] = m.predict_proba(X_tr_df.iloc[fold_val])[:, 1]

        # Step 2: Search best threshold on OOF predictions
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.02, 0.60, 0.01):
            f = f1_score(y_tr, (oof_proba >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

        # Step 3: Refit on full training set, evaluate on test with selected threshold
        model.fit(X_tr_df, y_tr)
        proba = model.predict_proba(X_te_df)[:, 1]
        pred_bin = (proba >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba)
        prauc = average_precision_score(y_te, proba)
        acc = accuracy_score(y_te, pred_bin)
        bacc = balanced_accuracy_score(y_te, pred_bin)
        f1 = f1_score(y_te, pred_bin, zero_division=0)
        cm = confusion_matrix(y_te, pred_bin)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            'proba': proba, 'auc': auc, 'prauc': prauc,
            'acc': acc, 'bacc': bacc, 'f1': f1, 'threshold': best_thr,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'model': model, 'color': color, 'ls': ls,
        }

        marker = " *" if auc == max(r['auc'] for r in results.values()) else "  "
        print(f"{marker}{name:<22s} {auc:>5.3f} {prauc:>7.3f} {best_thr:>4.2f} {acc:>5.3f} {bacc:>7.3f} {f1:>5.3f}  {tp:>4} {fp:>5} {fn:>4} {tn:>5}")

    print(f"{'='*95}")
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"  Best AUC: {best_name} ({results[best_name]['auc']:.3f})")
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")

    # ================= 绘图 1: 多模型 Hazard 曲线对比 =================
    INTERVAL_ORDER = {"0M->1M": 0, "1M->3M": 1, "3M->6M": 2,
                      "6M->12M": 3, "12M->18M": 4, "18M->24M": 5}

    df_test = df_long_test.copy()
    empirical = df_test.groupby('Interval_Name').agg(
        N=('Y_Relapse', 'count'), Events=('Y_Relapse', 'sum'),
        Actual=('Y_Relapse', 'mean')
    ).reset_index()
    empirical['sk'] = empirical['Interval_Name'].map(INTERVAL_ORDER)
    empirical = empirical.sort_values('sk').reset_index(drop=True)
    x_labels = empirical['Interval_Name'].tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_labels, empirical['Actual'], marker='o', lw=2.5, color='crimson',
            label='Actual Hazard', zorder=10)
    ax.fill_between(x_labels, 0, empirical['Actual'], color='crimson', alpha=0.08)

    for idx, row in empirical.iterrows():
        ax.text(idx, row['Actual'] + 0.006,
                f"{row['Events']:.0f}/{row['N']:.0f}", ha='center', fontsize=8, color='crimson')

    for name, r in results.items():
        df_test['_pred'] = r['proba']
        pred_hz = df_test.groupby('Interval_Name')['_pred'].mean().reset_index()
        pred_hz['sk'] = pred_hz['Interval_Name'].map(INTERVAL_ORDER)
        pred_hz = pred_hz.sort_values('sk')
        ax.plot(x_labels, pred_hz['_pred'].values, marker='s', lw=1.5,
                color=r['color'], linestyle=r['ls'], alpha=0.85,
                label=f"{name} (AUC={r['auc']:.3f})")

    ax.set_title("Multi-Model Hazard Curve Comparison (Strict Temporal Validation)", fontsize=13, pad=12)
    ax.set_xlabel("Follow-up Intervals", fontsize=11)
    ax.set_ylabel("P(Relapse | Normal at t)", fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(alpha=0.3)
    fig.savefig(Config.OUT_DIR / "Hazard_Curve_Strict.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= 绘图 2: AUC / PR-AUC 横向对比柱状图 =================
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    praucs = [results[n]['prauc'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, aucs, w, label='ROC-AUC', color='steelblue')
    bars2 = ax.bar(x + w/2, praucs, w, label='PR-AUC', color='coral')

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: ROC-AUC vs PR-AUC", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(aucs) + 0.15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Model_Comparison_Bar.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= 绘图 3: SHAP (AUC 最好的树模型) =================
    tree_candidates = {n: r for n, r in results.items()
                       if hasattr(r['model'], 'feature_importances_')}
    if tree_candidates:
        best_tree = max(tree_candidates, key=lambda k: tree_candidates[k]['auc'])
        model_for_shap = tree_candidates[best_tree]['model']
        print(f"\n  SHAP analysis on: {best_tree}")
        try:
            if isinstance(model_for_shap, xgb.XGBClassifier):
                dm = xgb.DMatrix(X_tr_df, feature_names=feat_names)
                contribs = model_for_shap.get_booster().predict(dm, pred_contribs=True)
                shap_values = contribs[:, :-1]
            else:
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values = explainer.shap_values(X_tr_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_tr_df, feature_names=feat_names,
                              max_display=20, show=False)
            plt.title(f"What Drives Relapse? ({best_tree})", fontsize=14, pad=20)
            plt.savefig(Config.OUT_DIR / "Hazard_SHAP_Relapse.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  SHAP failed for {best_tree}: {e}")

def main():
    print("=" * 80)
    print("  Multi-State Relapse Analysis")
    print("=" * 80)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, pids = load_data()
    n_static = X_s_raw.shape[1]
    n_seq = ft3_raw.shape[1]
    n_eval = eval_raw.shape[1]
    print(f"  Records: {len(pids)}  Unique patients: {len(set(pids))}")

    # --- Temporal split ---
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask

    # --- MissForest imputation: fit on train, transform both ---
    import joblib
    imputer_path = Config.OUT_DIR / "missforest_imputer.pkl"
    raw_all = build_impute_matrix(X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw)

    if imputer_path.exists():
        print(f"\n  Loading cached MissForest from {imputer_path}")
        imputer = joblib.load(imputer_path)
    else:
        print(f"\n  Fitting MissForest on train set ({tr_mask.sum()} records)...")
        imputer = fit_missforest(raw_all[tr_mask])
        joblib.dump(imputer, imputer_path)
        print(f"  Saved to {imputer_path}")
    print(f"  Imputing train & test...")

    filled_tr = apply_missforest(raw_all[tr_mask], imputer, n_eval)
    filled_te = apply_missforest(raw_all[te_mask], imputer, n_eval)
    xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, n_seq)
    xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, n_seq)

    # --- Phase 1: Transition heatmaps (full imputed data) ---
    print(f"\n--- Phase 1: State Transition Heatmaps (Doctor Labels, MissForest imputed) ---")
    ev_full = np.vstack([ev_tr, ev_te])
    S_full = build_states_from_labels(ev_full)
    plot_transition_heatmaps(S_full)

    # --- Phase 2: Model comparison ---
    print(f"\n--- Phase 2: Multi-Model Comparison (Temporal Split) ---")
    def build_set(xs, ft3, ft4, tsh, ev, mask_pids):
        X_d = np.stack([ft3, ft4, tsh], axis=-1)
        S = build_states_from_labels(ev)
        return build_long_format_data(xs, X_d, S, mask_pids)

    df_tr = build_set(xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr, pids[tr_mask])
    df_te = build_set(xs_te, ft3_te, ft4_te, tsh_te, ev_te, pids[te_mask])
    train_and_evaluate_hazard_strict(df_tr, df_te)

if __name__ == "__main__":
    main()