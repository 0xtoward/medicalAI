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
                             f1_score, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
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
import joblib
from sklearn.base import clone

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


def split_imputed(arr, n_static, n_lab, n_eval):
    """Split imputed matrix back into components."""
    i = 0
    X_s = arr[:, i:i+n_static];       i += n_static
    ft3 = arr[:, i:i+n_lab];          i += n_lab
    ft4 = arr[:, i:i+n_lab];          i += n_lab
    tsh = arr[:, i:i+n_lab];          i += n_lab
    evals = arr[:, i:i+n_eval]
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
def build_long_format_data(X_s, X_d, S_matrix, pids, target_k=None):
    """Build long-format rows. If target_k is set, only build for interval k→k+1."""
    records = []
    N, T = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(T - 1)
    for i in range(N):
        pid = pids[i]
        xs = X_s[i]
        for k in k_range:
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
def get_tune_specs():
    """Return {name: (estimator, param_grid, color, linestyle, n_iter)}."""
    S = Config.SEED
    return {
        "Logistic Reg.": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, random_state=S))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__penalty': ['l1', 'l2'], 'lr__solver': ['saga']},
            "#1f77b4", "-.", 14
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=S, n_jobs=1),
            {'n_estimators': [100, 200, 300, 500],
             'max_depth': [3, 5, 7, 10, None],
             'min_samples_leaf': [3, 5, 10, 20],
             'max_features': ['sqrt', 'log2', 0.5]},
            "#ff7f0e", "--", 20
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=S),
            {'n_estimators': [50, 100, 200, 300],
             'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]},
            "#9467bd", "--", 15
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=S),
            {'n_estimators': [100, 200, 300, 500],
             'learning_rate': [0.01, 0.03, 0.05, 0.1],
             'max_depth': [2, 3, 4, 5],
             'subsample': [0.6, 0.7, 0.8, 1.0],
             'min_samples_leaf': [5, 10, 20]},
            "#2ca02c", "--", 20
        ),
        "XGBoost": (
            xgb.XGBClassifier(eval_metric='logloss', random_state=S,
                              n_jobs=1, verbosity=0),
            {'n_estimators': [100, 200, 300, 500],
             'learning_rate': [0.01, 0.03, 0.05, 0.1],
             'max_depth': [2, 3, 4, 5],
             'subsample': [0.6, 0.7, 0.8, 1.0],
             'colsample_bytree': [0.6, 0.8, 1.0],
             'min_child_weight': [1, 3, 5]},
            "#d62728", "--", 20
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=S, n_jobs=1, verbosity=-1),
            {'n_estimators': [100, 200, 300, 500],
             'learning_rate': [0.01, 0.03, 0.05, 0.1],
             'max_depth': [2, 3, 4, 5, -1],
             'subsample': [0.6, 0.7, 0.8, 1.0],
             'colsample_bytree': [0.6, 0.8, 1.0],
             'min_child_samples': [5, 10, 20]},
            "#e377c2", "--", 20
        ),
        "MLP": (
            Pipeline([('scaler', StandardScaler()),
                      ('mlp', MLPClassifier(max_iter=1000, early_stopping=True,
                                            validation_fraction=0.15,
                                            n_iter_no_change=50, random_state=S))]),
            {'mlp__hidden_layer_sizes': [(32,), (64, 32), (128, 64), (32, 16)],
             'mlp__alpha': [0.001, 0.01, 0.05, 0.1],
             'mlp__learning_rate_init': [0.0005, 0.001, 0.005, 0.01]},
            "#8c564b", ":", 16
        ),
        "Cox PH": (
            CoxPHWrapper(penalizer=0.1),
            {'penalizer': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
            "#bcbd22", "-.", 7
        ),
    }


def run_shap_analysis(model_name, model, X_tr_df, feat_names):
    """Run SHAP on the selected best model with a model-appropriate explainer."""
    print(f"\n  SHAP analysis on: {model_name}")
    try:
        if model_name == "Logistic Reg." and isinstance(model, Pipeline):
            scaler = model.named_steps['scaler']
            lr = model.named_steps['lr']
            X_scaled = scaler.transform(X_tr_df)
            explainer = shap.LinearExplainer(lr, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_scaled, feature_names=feat_names,
                max_display=20, show=False
            )
        elif isinstance(model, xgb.XGBClassifier):
            dm = xgb.DMatrix(X_tr_df, feature_names=feat_names)
            contribs = model.get_booster().predict(dm, pred_contribs=True)
            shap_values = contribs[:, :-1]
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_tr_df, feature_names=feat_names,
                max_display=20, show=False
            )
        elif hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_tr_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_tr_df, feature_names=feat_names,
                max_display=20, show=False
            )
        else:
            # Generic fallback for models without native SHAP support.
            bg = shap.sample(X_tr_df, min(100, len(X_tr_df)), random_state=Config.SEED)
            eval_x = X_tr_df.iloc[:min(300, len(X_tr_df))]
            explainer = shap.Explainer(
                lambda data: model.predict_proba(pd.DataFrame(data, columns=feat_names))[:, 1],
                bg
            )
            shap_values = explainer(eval_x)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, eval_x, feature_names=feat_names,
                max_display=20, show=False
            )

        plt.title(f"What Drives Relapse? ({model_name})", fontsize=14, pad=20)
        plt.savefig(Config.OUT_DIR / "Hazard_SHAP_Relapse.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")


def compute_binary_metrics(y_true, proba, threshold):
    """Compute discrimination/calibration/classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'auc': roc_auc_score(y_true, proba),
        'prauc': average_precision_score(y_true, proba),
        'brier': brier_score_loss(y_true, proba),
        'acc': accuracy_score(y_true, pred),
        'bacc': balanced_accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred, zero_division=0),
        'recall': tp / (tp + fn) if (tp + fn) else np.nan,
        'specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'threshold': threshold,
    }


def compute_calibration_stats(y_true, proba):
    """Compute calibration-in-the-large style intercept and slope."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_true)) < 2:
        return {'intercept': np.nan, 'slope': np.nan}

    logit_p = np.log(proba / (1 - proba)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
        lr.fit(logit_p, y_true)
        return {
            'intercept': float(lr.intercept_[0]),
            'slope': float(lr.coef_[0, 0]),
        }
    except Exception:
        return {'intercept': np.nan, 'slope': np.nan}


def bootstrap_group_cis(eval_df, threshold, group_col='Patient_ID', n_boot=800):
    """Cluster bootstrap CIs by patient to respect repeated intervals."""
    group_values = eval_df[group_col].dropna().unique()
    if len(group_values) == 0:
        return {}

    grouped = {g: eval_df.loc[eval_df[group_col] == g] for g in group_values}
    rng = np.random.default_rng(Config.SEED)
    store = {k: [] for k in ['auc', 'prauc', 'brier', 'recall', 'specificity', 'cal_intercept', 'cal_slope']}

    for _ in range(n_boot):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        sampled_df = pd.concat([grouped[g] for g in sampled_groups], ignore_index=True)
        y_boot = sampled_df['Y'].values
        if len(np.unique(y_boot)) < 2:
            continue
        m = compute_binary_metrics(y_boot, sampled_df['proba'].values, threshold)
        cal = compute_calibration_stats(y_boot, sampled_df['proba'].values)
        m['cal_intercept'] = cal['intercept']
        m['cal_slope'] = cal['slope']
        for key in store:
            if np.isfinite(m[key]):
                store[key].append(m[key])

    cis = {}
    for key, values in store.items():
        if len(values) < 20:
            cis[key] = (np.nan, np.nan)
        else:
            cis[key] = (float(np.percentile(values, 2.5)),
                        float(np.percentile(values, 97.5)))
    return cis


def format_ci(cis, key):
    """Format CI text for printing."""
    low, high = cis.get(key, (np.nan, np.nan))
    if np.isnan(low) or np.isnan(high):
        return "NA"
    return f"[{low:.3f}, {high:.3f}]"


def save_calibration_figure(y_true, proba, title, out_path, n_bins=6):
    """Save calibration curve + prediction histogram."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy='quantile')

    fig = plt.figure(figsize=(7.5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.plot([0, 1], [0, 1], '--', color='gray', lw=1.5, label='Perfect calibration')
    ax1.plot(mean_pred, frac_pos, marker='o', lw=2, color='navy', label='Model')
    ax1.set_title(title, fontsize=13)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed event rate")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.hist(proba, bins=20, color='steelblue', alpha=0.85, edgecolor='white')
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_threshold_sensitivity_figure(y_true, proba, best_thr, title, out_path):
    """Show how recall/specificity/F1 vary across thresholds."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    thresholds = np.arange(0.02, 0.61, 0.01)
    rows = []
    for thr in thresholds:
        m = compute_binary_metrics(y_true, proba, thr)
        rows.append({
            'threshold': thr,
            'recall': m['recall'],
            'specificity': m['specificity'],
            'f1': m['f1'],
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(df['threshold'], df['recall'], lw=2, label='Recall')
    ax.plot(df['threshold'], df['specificity'], lw=2, label='Specificity')
    ax.plot(df['threshold'], df['f1'], lw=2, label='F1')
    ax.axvline(best_thr, color='black', linestyle='--', alpha=0.7, label=f'Chosen threshold={best_thr:.2f}')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compute_dca(y_true, proba, thresholds=None):
    """Decision curve analysis net benefit."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    if thresholds is None:
        thresholds = np.arange(0.02, 0.51, 0.01)

    n = len(y_true)
    prevalence = y_true.mean()
    rows = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        odds = thr / (1 - thr)
        nb_model = tp / n - fp / n * odds
        nb_all = prevalence - (1 - prevalence) * odds
        rows.append({
            'threshold': thr,
            'model': nb_model,
            'treat_all': nb_all,
            'treat_none': 0.0
        })
    return pd.DataFrame(rows)


def save_dca_figure(y_true, proba, title, out_path):
    """Save DCA figure."""
    dca_df = compute_dca(y_true, proba)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dca_df['threshold'], dca_df['model'], lw=2.2, color='darkgreen', label='Model')
    ax.plot(dca_df['threshold'], dca_df['treat_all'], lw=1.6, color='crimson', linestyle='--', label='Treat all')
    ax.plot(dca_df['threshold'], dca_df['treat_none'], lw=1.6, color='black', linestyle=':', label='Treat none')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def aggregate_patient_level(df_long, proba, method='product'):
    """Aggregate interval risks into patient-level 'any relapse during follow-up' risk."""
    tmp = df_long[['Patient_ID', 'Interval_Name', 'Y_Relapse']].copy().reset_index(drop=True)
    tmp['proba'] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)

    patient_rows = []
    for pid, g in tmp.groupby('Patient_ID', sort=False):
        probs = g['proba'].values
        if method == 'product':
            agg_proba = float(1 - np.prod(1 - probs))
        elif method == 'max':
            agg_proba = float(np.max(probs))
        elif method == 'mean':
            agg_proba = float(np.mean(probs))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        patient_rows.append({
            'Patient_ID': pid,
            'Y': int(g['Y_Relapse'].max()),
            'proba': agg_proba,
            'Max_Interval_Risk': float(np.max(probs)),
            'Mean_Interval_Risk': float(np.mean(probs)),
            'Intervals': int(len(g)),
            'First_Window': g['Interval_Name'].iloc[0],
            'Last_Window': g['Interval_Name'].iloc[-1],
            'Aggregation': method,
        })
    return pd.DataFrame(patient_rows)


def select_best_threshold(y_true, proba, low=0.02, high=0.95, step=0.01):
    """Select threshold by F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(low, high + 1e-9, step):
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def save_patient_risk_strata(train_patient_df, test_patient_df, out_path):
    """Use train-set quartiles for patient-level risk stratification."""
    train_scores = np.clip(train_patient_df['proba'].values, 0, 1)
    bins = np.quantile(train_scores, [0, 0.25, 0.5, 0.75, 1.0])
    bins[0] = min(bins[0], 0.0)
    bins[-1] = max(bins[-1], 1.0)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6

    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    risk_group = pd.cut(test_patient_df['proba'], bins=bins, include_lowest=True, labels=labels)
    strat_df = test_patient_df.copy()
    strat_df['Risk_Group'] = risk_group.astype(str)

    summary = strat_df.groupby('Risk_Group').agg(
        N=('Y', 'size'),
        Observed=('Y', 'mean'),
        Predicted=('proba', 'mean')
    ).reindex(labels).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(summary['Risk_Group'], summary['Observed'], color='cornflowerblue', alpha=0.85, label='Observed relapse rate')
    ax.plot(summary['Risk_Group'], summary['Predicted'], marker='o', lw=2, color='darkorange', label='Mean predicted risk')
    for bar, n in zip(bars, summary['N'].fillna(0).astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha='center', fontsize=9)
    ax.set_ylim(0, min(1.0, float(np.nanmax([summary['Observed'].max(), summary['Predicted'].max()]) + 0.12)))
    ax.set_title("Patient-Level Risk Stratification (Q1-Q4)", fontsize=13)
    ax.set_xlabel("Risk groups from train-set quartiles")
    ax.set_ylabel("Relapse probability")
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_patient_aggregation_sensitivity(df_long_train, df_long_test, oof_proba, test_proba):
    """Compare patient-level conclusions across aggregation rules."""
    methods = {
        'Product_AnyRisk': 'product',
        'Max_Interval_Risk': 'max',
        'Mean_Interval_Risk': 'mean',
    }
    rows = []
    curves = {}
    for label, method in methods.items():
        train_df = aggregate_patient_level(df_long_train, oof_proba, method=method)
        test_df = aggregate_patient_level(df_long_test, test_proba, method=method)
        thr = select_best_threshold(train_df['Y'].values, train_df['proba'].values,
                                    low=0.05, high=0.95, step=0.01)
        m = compute_binary_metrics(test_df['Y'].values, test_df['proba'].values, thr)
        cal = compute_calibration_stats(test_df['Y'].values, test_df['proba'].values)
        rows.append({
            'Aggregation': label,
            'Threshold': thr,
            'AUC': m['auc'],
            'PR_AUC': m['prauc'],
            'Brier': m['brier'],
            'Recall': m['recall'],
            'Specificity': m['specificity'],
            'Calibration_Intercept': cal['intercept'],
            'Calibration_Slope': cal['slope'],
        })
        curves[label] = test_df
    return pd.DataFrame(rows), curves


def save_patient_aggregation_sensitivity_figure(summary_df, out_path):
    """Visualize patient-level aggregation sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(summary_df))
    labels = summary_df['Aggregation'].tolist()

    axes[0].bar(x - 0.17, summary_df['AUC'], 0.34, label='AUC', color='steelblue')
    axes[0].bar(x + 0.17, summary_df['PR_AUC'], 0.34, label='PR-AUC', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Patient-Level Aggregation Sensitivity")
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.17, summary_df['Calibration_Intercept'], 0.34, label='Intercept', color='slateblue')
    axes[1].bar(x + 0.17, summary_df['Calibration_Slope'], 0.34, label='Slope', color='darkorange')
    axes[1].axhline(0.0, color='gray', linestyle='--', lw=1)
    axes[1].axhline(1.0, color='gray', linestyle=':', lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_title("Calibration Statistics by Aggregation")
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_window_sensitivity(df_long_test, proba, threshold):
    """Check if results depend heavily on specific follow-up windows."""
    tmp = df_long_test[['Patient_ID', 'Interval_Name', 'Y_Relapse']].copy().reset_index(drop=True)
    tmp['Y'] = tmp['Y_Relapse'].astype(int)
    tmp['proba'] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    scenarios = {
        'All_Windows': np.ones(len(tmp), dtype=bool),
        'Exclude_1M_3M': tmp['Interval_Name'].values != '1M->3M',
        'Exclude_12Mplus': ~tmp['Interval_Name'].isin(['12M->18M', '18M->24M']).values,
        'Only_3Mplus': tmp['Interval_Name'].isin(['3M->6M', '6M->12M', '12M->18M', '18M->24M']).values,
    }
    rows = []
    for label, mask in scenarios.items():
        sub = tmp.loc[mask].copy()
        if len(sub) == 0 or len(np.unique(sub['Y'])) < 2:
            continue
        m = compute_binary_metrics(sub['Y'].values, sub['proba'].values, threshold)
        rows.append({
            'Scenario': label,
            'Intervals': len(sub),
            'Events': int(sub['Y'].sum()),
            'AUC': m['auc'],
            'PR_AUC': m['prauc'],
            'Brier': m['brier'],
            'Recall': m['recall'],
            'Specificity': m['specificity'],
        })
    return pd.DataFrame(rows)


def train_and_evaluate_hazard_strict(df_long_train, df_long_test):
    feat_cols = Config.STATIC_NAMES + [
        "FT3_Current", "FT4_Current", "logTSH_Current",
        "Ever_Hyper_Before", "Ever_Hypo_Before", "Time_In_Normal",
        "Delta_FT4_k0", "Delta_TSH_k0", "Delta_FT4_1step", "Delta_TSH_1step"
    ]

    # Encode categorical columns using training-set categories only
    interval_cats = sorted(df_long_train['Interval_Name'].unique())
    prev_state_cats = sorted(df_long_train['Prev_State'].unique())

    def make_features(df):
        out = df[feat_cols].copy().reset_index(drop=True)
        for cat in interval_cats:
            out[f'Window_{cat}'] = (df['Interval_Name'].values == cat).astype(float)
        for cat in prev_state_cats:
            out[f'PrevState_{cat}'] = (df['Prev_State'].values == cat).astype(float)
        return out

    X_tr_df = make_features(df_long_train)
    X_te_df = make_features(df_long_test)
    feat_names = list(X_tr_df.columns)
    y_tr = df_long_train['Y_Relapse'].values
    y_te = df_long_test['Y_Relapse'].values
    groups_tr = df_long_train['Patient_ID'].values

    n_tr_patients = df_long_train['Patient_ID'].nunique()
    n_te_patients = df_long_test['Patient_ID'].nunique()
    print(f"\n  Train: {n_tr_patients} patients, {len(y_tr)} intervals (relapse: {y_tr.sum()})")
    print(f"  Test : {n_te_patients} patients, {len(y_te)} intervals (relapse: {y_te.sum()})")
    print(f"  Features: {len(feat_names)}")

    # ================= Tune ALL models (RandomizedSearchCV + GroupKFold) =================
    print("\n  Tuning ALL models (RandomizedSearchCV + GroupKFold)...")
    S = Config.SEED
    gkf = GroupKFold(n_splits=3)
    tune_specs = get_tune_specs()
    tuned_models = {}

    for name, (base_est, param_grid, color, ls, n_iter) in tune_specs.items():
        njobs = 1 if isinstance(base_est, CoxPHWrapper) else -1
        rs = RandomizedSearchCV(base_est, param_grid, n_iter=n_iter, cv=gkf,
                                scoring='average_precision', random_state=S, n_jobs=njobs)
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        tuned_models[name] = (rs.best_estimator_, color, ls)
        print(f"    {name:<22s} CV PR-AUC={rs.best_score_:.3f}  {rs.best_params_}")

    # Build Stacking from tuned base models
    stacking_est = StackingClassifier(
        estimators=[
            ('lr', clone(tuned_models['Logistic Reg.'][0])),
            ('rf', clone(tuned_models['Random Forest'][0])),
            ('gbc', clone(tuned_models['GradientBoosting'][0])),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=S),
        cv=3, n_jobs=1, passthrough=False
    )
    tuned_models["Stacking"] = (stacking_est, "#17becf", "-")
    print(f"    {'Stacking':<22s} Built from tuned LR + RF + GBC")

    # ================= OOF threshold selection + refit =================
    print("\n  OOF threshold selection + evaluation...")
    results = {}

    print(f"\n{'='*95}")
    print(f"  {'Model':<22s} {'AUC':>6} {'PR-AUC':>8} {'Thr':>5} {'Acc':>6} {'BalAcc':>8} {'F1':>6}  {'TP':>4} {'FP':>5} {'FN':>4} {'TN':>5}")
    print(f"{'='*95}")

    for name, (model, color, ls) in tuned_models.items():
        oof_proba = np.zeros(len(y_tr))
        for fold_tr, fold_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[fold_tr], y_tr[fold_tr])
            oof_proba[fold_val] = m.predict_proba(X_tr_df.iloc[fold_val])[:, 1]

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.02, 0.60, 0.01):
            f = f1_score(y_tr, (oof_proba >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

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
            'model': model, 'color': color, 'ls': ls, 'oof_proba': oof_proba,
        }

        marker = " *" if auc == max(r['auc'] for r in results.values()) else "  "
        print(f"{marker}{name:<22s} {auc:>5.3f} {prauc:>7.3f} {best_thr:>4.2f} {acc:>5.3f} {bacc:>7.3f} {f1:>5.3f}  {tp:>4} {fp:>5} {fn:>4} {tn:>5}")

    print(f"{'='*95}")
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"  Best AUC: {best_name} ({results[best_name]['auc']:.3f})")
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")

    # ================= 评估增强: interval-level CI / calibration / DCA =================
    best_eval_name = best_prauc_name
    best_eval = results[best_eval_name]
    interval_eval_df = pd.DataFrame({
        'Patient_ID': df_long_test['Patient_ID'].values,
        'Y': y_te,
        'proba': best_eval['proba']
    })
    interval_cis = bootstrap_group_cis(interval_eval_df, best_eval['threshold'], group_col='Patient_ID')
    interval_metrics = compute_binary_metrics(y_te, best_eval['proba'], best_eval['threshold'])
    interval_cal = compute_calibration_stats(y_te, best_eval['proba'])

    print(f"\n  Interval-level report on best PR-AUC model: {best_eval_name}")
    print(f"    AUC         = {interval_metrics['auc']:.3f}  95% CI {format_ci(interval_cis, 'auc')}")
    print(f"    PR-AUC      = {interval_metrics['prauc']:.3f}  95% CI {format_ci(interval_cis, 'prauc')}")
    print(f"    Brier       = {interval_metrics['brier']:.3f}  95% CI {format_ci(interval_cis, 'brier')}")
    print(f"    Recall      = {interval_metrics['recall']:.3f}  95% CI {format_ci(interval_cis, 'recall')}")
    print(f"    Specificity = {interval_metrics['specificity']:.3f}  95% CI {format_ci(interval_cis, 'specificity')}")
    print(f"    Cal. Intcp  = {interval_cal['intercept']:.3f}  95% CI {format_ci(interval_cis, 'cal_intercept')}")
    print(f"    Cal. Slope  = {interval_cal['slope']:.3f}  95% CI {format_ci(interval_cis, 'cal_slope')}")
    print(f"    Threshold   = {interval_metrics['threshold']:.2f}")

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

    save_calibration_figure(
        y_te, best_eval['proba'],
        f"Calibration Curve ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "Calibration_Interval.png"
    )
    save_dca_figure(
        y_te, best_eval['proba'],
        f"Decision Curve Analysis ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "DCA_Interval.png"
    )
    save_threshold_sensitivity_figure(
        y_te, best_eval['proba'], best_eval['threshold'],
        f"Threshold Sensitivity ({best_eval_name}, interval-level)",
        Config.OUT_DIR / "Threshold_Sensitivity_Interval.png"
    )

    # ================= patient-level 汇总: any relapse during follow-up =================
    patient_tr = aggregate_patient_level(df_long_train, best_eval['oof_proba'])
    patient_te = aggregate_patient_level(df_long_test, best_eval['proba'])
    patient_thr = select_best_threshold(patient_tr['Y'].values, patient_tr['proba'].values,
                                        low=0.05, high=0.95, step=0.01)
    patient_metrics = compute_binary_metrics(patient_te['Y'].values, patient_te['proba'].values, patient_thr)
    patient_cis = bootstrap_group_cis(patient_te, patient_thr, group_col='Patient_ID')
    patient_cal = compute_calibration_stats(patient_te['Y'].values, patient_te['proba'].values)

    print(f"\n  Patient-level report (endpoint: any relapse during follow-up)")
    print(f"    Patients    = train {len(patient_tr)} / test {len(patient_te)}")
    print(f"    AUC         = {patient_metrics['auc']:.3f}  95% CI {format_ci(patient_cis, 'auc')}")
    print(f"    PR-AUC      = {patient_metrics['prauc']:.3f}  95% CI {format_ci(patient_cis, 'prauc')}")
    print(f"    Brier       = {patient_metrics['brier']:.3f}  95% CI {format_ci(patient_cis, 'brier')}")
    print(f"    Recall      = {patient_metrics['recall']:.3f}  95% CI {format_ci(patient_cis, 'recall')}")
    print(f"    Specificity = {patient_metrics['specificity']:.3f}  95% CI {format_ci(patient_cis, 'specificity')}")
    print(f"    Cal. Intcp  = {patient_cal['intercept']:.3f}  95% CI {format_ci(patient_cis, 'cal_intercept')}")
    print(f"    Cal. Slope  = {patient_cal['slope']:.3f}  95% CI {format_ci(patient_cis, 'cal_slope')}")
    print(f"    Threshold   = {patient_thr:.2f}")

    save_calibration_figure(
        patient_te['Y'].values, patient_te['proba'].values,
        f"Calibration Curve ({best_eval_name}, patient-level)",
        Config.OUT_DIR / "Calibration_Patient.png"
    )
    save_dca_figure(
        patient_te['Y'].values, patient_te['proba'].values,
        f"Decision Curve Analysis ({best_eval_name}, patient-level)",
        Config.OUT_DIR / "DCA_Patient.png"
    )
    save_threshold_sensitivity_figure(
        patient_te['Y'].values, patient_te['proba'].values, patient_thr,
        f"Threshold Sensitivity ({best_eval_name}, patient-level)",
        Config.OUT_DIR / "Threshold_Sensitivity_Patient.png"
    )
    save_patient_risk_strata(
        patient_tr, patient_te,
        Config.OUT_DIR / "Patient_Risk_Q1Q4.png"
    )

    # ================= 敏感性分析 1: patient-level 汇总规则 =================
    patient_sens_df, _ = evaluate_patient_aggregation_sensitivity(
        df_long_train, df_long_test, best_eval['oof_proba'], best_eval['proba']
    )
    print(f"\n  Patient-level aggregation sensitivity")
    for _, row in patient_sens_df.iterrows():
        print(f"    {row['Aggregation']:<18s} AUC={row['AUC']:.3f}  PR-AUC={row['PR_AUC']:.3f}  "
              f"Brier={row['Brier']:.3f}  Recall={row['Recall']:.3f}  Specificity={row['Specificity']:.3f}")
    patient_sens_df.to_csv(Config.OUT_DIR / "Patient_Aggregation_Sensitivity.csv", index=False)
    save_patient_aggregation_sensitivity_figure(
        patient_sens_df,
        Config.OUT_DIR / "Patient_Aggregation_Sensitivity.png"
    )

    # ================= 敏感性分析 2: 时间窗口 =================
    window_sens_df = evaluate_window_sensitivity(df_long_test, best_eval['proba'], best_eval['threshold'])
    print(f"\n  Window sensitivity (interval-level best model)")
    for _, row in window_sens_df.iterrows():
        print(f"    {row['Scenario']:<18s} n={int(row['Intervals'])}  events={int(row['Events'])}  "
              f"AUC={row['AUC']:.3f}  PR-AUC={row['PR_AUC']:.3f}  Brier={row['Brier']:.3f}")
    window_sens_df.to_csv(Config.OUT_DIR / "Window_Sensitivity.csv", index=False)

    # ================= 绘图 3: SHAP (优先解释 PR-AUC 最优模型) =================
    best_shap_name = best_prauc_name
    best_shap_model = results[best_shap_name]['model']
    run_shap_analysis(best_shap_name, best_shap_model, X_tr_df, feat_names)

def main():
    print("=" * 80)
    print("  Multi-State Relapse Analysis (Temporal-Safe MissForest)")
    print("=" * 80)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, pids = load_data()
    n_static = X_s_raw.shape[1]
    print(f"  Records: {len(pids)}  Unique patients: {len(set(pids))}")

    # --- Temporal split by enrollment order ---
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    n_train_patients = len(train_pids)
    n_test_patients = len(unique_pids) - n_train_patients
    print(f"  Train: {n_train_patients} patients, {tr_mask.sum()} records")
    print(f"  Test:  {n_test_patients} patients, {te_mask.sum()} records")

    # --- Phase 1: Transition heatmaps (depth-6 imputer for descriptive view) ---
    print(f"\n--- Phase 1: State Transition Heatmaps ---")
    max_depth = eval_raw.shape[1]  # 6
    hm_raw = np.hstack([X_s_raw, ft3_raw[:, :max_depth], ft4_raw[:, :max_depth],
                         tsh_raw[:, :max_depth], eval_raw])
    hm_imp_path = Config.OUT_DIR / f"missforest_depth{max_depth}.pkl"
    if hm_imp_path.exists():
        hm_imputer = joblib.load(hm_imp_path)
        print(f"  MissForest depth-{max_depth}: loaded from cache")
    else:
        print(f"  MissForest depth-{max_depth}: fitting on train ({n_train_patients} patients, {tr_mask.sum()} records)...")
        hm_imputer = fit_missforest(hm_raw[tr_mask])
        joblib.dump(hm_imputer, hm_imp_path)
    hm_filled = apply_missforest(hm_raw, hm_imputer, max_depth)
    _, _, _, _, ev_hm = split_imputed(hm_filled, n_static, max_depth, max_depth)
    S_full = build_states_from_labels(ev_hm)
    plot_transition_heatmaps(S_full)

    # --- Phase 2: Temporal-safe per-depth MissForest + long-format pooling ---
    print(f"\n--- Phase 2: Building temporal-safe long-format data ---")
    df_tr_parts, df_te_parts = [], []

    for depth in range(1, 7):
        k = depth - 1
        n_lab = depth
        n_ev = depth
        interval_name = f"{Config.TIME_STAMPS[k]}->{Config.TIME_STAMPS[k+1]}"

        raw = np.hstack([X_s_raw,
                         ft3_raw[:, :n_lab], ft4_raw[:, :n_lab], tsh_raw[:, :n_lab],
                         eval_raw[:, :n_ev]])

        imp_path = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        if imp_path.exists():
            imputer = joblib.load(imp_path)
        else:
            print(f"  MissForest depth-{depth} ({interval_name}): fitting on train ({n_train_patients} patients, {tr_mask.sum()} records)...")
            imputer = fit_missforest(raw[tr_mask])
            joblib.dump(imputer, imp_path)

        filled_tr = apply_missforest(raw[tr_mask], imputer, n_ev)
        filled_te = apply_missforest(raw[te_mask], imputer, n_ev)

        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, n_lab, n_ev)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, n_lab, n_ev)

        S_tr = build_states_from_labels(ev_tr)
        S_te = build_states_from_labels(ev_te)

        X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_tr_parts.append(build_long_format_data(xs_tr, X_d_tr, S_tr, pids[tr_mask], target_k=k))
        df_te_parts.append(build_long_format_data(xs_te, X_d_te, S_te, pids[te_mask], target_k=k))

        n_tr_k = len(df_tr_parts[-1])
        n_te_k = len(df_te_parts[-1])
        print(f"    depth-{depth} ({interval_name}): train {n_tr_k}  test {n_te_k} rows")

    df_tr = pd.concat(df_tr_parts, ignore_index=True)
    df_te = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_tr)}  test {len(df_te)} rows")

    # --- Phase 3: Model comparison ---
    print(f"\n--- Phase 3: Multi-Model Comparison ---")
    train_and_evaluate_hazard_strict(df_tr, df_te)

    print(f"\n  All plots saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()