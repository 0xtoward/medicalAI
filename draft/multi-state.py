import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, RandomForestRegressor,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"

# ==========================================
# 1. Config
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
        'Eval_Cols': [35, 44, 53, 62, 71, 80],
    }
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos",
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb",
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose"]

    TIME_STAMPS = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
    STATE_NAMES = ["Hyper", "Normal", "Hypo"]
    N_CLASSES = 3

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.SEED)


# ==========================================
# 2. Data loading & imputation (same as before)
# ==========================================
def load_data():
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:].reset_index(drop=True)
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()
    valid_idx = df.iloc[:, Config.COL_IDX['Outcome']].dropna().index
    df = df.loc[valid_idx].reset_index(drop=True)

    X_s = df.iloc[:, Config.COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').values
    gt = df.iloc[:, 3].astype(str).str.strip()
    X_s[:, 0] = np.where((gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1), 1.0, 0.0)

    ft3 = df.iloc[:, Config.COL_IDX['FT3_Sequence']].apply(pd.to_numeric, errors='coerce').values
    ft4 = df.iloc[:, Config.COL_IDX['FT4_Sequence']].apply(pd.to_numeric, errors='coerce').values
    tsh = df.iloc[:, Config.COL_IDX['TSH_Sequence']].apply(pd.to_numeric, errors='coerce').values
    ev = df.iloc[:, Config.COL_IDX['Eval_Cols']].apply(pd.to_numeric, errors='coerce').values

    return X_s, ft3, ft4, tsh, ev, df['Patient_ID'].values


def build_impute_matrix(X_s, ft3, ft4, tsh, evals):
    return np.hstack([X_s, ft3, ft4, tsh, evals])


def split_imputed(arr, n_static, n_seq):
    i = 0
    X_s = arr[:, i:i+n_static]; i += n_static
    ft3 = arr[:, i:i+n_seq];    i += n_seq
    ft4 = arr[:, i:i+n_seq];    i += n_seq
    tsh = arr[:, i:i+n_seq];    i += n_seq
    evals = arr[:, i:]
    return X_s, ft3, ft4, tsh, evals


def fit_missforest(raw_matrix):
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                        random_state=Config.SEED, n_jobs=1),
        max_iter=10, random_state=Config.SEED
    )
    imputer.fit(raw_matrix)
    return imputer


def apply_missforest(raw_matrix, imputer, n_eval_cols):
    filled = imputer.transform(raw_matrix)
    eval_start = filled.shape[1] - n_eval_cols
    filled[:, eval_start:] = np.clip(np.round(filled[:, eval_start:]), 1, 3)
    return filled


def build_states_from_labels(eval_imputed):
    """0M=all Hyper; 1M-24M from doctor labels: 1→Hyper(0), 3→Normal(1), 2→Hypo(2)."""
    N = eval_imputed.shape[0]
    T = len(Config.TIME_STAMPS)
    S = np.zeros((N, T), dtype=int)
    label_map = {1: 0, 3: 1, 2: 2}
    for t in range(eval_imputed.shape[1]):
        for i in range(N):
            S[i, t + 1] = label_map.get(int(round(eval_imputed[i, t])), 0)
    return S


# ==========================================
# 3. Transition heatmaps
# ==========================================
def plot_transition_heatmaps(S_matrix):
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
        print(f"  [{label}]  transition: {(mat.sum()-np.trace(mat))/mat.sum():.1%}  "
              f"relapse(N->H): {mat[1, 0]}")

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close(fig)
    print(f"\n  Total transitions: {total.sum()}  Stable: {np.trace(total)}  "
          f"Relapse(N->H): {total[1,0]}  Hypo(N->Hp): {total[1,2]}")


# ==========================================
# 4. Feature engineering: long format with 3-class target
# ==========================================
def build_long_format_data(X_s, X_d, S_matrix, pids):
    """For each (patient, interval) where S_k == Normal, predict S_{k+1} ∈ {Hyper, Normal, Hypo}."""
    records = []
    N, T = S_matrix.shape
    for i in range(N):
        pid, xs = pids[i], X_s[i]
        for k in range(T - 1):
            if S_matrix[i, k] != 1:
                continue
            labs_k = X_d[i, k, :]
            labs_0 = X_d[i, 0, :]
            y_next = S_matrix[i, k + 1]   # 0=Hyper, 1=Normal, 2=Hypo

            hist = S_matrix[i, :k]
            prev_state = S_matrix[i, k - 1] if k > 0 else -1
            ever_hyper = int(0 in hist)
            ever_hypo = int(2 in hist)
            time_in_normal = int(np.sum(hist == 1))

            delta_ft4_k0 = labs_k[1] - labs_0[1]
            delta_tsh_k0 = (np.log1p(np.clip(labs_k[2], 0, None))
                            - np.log1p(np.clip(labs_0[2], 0, None)))
            if k > 0:
                labs_prev = X_d[i, k - 1, :]
                delta_ft4_1s = labs_k[1] - labs_prev[1]
                delta_tsh_1s = (np.log1p(np.clip(labs_k[2], 0, None))
                                - np.log1p(np.clip(labs_prev[2], 0, None)))
            else:
                delta_ft4_1s = delta_tsh_1s = 0.0

            records.append({
                "Patient_ID": pid,
                "Interval_ID": k,
                "Interval_Name": f"{Config.TIME_STAMPS[k]}->{Config.TIME_STAMPS[k+1]}",
                "Y_NextState": y_next,
                **dict(zip(Config.STATIC_NAMES, xs)),
                "FT3_Current": labs_k[0],
                "FT4_Current": labs_k[1],
                "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                "Prev_State": str(prev_state),
                "Ever_Hyper_Before": ever_hyper,
                "Ever_Hypo_Before": ever_hypo,
                "Time_In_Normal": time_in_normal,
                "Delta_FT4_k0": delta_ft4_k0,
                "Delta_TSH_k0": delta_tsh_k0,
                "Delta_FT4_1step": delta_ft4_1s,
                "Delta_TSH_1step": delta_tsh_1s,
            })
    return pd.DataFrame(records)


# ==========================================
# 5. Model zoo (multi-class compatible)
# ==========================================
def get_model_zoo():
    S = Config.SEED
    return {
        "Logistic Reg.": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, C=0.1,
                                                multi_class='multinomial', random_state=S))]),
            "#1f77b4", "-."
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, max_depth=5,
                                   random_state=S, n_jobs=1),
            "#ff7f0e", "--"
        ),
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                               algorithm='SAMME', random_state=S),
            "#9467bd", "--"
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=3, subsample=0.8, random_state=S),
            "#2ca02c", "--"
        ),
        "XGBoost": (
            xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                              subsample=0.8, eval_metric='mlogloss',
                              random_state=S, n_jobs=1, verbosity=0),
            "#d62728", "--"
        ),
        "LightGBM": (
            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                               subsample=0.8, random_state=S,
                               n_jobs=1, verbosity=-1),
            "#e377c2", "--"
        ),
        "MLP": (
            Pipeline([('scaler', StandardScaler()),
                      ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                            learning_rate_init=0.001, alpha=0.01,
                                            early_stopping=True, validation_fraction=0.15,
                                            n_iter_no_change=50, random_state=S))]),
            "#8c564b", ":"
        ),
        "Stacking": (
            StackingClassifier(
                estimators=[
                    ('lr', Pipeline([('s', StandardScaler()),
                                     ('lr', LogisticRegression(max_iter=2000, C=0.1,
                                                               multi_class='multinomial',
                                                               random_state=S))])),
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=4,
                                                  random_state=S, n_jobs=1)),
                    ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                       max_depth=3, random_state=S)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, multi_class='multinomial',
                                                    random_state=S),
                cv=3, n_jobs=1, passthrough=False
            ),
            "#17becf", "-"
        ),
    }


# ==========================================
# 6. Multi-class evaluation + plots
# ==========================================
def train_and_evaluate(df_tr, df_te):
    feat_cols = Config.STATIC_NAMES + [
        "FT3_Current", "FT4_Current", "logTSH_Current",
        "Ever_Hyper_Before", "Ever_Hypo_Before", "Time_In_Normal",
        "Delta_FT4_k0", "Delta_TSH_k0", "Delta_FT4_1step", "Delta_TSH_1step"
    ]
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    dummies = pd.get_dummies(df_all['Interval_Name'], prefix='Win')
    prev_d = pd.get_dummies(df_all['Prev_State'], prefix='Prev')
    X_df = pd.concat([df_all[feat_cols], dummies, prev_d], axis=1)

    n_tr = len(df_tr)
    feat_names = list(X_df.columns)
    X_tr_df, y_tr = X_df.iloc[:n_tr], df_tr['Y_NextState'].values
    X_te_df, y_te = X_df.iloc[n_tr:], df_te['Y_NextState'].values
    groups_tr = df_tr['Patient_ID'].values

    cls_counts = {c: int((y_tr == c).sum()) for c in range(3)}
    cls_counts_te = {c: int((y_te == c).sum()) for c in range(3)}
    print(f"\n  Train: {df_tr['Patient_ID'].nunique()} patients, {n_tr} intervals")
    print(f"    class dist: Hyper={cls_counts[0]}  Normal={cls_counts[1]}  Hypo={cls_counts[2]}")
    print(f"  Test : {df_te['Patient_ID'].nunique()} patients, {len(y_te)} intervals")
    print(f"    class dist: Hyper={cls_counts_te[0]}  Normal={cls_counts_te[1]}  Hypo={cls_counts_te[2]}")

    # ---- Hyperparameter tuning (GroupKFold) ----
    print("\n  Tuning (GroupKFold)...")
    S = Config.SEED
    gkf = GroupKFold(n_splits=3)
    tuned = {}

    tune_specs = {
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=S),
            {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [2, 3, 5], 'subsample': [0.7, 0.8, 1.0]}
        ),
        "XGBoost": (
            xgb.XGBClassifier(eval_metric='mlogloss', random_state=S, n_jobs=1, verbosity=0),
            {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1],
             'max_depth': [2, 3, 5], 'subsample': [0.7, 0.8, 1.0]}
        ),
    }
    for tname, (base, grid) in tune_specs.items():
        rs = RandomizedSearchCV(base, grid, n_iter=10, cv=gkf,
                                scoring='f1_macro', random_state=S, n_jobs=-1)
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        tuned[tname] = rs.best_estimator_
        print(f"    {tname}: {rs.best_params_}  (CV macro-F1={rs.best_score_:.3f})")

    model_zoo = get_model_zoo()
    for tname, est in tuned.items():
        _, color, ls = model_zoo[tname]
        model_zoo[tname] = (est, color, ls)

    # ---- OOF + evaluation ----
    print("\n  Training & evaluating (OOF on train, then refit)...")
    results = {}

    header = (f"  {'Model':<18s} {'mAUC':>5} {'F1mac':>6} {'BalAcc':>6}"
              f"  {'AUC_H':>5} {'AUC_N':>5} {'AUC_Hp':>5}"
              f"  {'F1_H':>4} {'F1_N':>4} {'F1_Hp':>4}")
    print(f"\n{'='*90}")
    print(header)
    print(f"{'='*90}")

    for name, (model, color, ls) in model_zoo.items():
        # OOF to get reliable train-set probabilities
        oof_proba = np.zeros((n_tr, Config.N_CLASSES))
        for fold_tr, fold_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[fold_tr], y_tr[fold_tr])
            oof_proba[fold_val] = m.predict_proba(X_tr_df.iloc[fold_val])

        # Refit on full train, predict test
        model.fit(X_tr_df, y_tr)
        proba = model.predict_proba(X_te_df)
        pred = proba.argmax(axis=1)

        # Metrics
        y_te_bin = label_binarize(y_te, classes=[0, 1, 2])
        try:
            macro_auc = roc_auc_score(y_te_bin, proba, multi_class='ovr', average='macro')
        except ValueError:
            macro_auc = 0.5
        per_class_auc = []
        for c in range(Config.N_CLASSES):
            try:
                per_class_auc.append(roc_auc_score((y_te == c).astype(int), proba[:, c]))
            except ValueError:
                per_class_auc.append(0.5)

        f1_mac = f1_score(y_te, pred, average='macro', zero_division=0)
        bacc = balanced_accuracy_score(y_te, pred)
        f1_per = f1_score(y_te, pred, average=None, zero_division=0)
        cm = confusion_matrix(y_te, pred, labels=[0, 1, 2])

        results[name] = {
            'proba': proba, 'pred': pred, 'macro_auc': macro_auc,
            'per_auc': per_class_auc, 'f1_mac': f1_mac, 'bacc': bacc,
            'f1_per': f1_per, 'cm': cm, 'model': model, 'color': color, 'ls': ls,
        }

        star = " *" if macro_auc == max(r['macro_auc'] for r in results.values()) else "  "
        print(f"{star}{name:<18s} {macro_auc:>4.3f} {f1_mac:>5.3f} {bacc:>5.3f}"
              f"  {per_class_auc[0]:>4.3f} {per_class_auc[1]:>4.3f} {per_class_auc[2]:>4.3f}"
              f"  {f1_per[0]:>.2f} {f1_per[1]:>.2f} {f1_per[2]:>.2f}")

    print(f"{'='*90}")
    best = max(results, key=lambda k: results[k]['macro_auc'])
    print(f"  Best macro-AUC: {best} ({results[best]['macro_auc']:.3f})")
    best_f1 = max(results, key=lambda k: results[k]['f1_mac'])
    print(f"  Best macro-F1:  {best_f1} ({results[best_f1]['f1_mac']:.3f})")

    # ---- Classification report for the best model ----
    print(f"\n  Classification Report ({best}):")
    print(classification_report(y_te, results[best]['pred'],
                                target_names=Config.STATE_NAMES, zero_division=0))

    # ================= Plot 1: State probability by interval =================
    INTERVAL_ORDER = {"0M->1M": 0, "1M->3M": 1, "3M->6M": 2,
                      "6M->12M": 3, "12M->18M": 4, "18M->24M": 5}

    df_test = df_te.copy()
    empirical = df_test.groupby('Interval_Name')['Y_NextState'].value_counts(normalize=True).unstack(fill_value=0)
    for c in range(3):
        if c not in empirical.columns:
            empirical[c] = 0.0
    empirical = empirical[[0, 1, 2]]
    empirical.columns = Config.STATE_NAMES
    empirical['sk'] = empirical.index.map(INTERVAL_ORDER)
    empirical = empirical.sort_values('sk').drop(columns='sk')

    colors_state = ['#d62728', '#2ca02c', '#1f77b4']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: actual stacked bar
    empirical.plot.bar(stacked=True, ax=axes[0], color=colors_state, edgecolor='white', width=0.7)
    axes[0].set_title("Actual Next-State Distribution\n(Given Normal at t)", fontsize=12)
    axes[0].set_ylabel("Proportion")
    axes[0].set_xlabel("Interval")
    axes[0].legend(title="Next State", fontsize=9)
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylim(0, 1.05)

    # Right: best model predicted probabilities
    df_test['P_Hyper'] = results[best]['proba'][:, 0]
    df_test['P_Normal'] = results[best]['proba'][:, 1]
    df_test['P_Hypo'] = results[best]['proba'][:, 2]
    pred_by_int = df_test.groupby('Interval_Name')[['P_Hyper', 'P_Normal', 'P_Hypo']].mean()
    pred_by_int['sk'] = pred_by_int.index.map(INTERVAL_ORDER)
    pred_by_int = pred_by_int.sort_values('sk').drop(columns='sk')
    pred_by_int.columns = Config.STATE_NAMES

    pred_by_int.plot.bar(stacked=True, ax=axes[1], color=colors_state, edgecolor='white', width=0.7)
    axes[1].set_title(f"Predicted Next-State Distribution ({best})\nmacro-AUC={results[best]['macro_auc']:.3f}",
                      fontsize=12)
    axes[1].set_ylabel("Mean Predicted Probability")
    axes[1].set_xlabel("Interval")
    axes[1].legend(title="Next State", fontsize=9)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "NextState_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 2: Model comparison bar (macro-AUC + macro-F1) =================
    names = list(results.keys())
    m_aucs = [results[n]['macro_auc'] for n in names]
    m_f1s = [results[n]['f1_mac'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, m_aucs, w, label='Macro AUC (OVR)', color='steelblue')
    b2 = ax.bar(x + w/2, m_f1s, w, label='Macro F1', color='coral')
    for b in b1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("3-Class Next-State: Model Comparison", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(m_aucs + m_f1s) + 0.15)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Model_Comparison_Bar.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 3: Confusion matrix of the best model =================
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_best = results[best]['cm']
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.STATE_NAMES, yticklabels=Config.STATE_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {best}\nmacro-AUC={results[best]['macro_auc']:.3f}  "
                 f"macro-F1={results[best]['f1_mac']:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 4: Per-class AUC comparison =================
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.25
    for c, (sname, col) in enumerate(zip(Config.STATE_NAMES, colors_state)):
        vals = [results[n]['per_auc'][c] for n in names]
        ax.bar(x + (c - 1) * w, vals, w, label=f'AUC: {sname}', color=col, alpha=0.85)
        for xi, v in zip(x + (c - 1) * w, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("OVR AUC")
    ax.set_title("Per-Class AUC by Model (Hyper = Relapse)", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "PerClass_AUC.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 5: SHAP for Hyper class (best compatible tree model) =================
    # GBC TreeExplainer doesn't support multi-class; RF, XGB, LGBM do
    shap_ok = {n: r for n, r in results.items()
               if isinstance(r['model'], (RandomForestClassifier, xgb.XGBClassifier,
                                          lgb.LGBMClassifier))}
    if shap_ok:
        best_tree = max(shap_ok, key=lambda k: shap_ok[k]['macro_auc'])
        model_shap = shap_ok[best_tree]['model']
        print(f"\n  SHAP analysis (Hyper class) on: {best_tree}")
        try:
            if isinstance(model_shap, xgb.XGBClassifier):
                dm = xgb.DMatrix(X_tr_df, feature_names=feat_names)
                contribs = model_shap.get_booster().predict(dm, pred_contribs=True)
                shap_vals = contribs[:, 0, :-1] if contribs.ndim == 3 else contribs[:, :-1]
            else:
                explainer = shap.TreeExplainer(model_shap)
                sv = explainer.shap_values(X_tr_df)
                shap_vals = sv[0] if isinstance(sv, list) else sv[:, :, 0]

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_tr_df, feature_names=feat_names,
                              max_display=20, show=False)
            plt.title(f"What Drives Relapse (Hyper)? ({best_tree})", fontsize=14, pad=20)
            plt.savefig(Config.OUT_DIR / "SHAP_Hyper.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  SHAP failed: {e}")


# ==========================================
# 7. Main
# ==========================================
def main():
    print("=" * 80)
    print("  Multi-State 3-Class Next-State Prediction")
    print("  P(S_{k+1} in {Hyper, Normal, Hypo} | S_k = Normal)")
    print("=" * 80)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, pids = load_data()
    n_static, n_seq, n_eval = X_s_raw.shape[1], ft3_raw.shape[1], eval_raw.shape[1]
    print(f"  Records: {len(pids)}  Unique patients: {len(set(pids))}")

    # Temporal split
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask

    # MissForest (train-only fit, cached)
    imputer_path = Config.OUT_DIR / "missforest_imputer.pkl"
    raw_all = build_impute_matrix(X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw)

    if imputer_path.exists():
        print(f"\n  Loading cached MissForest from {imputer_path}")
        imputer = joblib.load(imputer_path)
    else:
        print(f"\n  Fitting MissForest on train ({tr_mask.sum()} records)...")
        imputer = fit_missforest(raw_all[tr_mask])
        joblib.dump(imputer, imputer_path)
        print(f"  Saved to {imputer_path}")
    print("  Imputing train & test...")

    filled_tr = apply_missforest(raw_all[tr_mask], imputer, n_eval)
    filled_te = apply_missforest(raw_all[te_mask], imputer, n_eval)
    xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, n_seq)
    xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, n_seq)

    # Phase 1: Transition heatmaps
    print("\n--- Phase 1: State Transition Heatmaps ---")
    ev_full = np.vstack([ev_tr, ev_te])
    S_full = build_states_from_labels(ev_full)
    plot_transition_heatmaps(S_full)

    # Phase 2: 3-class model comparison
    print("\n--- Phase 2: 3-Class Next-State Prediction ---")
    def build_set(xs, ft3, ft4, tsh, ev, mask_pids):
        X_d = np.stack([ft3, ft4, tsh], axis=-1)
        S = build_states_from_labels(ev)
        return build_long_format_data(xs, X_d, S, mask_pids)

    df_tr = build_set(xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr, pids[tr_mask])
    df_te = build_set(xs_te, ft3_te, ft4_te, tsh_te, ev_te, pids[te_mask])
    train_and_evaluate(df_tr, df_te)

    print(f"\n  All plots saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
