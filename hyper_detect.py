"""
hyper_detect.py — 纯二分类甲亢截获模型
目标: P(Hyper | 3M or 6M 截面数据)
  • MissForest (train-only)
  • 8 models + RandomizedSearchCV + OOF threshold
  • Stacking / VotingClassifier
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             f1_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, RandomForestRegressor,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import shap

plt.rcParams["font.family"] = "DejaVu Sans"


# ==========================================
# 1. Config
# ==========================================
class Config:
    FILE_PATH = "1003.xlsx"
    SEED = 42
    OUT_DIR = Path("./hyper_detect_result/")

    COL_IDX = {
        'ID': 0, 'Outcome': 14,
        'Static_Feats': [3, 4, 5, 6, 7, 8, 9, 11, 12, 19, 20, 21, 22, 23, 24, 25],
        'FT3_Sequence': [16, 29, 38, 47, 56, 65, 74],
        'FT4_Sequence': [17, 30, 39, 48, 57, 66, 75],
        'TSH_Sequence': [18, 31, 40, 49, 58, 67, 76],
    }
    STATIC_NAMES = ["Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos",
                    "ThyroidW", "RAI3d", "TreatCount", "TGAb", "TPOAb",
                    "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose"]


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.SEED)


# ==========================================
# 2. Data loading & imputation (from all2.py)
# ==========================================
def load_data_raw():
    df = pd.read_excel(Config.FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    df['Patient_ID'] = df.iloc[:, Config.COL_IDX['ID']].ffill()

    X_s = df.iloc[:, Config.COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').values
    gt = df.iloc[:, 3].astype(str).str.strip()
    gender = np.zeros(len(df), dtype=np.float32)
    gender[(gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1)] = 1.0
    X_s[:, 0] = gender

    y_raw = df.iloc[:, Config.COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    ft3 = df.iloc[:, Config.COL_IDX['FT3_Sequence']].apply(pd.to_numeric, errors='coerce').values
    ft4 = df.iloc[:, Config.COL_IDX['FT4_Sequence']].apply(pd.to_numeric, errors='coerce').values
    tsh = df.iloc[:, Config.COL_IDX['TSH_Sequence']].apply(pd.to_numeric, errors='coerce').values

    valid_idx = y_raw.dropna().index
    take = df.index.get_indexer(valid_idx)
    y = y_raw.loc[valid_idx].values.astype(int)
    return X_s[take], ft3[take], ft4[take], tsh[take], y, df.loc[valid_idx, 'Patient_ID'].values


def fit_missforest(matrix):
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                        random_state=Config.SEED, n_jobs=1),
        max_iter=10, random_state=Config.SEED
    )
    imputer.fit(matrix)
    return imputer


def split_imputed(arr, n_static, n_seq):
    i = 0
    X_s = arr[:, i:i+n_static]; i += n_static
    ft3 = arr[:, i:i+n_seq];    i += n_seq
    ft4 = arr[:, i:i+n_seq];    i += n_seq
    tsh = arr[:, i:i+n_seq]
    return X_s, ft3, ft4, tsh


def extract_flat_features(X_s, X_d, seq_len):
    X_d_t = X_d[:, :seq_len, :]
    X_d_flat = X_d_t.reshape(X_d_t.shape[0], -1)
    deltas = []
    names = list(Config.STATIC_NAMES)

    times = ["0M", "1M", "3M", "6M", "12M", "18M", "24M"]
    for t in range(seq_len):
        for h in ["FT3", "FT4", "TSH"]:
            names.append(f"{h}_{times[t]}")

    if seq_len >= 3:
        deltas.append(X_d_t[:, 2, :] - X_d_t[:, 0, :])
        for h in ["FT3", "FT4", "TSH"]:
            names.append(f"D_{h}_3M-0M")
    if seq_len >= 4:
        deltas.append(X_d_t[:, 3, :] - X_d_t[:, 0, :])
        for h in ["FT3", "FT4", "TSH"]:
            names.append(f"D_{h}_6M-0M")
        deltas.append(X_d_t[:, 3, :] - X_d_t[:, 2, :])
        for h in ["FT3", "FT4", "TSH"]:
            names.append(f"D_{h}_6M-3M")

    parts = [X_s, X_d_flat] + deltas
    return np.concatenate(parts, axis=1), names


# ==========================================
# 3. Tuning specs for ALL models
# ==========================================
def get_tune_specs():
    """Each entry: (base_estimator, param_grid, color, linestyle, n_iter)."""
    S = Config.SEED
    return {
        "Logistic Reg.": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, random_state=S))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__penalty': ['l1', 'l2'],
             'lr__solver': ['saga']},
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
            xgb.XGBClassifier(eval_metric='logloss', random_state=S, n_jobs=1, verbosity=0),
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
            {'mlp__hidden_layer_sizes': [(32,), (64, 32), (128, 64), (64, 32, 16)],
             'mlp__alpha': [0.001, 0.01, 0.05, 0.1],
             'mlp__learning_rate_init': [0.0005, 0.001, 0.005, 0.01]},
            "#8c564b", ":", 16
        ),
    }


# ==========================================
# 4. Evaluation + plots
# ==========================================
def plot_shap_binary(model, X_tr_df, feat_names, title, filename):
    print(f"  SHAP: {title}...")
    try:
        if isinstance(model, xgb.XGBClassifier):
            dm = xgb.DMatrix(X_tr_df, feature_names=feat_names)
            contribs = model.get_booster().predict(dm, pred_contribs=True)
            sv = contribs[:, :-1]
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_tr_df)
            if isinstance(sv, list):
                sv = sv[1]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_tr_df, feature_names=feat_names, max_display=20, show=False)
        plt.title(title, fontsize=14, pad=20)
        plt.savefig(Config.OUT_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    SHAP failed: {e}")


def run_experiment(landmark_name, seq_len):
    print(f"\n{'='*100}")
    print(f"  {landmark_name}  —  Binary: Hyper vs Non-Hyper")
    print(f"{'='*100}")

    # ---- Data ----
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, y_raw, pids = load_data_raw()
    n_static = X_s_raw.shape[1]

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    train_idx = np.where(np.array([pid in train_pids for pid in pids]))[0]
    test_idx = np.where(np.array([pid not in train_pids for pid in pids]))[0]
    n_train_patients = len(train_pids)
    n_test_patients = len(unique_pids) - n_train_patients

    # Temporal-safe MissForest: only use columns ≤ seq_len (no future data)
    ft3_cut = ft3_raw[:, :seq_len]
    ft4_cut = ft4_raw[:, :seq_len]
    tsh_cut = tsh_raw[:, :seq_len]
    raw_all = np.hstack([X_s_raw, ft3_cut, ft4_cut, tsh_cut])

    imputer_path = Config.OUT_DIR / f"missforest_seqlen{seq_len}.pkl"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        print(f"  MissForest (seq≤{seq_len}): loaded from cache")
    else:
        print(f"  MissForest (seq≤{seq_len}): fitting on train ({n_train_patients} patients, {len(train_idx)} records)...")
        imputer = fit_missforest(raw_all[train_idx])
        joblib.dump(imputer, imputer_path)
        print(f"  MissForest (seq≤{seq_len}): saved to {imputer_path}")

    filled_tr = imputer.transform(raw_all[train_idx])
    filled_te = imputer.transform(raw_all[test_idx])
    xs_tr, f3_tr, f4_tr, ts_tr = split_imputed(filled_tr, n_static, seq_len)
    xs_te, f3_te, f4_te, ts_te = split_imputed(filled_te, n_static, seq_len)

    Xd_tr = np.stack([f3_tr, f4_tr, ts_tr], axis=-1)
    Xd_te = np.stack([f3_te, f4_te, ts_te], axis=-1)
    X_tr_feat, feat_names = extract_flat_features(xs_tr, Xd_tr, seq_len)
    X_te_feat, _ = extract_flat_features(xs_te, Xd_te, seq_len)

    scaler = StandardScaler().fit(X_tr_feat)
    X_tr = scaler.transform(X_tr_feat)
    X_te = scaler.transform(X_te_feat)

    # Binary label: 1=Hyper (outcome==1), 0=Non-Hyper
    y_tr = (y_raw[train_idx] == 1).astype(int)
    y_te = (y_raw[test_idx] == 1).astype(int)
    groups_tr = pids[train_idx]

    print(f"  Train: {n_train_patients} patients, {len(y_tr)} records  (Hyper={y_tr.sum()}, Non-Hyper={len(y_tr)-y_tr.sum()})")
    print(f"  Test:  {n_test_patients} patients, {len(y_te)} records  (Hyper={y_te.sum()}, Non-Hyper={len(y_te)-y_te.sum()})")
    print(f"  Features: {len(feat_names)}")

    X_tr_df = pd.DataFrame(X_tr, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, columns=feat_names)

    # ---- Hyperparameter tuning: ALL models (GroupKFold) ----
    print("\n  Tuning ALL models (RandomizedSearchCV + GroupKFold)...")
    S = Config.SEED
    gkf = GroupKFold(n_splits=3)
    model_zoo = {}

    for name, (base, grid, color, ls, n_iter) in get_tune_specs().items():
        rs = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=gkf,
                                scoring='average_precision', random_state=S, n_jobs=-1)
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        model_zoo[name] = (rs.best_estimator_, color, ls)
        print(f"    {name:<18s} CV PR-AUC={rs.best_score_:.3f}  {rs.best_params_}")

    # Stacking: assemble from the tuned base models
    print("    Building Stacking from tuned base models...")
    stacking = StackingClassifier(
        estimators=[
            ('lr', clone(model_zoo['Logistic Reg.'][0])),
            ('rf', clone(model_zoo['Random Forest'][0])),
            ('xgb', clone(model_zoo['XGBoost'][0])),
        ],
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=S),
        cv=3, n_jobs=1, passthrough=False
    )
    model_zoo["Stacking"] = (stacking, "#17becf", "-")

    # ---- OOF threshold + evaluate ----
    print("\n  OOF threshold selection + evaluation...")
    results = {}

    hdr = (f"  {'Model':<18s} {'AUC':>5} {'PR-AUC':>7} {'Thr':>4} {'Acc':>5} "
           f"{'BalAcc':>6} {'F1':>5}  {'TP':>3} {'FP':>4} {'FN':>3} {'TN':>4}")
    print(f"\n{'='*85}")
    print(hdr)
    print(f"{'='*85}")

    for name, (model, color, ls) in model_zoo.items():
        oof = np.zeros(len(y_tr))
        for f_tr, f_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            m = clone(model)
            m.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
            oof[f_val] = m.predict_proba(X_tr_df.iloc[f_val])[:, 1]

        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.05, 0.60, 0.01):
            f = f1_score(y_tr, (oof >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

        model.fit(X_tr_df, y_tr)
        proba = model.predict_proba(X_te_df)[:, 1]
        pred = (proba >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba)
        prauc = average_precision_score(y_te, proba)
        acc = accuracy_score(y_te, pred)
        bacc = balanced_accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        cm = confusion_matrix(y_te, pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = dict(proba=proba, auc=auc, prauc=prauc, acc=acc, bacc=bacc,
                             f1=f1, thr=best_thr, tp=tp, fp=fp, fn=fn, tn=tn,
                             model=model, color=color, ls=ls)

        star = " *" if auc == max(r['auc'] for r in results.values()) else "  "
        print(f"{star}{name:<18s} {auc:>4.3f} {prauc:>6.3f} {best_thr:>3.2f} {acc:>4.3f} "
              f"{bacc:>5.3f} {f1:>4.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}")

    print(f"{'='*85}")
    best_auc_name = max(results, key=lambda k: results[k]['auc'])
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    print(f"  Best AUC:    {best_auc_name} ({results[best_auc_name]['auc']:.3f})")
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")

    tag = landmark_name.replace(" ", "_")[:4]

    # ================= Plot 1: ROC curves =================
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_te, r['proba'])
        ax.plot(fpr, tpr, lw=2, color=r['color'], linestyle=r['ls'],
                label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Hyper Detection ROC — {landmark_name}", fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    fig.savefig(Config.OUT_DIR / f"ROC_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 2: AUC + PR-AUC bar chart =================
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    praucs = [results[n]['prauc'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, aucs, w, label='ROC-AUC', color='steelblue')
    b2 = ax.bar(x + w/2, praucs, w, label='PR-AUC', color='coral')
    for b in b1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha='center', fontsize=7.5)
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha='center', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {landmark_name}", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(aucs) + 0.12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / f"Bar_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 3: Confusion matrix of best model =================
    best_name = best_auc_name
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_best = confusion_matrix(y_te, (results[best_name]['proba'] >= results[best_name]['thr']).astype(int))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Hyper', 'Hyper'], yticklabels=['Non-Hyper', 'Hyper'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"CM: {best_name}  AUC={results[best_name]['auc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / f"CM_{tag}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ================= Plot 4: SHAP (best tree model) =================
    tree_ok = {n: r for n, r in results.items()
               if isinstance(r['model'], (RandomForestClassifier, xgb.XGBClassifier,
                                          lgb.LGBMClassifier, GradientBoostingClassifier))}
    if tree_ok:
        bt = max(tree_ok, key=lambda k: tree_ok[k]['auc'])
        plot_shap_binary(tree_ok[bt]['model'], X_tr_df, feat_names,
                         title=f"SHAP: Hyper Detection ({bt}, {landmark_name})",
                         filename=f"SHAP_{tag}.png")


# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    run_experiment("3-Month", seq_len=3)
    run_experiment("6-Month", seq_len=4)
    print(f"\n  Done. Results in {Config.OUT_DIR.resolve()}")
