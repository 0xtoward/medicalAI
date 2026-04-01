import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
import scipy.integrate
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = scipy.integrate.trapezoid
from lifelines import CoxPHFitter
from scipy.special import expit
import joblib
from sklearn.base import clone
from imblearn.ensemble import BalancedRandomForestClassifier
# from tabpfn import TabPFNClassifier  # Disabled for the current manuscript version.
from utils.config import SEED, STATIC_NAMES, TIME_STAMPS, STATE_NAMES
from utils.data import (load_data as _load_data, fit_missforest,
                        apply_missforest, split_imputed, build_states_from_labels,
                        load_or_fit_depth_imputer as _load_or_fit_depth_imputer)
from utils.evaluation import (
    aggregate_patient_level as _aggregate_patient_level,
    bootstrap_group_cis as _bootstrap_group_cis,
    compute_binary_metrics as _compute_binary_metrics,
    compute_calibration_stats as _compute_calibration_stats,
    compute_dca as _compute_dca,
    evaluate_patient_aggregation_sensitivity as _evaluate_patient_aggregation_sensitivity,
    evaluate_window_sensitivity as _evaluate_window_sensitivity,
    format_ci as _format_ci,
    save_calibration_figure as _save_calibration_figure,
    save_dca_figure as _save_dca_figure,
    save_patient_aggregation_sensitivity_figure as _save_patient_aggregation_sensitivity_figure,
    save_patient_risk_strata as _save_patient_risk_strata,
    save_threshold_sensitivity_figure as _save_threshold_sensitivity_figure,
    select_best_threshold as _select_best_threshold,
    concordance_index_simple as _concordance_index_simple,
)
from utils.recurrence import derive_recurrent_survival_data as _derive_recurrent_survival_data
from utils.recurrence import build_interval_risk_data as _build_interval_risk_data
from utils.model_viz import save_logistic_regression_visuals
from utils.performance_panels import (
    build_binary_performance_long,
    export_metric_matrices,
    save_performance_heatmap_panels,
)
from utils.plot_style import PRIMARY_BLUE, PRIMARY_TEAL, apply_publication_style
from utils.shap_viz import save_binary_shap_suite

apply_publication_style()

# ==========================================
# 1. 基础配置
# ==========================================
class Config:
    OUT_DIR = Path("./results/relapse/")
    LEGACY_OUT_DIR = Path("./multistate_result/")

Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)


def load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=None, label=""):
    """Load a local/shared MissForest cache or fit a new one."""
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None
    if primary.exists():
        print(f"  MissForest {label}: loaded from cache")
    elif fallback is not None and fallback.exists():
        print(f"  MissForest {label}: loaded from legacy cache")
    else:
        print(f"  MissForest {label}: fitting on train ({len(raw_train)} records)...")
    return _load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=fallback_cache_path)


def compute_recurrent_c_index_from_intervals(df_long, proba):
    """Project interval predictions to spell starts and compute event-time ranking."""
    tmp = df_long.copy().reset_index(drop=True)
    tmp["proba"] = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    recurrent_df = _derive_recurrent_survival_data(tmp)
    if len(recurrent_df) == 0:
        return np.nan
    return _concordance_index_simple(
        recurrent_df["Gap_Time"].values,
        recurrent_df["proba"].values,
        recurrent_df["Event"].values,
    )


def save_lr_artifacts(model_name, model, feat_names, decision_threshold=None):
    """Visualize LR pipeline structure and parameter shapes."""
    if model_name not in {"Logistic Reg.", "Elastic LR"}:
        return
    try:
        prefix = "Logistic_Reg" if model_name == "Logistic Reg." else "Elastic_LR"
        save_logistic_regression_visuals(
            model_name,
            model,
            feat_names,
            Config.OUT_DIR,
            prefix=prefix,
            decision_threshold=decision_threshold,
            output_label="P(Relapse at next window)",
        )
    except Exception as e:
        print(f"  LR visualization failed for {model_name}: {e}")

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
        label = f"{TIME_STAMPS[i]} -> {TIME_STAMPS[i+1]}"
        mat = np.zeros((3, 3), dtype=int)
        for sf in range(3):
            for st in range(3):
                mat[sf, st] = np.sum((S_matrix[:, i] == sf) & (S_matrix[:, i+1] == st))
        total += mat

        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=STATE_NAMES, yticklabels=STATE_NAMES, ax=ax)
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
    """Build long-format rows using the shared recurrence feature definitions."""
    return _build_interval_risk_data(X_s, X_d, S_matrix, pids, target_k=target_k)

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
    """Return {name: (estimator, param_grid, color, linestyle, n_iter, n_jobs)}."""
    S = SEED
    return {
        "Logistic Reg.": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(max_iter=2000, random_state=S))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__penalty': ['l1', 'l2'], 'lr__solver': ['saga']},
            "#1f77b4", "-.", 14, -1
        ),
        "Elastic LR": (
            Pipeline([('scaler', StandardScaler()),
                      ('lr', LogisticRegression(
                          max_iter=4000, penalty='elasticnet', solver='saga', random_state=S
                      ))]),
            {'lr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
             'lr__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            "#4c78a8", "-.", 12, -1
        ),
        "SVM": (
            Pipeline([('scaler', StandardScaler()),
                      ('svc', SVC(
                          kernel='rbf', probability=True, class_weight='balanced',
                          random_state=S
                      ))]),
            {'svc__C': [0.1, 0.5, 1, 2, 5, 10],
             'svc__gamma': ['scale', 0.01, 0.05, 0.1, 0.5]},
            "#f58518", ":", 14, -1
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=S, n_jobs=1, class_weight='balanced'),
            {'n_estimators': [100, 200, 300, 500],
             'max_depth': [3, 5, 7, 10, None],
             'min_samples_leaf': [3, 5, 10, 20],
             'max_features': ['sqrt', 'log2', 0.5]},
            "#54a24b", "--", 20, -1
        ),
        "Balanced RF": (
            BalancedRandomForestClassifier(random_state=S, n_jobs=1),
            {'n_estimators': [100, 200, 300, 500],
             'max_depth': [2, 3, 4, 5],
             'min_samples_leaf': [1, 3, 5, 10],
             'sampling_strategy': ['all', 'not minority']},
            "#72b7b2", "--", 16, -1
        ),
        # "LightGBM": (
        #     lgb.LGBMClassifier(random_state=S, n_jobs=1, verbosity=-1),
        #     {'n_estimators': [100, 200, 300, 500],
        #      'learning_rate': [0.01, 0.03, 0.05, 0.1],
        #      'max_depth': [2, 3, 4, 5, -1],
        #      'subsample': [0.6, 0.7, 0.8, 1.0],
        #      'colsample_bytree': [0.6, 0.8, 1.0],
        #      'min_child_samples': [5, 10, 20]},
        #     "#b279a2", "--", 20, -1
        # ),
        "MLP": (
            Pipeline([('scaler', StandardScaler()),
                      ('mlp', MLPClassifier(max_iter=1000, early_stopping=True,
                                            validation_fraction=0.15,
                                            n_iter_no_change=50, random_state=S))]),
            {'mlp__hidden_layer_sizes': [(32,), (64, 32), (128, 64), (32, 16)],
             'mlp__alpha': [0.001, 0.01, 0.05, 0.1],
             'mlp__learning_rate_init': [0.0005, 0.001, 0.005, 0.01]},
            "#9d755d", ":", 16, -1
        ),
        # "TabPFN": (
        #     TabPFNClassifier(
        #         device='cpu',
        #         random_state=S,
        #         n_estimators=8,
        #         n_preprocessing_jobs=1,
        #         ignore_pretraining_limits=True
        #     ),
        #     None,
        #     "#bab0ac", "-", 0, 1
        # ),
        "Cox PH": (
            CoxPHWrapper(penalizer=0.1),
            {'penalizer': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
            "#bcbd22", "-.", 7, 1
        ),
    }


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    """Tune model with GroupKFold, or evaluate a fixed-config model."""
    if grid:
        rs = RandomizedSearchCV(
            base, grid, n_iter=n_iter, cv=cv,
            scoring='average_precision', random_state=SEED, n_jobs=n_jobs
        )
        rs.fit(X_tr_df, y_tr, groups=groups_tr)
        return rs.best_estimator_, rs.best_score_, rs.best_params_

    scores = []
    for f_tr, f_val in cv.split(X_tr_df, y_tr, groups=groups_tr):
        m = clone(base)
        m.fit(X_tr_df.iloc[f_tr], y_tr[f_tr])
        proba = m.predict_proba(X_tr_df.iloc[f_val])[:, 1]
        scores.append(average_precision_score(y_tr[f_val], proba))
    return clone(base), float(np.mean(scores)), None


def run_shap_analysis(model_name, model, X_tr_df, X_te_df, feat_names):
    """Save shared SHAP outputs for the selected best model."""
    print(f"\n  SHAP analysis on: {model_name}")
    try:
        save_binary_shap_suite(
            model_name=model_name,
            model=model,
            X_background=X_tr_df,
            X_local=X_te_df,
            feat_names=feat_names,
            out_dir=Config.OUT_DIR,
            summary_filename="Hazard_SHAP_Relapse.png",
            summary_title=f"What Drives Relapse? ({model_name})",
            seed=SEED,
            max_display=20,
        )
    except Exception as e:
        print(f"  SHAP failed for {model_name}: {e}")


def compute_binary_metrics(y_true, proba, threshold):
    """Compute discrimination/calibration/classification metrics."""
    return _compute_binary_metrics(y_true, proba, threshold)


def compute_calibration_stats(y_true, proba):
    """Compute calibration-in-the-large style intercept and slope."""
    return _compute_calibration_stats(y_true, proba)


def bootstrap_group_cis(eval_df, threshold, group_col='Patient_ID', n_boot=800):
    """Cluster bootstrap CIs by patient to respect repeated intervals."""
    return _bootstrap_group_cis(eval_df, threshold, group_col=group_col, n_boot=n_boot, seed=SEED)


def format_ci(cis, key):
    """Format CI text for printing."""
    return _format_ci(cis, key)


def save_calibration_figure(y_true, proba, title, out_path, n_bins=6):
    """Save calibration curve + prediction histogram."""
    return _save_calibration_figure(y_true, proba, title, out_path, n_bins=n_bins)


def save_threshold_sensitivity_figure(y_true, proba, best_thr, title, out_path):
    """Show how recall/specificity/F1 vary across thresholds."""
    return _save_threshold_sensitivity_figure(y_true, proba, best_thr, title, out_path)


def compute_dca(y_true, proba, thresholds=None):
    """Decision curve analysis net benefit."""
    return _compute_dca(y_true, proba, thresholds=thresholds)


def save_dca_figure(y_true, proba, title, out_path):
    """Save DCA figure."""
    return _save_dca_figure(y_true, proba, title, out_path)


def aggregate_patient_level(df_long, proba, method='product'):
    """Aggregate interval risks into patient-level 'any relapse during follow-up' risk."""
    return _aggregate_patient_level(df_long, proba, method=method)


def select_best_threshold(y_true, proba, low=0.02, high=0.95, step=0.01):
    """Select threshold by F1."""
    return _select_best_threshold(y_true, proba, low=low, high=high, step=step)


def save_patient_risk_strata(train_patient_df, test_patient_df, out_path):
    """Use train-set quartiles for patient-level risk stratification."""
    return _save_patient_risk_strata(train_patient_df, test_patient_df, out_path)


def evaluate_patient_aggregation_sensitivity(df_long_train, df_long_test, oof_proba, test_proba):
    """Compare patient-level conclusions across aggregation rules."""
    return _evaluate_patient_aggregation_sensitivity(df_long_train, df_long_test, oof_proba, test_proba)


def save_patient_aggregation_sensitivity_figure(summary_df, out_path):
    """Visualize patient-level aggregation sensitivity."""
    return _save_patient_aggregation_sensitivity_figure(summary_df, out_path)


def evaluate_window_sensitivity(df_long_test, proba, threshold):
    """Check if results depend heavily on specific follow-up windows."""
    return _evaluate_window_sensitivity(df_long_test, proba, threshold)


def train_and_evaluate_hazard_strict(df_long_train, df_long_test):
    feat_cols = STATIC_NAMES + [
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
        return out.apply(pd.to_numeric, errors='coerce').astype(float)

    X_tr_df = make_features(df_long_train)
    X_te_df = make_features(df_long_test)
    feat_names = list(X_tr_df.columns)
    y_tr = df_long_train['Y_Relapse'].values.astype(int)
    y_te = df_long_test['Y_Relapse'].values.astype(int)
    groups_tr = df_long_train['Patient_ID'].values

    print(f"\n  Train: {len(y_tr)} intervals (relapse: {y_tr.sum()})")
    print(f"  Test : {len(y_te)} intervals (relapse: {y_te.sum()})")
    print(f"  Features: {len(feat_names)}")

    # ================= Tune ALL models (RandomizedSearchCV + GroupKFold) =================
    print("\n  Tuning ALL models (RandomizedSearchCV + GroupKFold)...")
    S = SEED
    gkf = GroupKFold(n_splits=3)
    tune_specs = get_tune_specs()
    tuned_models = {}

    for name, (base_est, param_grid, color, ls, n_iter, n_jobs) in tune_specs.items():
        best_est, best_score, best_params = fit_candidate_model(
            base_est, param_grid, n_iter, n_jobs, gkf, X_tr_df, y_tr, groups_tr
        )
        tuned_models[name] = (best_est, color, ls)
        suffix = "[fixed config]" if best_params is None else str(best_params)
        print(f"    {name:<22s} CV PR-AUC={best_score:.3f}  {suffix}")

    # ================= OOF threshold selection + refit =================
    print("\n  OOF threshold selection + evaluation...")
    results = {}

    print(f"\n{'='*98}")
    print(f"  {'Model':<22s} {'AUC':>6} {'PR-AUC':>8} {'CIdx':>6} {'Thr':>5} {'Recall':>8} {'Spec':>7} {'F1':>6}  {'TP':>4} {'FP':>5} {'FN':>4} {'TN':>5}")
    print(f"{'='*98}")

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
        train_fit_proba = model.predict_proba(X_tr_df)[:, 1]
        proba = model.predict_proba(X_te_df)[:, 1]
        pred_bin = (proba >= best_thr).astype(int)

        auc = roc_auc_score(y_te, proba)
        prauc = average_precision_score(y_te, proba)
        f1 = f1_score(y_te, pred_bin, zero_division=0)
        cm = confusion_matrix(y_te, pred_bin)
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        c_index = compute_recurrent_c_index_from_intervals(df_long_test, proba)
        results[name] = {
            'proba': proba, 'auc': auc, 'prauc': prauc,
            'c_index': c_index,
            'recall': recall, 'specificity': specificity, 'f1': f1, 'threshold': best_thr,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'model': model, 'color': color, 'ls': ls,
            'oof_proba': oof_proba, 'train_fit_proba': train_fit_proba,
        }

        marker = " *" if auc == max(r['auc'] for r in results.values()) else "  "
        print(f"{marker}{name:<22s} {auc:>5.3f} {prauc:>7.3f} {c_index:>5.3f} {best_thr:>4.2f} {recall:>7.3f} {specificity:>7.3f} {f1:>5.3f}  {tp:>4} {fp:>5} {fn:>4} {tn:>5}")

    print(f"{'='*98}")
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"  Best AUC: {best_name} ({results[best_name]['auc']:.3f})")
    best_prauc_name = max(results, key=lambda k: results[k]['prauc'])
    print(f"  Best PR-AUC: {best_prauc_name} ({results[best_prauc_name]['prauc']:.3f})")
    best_cidx_name = max(results, key=lambda k: results[k]['c_index'])
    print(f"  Best C-index: {best_cidx_name} ({results[best_cidx_name]['c_index']:.3f})")

    perf_domains = {
        "Train_Fit": {"y_true": y_tr, "proba_key": "train_fit_proba"},
        "Validation_OOF": {"y_true": y_tr, "proba_key": "oof_proba"},
        "Test_Temporal": {"y_true": y_te, "proba_key": "proba"},
    }
    perf_metric_keys = ["prauc", "auc", "recall", "specificity", "f1"]
    perf_long_df = build_binary_performance_long(
        task_name="Rolling Landmark",
        results=results,
        domain_payloads=perf_domains,
        metric_keys=perf_metric_keys,
        threshold_key="threshold",
    )
    export_metric_matrices(perf_long_df, Config.OUT_DIR, prefix="Performance")
    save_performance_heatmap_panels(
        perf_long_df,
        Config.OUT_DIR / "Performance_Heatmaps.png",
        task_order=["Rolling Landmark"],
        domain_order=["Train_Fit", "Validation_OOF", "Test_Temporal"],
        metric_order=perf_metric_keys,
        title="Rolling Landmark Internal Performance Heatmaps",
    )

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
    cidxs = [results[n]['c_index'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.26
    bars1 = ax.bar(x - w, aucs, w, label='ROC-AUC', color=PRIMARY_BLUE)
    bars2 = ax.bar(x, praucs, w, label='PR-AUC', color=PRIMARY_TEAL)
    bars3 = ax.bar(x + w, cidxs, w, label='C-index', color='#38bdf8')

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)
    for b in bars3:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: ROC-AUC vs PR-AUC vs C-index", fontsize=13)
    ax.legend()
    ax.set_ylim(0, max(max(aucs), max(praucs), max(cidxs)) + 0.15)
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

    # ================= 绘图 3: SHAP suite (优先解释 PR-AUC 最优模型) =================
    best_shap_name = best_prauc_name
    best_shap_model = results[best_shap_name]['model']
    save_lr_artifacts(
        'Logistic Reg.',
        results['Logistic Reg.']['model'],
        feat_names,
        decision_threshold=results['Logistic Reg.']['threshold'],
    )
    run_shap_analysis(best_shap_name, best_shap_model, X_tr_df, X_te_df, feat_names)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached .pkl files')
    args = parser.parse_args()
    if args.clear_cache:
        from utils.data import clear_pkl_cache
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 80)
    print("  Multi-State Relapse Analysis (Temporal-Safe MissForest)")
    print("=" * 80)

    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    print(f"  Records: {len(pids)}")

    # --- Temporal split by enrollment order ---
    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    n_train_records = int(tr_mask.sum())
    n_test_records = int(te_mask.sum())
    print(f"  Train: {n_train_records} records")
    print(f"  Test:  {n_test_records} records")

    # --- Phase 1: Transition heatmaps (depth-6 imputer for descriptive view) ---
    print(f"\n--- Phase 1: State Transition Heatmaps ---")
    max_depth = eval_raw.shape[1]  # 6
    hm_raw = np.hstack([X_s_raw, ft3_raw[:, :max_depth], ft4_raw[:, :max_depth],
                         tsh_raw[:, :max_depth], eval_raw])
    hm_imp_path = Config.OUT_DIR / f"missforest_depth{max_depth}.pkl"
    hm_legacy_path = Config.LEGACY_OUT_DIR / f"missforest_depth{max_depth}.pkl"
    hm_imputer = load_or_fit_depth_imputer(
        hm_raw[tr_mask], hm_imp_path, fallback_cache_path=hm_legacy_path, label=f"depth-{max_depth}"
    )
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
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k+1]}"

        raw = np.hstack([X_s_raw,
                         ft3_raw[:, :n_lab], ft4_raw[:, :n_lab], tsh_raw[:, :n_lab],
                         eval_raw[:, :n_ev]])

        imp_path = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        legacy_imp_path = Config.LEGACY_OUT_DIR / f"missforest_depth{depth}.pkl"
        imputer = load_or_fit_depth_imputer(
            raw[tr_mask], imp_path, fallback_cache_path=legacy_imp_path, label=f"depth-{depth} ({interval_name})"
        )

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
