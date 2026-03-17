import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.integrate
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from utils.config import SEED, STATIC_NAMES, STATE_NAMES, TIME_STAMPS
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    clear_pkl_cache,
    fit_missforest,
    load_or_fit_depth_imputer as _load_or_fit_depth_imputer,
    load_data as _load_data,
    split_imputed,
)
from utils.evaluation import (
    compute_binary_metrics as _compute_binary_metrics,
    compute_calibration_stats as _compute_calibration_stats,
    concordance_index_simple as _concordance_index_simple,
    save_calibration_figure as _save_calibration_figure,
    select_best_threshold as _select_best_threshold,
)
from utils.recurrence import (
    build_interval_risk_data as _build_interval_risk_data,
    derive_recurrent_survival_data as _derive_recurrent_survival_data,
)

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid

try:
    from lifelines import CoxPHFitter
except ImportError:
    CoxPHFitter = None

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
except ImportError:
    RandomSurvivalForest = None
    Surv = None

plt.rcParams["font.family"] = "DejaVu Sans"
np.random.seed(SEED)

TIME_MONTHS = [0, 1, 3, 6, 12, 18, 24]
HORIZONS = [3, 6, 12]


class Config:
    OUT_DIR = Path("./results/recurrent_survival/")
    SHARED_MF_DIR = Path("./results/relapse/")
    LEGACY_SHARED_MF_DIR = Path("./multistate_result/")


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=None, label=""):
    """Reuse an existing MissForest cache when the training matrix is identical."""
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None
    if primary.exists():
        print(f"  MissForest {label}: loaded from {primary}")
    elif fallback is not None and fallback.exists():
        print(f"  MissForest {label}: reused shared cache {fallback}")
    else:
        print(f"  MissForest {label}: fitting on train ({len(raw_train)} records)...")
    return _load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=fallback_cache_path)


def plot_transition_heatmaps(S_matrix):
    """Plot state transition heatmaps for each consecutive interval."""
    n_intervals = S_matrix.shape[1] - 1
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
        label = f"{TIME_STAMPS[i]} -> {TIME_STAMPS[i + 1]}"
        mat = np.zeros((3, 3), dtype=int)
        for src in range(3):
            for dst in range(3):
                mat[src, dst] = np.sum((S_matrix[:, i] == src) & (S_matrix[:, i + 1] == dst))
        total += mat
        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=STATE_NAMES,
            yticklabels=STATE_NAMES,
            ax=ax,
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("State at t+1")
        ax.set_ylabel("State at t")
        print(
            f"  [{label}] transition: {(mat.sum() - np.trace(mat)) / mat.sum():.1%}  "
            f"relapse(Normal->Hyper): {mat[1, 0]}"
        )

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close(fig)
    print(
        f"\n  Total transitions: {total.sum()}  Stable: {np.trace(total)}  "
        f"Relapse(N->H): {total[1, 0]}  Normal->Hypo: {total[1, 2]}"
    )


def build_interval_risk_data(X_s, X_d, S_matrix, pids, target_k=None):
    """Build interval rows with shared relapse-history semantics.

    Ever_Hyper_Before is 1 only after a prior Normal -> Hyper relapse.
    """
    return _build_interval_risk_data(X_s, X_d, S_matrix, pids, target_k=target_k)


def derive_recurrent_survival_data(interval_df):
    """Collapse contiguous Normal intervals into recurrent risk spells."""
    return _derive_recurrent_survival_data(interval_df)


def build_recurrent_datasets():
    """Build interval-level and recurrent-survival datasets with leakage-safe features."""
    X_s_raw, ft3_raw, ft4_raw, tsh_raw, eval_raw, _, pids = _load_data()
    n_static = X_s_raw.shape[1]
    print(f"  Records: {len(pids)}")

    unique_pids = list(dict.fromkeys(pids))
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    tr_mask = np.array([p in train_pids for p in pids])
    te_mask = ~tr_mask
    print(f"  Train: {int(tr_mask.sum())} records")
    print(f"  Test:  {int(te_mask.sum())} records")

    print(f"\n--- Phase 1: State Transition Heatmaps ---")
    max_depth = eval_raw.shape[1]
    hm_raw = np.hstack(
        [X_s_raw, ft3_raw[:, :max_depth], ft4_raw[:, :max_depth], tsh_raw[:, :max_depth], eval_raw]
    )
    hm_imp_path = Config.OUT_DIR / f"missforest_depth{max_depth}.pkl"
    hm_shared_path = Config.SHARED_MF_DIR / f"missforest_depth{max_depth}.pkl"
    if not hm_shared_path.exists():
        hm_shared_path = Config.LEGACY_SHARED_MF_DIR / f"missforest_depth{max_depth}.pkl"
    hm_imputer = load_or_fit_depth_imputer(
        hm_raw[tr_mask], hm_imp_path, fallback_cache_path=hm_shared_path, label=f"depth-{max_depth}"
    )
    hm_filled = apply_missforest(hm_raw, hm_imputer, max_depth)
    _, _, _, _, ev_hm = split_imputed(hm_filled, n_static, max_depth, max_depth)
    S_full = build_states_from_labels(ev_hm)
    plot_transition_heatmaps(S_full)

    print(f"\n--- Phase 2: Building interval risk data ---")
    df_tr_parts, df_te_parts = [], []
    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"
        raw = np.hstack(
            [X_s_raw, ft3_raw[:, :depth], ft4_raw[:, :depth], tsh_raw[:, :depth], eval_raw[:, :depth]]
        )
        imp_path = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        shared_imp_path = Config.SHARED_MF_DIR / f"missforest_depth{depth}.pkl"
        if not shared_imp_path.exists():
            shared_imp_path = Config.LEGACY_SHARED_MF_DIR / f"missforest_depth{depth}.pkl"
        imputer = load_or_fit_depth_imputer(
            raw[tr_mask],
            imp_path,
            fallback_cache_path=shared_imp_path,
            label=f"depth-{depth} ({interval_name})",
        )

        filled_tr = apply_missforest(raw[tr_mask], imputer, depth)
        filled_te = apply_missforest(raw[te_mask], imputer, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        S_tr = build_states_from_labels(ev_tr)
        S_te = build_states_from_labels(ev_te)
        X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_tr_k = build_interval_risk_data(xs_tr, X_d_tr, S_tr, pids[tr_mask], target_k=k)
        df_te_k = build_interval_risk_data(xs_te, X_d_te, S_te, pids[te_mask], target_k=k)
        df_tr_parts.append(df_tr_k)
        df_te_parts.append(df_te_k)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr_k)}  test {len(df_te_k)} rows")

    interval_train = pd.concat(df_tr_parts, ignore_index=True)
    interval_test = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled interval rows: train {len(interval_train)}  test {len(interval_test)}")

    recurrent_train = derive_recurrent_survival_data(interval_train)
    recurrent_test = derive_recurrent_survival_data(interval_test)
    print(f"  Recurrent risk spells: train {len(recurrent_train)}  test {len(recurrent_test)}")

    interval_train.to_csv(Config.OUT_DIR / "interval_risk_train.csv", index=False)
    interval_test.to_csv(Config.OUT_DIR / "interval_risk_test.csv", index=False)
    recurrent_train.to_csv(Config.OUT_DIR / "recurrent_surv_train.csv", index=False)
    recurrent_test.to_csv(Config.OUT_DIR / "recurrent_surv_test.csv", index=False)
    return interval_train, interval_test, recurrent_train, recurrent_test


def make_feature_frames(train_df, test_df, feature_mode):
    """Create one-hot encoded feature frames for the chosen feature mode."""
    base_cols = STATIC_NAMES + ["FT3_Current", "FT4_Current", "logTSH_Current", "Prior_Relapse_Count"]
    if feature_mode == "markov_lite":
        feat_cols = base_cols
    elif feature_mode == "history_augmented":
        feat_cols = base_cols + [
            "Ever_Hyper_Before",
            "Ever_Hypo_Before",
            "Time_In_Normal",
            "Delta_FT4_k0",
            "Delta_TSH_k0",
            "Delta_FT4_1step",
            "Delta_TSH_1step",
        ]
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    cat_cols = ["Interval_Name", "Prev_State"]
    tr = train_df[feat_cols + cat_cols].copy().reset_index(drop=True)
    te = test_df[feat_cols + cat_cols].copy().reset_index(drop=True)

    for col in cat_cols:
        cats = sorted(train_df[col].astype(str).unique())
        for cat in cats:
            name = f"{col}_{cat}"
            tr[name] = (train_df[col].astype(str).values == cat).astype(float)
            te[name] = (test_df[col].astype(str).values == cat).astype(float)

    tr = tr.drop(columns=cat_cols)
    te = te.drop(columns=cat_cols)
    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)

    medians = tr.median(numeric_only=True)
    tr = tr.fillna(medians)
    te = te.fillna(medians)

    keep_cols = [c for c in tr.columns if tr[c].nunique(dropna=False) > 1]
    tr = tr[keep_cols]
    te = te[keep_cols]
    return tr, te, list(tr.columns)


def concordance_index_simple(time, score, event):
    """Simple concordance index without external dependencies."""
    return _concordance_index_simple(time, score, event)


def safe_binary_metrics(y_true, proba, threshold=0.5):
    """Compute binary metrics and guard against degenerate subsets."""
    return _compute_binary_metrics(y_true, proba, threshold)


def compute_calibration_stats(y_true, proba):
    """Compute calibration intercept and slope."""
    return _compute_calibration_stats(y_true, proba)


def select_best_threshold(y_true, proba, low=0.02, high=0.95, step=0.01):
    return _select_best_threshold(y_true, proba, low=low, high=high, step=step)


def km_curve(times, events):
    """Compute a simple Kaplan-Meier step curve."""
    df = pd.DataFrame({"time": times, "event": events}).sort_values("time")
    unique_times = sorted(df["time"].unique())
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    at_risk = len(df)
    for t in unique_times:
        sub = df[df["time"] == t]
        d = int(sub["event"].sum())
        c = int((1 - sub["event"]).sum())
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
            xs.extend([t, t])
            ys.extend([ys[-1], surv])
        at_risk -= d + c
    return np.array(xs), np.array(ys)


def save_event_order_curve(train_df, test_df):
    """Save cumulative relapse-free curves stratified by event order."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, df, title in [
        (axes[0], train_df, "Train relapse-free curve"),
        (axes[1], test_df, "Test relapse-free curve"),
    ]:
        for order, g in df.groupby("Event_Order", sort=True):
            if len(g) < 5:
                continue
            xs, ys = km_curve(g["Gap_Time"].values, g["Event"].values)
            ax.step(xs, ys, where="post", label=f"Event order {int(order)} (n={len(g)})")
        ax.set_xlabel("Months since (re-)entering Normal")
        ax.set_ylabel("Relapse-free probability")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "EventOrder_KM.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


class CoxRecurrenceModel:
    """Cox recurrent-survival wrapper with optional event-order stratification."""

    def __init__(self, name, strata_order=False, penalizer=0.05):
        self.name = name
        self.strata_order = strata_order
        self.penalizer = penalizer
        self.model = None
        self.feature_cols = None

    def fit(self, X_df, meta_df):
        if CoxPHFitter is None:
            raise RuntimeError("lifelines is not installed")
        self.feature_cols = list(X_df.columns)
        fit_df = X_df.copy()
        fit_df["Gap_Time"] = meta_df["Gap_Time"].values
        fit_df["Event"] = meta_df["Event"].values.astype(int)
        fit_df["Patient_ID"] = meta_df["Patient_ID"].values
        if self.strata_order:
            fit_df["Event_Order"] = meta_df["Event_Order"].values.astype(int)
        self.model = CoxPHFitter(penalizer=self.penalizer)
        kwargs = {"duration_col": "Gap_Time", "event_col": "Event", "cluster_col": "Patient_ID"}
        if self.strata_order:
            kwargs["strata"] = ["Event_Order"]
        self.model.fit(fit_df, **kwargs)
        return self

    def risk_score(self, X_df, meta_df):
        pred_df = X_df.copy()
        if self.strata_order:
            pred_df["Event_Order"] = meta_df["Event_Order"].values.astype(int)
        return self.model.predict_partial_hazard(pred_df).values.flatten()

    def risk_within(self, X_df, meta_df, horizons):
        pred_df = X_df.copy()
        if self.strata_order:
            pred_df["Event_Order"] = meta_df["Event_Order"].values.astype(int)
        surv_df = self.model.predict_survival_function(pred_df, times=np.asarray(horizons, dtype=float))
        return 1.0 - surv_df.T.values


class RSFRecurrenceModel:
    """Random survival forest on recurrent-survival origins."""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.feature_cols = None

    def fit(self, X_df, meta_df):
        if RandomSurvivalForest is None or Surv is None:
            raise RuntimeError("sksurv is not installed")
        self.feature_cols = list(X_df.columns)
        y = Surv.from_arrays(event=meta_df["Event"].astype(bool).values, time=meta_df["Gap_Time"].values)
        self.model = RandomSurvivalForest(
            n_estimators=400,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED,
        )
        self.model.fit(X_df.values, y)
        return self

    def risk_score(self, X_df, meta_df):
        return self.model.predict(X_df.values)

    def risk_within(self, X_df, meta_df, horizons):
        surv_fns = self.model.predict_survival_function(X_df.values, return_array=False)
        horizons = np.asarray(horizons, dtype=float)
        out = np.zeros((len(surv_fns), len(horizons)), dtype=float)
        for i, fn in enumerate(surv_fns):
            for j, h in enumerate(horizons):
                out[i, j] = 1.0 - float(fn(h))
        return out


def evaluate_horizons(test_df, risk_matrix, horizons, model_name):
    """Evaluate horizon-specific risk on rows with known status by that horizon."""
    rows = []
    for j, h in enumerate(horizons):
        # Known status: event within h, or survived/censored after h.
        known_mask = (test_df["Event"].values == 1) | (test_df["Gap_Time"].values >= h)
        sub = test_df.loc[known_mask].copy()
        if len(sub) == 0:
            continue
        y_h = ((sub["Event"].values == 1) & (sub["Gap_Time"].values <= h)).astype(int)
        proba_h = risk_matrix[known_mask, j]
        if len(np.unique(y_h)) < 2:
            auc = np.nan
            pr = np.nan
        else:
            auc = roc_auc_score(y_h, proba_h)
            pr = average_precision_score(y_h, proba_h)
        rows.append(
            {
                "Model": model_name,
                "Horizon_Months": h,
                "Known_N": len(sub),
                "Events": int(y_h.sum()),
                "AUC": auc,
                "PR_AUC": pr,
                "Brier": brier_score_loss(y_h, np.clip(proba_h, 1e-6, 1 - 1e-6)),
            }
        )
    return pd.DataFrame(rows)


def save_horizon_figure(horizon_df):
    """Save horizon-wise AUC/Brier figure."""
    if len(horizon_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    sns.lineplot(data=horizon_df, x="Horizon_Months", y="AUC", hue="Model", marker="o", ax=axes[0])
    axes[0].set_title("Horizon-specific AUC")
    axes[0].grid(alpha=0.3)
    sns.lineplot(data=horizon_df, x="Horizon_Months", y="Brier", hue="Model", marker="o", ax=axes[1])
    axes[1].set_title("Horizon-specific Brier")
    axes[1].grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Horizon_AUC_Brier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_performance_bar(perf_df):
    """Save summary bar plot for concordance and next-window metrics."""
    if len(perf_df) == 0:
        return
    plot_df = perf_df.melt(
        id_vars=["Model"], value_vars=["C_Index", "NextWindow_AUC", "NextWindow_PR_AUC"], var_name="Metric", value_name="Value"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Recurrent-Survival Model Performance")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Model_Comparison_Bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_calibration_figure(y_true, proba, title, out_path):
    """Save calibration figure for the selected best model."""
    return _save_calibration_figure(y_true, proba, title, out_path)


def save_patient_risk_strata(train_df, test_df, train_risk, test_risk):
    """Patient-level 12M recurrent relapse risk stratification."""
    train_tmp = train_df[["Patient_ID"]].copy()
    train_tmp["risk12"] = np.clip(train_risk, 0, 1)
    train_tmp["Y12"] = ((train_df["Event"].values == 1) & (train_df["Gap_Time"].values <= 12)).astype(int)
    test_tmp = test_df[["Patient_ID"]].copy()
    test_tmp["risk12"] = np.clip(test_risk, 0, 1)
    test_tmp["Y12"] = ((test_df["Event"].values == 1) & (test_df["Gap_Time"].values <= 12)).astype(int)

    train_pt = train_tmp.groupby("Patient_ID", sort=False).agg(Y=("Y12", "max"), risk12=("risk12", "max")).reset_index()
    test_pt = test_tmp.groupby("Patient_ID", sort=False).agg(Y=("Y12", "max"), risk12=("risk12", "max")).reset_index()

    bins = np.quantile(train_pt["risk12"].values, [0, 0.25, 0.5, 0.75, 1.0])
    bins[0] = min(bins[0], 0.0)
    bins[-1] = max(bins[-1], 1.0)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-6
    labels = ["Q1", "Q2", "Q3", "Q4"]
    test_pt["Risk_Group"] = pd.cut(test_pt["risk12"], bins=bins, include_lowest=True, labels=labels).astype(str)

    summary = (
        test_pt.groupby("Risk_Group")
        .agg(N=("Y", "size"), Observed=("Y", "mean"), Predicted=("risk12", "mean"))
        .reindex(labels)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(summary["Risk_Group"], summary["Observed"], color="cornflowerblue", alpha=0.85, label="Observed 12M relapse")
    ax.plot(summary["Risk_Group"], summary["Predicted"], marker="o", lw=2, color="darkorange", label="Mean predicted 12M risk")
    for bar, n in zip(bars, summary["N"].fillna(0).astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, min(1.0, float(np.nanmax([summary["Observed"].max(), summary["Predicted"].max()]) + 0.12)))
    ax.set_title("Patient-Level Recurrent Relapse Risk Stratification (12M)")
    ax.set_xlabel("Risk groups from train-set quartiles")
    ax.set_ylabel("12M relapse probability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Patient_Risk_Q1Q4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(interval_train, interval_test, recurrent_train, recurrent_test):
    """Fit recurrent-survival baselines and align them to next-window relapse metrics."""
    save_event_order_curve(recurrent_train, recurrent_test)

    model_specs = []
    if CoxPHFitter is not None:
        model_specs.extend(
            [
                ("AG Cox", "markov_lite", CoxRecurrenceModel("AG Cox", strata_order=False)),
                ("PWP Gap Cox", "markov_lite", CoxRecurrenceModel("PWP Gap Cox", strata_order=True)),
                ("AG Cox", "history_augmented", CoxRecurrenceModel("AG Cox", strata_order=False)),
                ("PWP Gap Cox", "history_augmented", CoxRecurrenceModel("PWP Gap Cox", strata_order=True)),
            ]
        )
    if RandomSurvivalForest is not None and Surv is not None:
        model_specs.extend(
            [
                ("Event-Specific RSF", "markov_lite", RSFRecurrenceModel("Event-Specific RSF")),
                ("Event-Specific RSF", "history_augmented", RSFRecurrenceModel("Event-Specific RSF")),
            ]
        )

    if len(model_specs) == 0:
        print("\n  No recurrent-survival libraries available. Dataset exports completed, but AG/PWP/RSF models were skipped.")
        return

    print(f"\n--- Phase 3: Recurrent-survival modeling ---")
    perf_rows = []
    horizon_rows = []
    model_results = {}

    for model_name, feature_mode, model in model_specs:
        label = f"{model_name} [{feature_mode}]"
        X_tr, X_te, feat_names = make_feature_frames(recurrent_train, recurrent_test, feature_mode)
        try:
            model.fit(X_tr, recurrent_train)
        except Exception as e:
            print(f"  {label:<36s} skipped: {e}")
            continue

        risk_score = np.asarray(model.risk_score(X_te, recurrent_test), dtype=float)
        c_index = concordance_index_simple(recurrent_test["Gap_Time"].values, risk_score, recurrent_test["Event"].values)

        unique_next = sorted(recurrent_test["Next_Window_Months"].astype(float).unique())
        risk_next_mat = model.risk_within(X_te, recurrent_test, unique_next)
        next_map = {h: risk_next_mat[:, idx] for idx, h in enumerate(unique_next)}
        next_risk = np.zeros(len(recurrent_test), dtype=float)
        for h in unique_next:
            mask = recurrent_test["Next_Window_Months"].values.astype(float) == h
            next_risk[mask] = next_map[h][mask]

        train_next_values = sorted(recurrent_train["Next_Window_Months"].astype(float).unique())
        train_next_mat = model.risk_within(X_tr, recurrent_train, train_next_values)
        train_next_map = {h: train_next_mat[:, idx] for idx, h in enumerate(train_next_values)}
        train_next_risk = np.zeros(len(recurrent_train), dtype=float)
        for h in train_next_values:
            mask = recurrent_train["Next_Window_Months"].values.astype(float) == h
            train_next_risk[mask] = train_next_map[h][mask]

        threshold = select_best_threshold(
            recurrent_train["Y_Relapse_Next"].values,
            train_next_risk,
            low=0.02,
            high=0.60,
            step=0.01,
        )
        next_metrics = safe_binary_metrics(recurrent_test["Y_Relapse_Next"].values, next_risk, threshold=threshold)
        cal = compute_calibration_stats(recurrent_test["Y_Relapse_Next"].values, next_risk)

        horizons = sorted(set(HORIZONS + unique_next))
        horizon_risk = model.risk_within(X_te, recurrent_test, horizons)
        horizon_df = evaluate_horizons(recurrent_test, horizon_risk, horizons, label)
        if len(horizon_df) > 0:
            horizon_rows.append(horizon_df)
            mean_h_brier = float(horizon_df["Brier"].mean())
        else:
            mean_h_brier = np.nan

        perf_rows.append(
            {
                "Model": label,
                "Feature_Mode": feature_mode,
                "C_Index": c_index,
                "NextWindow_AUC": next_metrics["auc"],
                "NextWindow_PR_AUC": next_metrics["prauc"],
                "NextWindow_Brier": next_metrics["brier"],
                "NextWindow_Recall": next_metrics["recall"],
                "NextWindow_Specificity": next_metrics["specificity"],
                "Calibration_Intercept": cal["intercept"],
                "Calibration_Slope": cal["slope"],
                "Threshold": threshold,
                "Mean_Horizon_Brier": mean_h_brier,
            }
        )
        model_results[label] = {
            "model": model,
            "feature_mode": feature_mode,
            "X_train": X_tr,
            "X_test": X_te,
            "next_risk": next_risk,
            "threshold": threshold,
            "horizons": horizons,
            "horizon_risk": horizon_risk,
            "feat_names": feat_names,
        }

        print(
            f"  {label:<36s} C-index={c_index:.3f}  next-AUC={next_metrics['auc']:.3f}  "
            f"next-PR-AUC={next_metrics['prauc']:.3f}  next-Brier={next_metrics['brier']:.3f}"
        )

    perf_df = pd.DataFrame(perf_rows).sort_values(["NextWindow_PR_AUC", "C_Index"], ascending=False)
    perf_df.to_csv(Config.OUT_DIR / "Model_Performance.csv", index=False)
    print(f"\n{perf_df[['Model', 'C_Index', 'NextWindow_AUC', 'NextWindow_PR_AUC', 'NextWindow_Brier']].to_string(index=False)}")

    if len(horizon_rows) > 0:
        horizon_df = pd.concat(horizon_rows, ignore_index=True)
        horizon_df.to_csv(Config.OUT_DIR / "Horizon_Metrics.csv", index=False)
        save_horizon_figure(horizon_df)

    save_performance_bar(perf_df)

    if len(perf_df) == 0:
        return

    best_cidx_name = perf_df.loc[perf_df["C_Index"].idxmax(), "Model"]
    best_auc_name = perf_df.loc[perf_df["NextWindow_AUC"].idxmax(), "Model"]
    best_prauc_name = perf_df.loc[perf_df["NextWindow_PR_AUC"].idxmax(), "Model"]
    best = model_results[best_prauc_name]
    best_cal = compute_calibration_stats(recurrent_test["Y_Relapse_Next"].values, best["next_risk"])

    summary_df = pd.DataFrame(
        [
            {"Metric": "Best C-index", "Model": best_cidx_name, "Value": float(perf_df.loc[perf_df["C_Index"].idxmax(), "C_Index"])},
            {"Metric": "Best next-window AUC", "Model": best_auc_name, "Value": float(perf_df.loc[perf_df["NextWindow_AUC"].idxmax(), "NextWindow_AUC"])},
            {"Metric": "Best next-window PR-AUC", "Model": best_prauc_name, "Value": float(perf_df.loc[perf_df["NextWindow_PR_AUC"].idxmax(), "NextWindow_PR_AUC"])},
        ]
    )
    summary_df.to_csv(Config.OUT_DIR / "Best_Model_Summary.csv", index=False)

    print(f"\n  Best C-index         : {best_cidx_name} ({summary_df.loc[0, 'Value']:.3f})")
    print(f"  Best next-window AUC : {best_auc_name} ({summary_df.loc[1, 'Value']:.3f})")
    print(f"  Best next-window PR-AUC: {best_prauc_name} ({summary_df.loc[2, 'Value']:.3f})")
    print(f"  Calibration reference model: {best_prauc_name}")
    print(f"    Next-window Brier       = {float(perf_df.loc[perf_df['Model'] == best_prauc_name, 'NextWindow_Brier'].iloc[0]):.3f}")
    print(f"    Calibration intercept   = {best_cal['intercept']:.3f}")
    print(f"    Calibration slope       = {best_cal['slope']:.3f}")

    save_calibration_figure(
        recurrent_test["Y_Relapse_Next"].values,
        best["next_risk"],
        f"Calibration Curve ({best_prauc_name}, next-window relapse)",
        Config.OUT_DIR / "Calibration_NextWindow.png",
    )

    risk12_train = best["model"].risk_within(best["X_train"], recurrent_train, [12])[:, 0]
    risk12_test = best["model"].risk_within(best["X_test"], recurrent_test, [12])[:, 0]
    save_patient_risk_strata(recurrent_train, recurrent_test, risk12_train, risk12_test)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear recurrent-survival .pkl caches")
    args = parser.parse_args()
    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 88)
    print("  Recurrent Relapse Survival Benchmark (AG / PWP / RSF-style)")
    print("=" * 88)

    interval_train, interval_test, recurrent_train, recurrent_test = build_recurrent_datasets()
    print(f"\n  Train recurrent events: {int(recurrent_train['Event'].sum())} / {len(recurrent_train)} spells")
    print(f"  Test recurrent events : {int(recurrent_test['Event'].sum())} / {len(recurrent_test)} spells")
    train_and_evaluate(interval_train, interval_test, recurrent_train, recurrent_test)
    print(f"\n  All outputs saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
