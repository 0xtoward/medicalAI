import os
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from utils.config import SEED, STATIC_NAMES, STATE_NAMES, TIME_STAMPS
from utils.data import (
    apply_missforest,
    build_states_from_labels,
    clear_pkl_cache,
    fit_missforest,
    load_data as _load_data,
    split_imputed,
)

plt.rcParams["font.family"] = "DejaVu Sans"

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import shap
except ImportError:
    shap = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:
    BalancedRandomForestClassifier = None


class Config:
    OUT_DIR = Path("./multistate_nextstate_result/")
    N_CLASSES = 3


Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)


def plot_transition_heatmaps(S_matrix):
    """Plot state transition heatmaps for descriptive overview."""
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
            f"Normal->Hyper: {mat[1, 0]}"
        )

    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Transition_Heatmaps.png", dpi=300)
    plt.close(fig)

    print(
        f"\n  Total transitions: {total.sum()}  Stable: {np.trace(total)}  "
        f"Normal->Hyper: {total[1, 0]}  Normal->Hypo: {total[1, 2]}"
    )


def build_multistate_long_format(X_s, X_d, S_matrix, pids, target_k=None):
    """Build long-format rows for next-state prediction among Normal-at-k intervals."""
    records = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)

    for i in range(n_samples):
        pid = pids[i]
        xs = X_s[i]
        for k in k_range:
            if S_matrix[i, k] != 1:
                continue

            labs_k = X_d[i, k, :]
            labs_0 = X_d[i, 0, :]
            y_next = int(S_matrix[i, k + 1])  # 0=Hyper, 1=Normal, 2=Hypo

            hist_states = S_matrix[i, :k]
            prev_state = int(S_matrix[i, k - 1]) if k > 0 else -1
            ever_hyper = int(0 in hist_states)
            ever_hypo = int(2 in hist_states)
            time_in_normal = int(np.sum(hist_states == 1))

            delta_ft4_k0 = labs_k[1] - labs_0[1]
            delta_tsh_k0 = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(
                np.clip(labs_0[2], 0, None)
            )
            if k > 0:
                labs_prev = X_d[i, k - 1, :]
                delta_ft4_1step = labs_k[1] - labs_prev[1]
                delta_tsh_1step = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(
                    np.clip(labs_prev[2], 0, None)
                )
            else:
                delta_ft4_1step = 0.0
                delta_tsh_1step = 0.0

            records.append(
                {
                    "Patient_ID": pid,
                    "Interval_ID": k,
                    "Interval_Name": f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}",
                    "Y_NextState": y_next,
                    **dict(zip(STATIC_NAMES, xs)),
                    "FT3_Current": labs_k[0],
                    "FT4_Current": labs_k[1],
                    "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                    "Prev_State": str(prev_state),
                    "Ever_Hyper_Before": ever_hyper,
                    "Ever_Hypo_Before": ever_hypo,
                    "Time_In_Normal": time_in_normal,
                    "Delta_FT4_k0": delta_ft4_k0,
                    "Delta_TSH_k0": delta_tsh_k0,
                    "Delta_FT4_1step": delta_ft4_1step,
                    "Delta_TSH_1step": delta_tsh_1step,
                }
            )
    return pd.DataFrame(records)


def get_tune_specs():
    """Return {name: (estimator, param_grid, color, linestyle, n_iter, n_jobs)}."""
    s = SEED
    specs = {
        "Logistic Reg.": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=3000,
                            multi_class="multinomial",
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                "lr__penalty": ["l2"],
                "lr__solver": ["lbfgs"],
            },
            "#1f77b4",
            "-.",
            10,
            -1,
        ),
        "Elastic LR": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=5000,
                            penalty="elasticnet",
                            solver="saga",
                            multi_class="multinomial",
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "lr__C": [0.001, 0.01, 0.1, 0.5, 1, 5],
                "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "#4c78a8",
            "-.",
            10,
            -1,
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            probability=True,
                            class_weight="balanced",
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "svc__C": [0.1, 0.5, 1, 2, 5, 10],
                "svc__gamma": ["scale", 0.01, 0.05, 0.1],
            },
            "#f58518",
            ":",
            12,
            -1,
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=s, n_jobs=1, class_weight="balanced_subsample"
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [3, 5, 10, 20],
                "max_features": ["sqrt", "log2", 0.5],
            },
            "#54a24b",
            "--",
            15,
            -1,
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=s),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_leaf": [5, 10, 20],
            },
            "#2ca02c",
            "--",
            12,
            -1,
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=s),
            {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
            "#9467bd",
            "--",
            10,
            -1,
        ),
        "MLP": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPClassifier(
                            max_iter=1200,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=50,
                            random_state=s,
                        ),
                    ),
                ]
            ),
            {
                "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                "mlp__alpha": [0.001, 0.01, 0.05, 0.1],
                "mlp__learning_rate_init": [0.0005, 0.001, 0.005],
            },
            "#9d755d",
            ":",
            10,
            -1,
        ),
    }
    if BalancedRandomForestClassifier is not None:
        specs["Balanced RF"] = (
            BalancedRandomForestClassifier(random_state=s, n_jobs=1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "min_samples_leaf": [1, 3, 5, 10],
                "sampling_strategy": ["all", "not minority"],
            },
            "#72b7b2",
            "--",
            12,
            -1,
        )
    if xgb is not None:
        specs["XGBoost"] = (
            xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=Config.N_CLASSES,
                eval_metric="mlogloss",
                random_state=s,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            "#d62728",
            "--",
            12,
            -1,
        )
    if lgb is not None:
        specs["LightGBM"] = (
            lgb.LGBMClassifier(
                objective="multiclass",
                num_class=Config.N_CLASSES,
                random_state=s,
                n_jobs=1,
                verbosity=-1,
            ),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5, -1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_samples": [5, 10, 20],
            },
            "#b279a2",
            "--",
            12,
            -1,
        )
    return specs


def fit_candidate_model(base, grid, n_iter, n_jobs, cv, X_tr_df, y_tr, groups_tr):
    rs = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc_ovo_weighted",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    rs.fit(X_tr_df, y_tr, groups=groups_tr)
    return rs.best_estimator_, rs.best_score_, rs.best_params_


def compute_multiclass_metrics(y_true, proba, labels=(0, 1, 2)):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)

    y_bin = label_binarize(y_true, classes=list(labels))
    metrics = {
        "acc": accuracy_score(y_true, pred),
        "bacc": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "auc_ovo_weighted": roc_auc_score(
            y_true, proba, multi_class="ovo", average="weighted"
        ),
        "auc_ovr_macro": roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro"
        ),
        "pr_auc_macro": average_precision_score(y_bin, proba, average="macro"),
        "cm": confusion_matrix(y_true, pred, labels=list(labels)),
    }

    per_class_auc = {}
    per_class_pr = {}
    for idx, state_name in enumerate(STATE_NAMES):
        y_one = (y_true == idx).astype(int)
        if len(np.unique(y_one)) < 2:
            per_class_auc[state_name] = np.nan
            per_class_pr[state_name] = np.nan
            continue
        per_class_auc[state_name] = roc_auc_score(y_one, proba[:, idx])
        per_class_pr[state_name] = average_precision_score(y_one, proba[:, idx])

    metrics["per_class_auc"] = per_class_auc
    metrics["per_class_pr"] = per_class_pr
    metrics["pred"] = pred
    return metrics


def plot_metric_bars(results):
    names = list(results.keys())
    aucs = [results[n]["auc_ovo_weighted"] for n in names]
    praucs = [results[n]["pr_auc_macro"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, aucs, w, label="Weighted OVO AUC", color="steelblue")
    ax.bar(x, praucs, w, label="Macro PR-AUC", color="coral")
    ax.bar(x + w, f1s, w, label="Macro F1", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Multi-State Next-State Model Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Model_Comparison_Bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_auc(results):
    rows = []
    for model_name, r in results.items():
        for state_name, auc in r["per_class_auc"].items():
            rows.append({"Model": model_name, "State": state_name, "AUC": auc})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=df, x="Model", y="AUC", hue="State", ax=ax)
    ax.set_title("Per-Class One-vs-Rest ROC-AUC")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "PerClass_AUC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(best_name, metrics):
    cm = metrics["cm"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=STATE_NAMES,
        yticklabels=STATE_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {best_name}")
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "Confusion_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_next_state_distribution(df_long):
    counts = (
        df_long["Y_NextState"]
        .map({0: "Hyper", 1: "Normal", 2: "Hypo"})
        .value_counts()
        .reindex(STATE_NAMES)
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(counts.index, counts.values, color=["tomato", "seagreen", "royalblue"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{int(val)}", ha="center")
    ax.set_ylabel("Intervals")
    ax.set_title("Next-State Distribution (Normal-at-k intervals)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Config.OUT_DIR / "NextState_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_shap_analysis(best_name, best_model, X_tr_df):
    """Run SHAP on the best tree-compatible model for Hyper next-state."""
    if shap is None:
        print("  SHAP skipped: shap is not installed")
        return
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_tr_df)
        if isinstance(shap_values, list):
            shap_target = shap_values[0]
        else:
            shap_target = shap_values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_target,
            X_tr_df,
            feature_names=list(X_tr_df.columns),
            max_display=20,
            show=False,
        )
        plt.title(f"SHAP: Hyper Next-State Risk ({best_name})", fontsize=14, pad=20)
        plt.savefig(Config.OUT_DIR / "SHAP_Hyper.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  SHAP failed for {best_name}: {e}")


def train_and_evaluate_multistate(df_tr, df_te):
    feat_cols = STATIC_NAMES + [
        "FT3_Current",
        "FT4_Current",
        "logTSH_Current",
        "Ever_Hyper_Before",
        "Ever_Hypo_Before",
        "Time_In_Normal",
        "Delta_FT4_k0",
        "Delta_TSH_k0",
        "Delta_FT4_1step",
        "Delta_TSH_1step",
    ]

    interval_cats = sorted(df_tr["Interval_Name"].unique())
    prev_state_cats = sorted(df_tr["Prev_State"].unique())

    def make_features(df):
        out = df[feat_cols].copy().reset_index(drop=True)
        for cat in interval_cats:
            out[f"Window_{cat}"] = (df["Interval_Name"].values == cat).astype(float)
        for cat in prev_state_cats:
            out[f"PrevState_{cat}"] = (df["Prev_State"].values == cat).astype(float)
        return out

    X_tr_df = make_features(df_tr)
    X_te_df = make_features(df_te)
    y_tr = df_tr["Y_NextState"].values.astype(int)
    y_te = df_te["Y_NextState"].values.astype(int)
    groups_tr = df_tr["Patient_ID"].values

    print(f"\n  Train: {len(y_tr)} intervals")
    print(f"  Test : {len(y_te)} intervals")
    print(
        "  Next-state counts (train): "
        f"Hyper={int(np.sum(y_tr == 0))}, Normal={int(np.sum(y_tr == 1))}, Hypo={int(np.sum(y_tr == 2))}"
    )
    print(
        "  Next-state counts (test) : "
        f"Hyper={int(np.sum(y_te == 0))}, Normal={int(np.sum(y_te == 1))}, Hypo={int(np.sum(y_te == 2))}"
    )
    print(f"  Features: {X_tr_df.shape[1]}")

    print("\n  Tuning ALL models (RandomizedSearchCV + GroupKFold)...")
    gkf = GroupKFold(n_splits=3)
    model_zoo = {}
    for name, (base, grid, color, ls, n_iter, n_jobs) in get_tune_specs().items():
        best_est, best_score, best_params = fit_candidate_model(
            base, grid, n_iter, n_jobs, gkf, X_tr_df, y_tr, groups_tr
        )
        model_zoo[name] = (best_est, color, ls)
        print(f"    {name:<18s} CV weighted-OVO-AUC={best_score:.3f}  {best_params}")

    stacking = StackingClassifier(
        estimators=[
            ("lr", clone(model_zoo["Logistic Reg."][0])),
            ("rf", clone(model_zoo["Random Forest"][0])),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000, multi_class="multinomial", random_state=SEED
        ),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )
    if "LightGBM" in model_zoo:
        stacking.estimators.append(("lgbm", clone(model_zoo["LightGBM"][0])))
    elif "XGBoost" in model_zoo:
        stacking.estimators.append(("xgb", clone(model_zoo["XGBoost"][0])))
    stacking.fit(X_tr_df, y_tr)
    model_zoo["Stacking"] = (stacking, "#17becf", "-")
    print(f"    {'Stacking':<18s} Built from tuned base learners")

    print("\n  Final evaluation on test set...")
    print(f"\n{'=' * 98}")
    print(
        f"  {'Model':<18s} {'AUC(OVO-w)':>10} {'PR-AUC(m)':>10} {'Acc':>7} "
        f"{'BalAcc':>8} {'F1-macro':>10}"
    )
    print(f"{'=' * 98}")

    results = {}
    for name, (model, color, ls) in model_zoo.items():
        model.fit(X_tr_df, y_tr)
        proba = model.predict_proba(X_te_df)
        metrics = compute_multiclass_metrics(y_te, proba)
        metrics["model"] = model
        metrics["proba"] = proba
        metrics["color"] = color
        metrics["ls"] = ls
        results[name] = metrics
        print(
            f"  {name:<18s} {metrics['auc_ovo_weighted']:>10.3f} {metrics['pr_auc_macro']:>10.3f} "
            f"{metrics['acc']:>7.3f} {metrics['bacc']:>8.3f} {metrics['f1_macro']:>10.3f}"
        )

    print(f"{'=' * 98}")
    best_auc_name = max(results, key=lambda k: results[k]["auc_ovo_weighted"])
    best_pr_name = max(results, key=lambda k: results[k]["pr_auc_macro"])
    print(f"  Best weighted OVO AUC: {best_auc_name} ({results[best_auc_name]['auc_ovo_weighted']:.3f})")
    print(f"  Best macro PR-AUC   : {best_pr_name} ({results[best_pr_name]['pr_auc_macro']:.3f})")

    plot_metric_bars(results)
    plot_per_class_auc(results)
    plot_confusion(best_auc_name, results[best_auc_name])
    plot_next_state_distribution(df_te)

    best_tree_name = None
    for candidate in [best_auc_name, "LightGBM", "XGBoost", "Random Forest", "Balanced RF", "GradientBoosting"]:
        if candidate in results and hasattr(results[candidate]["model"], "feature_importances_"):
            best_tree_name = candidate
            break
    if best_tree_name:
        run_shap_analysis(best_tree_name, results[best_tree_name]["model"], X_tr_df)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached .pkl files")
    args = parser.parse_args()
    if args.clear_cache:
        clear_pkl_cache(Config.OUT_DIR)

    print("=" * 88)
    print("  Multi-State Next-State Prediction (Temporal-Safe MissForest)")
    print("=" * 88)

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

    print("\n--- Phase 1: State Transition Heatmaps ---")
    max_depth = eval_raw.shape[1]
    hm_raw = np.hstack(
        [X_s_raw, ft3_raw[:, :max_depth], ft4_raw[:, :max_depth], tsh_raw[:, :max_depth], eval_raw]
    )
    hm_imp_path = Config.OUT_DIR / f"missforest_depth{max_depth}.pkl"
    if hm_imp_path.exists():
        hm_imputer = joblib.load(hm_imp_path)
        print(f"  MissForest depth-{max_depth}: loaded from cache")
    else:
        print(f"  MissForest depth-{max_depth}: fitting on train ({int(tr_mask.sum())} records)...")
        hm_imputer = fit_missforest(hm_raw[tr_mask])
        joblib.dump(hm_imputer, hm_imp_path)
    hm_filled = apply_missforest(hm_raw, hm_imputer, max_depth)
    _, _, _, _, ev_hm = split_imputed(hm_filled, n_static, max_depth, max_depth)
    S_full = build_states_from_labels(ev_hm)
    plot_transition_heatmaps(S_full)

    print("\n--- Phase 2: Building temporal-safe long-format data ---")
    df_tr_parts, df_te_parts = [], []
    for depth in range(1, 7):
        k = depth - 1
        interval_name = f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}"
        raw = np.hstack(
            [
                X_s_raw,
                ft3_raw[:, :depth],
                ft4_raw[:, :depth],
                tsh_raw[:, :depth],
                eval_raw[:, :depth],
            ]
        )
        imp_path = Config.OUT_DIR / f"missforest_depth{depth}.pkl"
        if imp_path.exists():
            imputer = joblib.load(imp_path)
        else:
            print(f"  MissForest depth-{depth} ({interval_name}): fitting on train ({int(tr_mask.sum())} records)...")
            imputer = fit_missforest(raw[tr_mask])
            joblib.dump(imputer, imp_path)

        filled_tr = apply_missforest(raw[tr_mask], imputer, depth)
        filled_te = apply_missforest(raw[te_mask], imputer, depth)
        xs_tr, ft3_tr, ft4_tr, tsh_tr, ev_tr = split_imputed(filled_tr, n_static, depth, depth)
        xs_te, ft3_te, ft4_te, tsh_te, ev_te = split_imputed(filled_te, n_static, depth, depth)
        S_tr = build_states_from_labels(ev_tr)
        S_te = build_states_from_labels(ev_te)
        X_d_tr = np.stack([ft3_tr, ft4_tr, tsh_tr], axis=-1)
        X_d_te = np.stack([ft3_te, ft4_te, tsh_te], axis=-1)

        df_tr_k = build_multistate_long_format(xs_tr, X_d_tr, S_tr, pids[tr_mask], target_k=k)
        df_te_k = build_multistate_long_format(xs_te, X_d_te, S_te, pids[te_mask], target_k=k)
        df_tr_parts.append(df_tr_k)
        df_te_parts.append(df_te_k)
        print(f"    depth-{depth} ({interval_name}): train {len(df_tr_k)}  test {len(df_te_k)} rows")

    df_tr = pd.concat(df_tr_parts, ignore_index=True)
    df_te = pd.concat(df_te_parts, ignore_index=True)
    print(f"  Pooled: train {len(df_tr)}  test {len(df_te)} rows")

    print("\n--- Phase 3: Multi-State Next-State Modeling ---")
    train_and_evaluate_multistate(df_tr, df_te)
    print(f"\n  All plots saved to {Config.OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
