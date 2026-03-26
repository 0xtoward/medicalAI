"""Leakage-safe feature selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class BinaryFeatureSelectionResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    selected_features: list[str]
    summary_df: pd.DataFrame
    best_c: float
    cv_score: float
    original_feature_count: int


def _safe_average_precision(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Guard against degenerate folds without both classes."""
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, proba))


def select_binary_features_with_l1(
    X_tr_df: pd.DataFrame,
    X_te_df: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    seed: int = 42,
    min_features: int = 8,
    c_grid: list[float] | None = None,
) -> BinaryFeatureSelectionResult:
    """Fit an L1 logistic selector on train only and export the feature manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr_df = X_tr_df.copy()
    X_te_df = X_te_df.copy()
    y_tr = np.asarray(y_tr, dtype=int)
    groups_tr = np.asarray(groups_tr)

    if c_grid is None:
        c_grid = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    unique_groups = np.unique(groups_tr)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2 or np.unique(y_tr).size < 2:
        summary_df = pd.DataFrame(
            {
                "feature": X_tr_df.columns,
                "coefficient": np.nan,
                "abs_coefficient": np.nan,
                "selected": True,
            }
        )
        summary_df.to_csv(out_dir / f"{prefix}_Feature_Selection.csv", index=False)
        return BinaryFeatureSelectionResult(
            X_train=X_tr_df,
            X_test=X_te_df,
            selected_features=list(X_tr_df.columns),
            summary_df=summary_df,
            best_c=np.nan,
            cv_score=np.nan,
            original_feature_count=X_tr_df.shape[1],
        )

    gkf = GroupKFold(n_splits=n_splits)
    candidates: list[dict] = []

    for c_value in c_grid:
        fold_scores: list[float] = []
        for fold_tr, fold_val in gkf.split(X_tr_df, y_tr, groups=groups_tr):
            scaler = StandardScaler()
            X_fold_tr = scaler.fit_transform(X_tr_df.iloc[fold_tr])
            X_fold_val = scaler.transform(X_tr_df.iloc[fold_val])
            selector = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=c_value,
                class_weight="balanced",
                max_iter=6000,
                random_state=seed,
            )
            selector.fit(X_fold_tr, y_tr[fold_tr])
            proba = selector.predict_proba(X_fold_val)[:, 1]
            fold_scores.append(_safe_average_precision(y_tr[fold_val], proba))

        full_scaler = StandardScaler()
        X_full = full_scaler.fit_transform(X_tr_df)
        full_selector = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=c_value,
            class_weight="balanced",
            max_iter=6000,
            random_state=seed,
        )
        full_selector.fit(X_full, y_tr)
        coef = full_selector.coef_.ravel()
        coef_abs = np.abs(coef)
        support = coef_abs > 1e-8
        candidates.append(
            {
                "c_value": c_value,
                "cv_score": float(np.nanmean(fold_scores)),
                "coef": coef,
                "coef_abs": coef_abs,
                "support": support,
                "n_selected": int(support.sum()),
            }
        )

    eligible = [c for c in candidates if c["n_selected"] >= min_features]
    if not eligible:
        eligible = [c for c in candidates if c["n_selected"] > 0]
    if not eligible:
        eligible = candidates

    best = sorted(eligible, key=lambda x: (-x["cv_score"], x["n_selected"], x["c_value"]))[0]
    support = best["support"].copy()

    if support.sum() < min_features:
        top_k = min(min_features, len(support))
        top_idx = np.argsort(best["coef_abs"])[::-1][:top_k]
        support[:] = False
        support[top_idx] = True

    selected_features = list(X_tr_df.columns[support])
    X_tr_selected = X_tr_df.loc[:, selected_features].copy()
    X_te_selected = X_te_df.loc[:, selected_features].copy()

    summary_df = pd.DataFrame(
        {
            "feature": X_tr_df.columns,
            "coefficient": best["coef"],
            "abs_coefficient": best["coef_abs"],
            "selected": support,
        }
    ).sort_values(["selected", "abs_coefficient"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(out_dir / f"{prefix}_Feature_Selection.csv", index=False)

    report_path = out_dir / f"{prefix}_Feature_Selection_Report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"original_features={X_tr_df.shape[1]}",
                f"selected_features={len(selected_features)}",
                f"best_c={best['c_value']}",
                f"cv_pr_auc={best['cv_score']:.6f}",
                "selected_feature_names=" + ", ".join(selected_features),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return BinaryFeatureSelectionResult(
        X_train=X_tr_selected,
        X_test=X_te_selected,
        selected_features=selected_features,
        summary_df=summary_df,
        best_c=float(best["c_value"]),
        cv_score=float(best["cv_score"]),
        original_feature_count=X_tr_df.shape[1],
    )
