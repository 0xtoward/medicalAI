"""Shared SHAP plotting helpers for binary risk models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _ensure_frame(X, feature_names):
    cols = list(feature_names)
    if isinstance(X, pd.DataFrame):
        return X.loc[:, cols].copy()
    arr = np.asarray(X)
    return pd.DataFrame(arr, columns=cols)


def _select_binary_view(values, base_values):
    base = base_values
    if isinstance(values, list):
        class_idx = 1 if len(values) > 1 else 0
        values = np.asarray(values[class_idx])
        if isinstance(base, (list, tuple, np.ndarray)):
            base = np.asarray(base).reshape(-1)[class_idx]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
            if isinstance(base, (list, tuple, np.ndarray)):
                flat = np.asarray(base).reshape(-1)
                base = flat[1] if flat.size > 1 else flat[0]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values for binary output, got shape {values.shape}")
    return values, base


def _normalize_base_values(base_values, n_samples):
    base = np.asarray(base_values, dtype=float)
    if base.ndim == 0:
        return np.full(n_samples, float(base))
    flat = base.reshape(-1)
    if flat.size == n_samples:
        return flat.astype(float)
    if flat.size == 0:
        return np.zeros(n_samples, dtype=float)
    return np.full(n_samples, float(flat[0]))


def _make_explanation(values, base_values, data_values, feature_names):
    values, base_values = _select_binary_view(values, base_values)
    base_arr = _normalize_base_values(base_values, values.shape[0])
    return shap.Explanation(
        values=values,
        base_values=base_arr,
        data=np.asarray(data_values),
        feature_names=list(feature_names),
    )


def _fit_binary_explainer(model_name, model, X_background, feature_names, seed=42, max_eval=300):
    X_bg = _ensure_frame(X_background, feature_names)

    if model_name in {"Logistic Reg.", "Elastic LR"} and isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        lr = model.named_steps.get("lr")
        if scaler is not None and lr is not None:
            bg_scaled = pd.DataFrame(scaler.transform(X_bg), columns=list(feature_names), index=X_bg.index)
            explainer = shap.LinearExplainer(lr, bg_scaled.values)

            def explain(X):
                X_use = _ensure_frame(X, feature_names)
                X_plot = pd.DataFrame(
                    scaler.transform(X_use),
                    columns=list(feature_names),
                    index=X_use.index,
                )
                explanation = _make_explanation(
                    explainer.shap_values(X_plot.values),
                    explainer.expected_value,
                    X_plot.values,
                    feature_names,
                )
                return explanation, X_plot

            return explain

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)

        def explain(X):
            X_use = _ensure_frame(X, feature_names)
            explanation = _make_explanation(
                explainer.shap_values(X_use),
                explainer.expected_value,
                X_use.values,
                feature_names,
            )
            return explanation, X_use

        return explain

    bg = shap.sample(X_bg, min(100, len(X_bg)), random_state=seed)
    explainer = shap.Explainer(
        lambda data: model.predict_proba(pd.DataFrame(data, columns=list(feature_names)))[:, 1],
        bg,
    )

    def explain(X):
        X_use = _ensure_frame(X, feature_names).iloc[: min(max_eval, len(X))].copy()
        raw_explanation = explainer(X_use)
        explanation = _make_explanation(
            raw_explanation.values,
            raw_explanation.base_values,
            X_use.values,
            feature_names,
        )
        return explanation, X_use

    return explain


def _save_summary_plot(explanation, X_plot, out_path, title, max_display=20, plot_type=None):
    plt.figure(figsize=(10, 8))
    kwargs = {}
    if plot_type is not None:
        kwargs["plot_type"] = plot_type
    shap.summary_plot(
        explanation.values,
        X_plot,
        feature_names=list(explanation.feature_names),
        max_display=max_display,
        show=False,
        **kwargs,
    )
    plt.title(title, fontsize=14, pad=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_local_plot(explanation, sample_idx, out_path, title, max_display=12):
    try:
        shap.plots.waterfall(explanation[sample_idx], max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9.5, 6.2)
    except Exception:
        plt.close("all")
        plt.figure(figsize=(12, 3.6))
        base_values = np.asarray(explanation.base_values).reshape(-1)
        shap.force_plot(
            float(base_values[sample_idx]),
            explanation.values[sample_idx],
            features=explanation.data[sample_idx],
            feature_names=list(explanation.feature_names),
            matplotlib=True,
            show=False,
            text_rotation=12,
        )
        fig = plt.gcf()
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _local_case_indices(proba):
    order = np.argsort(np.asarray(proba, dtype=float))
    if len(order) == 0:
        return []
    if len(order) == 1:
        return [("High-risk example", int(order[-1]))]
    return [
        ("High-risk example", int(order[-1])),
        ("Low-risk example", int(order[0])),
    ]


def save_binary_shap_suite(
    model_name,
    model,
    X_background,
    feat_names,
    out_dir,
    summary_filename,
    summary_title,
    X_local=None,
    seed=42,
    max_display=20,
):
    """Save global and local SHAP views for a binary classifier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explain = _fit_binary_explainer(model_name, model, X_background, feat_names, seed=seed)
    global_explanation, X_global_plot = explain(X_background)
    summary_path = out_dir / summary_filename
    stem = summary_path.stem
    suffix = summary_path.suffix or ".png"

    _save_summary_plot(global_explanation, X_global_plot, summary_path, summary_title, max_display=max_display)
    _save_summary_plot(
        global_explanation,
        X_global_plot,
        out_dir / f"{stem}_Bar{suffix}",
        f"{summary_title} - Mean |SHAP|",
        max_display=max_display,
        plot_type="bar",
    )

    X_local_df = X_background if X_local is None else X_local
    local_explanation, X_local_plot = explain(X_local_df)
    X_local_model = _ensure_frame(X_local_df, feat_names).iloc[: len(X_local_plot)].copy()
    local_proba = np.asarray(model.predict_proba(X_local_model)[:, 1], dtype=float)

    filename_map = {
        "High-risk example": out_dir / f"{stem}_Local_HighRisk{suffix}",
        "Low-risk example": out_dir / f"{stem}_Local_LowRisk{suffix}",
    }
    for label, idx in _local_case_indices(local_proba):
        _save_local_plot(
            local_explanation,
            idx,
            filename_map[label],
            f"{summary_title} - {label} (pred={local_proba[idx]:.3f})",
        )
