"""Lightweight feature builders shared by training and inference."""

from __future__ import annotations

import pandas as pd

from thyroid_app.constants import RELAPSE_FEATURE_COLUMNS


def make_relapse_features(
    df: pd.DataFrame,
    interval_categories: list[str],
    prev_state_categories: list[str],
) -> pd.DataFrame:
    """Build the relapse model feature frame with fixed one-hot ordering."""
    out = df[RELAPSE_FEATURE_COLUMNS].copy().reset_index(drop=True)
    for category in interval_categories:
        out[f"Window_{category}"] = (df["Interval_Name"].values == category).astype(float)
    for category in prev_state_categories:
        out[f"PrevState_{category}"] = (df["Prev_State"].values == category).astype(float)
    return out.apply(pd.to_numeric, errors="coerce").astype(float)

