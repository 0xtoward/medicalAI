"""Shared data loading, splitting, imputation, and feature engineering."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from utils.config import FILE_PATH, COL_IDX, SEED, STATIC_NAMES, TIME_STAMPS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(file_path=None):
    """Unified loader for the 1003.xlsx dataset.

    Returns
    -------
    X_s        : ndarray (N, 16)   – static features (gender already encoded)
    ft3_raw    : ndarray (N, 7)    – FT3 at 7 time points (may contain NaN)
    ft4_raw    : ndarray (N, 7)
    tsh_raw    : ndarray (N, 7)
    eval_raw   : ndarray (N, 6)    – doctor evaluation labels 1M–24M
    y_outcome  : ndarray (N,)      – raw outcome column (1/2/3), int
    pids       : ndarray (N,)      – Patient_ID (forward-filled)
    """
    fp = file_path or FILE_PATH
    df = pd.read_excel(fp, header=None, engine='openpyxl').iloc[2:]
    df['Patient_ID'] = df.iloc[:, COL_IDX['ID']].ffill()

    X_s = df.iloc[:, COL_IDX['Static_Feats']].apply(pd.to_numeric, errors='coerce').values
    gt = df.iloc[:, 3].astype(str).str.strip()
    gender = np.zeros(len(df), dtype=np.float32)
    gender[(gt == "男") | (gt.str.upper() == "M") | (df.iloc[:, 3] == 1)] = 1.0
    X_s[:, 0] = gender

    y_raw = df.iloc[:, COL_IDX['Outcome']].apply(pd.to_numeric, errors='coerce')
    ft3 = df.iloc[:, COL_IDX['FT3_Sequence']].apply(pd.to_numeric, errors='coerce').values
    ft4 = df.iloc[:, COL_IDX['FT4_Sequence']].apply(pd.to_numeric, errors='coerce').values
    tsh = df.iloc[:, COL_IDX['TSH_Sequence']].apply(pd.to_numeric, errors='coerce').values
    eval_raw = df.iloc[:, COL_IDX['Eval_Cols']].apply(pd.to_numeric, errors='coerce').values

    valid_idx = y_raw.dropna().index
    take = df.index.get_indexer(valid_idx)
    y = y_raw.loc[valid_idx].values.astype(int)
    pids = df.loc[valid_idx, 'Patient_ID'].values

    return X_s[take], ft3[take], ft4[take], tsh[take], eval_raw[take], y, pids


# ---------------------------------------------------------------------------
# Temporal split (by enrollment order)
# ---------------------------------------------------------------------------
def temporal_split(pids, ratio=0.8):
    """Split by patient enrollment order.

    Returns (train_mask, test_mask) as boolean arrays.
    """
    unique = list(dict.fromkeys(pids))
    cutoff = int(len(unique) * ratio)
    train_set = set(unique[:cutoff])
    tr = np.array([p in train_set for p in pids])
    return tr, ~tr


# ---------------------------------------------------------------------------
# MissForest imputation
# ---------------------------------------------------------------------------
def fit_missforest(train_matrix, seed=None):
    """Fit IterativeImputer (MissForest) on training data."""
    s = seed if seed is not None else SEED
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=5,
                                        random_state=s, n_jobs=1),
        max_iter=10, random_state=s
    )
    imputer.fit(train_matrix)
    return imputer


def split_imputed(arr, n_static, n_lab, n_eval=0):
    """Split imputed matrix back into components.

    Returns (X_s, ft3, ft4, tsh) if n_eval==0,
    or      (X_s, ft3, ft4, tsh, evals) if n_eval>0.
    """
    i = 0
    X_s  = arr[:, i:i+n_static]; i += n_static
    ft3  = arr[:, i:i+n_lab];    i += n_lab
    ft4  = arr[:, i:i+n_lab];    i += n_lab
    tsh  = arr[:, i:i+n_lab];    i += n_lab
    if n_eval > 0:
        evals = arr[:, i:i+n_eval]
        return X_s, ft3, ft4, tsh, evals
    return X_s, ft3, ft4, tsh


def apply_missforest(raw_matrix, imputer, n_eval_cols=0):
    """Transform with fitted imputer, then round eval cols to {1,2,3}."""
    filled = imputer.transform(raw_matrix)
    if n_eval_cols > 0:
        ev_start = filled.shape[1] - n_eval_cols
        filled[:, ev_start:] = np.clip(np.round(filled[:, ev_start:]), 1, 3)
    return filled


def missforest_cached(raw_all, train_mask, cache_path, n_eval_cols=0, seed=None):
    """Fit-or-load MissForest, transform train & test.

    Returns (filled_train, filled_test).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        imputer = joblib.load(cache_path)
        print(f"  MissForest: loaded from {cache_path.name}")
    else:
        print(f"  MissForest: fitting on train ({train_mask.sum()} samples)...")
        imputer = fit_missforest(raw_all[train_mask], seed=seed)
        joblib.dump(imputer, cache_path)
        print(f"  MissForest: saved to {cache_path.name}")

    filled_tr = apply_missforest(raw_all[train_mask], imputer, n_eval_cols)
    filled_te = apply_missforest(raw_all[~train_mask], imputer, n_eval_cols)
    return filled_tr, filled_te


def load_or_fit_depth_imputer(raw_train, cache_path, fallback_cache_path=None):
    """Load a local/shared MissForest cache or fit a new one."""
    primary = Path(cache_path)
    fallback = Path(fallback_cache_path) if fallback_cache_path is not None else None

    if primary.exists():
        return joblib.load(primary)

    if fallback is not None and fallback.exists():
        return joblib.load(fallback)

    imputer = fit_missforest(raw_train)
    joblib.dump(imputer, primary)
    return imputer


# ---------------------------------------------------------------------------
# Feature engineering (fixed-landmark scripts)
# ---------------------------------------------------------------------------
def extract_flat_features(X_s, X_d, seq_len, static_names=None):
    """Flatten time-series + compute deltas for fixed-landmark models.

    X_d : ndarray (N, T, 3) – stacked [ft3, ft4, tsh]
    Returns (X_feat, feature_names).
    """
    snames = static_names or STATIC_NAMES
    X_d_t = X_d[:, :seq_len, :]
    X_d_flat = X_d_t.reshape(X_d_t.shape[0], -1)

    names = list(snames)
    for t in range(seq_len):
        for h in ["FT3", "FT4", "TSH"]:
            names.append(f"{h}_{TIME_STAMPS[t]}")

    deltas = []
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


# ---------------------------------------------------------------------------
# State building (rolling-landmark scripts)
# ---------------------------------------------------------------------------
def build_states_from_labels(eval_imputed):
    """Build state matrix from imputed doctor evaluation labels.

    0M: all Hyper (pre-treatment).
    1M-24M: from labels (1=甲亢→0, 3=正常→1, 2=甲减→2).
    """
    N = eval_imputed.shape[0]
    T = eval_imputed.shape[1] + 1
    S = np.zeros((N, T), dtype=int)
    label_map = {1: 0, 3: 1, 2: 2}
    for t in range(eval_imputed.shape[1]):
        for i in range(N):
            v = int(round(eval_imputed[i, t]))
            S[i, t + 1] = label_map.get(v, 0)
    return S


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------
def clear_pkl_cache(out_dir):
    """Remove all .pkl files from the output directory."""
    out = Path(out_dir)
    removed = 0
    for f in out.glob("*.pkl"):
        f.unlink()
        removed += 1
    if removed:
        print(f"  Cleared {removed} cached .pkl file(s) from {out}")
    else:
        print(f"  No .pkl cache found in {out}")
