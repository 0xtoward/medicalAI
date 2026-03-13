#!/usr/bin/env python3
"""Regenerate Logistic_Reg_Coefficient_Flow.png from existing coefficients CSV.
Run this after updating the visualization code, without needing to rerun full repluse."""

from pathlib import Path
import pandas as pd

from utils.model_viz import _save_lr_coefficient_flow

OUT_DIR = Path("./results/repluse/")
CSV_PATH = OUT_DIR / "Logistic_Reg_Coefficients.csv"

# Intercept and threshold from last repluse run (update if needed)
INTERCEPT = -3.07
THRESHOLD = 0.15

def main():
    if not CSV_PATH.exists():
        print(f"NotFound: {CSV_PATH}. Run repluse.py first to generate it.")
        return
    df = pd.read_csv(CSV_PATH)
    df["is_active"] = df["is_active"].astype(bool)
    out_path = OUT_DIR / "Logistic_Reg_Coefficient_Flow.png"
    _save_lr_coefficient_flow(
        "Logistic Reg.",
        df,
        INTERCEPT,
        out_path,
        top_n=6,
        decision_threshold=THRESHOLD,
        output_label="P(Relapse at next window)",
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
