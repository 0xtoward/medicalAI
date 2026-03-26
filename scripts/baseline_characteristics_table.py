"""Generate a publication-style baseline characteristics table."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from utils.data import load_data, temporal_split


OUT_DIR = Path("results/cohort_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)


OUTCOME_MAP = {1: "Hyper", 2: "Hypo", 3: "Normal"}


def format_median_iqr(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    q1 = vals.quantile(0.25)
    med = vals.median()
    q3 = vals.quantile(0.75)
    return f"{med:.2f} ({q1:.2f}, {q3:.2f})"


def format_binary(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    yes = int((vals > 0.5).sum())
    total = int(vals.shape[0])
    return f"{yes} ({yes / total * 100:.1f}%)"


def format_p(value: float) -> str:
    if np.isnan(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def chi_square_p(series: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": series, "group": groups}).dropna()
    if frame.empty or frame["group"].nunique() < 2 or frame["value"].nunique() < 2:
        return np.nan
    table = pd.crosstab(frame["group"], frame["value"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    return float(chi2_contingency(table)[1])


def continuous_p(series: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(series, errors="coerce"), "group": groups}).dropna()
    if frame.empty or frame["group"].nunique() < 2:
        return np.nan
    grouped = [g["value"].values for _, g in frame.groupby("group", sort=False)]
    if any(len(arr) == 0 for arr in grouped):
        return np.nan
    if len(grouped) == 2:
        return float(mannwhitneyu(grouped[0], grouped[1], alternative="two-sided").pvalue)
    return float(kruskal(*grouped).pvalue)


def build_feature_frame() -> pd.DataFrame:
    X_s, ft3, ft4, tsh, _, y, pids = load_data()
    train_mask, test_mask = temporal_split(pids)

    columns = [
        "Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos", "ThyroidW", "RAI3d",
        "TreatCount", "TGAb", "TPOAb", "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose",
    ]
    df = pd.DataFrame(X_s, columns=columns)
    df["FT3_0M"] = ft3[:, 0]
    df["FT4_0M"] = ft4[:, 0]
    df["TSH_0M"] = tsh[:, 0]
    df["Outcome"] = pd.Series(y).map(OUTCOME_MAP)
    df["Split"] = np.where(train_mask, "Training", "Test")
    return df


def section_row(label: str) -> dict[str, str]:
    return {
        "Characteristic": label,
        "Overall": "",
        "Hyper": "",
        "Normal": "",
        "Hypo": "",
        "P_Outcome": "",
        "Training": "",
        "Test": "",
        "P_Split": "",
    }


def subrow(label: str, values: dict[str, str], p_outcome: str = "", p_split: str = "") -> dict[str, str]:
    return {
        "Characteristic": label,
        "Overall": values.get("Overall", ""),
        "Hyper": values.get("Hyper", ""),
        "Normal": values.get("Normal", ""),
        "Hypo": values.get("Hypo", ""),
        "P_Outcome": p_outcome,
        "Training": values.get("Training", ""),
        "Test": values.get("Test", ""),
        "P_Split": p_split,
    }


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    outcome_order = ["Hyper", "Normal", "Hypo"]
    split_order = ["Training", "Test"]

    def group_value(column: str, group_col: str, group_name: str, formatter) -> str:
        return formatter(df.loc[df[group_col] == group_name, column])

    rows: list[dict[str, str]] = []

    rows.append(section_row("Outcome class, n (%)"))
    outcome_counts = df["Outcome"].value_counts()
    split_p = chi2_contingency(pd.crosstab(df["Split"], df["Outcome"]))[1]
    for idx, outcome in enumerate(outcome_order):
        values = {
            "Overall": f"{int(outcome_counts.get(outcome, 0))} ({outcome_counts.get(outcome, 0) / len(df) * 100:.1f}%)",
            "Hyper": f"{int((df['Outcome'] == outcome).sum()) if outcome == 'Hyper' else 0} ({(df['Outcome'] == outcome).mean() * 100:.1f}%)" if outcome == "Hyper" else f"0 (0.0%)",
            "Normal": f"{int((df['Outcome'] == outcome).sum()) if outcome == 'Normal' else 0} ({(df['Outcome'] == outcome).mean() * 100:.1f}%)" if outcome == "Normal" else f"0 (0.0%)",
            "Hypo": f"{int((df['Outcome'] == outcome).sum()) if outcome == 'Hypo' else 0} ({(df['Outcome'] == outcome).mean() * 100:.1f}%)" if outcome == "Hypo" else f"0 (0.0%)",
            "Training": f"{int(((df['Split'] == 'Training') & (df['Outcome'] == outcome)).sum())} ({((df['Split'] == 'Training') & (df['Outcome'] == outcome)).sum() / (df['Split'] == 'Training').sum() * 100:.1f}%)",
            "Test": f"{int(((df['Split'] == 'Test') & (df['Outcome'] == outcome)).sum())} ({((df['Split'] == 'Test') & (df['Outcome'] == outcome)).sum() / (df['Split'] == 'Test').sum() * 100:.1f}%)",
        }
        rows.append(subrow(f"  {outcome}", values, p_split=format_p(split_p) if idx == 0 else ""))

    rows.append(section_row("Demographics"))
    rows.append(
        subrow(
            "Male, n (%)",
            {
                "Overall": format_binary(df["Sex"]),
                "Hyper": group_value("Sex", "Outcome", "Hyper", format_binary),
                "Normal": group_value("Sex", "Outcome", "Normal", format_binary),
                "Hypo": group_value("Sex", "Outcome", "Hypo", format_binary),
                "Training": group_value("Sex", "Split", "Training", format_binary),
                "Test": group_value("Sex", "Split", "Test", format_binary),
            },
            p_outcome=format_p(chi_square_p((df["Sex"] > 0.5).astype(int), df["Outcome"])),
            p_split=format_p(chi_square_p((df["Sex"] > 0.5).astype(int), df["Split"])),
        )
    )

    continuous_rows = [
        ("Age", "Age, median (IQR)"),
        ("Height", "Height, median (IQR)"),
        ("Weight", "Weight, median (IQR)"),
        ("BMI", "BMI, median (IQR)"),
        ("ThyroidW", "Thyroid weight, median (IQR)"),
        ("RAI3d", "RAI 3d uptake, median (IQR)"),
        ("TreatCount", "Treatment count, median (IQR)"),
        ("Dose", "Dose, median (IQR)"),
        ("Uptake24h", "24h uptake, median (IQR)"),
        ("MaxUptake", "Max uptake, median (IQR)"),
        ("HalfLife", "Half-life, median (IQR)"),
        ("TGAb", "TGAb, median (IQR)"),
        ("TPOAb", "TPOAb, median (IQR)"),
        ("TRAb", "TRAb, median (IQR)"),
        ("FT3_0M", "FT3 0M, median (IQR)"),
        ("FT4_0M", "FT4 0M, median (IQR)"),
        ("TSH_0M", "TSH 0M, median (IQR)"),
    ]

    rows.append(section_row("Disease burden / treatment"))
    rows.append(
        subrow(
            "Exophthalmos, n (%)",
            {
                "Overall": format_binary(df["Exophthalmos"]),
                "Hyper": group_value("Exophthalmos", "Outcome", "Hyper", format_binary),
                "Normal": group_value("Exophthalmos", "Outcome", "Normal", format_binary),
                "Hypo": group_value("Exophthalmos", "Outcome", "Hypo", format_binary),
                "Training": group_value("Exophthalmos", "Split", "Training", format_binary),
                "Test": group_value("Exophthalmos", "Split", "Test", format_binary),
            },
            p_outcome=format_p(chi_square_p((pd.to_numeric(df["Exophthalmos"], errors="coerce") > 0.5).astype(float), df["Outcome"])),
            p_split=format_p(chi_square_p((pd.to_numeric(df["Exophthalmos"], errors="coerce") > 0.5).astype(float), df["Split"])),
        )
    )

    for key, label in continuous_rows[:11]:
        rows.append(
            subrow(
                label,
                {
                    "Overall": format_median_iqr(df[key]),
                    "Hyper": group_value(key, "Outcome", "Hyper", format_median_iqr),
                    "Normal": group_value(key, "Outcome", "Normal", format_median_iqr),
                    "Hypo": group_value(key, "Outcome", "Hypo", format_median_iqr),
                    "Training": group_value(key, "Split", "Training", format_median_iqr),
                    "Test": group_value(key, "Split", "Test", format_median_iqr),
                },
                p_outcome=format_p(continuous_p(df[key], df["Outcome"])),
                p_split=format_p(continuous_p(df[key], df["Split"])),
            )
        )

    rows.append(section_row("Immunology / baseline labs"))
    for key, label in continuous_rows[11:]:
        rows.append(
            subrow(
                label,
                {
                    "Overall": format_median_iqr(df[key]),
                    "Hyper": group_value(key, "Outcome", "Hyper", format_median_iqr),
                    "Normal": group_value(key, "Outcome", "Normal", format_median_iqr),
                    "Hypo": group_value(key, "Outcome", "Hypo", format_median_iqr),
                    "Training": group_value(key, "Split", "Training", format_median_iqr),
                    "Test": group_value(key, "Split", "Test", format_median_iqr),
                },
                p_outcome=format_p(continuous_p(df[key], df["Outcome"])),
                p_split=format_p(continuous_p(df[key], df["Split"])),
            )
        )

    return pd.DataFrame(rows)


def render_table(table_df: pd.DataFrame, out_path: Path) -> None:
    headers = [
        "Characteristic",
        f"Overall, N = {table_df.attrs['n_total']}",
        f"Hyper, N = {table_df.attrs['n_hyper']}",
        f"Normal, N = {table_df.attrs['n_normal']}",
        f"Hypo, N = {table_df.attrs['n_hypo']}",
        "P-value¹",
        f"Training, N = {table_df.attrs['n_train']}",
        f"Test, N = {table_df.attrs['n_test']}",
        "P-value²",
    ]

    cell_text = table_df[
        ["Characteristic", "Overall", "Hyper", "Normal", "Hypo", "P_Outcome", "Training", "Test", "P_Split"]
    ].values.tolist()

    n_rows = len(cell_text)
    fig_h = max(12, 0.42 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(21, fig_h))
    ax.axis("off")

    col_widths = [0.24, 0.13, 0.12, 0.12, 0.12, 0.07, 0.12, 0.12, 0.07]
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        colLoc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.01, 0.06, 0.98, 0.92],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    section_color = "#f3d6d6"
    stripe_color = "#f9ecec"
    edge_color = "#d97b7b"

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(weight="bold", color="#222222")
            cell.set_linewidth(0.8)
        else:
            is_section = all(v == "" for v in cell_text[r - 1][1:])
            if is_section:
                cell.set_facecolor(section_color)
                if c == 0:
                    cell.set_text_props(weight="bold", ha="left")
                else:
                    cell.get_text().set_text("")
            else:
                cell.set_facecolor(stripe_color if r % 2 == 0 else "#ffffff")
                if c == 0:
                    cell.set_text_props(ha="left")

    fig.text(
        0.01,
        0.02,
        "Table 1. Baseline demographic, treatment, immunologic, and baseline laboratory characteristics.\n"
        "¹ P-value compares Hyper vs Normal vs Hypo groups. ² P-value compares training vs test split.\n"
        "Categorical variables use chi-square tests; continuous variables use Mann-Whitney U or Kruskal-Wallis tests.",
        fontsize=9,
        ha="left",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_feature_frame()
    table_df = make_summary_table(df)
    table_df.attrs["n_total"] = len(df)
    table_df.attrs["n_hyper"] = int((df["Outcome"] == "Hyper").sum())
    table_df.attrs["n_normal"] = int((df["Outcome"] == "Normal").sum())
    table_df.attrs["n_hypo"] = int((df["Outcome"] == "Hypo").sum())
    table_df.attrs["n_train"] = int((df["Split"] == "Training").sum())
    table_df.attrs["n_test"] = int((df["Split"] == "Test").sum())

    csv_path = OUT_DIR / "Baseline_Characteristics_Table.csv"
    png_path = OUT_DIR / "Baseline_Characteristics_Table.png"

    table_df.to_csv(csv_path, index=False)
    render_table(table_df, png_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved PNG to {png_path}")


if __name__ == "__main__":
    main()
