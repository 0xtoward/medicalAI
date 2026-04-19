"""Generate a patient-level baseline characteristics table."""

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

from utils.data import load_data


OUT_DIR = Path("results/cohort_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.15


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
    """Return one baseline row per patient with cohort split labels."""
    X_s, ft3, ft4, tsh, _, _, pids = load_data()

    columns = [
        "Sex", "Age", "Height", "Weight", "BMI", "Exophthalmos", "ThyroidW", "RAI3d",
        "TreatCount", "TGAb", "TPOAb", "TRAb", "Uptake24h", "MaxUptake", "HalfLife", "Dose",
    ]
    df = pd.DataFrame(X_s, columns=columns)
    df["FT3_0M"] = ft3[:, 0]
    df["FT4_0M"] = ft4[:, 0]
    df["TSH_0M"] = tsh[:, 0]
    df["PID"] = pids

    # Patient-level baseline table: keep the first baseline entry per patient.
    df = df.drop_duplicates("PID", keep="first").copy()

    unique_pids = list(dict.fromkeys(pids))
    dev_cutoff = int(len(unique_pids) * 0.8)
    train_patient_order = unique_pids[:dev_cutoff]
    split_idx = int(len(train_patient_order) * (1 - VAL_FRACTION))
    fit_pids = set(train_patient_order[:split_idx])
    val_pids = set(train_patient_order[split_idx:])

    def assign_split(pid: object) -> str:
        if pid in fit_pids:
            return "Training"
        if pid in val_pids:
            return "Validation"
        return "Temporal test"

    df["Split"] = df["PID"].map(assign_split)
    return df


def section_row(label: str) -> dict[str, str]:
    return {
        "Characteristic": label,
        "Overall": "",
        "Training": "",
        "Validation": "",
        "TemporalValidation": "",
        "P_Cohort": "",
    }


def subrow(label: str, values: dict[str, str], p_cohort: str = "") -> dict[str, str]:
    return {
        "Characteristic": label,
        "Overall": values.get("Overall", ""),
        "Training": values.get("Training", ""),
        "Validation": values.get("Validation", ""),
        "TemporalValidation": values.get("TemporalValidation", ""),
        "P_Cohort": p_cohort,
    }


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    split_order = ["Training", "Validation", "Temporal test"]

    def group_value(column: str, split_name: str, formatter) -> str:
        return formatter(df.loc[df["Split"] == split_name, column])

    rows: list[dict[str, str]] = []

    rows.append(section_row("Demographics"))
    rows.append(
        subrow(
            "Male, n (%)",
            {
                    "Overall": format_binary(df["Sex"]),
                    "Training": group_value("Sex", "Training", format_binary),
                    "Validation": group_value("Sex", "Validation", format_binary),
                    "TemporalValidation": group_value("Sex", "Temporal test", format_binary),
                },
            p_cohort=format_p(chi_square_p((df["Sex"] > 0.5).astype(int), df["Split"])),
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
                    "Training": group_value("Exophthalmos", "Training", format_binary),
                    "Validation": group_value("Exophthalmos", "Validation", format_binary),
                    "TemporalValidation": group_value("Exophthalmos", "Temporal test", format_binary),
                },
            p_cohort=format_p(
                chi_square_p(
                    (pd.to_numeric(df["Exophthalmos"], errors="coerce") > 0.5).astype(float),
                    df["Split"],
                )
            ),
        )
    )

    for key, label in continuous_rows[:11]:
        rows.append(
            subrow(
                label,
                {
                    "Overall": format_median_iqr(df[key]),
                    "Training": group_value(key, "Training", format_median_iqr),
                    "Validation": group_value(key, "Validation", format_median_iqr),
                    "TemporalValidation": group_value(key, "Temporal test", format_median_iqr),
                },
                p_cohort=format_p(continuous_p(df[key], df["Split"])),
            )
        )

    rows.append(section_row("Immunology / baseline labs"))
    for key, label in continuous_rows[11:]:
        rows.append(
            subrow(
                label,
                {
                    "Overall": format_median_iqr(df[key]),
                    "Training": group_value(key, "Training", format_median_iqr),
                    "Validation": group_value(key, "Validation", format_median_iqr),
                    "TemporalValidation": group_value(key, "Temporal test", format_median_iqr),
                },
                p_cohort=format_p(continuous_p(df[key], df["Split"])),
            )
        )

    return pd.DataFrame(rows)


def render_table(table_df: pd.DataFrame, out_path: Path) -> None:
    headers = [
        "Characteristic",
        f"Overall, N = {table_df.attrs['n_total']}",
        f"Training, N = {table_df.attrs['n_train']}",
        f"Validation, N = {table_df.attrs['n_val']}",
        f"Temporal validation, N = {table_df.attrs['n_test']}",
        "P-value",
    ]

    cell_text = table_df[
        ["Characteristic", "Overall", "Training", "Validation", "TemporalValidation", "P_Cohort"]
    ].values.tolist()

    n_rows = len(cell_text)
    fig_h = max(11, 0.40 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    col_widths = [0.30, 0.16, 0.14, 0.14, 0.16, 0.10]
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
        "Table 1. Patient-level baseline demographic, treatment, immunologic, and pretreatment laboratory characteristics.\n"
        "P-values compare Training vs Validation vs Temporal validation cohorts.\n"
        "Categorical variables use chi-square tests; continuous variables use Kruskal-Wallis tests.",
        fontsize=9,
        ha="left",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_feature_frame()
    table_df = make_summary_table(df)
    table_df.attrs["n_total"] = len(df)
    table_df.attrs["n_train"] = int((df["Split"] == "Training").sum())
    table_df.attrs["n_val"] = int((df["Split"] == "Validation").sum())
    table_df.attrs["n_test"] = int((df["Split"] == "Temporal test").sum())

    csv_path = OUT_DIR / "Baseline_Characteristics_Table.csv"
    png_path = OUT_DIR / "Baseline_Characteristics_Table.png"

    table_df.to_csv(csv_path, index=False)
    render_table(table_df, png_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved PNG to {png_path}")


if __name__ == "__main__":
    main()
