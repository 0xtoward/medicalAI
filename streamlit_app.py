"""English Streamlit app for the Thyroid RAI project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from thyroid_app.constants import (
    FIXED_LANDMARKS,
    LAB_FIELD_RANGES,
    RELAPSE_CURRENT_LANDMARKS,
    SERIES_NAMES,
    STATIC_FIELD_HELP,
    STATIC_FIELD_RANGES,
    STATE_TO_CODE,
    required_state_timestamps_for_landmark,
    required_timestamps_for_landmark,
)
from thyroid_app.content import ADVANCED_SECTIONS, FIXED_PLOTS, OVERVIEW_PLOTS, RELAPSE_PLOTS
from thyroid_app.inference import load_manifest, predict_fixed_landmark, predict_relapse


ARTIFACTS_DIR = "artifacts"


@st.cache_resource
def get_manifest() -> dict:
    return load_manifest(ARTIFACTS_DIR)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(240, 204, 160, 0.28), transparent 30%),
                radial-gradient(circle at top right, rgba(115, 148, 122, 0.18), transparent 28%),
                linear-gradient(180deg, #f8f4ee 0%, #f2ece3 100%);
        }
        h1, h2, h3 {
            font-family: "IBM Plex Serif", Georgia, serif;
            letter-spacing: -0.02em;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(55, 67, 43, 0.12);
            box-shadow: 0 10px 35px rgba(72, 66, 54, 0.08);
        }
        .note-card {
            padding: 0.9rem 1rem;
            border-left: 4px solid #7f6045;
            background: rgba(255, 252, 247, 0.92);
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_image(path_str: str) -> Path | None:
    path = Path(path_str)
    return path if path.exists() else None


def _display_plot_grid(plot_specs: list[tuple[str, str]], columns: int = 2) -> None:
    cols = st.columns(columns)
    for idx, (title, path_str) in enumerate(plot_specs):
        path = _load_image(path_str)
        with cols[idx % columns]:
            st.markdown(f"**{title}**")
            if path is None:
                st.warning(f"Missing asset: `{path_str}`")
            else:
                st.image(str(path), use_column_width=True)


def _metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div style="font-size:0.88rem; text-transform:uppercase; letter-spacing:0.08em; color:#6c6358;">{title}</div>
          <div style="font-size:1.8rem; font-weight:600; color:#2f3a28; margin:0.1rem 0 0.35rem 0;">{value}</div>
          <div style="font-size:0.92rem; color:#645d54;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_case_form(prefix: str, landmark: str, include_states: bool) -> dict:
    required_timestamps = required_timestamps_for_landmark(landmark)
    case_payload = {"landmark": landmark, "static": {}, "series": {lab: {} for lab in SERIES_NAMES}, "states": {}}

    st.subheader("Static Features")
    static_cols = st.columns(2)
    for idx, field_name in enumerate(STATIC_FIELD_RANGES):
        lower, upper = STATIC_FIELD_RANGES[field_name]
        with static_cols[idx % 2]:
            case_payload["static"][field_name] = st.number_input(
                field_name,
                min_value=float(lower),
                max_value=float(upper),
                value=float(lower if field_name in {"Sex", "Exophthalmos", "TreatCount"} else max(lower, min(upper, (lower + upper) / 5))),
                help=STATIC_FIELD_HELP.get(field_name, ""),
                key=f"{prefix}_{field_name}",
            )

    st.subheader("Laboratory Trajectory")
    for timestamp in required_timestamps:
        st.markdown(f"**{timestamp}**")
        cols = st.columns(3)
        for col_idx, lab_name in enumerate(SERIES_NAMES):
            lower, upper = LAB_FIELD_RANGES[lab_name]
            with cols[col_idx]:
                case_payload["series"][lab_name][timestamp] = st.number_input(
                    f"{lab_name} at {timestamp}",
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(max(lower, min(upper, (lower + upper) / 8))),
                    key=f"{prefix}_{lab_name}_{timestamp}",
                )

    if include_states:
        st.subheader("Clinician State History")
        for timestamp in required_state_timestamps_for_landmark(landmark):
            case_payload["states"][timestamp] = st.selectbox(
                f"State at {timestamp}",
                options=list(STATE_TO_CODE.keys()),
                index=1 if timestamp == landmark else 0,
                key=f"{prefix}_state_{timestamp}",
            )

    return case_payload


def _render_top_drivers(drivers: list[dict]) -> None:
    if not drivers:
        st.caption("No feature-attribution summary is available for this prediction.")
        return
    rows = []
    for driver in drivers:
        numeric_key = "contribution" if "contribution" in driver else "importance" if "importance" in driver else "coefficient"
        rows.append({"Feature": driver["feature"], "Value": round(float(driver[numeric_key]), 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_overview(manifest: dict) -> None:
    st.title("Thyroid RAI Course Modeling")
    st.markdown(
        """
        <div class="note-card">
        Research dashboard first, calculator second. This app packages the manuscript-facing outputs for Graves disease after RAI while exposing real inference for the main rolling relapse task and the fixed-landmark baseline models.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    with cols[0]:
        _metric_card(
            "Mainline AUC",
            f"{manifest['tasks']['relapse']['test_metrics']['auc']:.3f}",
            "Interval-level next-window relapse discrimination on the held-out test set.",
        )
    with cols[1]:
        _metric_card(
            "Mainline PR-AUC",
            f"{manifest['tasks']['relapse']['test_metrics']['prauc']:.3f}",
            "Precision-recall performance for the deployed relapse model.",
        )
    with cols[2]:
        _metric_card(
            "Patient-Level AUC",
            f"{manifest['tasks']['relapse']['patient_test_metrics']['auc']:.3f}",
            "Aggregated patient-level performance from the exported relapse artifact.",
        )

    st.header("Study Figures")
    _display_plot_grid(OVERVIEW_PLOTS, columns=2)

    st.header("Deployment Notes")
    st.markdown(
        """
        - The relapse calculator is the primary deployment task: `Normal -> Hyper` at the next follow-up window.
        - The fixed-landmark calculator estimates whether the patient is still `Hyper` at `3M` or `6M`.
        - Recurrent-survival and two-stage physiology remain benchmark displays only in V1.
        """
    )


def render_relapse_calculator(manifest: dict) -> None:
    st.title("Relapse Calculator")
    st.caption("Predict next-window `Normal -> Hyper` relapse risk for patients who are currently in `Normal`.")

    sample_case = manifest["sample_cases"]["relapse"]
    use_sample = st.toggle("Load exported sample case", value=False)
    landmark = st.selectbox("Current landmark", RELAPSE_CURRENT_LANDMARKS, index=0 if not use_sample else RELAPSE_CURRENT_LANDMARKS.index(sample_case["landmark"]))
    case_payload = sample_case if use_sample and sample_case["landmark"] == landmark else _build_case_form("relapse", landmark, include_states=True)
    if use_sample and sample_case["landmark"] != landmark:
        st.info("The exported sample case belongs to a different landmark. The manual form is shown instead.")

    if st.button("Run relapse prediction", type="primary"):
        result = predict_relapse(case_payload, ARTIFACTS_DIR)
        if result["validation_messages"]:
            for message in result["validation_messages"]:
                st.error(message)
            return

        cols = st.columns(3)
        with cols[0]:
            _metric_card("Next-Window Risk", f"{result['predicted_probability']:.1%}", result["decision_label"])
        with cols[1]:
            _metric_card("Decision Threshold", f"{result['decision_threshold']:.2f}", result["model_name"])
        with cols[2]:
            _metric_card("Projected Follow-up Risk", f"{result['projected_any_relapse_risk']:.1%}", result["projected_risk_group"])

        st.markdown(
            f"""
            <div class="note-card">
            {result['projection_note']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Top Drivers")
        _render_top_drivers(result["top_drivers"])

    st.header("Reference Figures")
    _display_plot_grid(RELAPSE_PLOTS, columns=2)


def render_fixed_calculator(manifest: dict) -> None:
    st.title("Fixed Landmark Calculator")
    st.caption("Estimate whether the patient is still `Hyper` at the selected landmark.")

    landmark = st.radio("Landmark", options=list(FIXED_LANDMARKS.keys()), horizontal=True)
    use_sample = st.toggle("Load exported sample case", value=False, key=f"fixed_sample_{landmark}")
    sample_case = manifest["sample_cases"]["fixed"][landmark]
    case_payload = sample_case if use_sample else _build_case_form(f"fixed_{landmark}", landmark, include_states=False)

    if st.button("Run fixed-landmark prediction", type="primary"):
        result = predict_fixed_landmark(case_payload, landmark, ARTIFACTS_DIR)
        if result["validation_messages"]:
            for message in result["validation_messages"]:
                st.error(message)
            return

        cols = st.columns(3)
        with cols[0]:
            _metric_card("Hyper Probability", f"{result['predicted_probability']:.1%}", result["decision_label"])
        with cols[1]:
            _metric_card("Decision Threshold", f"{result['decision_threshold']:.2f}", result["model_name"])
        with cols[2]:
            task_key = "fixed_3m" if landmark == "3M" else "fixed_6m"
            _metric_card("Test PR-AUC", f"{manifest['tasks'][task_key]['test_metrics']['prauc']:.3f}", "Exported artifact performance")

        st.subheader("Top Drivers")
        _render_top_drivers(result["top_drivers"])

    st.header("Reference Figures")
    _display_plot_grid(FIXED_PLOTS[landmark], columns=2)


def render_advanced_benchmarks() -> None:
    st.title("Advanced Benchmarks")
    st.caption("Display-only benchmark modules. No live inference is exposed in V1.")

    for section in ADVANCED_SECTIONS:
        st.header(section["title"])
        st.write(section["summary"])
        _display_plot_grid(section["plots"], columns=2)
        for table_path in section["tables"]:
            path = Path(table_path)
            if path.exists():
                st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
            else:
                st.warning(f"Missing table: `{table_path}`")


def main() -> None:
    st.set_page_config(page_title="Thyroid RAI App", layout="wide")
    _inject_styles()
    manifest = get_manifest()
    page = st.sidebar.radio(
        "Pages",
        ["Overview", "Relapse Calculator", "Fixed Landmark Calculator", "Advanced Benchmarks"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Artifacts are loaded from `artifacts/` and cached for the session.")

    if page == "Overview":
        render_overview(manifest)
    elif page == "Relapse Calculator":
        render_relapse_calculator(manifest)
    elif page == "Fixed Landmark Calculator":
        render_fixed_calculator(manifest)
    else:
        render_advanced_benchmarks()


if __name__ == "__main__":
    main()
