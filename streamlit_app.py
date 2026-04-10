"""Chinese Streamlit app for the current 3M and relapse deployment line."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from thyroid_app.constants import (
    LAB_FIELD_RANGES,
    RELAPSE_CURRENT_LANDMARKS,
    SERIES_NAMES,
    STATIC_FIELD_HELP,
    STATIC_FIELD_RANGES,
    STATE_TO_CODE,
    required_state_timestamps_for_landmark,
    required_timestamps_for_landmark,
)
from thyroid_app.content import FIXED_3M_PLOTS, OVERVIEW_PLOTS, RELAPSE_PLOTS
from thyroid_app.inference import load_manifest, predict_fixed_landmark, predict_relapse


ARTIFACTS_DIR = "artifacts"
STATE_LABELS = {"Hyper": "甲亢", "Normal": "正常", "Hypo": "甲减"}
STATIC_LABELS = {
    "Sex": "Sex / 性别",
    "Age": "Age / 年龄",
    "Height": "Height / 身高",
    "Weight": "Weight / 体重",
    "BMI": "BMI",
    "Exophthalmos": "Exophthalmos / 突眼",
    "ThyroidW": "ThyroidW / 甲状腺重量",
    "RAI3d": "RAI3d",
    "TreatCount": "TreatCount / 既往治疗次数",
    "TGAb": "TGAb",
    "TPOAb": "TPOAb",
    "TRAb": "TRAb",
    "Uptake24h": "Uptake24h / 24h摄取率",
    "MaxUptake": "MaxUptake / 最大摄取率",
    "HalfLife": "HalfLife / 有效半衰期",
    "Dose": "Dose / RAI剂量",
}


@st.cache_resource
def get_manifest() -> dict:
    return load_manifest(ARTIFACTS_DIR)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:
                radial-gradient(circle at 0% 0%, rgba(96, 165, 250, 0.18), transparent 24%),
                radial-gradient(circle at 100% 0%, rgba(45, 212, 191, 0.16), transparent 22%),
                linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%);
            --card-bg: rgba(255,255,255,0.88);
            --card-border: rgba(15, 23, 42, 0.08);
            --card-shadow: rgba(15, 23, 42, 0.08);
            --title: #0f172a;
            --body: #334155;
            --muted: #64748b;
            --accent: #0f766e;
        }
        .stApp { background: var(--bg); }
        html, body, [class*="css"]  {
            font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", sans-serif;
        }
        h1, h2, h3 { color: var(--title); letter-spacing: -0.02em; }
        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: 0 12px 32px var(--card-shadow);
        }
        .metric-label {
            font-size: 0.84rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--title);
            line-height: 1.1;
            margin: 0.15rem 0 0.35rem 0;
        }
        .metric-caption {
            font-size: 0.93rem;
            color: var(--body);
        }
        .note-card {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.82);
            border-left: 4px solid var(--accent);
            color: var(--body);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{title}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-caption">{caption}</div>
        </div>
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
                st.warning(f"缺少图像文件：`{path_str}`")
            else:
                st.image(str(path), use_column_width=True)


def _render_top_drivers(drivers: list[dict]) -> None:
    if not drivers:
        st.caption("当前预测未返回可展示的主要驱动因素。")
        return
    rows = []
    for driver in drivers:
        if "contribution" in driver:
            score = float(driver["contribution"])
        elif "importance" in driver:
            score = float(driver["importance"])
        elif "coefficient" in driver:
            score = float(driver["coefficient"])
        else:
            score = 0.0
        rows.append({"变量": driver["feature"], "数值": round(score, 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_case_form(prefix: str, landmark: str, include_states: bool) -> dict:
    required_timestamps = required_timestamps_for_landmark(landmark)
    payload = {"landmark": landmark, "static": {}, "series": {lab: {} for lab in SERIES_NAMES}, "states": {}}

    st.subheader("静态临床信息")
    static_cols = st.columns(2)
    for idx, field_name in enumerate(STATIC_FIELD_RANGES):
        lower, upper = STATIC_FIELD_RANGES[field_name]
        label = STATIC_LABELS.get(field_name, field_name)
        with static_cols[idx % 2]:
            if field_name in {"Sex", "Exophthalmos"}:
                option_map = {"否 / 女 / 0": 0.0, "是 / 男 / 1": 1.0}
                current = st.selectbox(
                    label,
                    options=list(option_map.keys()),
                    index=0,
                    help=STATIC_FIELD_HELP.get(field_name, ""),
                    key=f"{prefix}_{field_name}",
                )
                payload["static"][field_name] = option_map[current]
            else:
                default = lower if field_name in {"TreatCount"} else max(lower, min(upper, (lower + upper) / 5))
                payload["static"][field_name] = st.number_input(
                    label,
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(default),
                    help=STATIC_FIELD_HELP.get(field_name, ""),
                    key=f"{prefix}_{field_name}",
                )

    st.subheader("实验室轨迹")
    for timestamp in required_timestamps:
        st.markdown(f"**{timestamp}**")
        cols = st.columns(3)
        for col_idx, lab_name in enumerate(SERIES_NAMES):
            lower, upper = LAB_FIELD_RANGES[lab_name]
            with cols[col_idx]:
                payload["series"][lab_name][timestamp] = st.number_input(
                    f"{lab_name} @ {timestamp}",
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(max(lower, min(upper, (lower + upper) / 8))),
                    key=f"{prefix}_{lab_name}_{timestamp}",
                )

    if include_states:
        st.subheader("临床状态历史")
        label_to_code = {f"{v} / {k}": k for k, v in STATE_LABELS.items()}
        for timestamp in required_state_timestamps_for_landmark(landmark):
            choice = st.selectbox(
                f"{timestamp} 状态",
                options=list(label_to_code.keys()),
                index=1 if timestamp == landmark else 0,
                key=f"{prefix}_state_{timestamp}",
            )
            payload["states"][timestamp] = label_to_code[choice]

    return payload


def render_overview(manifest: dict) -> None:
    st.title("RAI 随访风险工作台")
    st.markdown(
        """
        <div class="note-card">
        本界面提供两个实际部署功能：`3M 早期反应评估` 与 `动态复发预警`。
        前者用于判断患者在 3 个月时点是否仍处于高风险早期反应状态，后者用于在当前随访时点更新下一时间窗复发风险。
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        _metric_card(
            "3M Accuracy",
            f"{manifest['tasks']['fixed_3m']['test_metrics']['acc']:.3f}",
            "当前 3M 部署模型的时间外推测试准确率",
        )
    with cols[1]:
        _metric_card(
            "Validation PR-AUC",
            f"{manifest['tasks']['relapse']['validation_metrics']['prauc']:.3f}",
            "动态复发模型内部验证集 PR-AUC",
        )
    with cols[2]:
        _metric_card(
            "Temporal PR-AUC",
            f"{manifest['tasks']['relapse']['test_metrics']['prauc']:.3f}",
            "动态复发模型时间外推测试 PR-AUC",
        )
    with cols[3]:
        _metric_card(
            "Direct Baseline",
            f"{manifest['tasks']['relapse_direct']['test_metrics']['prauc']:.3f}",
            "dynamic-only 基线的时间外推测试 PR-AUC",
        )

    st.header("方法与主结果")
    _display_plot_grid(OVERVIEW_PLOTS, columns=2)

    st.header("使用建议")
    st.markdown(
        """
        - `3M 早期反应评估` 适合在 3 个月时点判断患者是否仍处于较高早期反应风险。
        - `动态复发预警` 适合在当前状态为 `Normal` 的随访时点，评估下一时间窗复发风险。
        - 当前 relapse 模块的最终风险由多头 backbone 与冻结逻辑回归融合层共同完成，不是单一线性基线。
        """
    )


def render_fixed_3m_calculator(manifest: dict) -> None:
    st.title("3M 早期反应评估")
    st.caption("输入基线信息及 `0M/1M/3M` 实验室指标，评估患者在 3 个月时点是否仍处于高风险早期反应状态。")

    sample_case = manifest["sample_cases"]["fixed"]["3M"]
    use_sample = st.toggle("载入导出示例", value=False, key="fixed3m_sample")
    payload = sample_case if use_sample else _build_case_form("fixed3m", "3M", include_states=False)

    if st.button("运行 3M 评估", type="primary"):
        result = predict_fixed_landmark(payload, "3M", ARTIFACTS_DIR)
        if result["validation_messages"]:
            for msg in result["validation_messages"]:
                st.error(msg)
            return

        cols = st.columns(4)
        with cols[0]:
            _metric_card("3M 风险概率", f"{result['predicted_probability']:.1%}", result["decision_label"])
        with cols[1]:
            _metric_card("判定阈值", f"{result['decision_threshold']:.2f}", result["model_name"])
        with cols[2]:
            _metric_card(
                "时间外推准确率",
                f"{manifest['tasks']['fixed_3m']['test_metrics']['acc']:.3f}",
                "当前 app 载入的 3M 部署模型",
            )
        with cols[3]:
            _metric_card(
                "时间外推 AUC",
                f"{manifest['tasks']['fixed_3m']['test_metrics']['auc']:.3f}",
                "用于理解模型整体判别能力",
            )

        if result.get("routed") is not None and "route_feature" in result:
            st.markdown(
                f"""
                <div class="note-card">
                当前路由规则：`{result['route_feature']}` 与阈值 `{result['route_cutoff']:.4f}` 比较。
                当前个体取值为 `{result['route_value']:.4f}`，因此{('进入了专家子模型' if result.get('routed') else '仍由主融合模型处理')}。
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("主要驱动因素")
        _render_top_drivers(result["top_drivers"])

    st.header("参考图")
    _display_plot_grid(FIXED_3M_PLOTS, columns=2)


def render_relapse_calculator(manifest: dict) -> None:
    st.title("动态复发预警")
    st.caption("输入当前 landmark 之前的静态信息、实验室轨迹与临床状态，更新患者在下一时间窗内的复发风险。当前状态必须为 `Normal`。")

    sample_case = manifest["sample_cases"]["relapse"]
    use_sample = st.toggle("载入导出示例", value=False, key="relapse_sample")
    current_landmarks = manifest["tasks"]["relapse"]["current_landmarks"]
    default_idx = current_landmarks.index(sample_case["landmark"]) if sample_case["landmark"] in current_landmarks else 0
    landmark = st.selectbox("当前 landmark", current_landmarks, index=default_idx if use_sample else 0)
    payload = sample_case if use_sample and sample_case["landmark"] == landmark else _build_case_form("relapse", landmark, include_states=True)
    if use_sample and sample_case["landmark"] != landmark:
        st.info("示例病例对应的当前 landmark 与所选不一致，已切换为手动录入模式。")

    if st.button("运行动态复发预警", type="primary"):
        result = predict_relapse(payload, ARTIFACTS_DIR)
        if result["validation_messages"]:
            for msg in result["validation_messages"]:
                st.error(msg)
            return

        cols = st.columns(4)
        with cols[0]:
            _metric_card("下一窗口复发风险", f"{result['predicted_probability']:.1%}", result["decision_label"])
        with cols[1]:
            _metric_card("预警阈值", f"{result['decision_threshold']:.2f}", result["model_name"])
        with cols[2]:
            landmark_signal = result.get("landmark_signal", result.get("predicted_probability", 0.0))
            _metric_card("3M Landmark 信号", f"{landmark_signal:.1%}", result.get("landmark_source", ""))
        with cols[3]:
            _metric_card("累计随访风险", f"{result['projected_any_relapse_risk']:.1%}", result["projected_risk_group"])

        if "gate_static" in result:
            gate_cols = st.columns(3)
            with gate_cols[0]:
                _metric_card("Gate Static", f"{result['gate_static']:.2f}", "静态分支权重")
            with gate_cols[1]:
                _metric_card("Gate Local", f"{result['gate_local']:.2f}", "局部分支权重")
            with gate_cols[2]:
                _metric_card("Gate Global", f"{result['gate_global']:.2f}", "全局分支权重")

        st.markdown(
            f"""
            <div class="note-card">
            {result['projection_note']}
            当前 `3M landmark` 信号来源：`{result.get('landmark_source', 'n/a')}`。
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "next_hyper_prob" in result:
            st.caption(f"辅助 next-state 头给出的下一状态 `Hyper` 概率：{result['next_hyper_prob']:.1%}")

        st.subheader("主要驱动因素")
        _render_top_drivers(result["top_drivers"])

    st.header("参考图")
    _display_plot_grid(RELAPSE_PLOTS, columns=2)


def main() -> None:
    st.set_page_config(page_title="RAI 随访风险工作台", layout="wide")
    _inject_styles()
    manifest = get_manifest()

    page = st.sidebar.radio("页面", ["总览", "3M 早期反应评估", "动态复发预警"])
    st.sidebar.markdown("---")
    st.sidebar.caption("模型与图表均从 `artifacts/` 和 `results/` 目录加载。")
    st.sidebar.caption("当前仅保留两个部署功能：`3M` 与 `relapse`。")

    if page == "总览":
        render_overview(manifest)
    elif page == "3M 早期反应评估":
        render_fixed_3m_calculator(manifest)
    else:
        render_relapse_calculator(manifest)


if __name__ == "__main__":
    main()
