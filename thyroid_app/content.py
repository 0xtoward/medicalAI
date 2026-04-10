"""Static display content for the Chinese Streamlit app."""

OVERVIEW_PLOTS = [
    ("主方法图", "m1.png"),
    ("最终判别能力", "results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_FormalWinner_Discrimination.png"),
    ("患者级风险分层", "results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_Patient_Risk_Q1Q4.png"),
    ("端到端 SHAP 重要性", "results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_EndToEnd_SHAP_Bar.png"),
]

RELAPSE_PLOTS = [
    ("判别能力", "results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_FormalWinner_Discrimination.png"),
    ("校准曲线", "results/relapse_teacher_frozen_fuse/Calibration_Temporal.png"),
    ("决策曲线分析", "results/relapse_teacher_frozen_fuse/DCA_Temporal.png"),
    ("阈值敏感性", "results/relapse_teacher_frozen_fuse/Threshold_Sensitivity_Temporal.png"),
    ("混淆矩阵", "results/relapse_teacher_frozen_fuse/Confusion_Temporal.png"),
    ("端到端 SHAP", "results/relapse_teacher_frozen_fuse/TeacherFrozenFuse_EndToEnd_SHAP_Bar.png"),
]

FIXED_3M_PLOTS = [
    ("性能热图", "results/fixed_landmark_binary/Performance_Heatmaps.png"),
    ("3M ROC 曲线", "results/fixed_landmark_binary/ROC_3-Mo.png"),
    ("3M 校准曲线", "results/fixed_landmark_binary/Calibration_3-Mo.png"),
    ("3M 决策曲线分析", "results/fixed_landmark_binary/DCA_3-Mo.png"),
    ("3M 阈值敏感性", "results/fixed_landmark_binary/Threshold_Sensitivity_3-Mo.png"),
    ("3M 混淆矩阵", "results/fixed_landmark_binary/CM_3-Mo.png"),
]
