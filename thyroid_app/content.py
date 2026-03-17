"""Static display content and plot configuration for the Streamlit app."""

OVERVIEW_PLOTS = [
    ("Methodology", "methodology.png"),
    ("Transition Heatmaps", "results/relapse/Transition_Heatmaps.png"),
    ("Relapse Model Comparison", "results/relapse/Model_Comparison_Bar.png"),
    ("Patient Risk Stratification", "results/relapse/Patient_Risk_Q1Q4.png"),
]

RELAPSE_PLOTS = [
    ("Hazard Alignment", "results/relapse/Hazard_Curve_Strict.png"),
    ("Interval Calibration", "results/relapse/Calibration_Interval.png"),
    ("Patient-Level DCA", "results/relapse/DCA_Patient.png"),
    ("Threshold Sensitivity", "results/relapse/Threshold_Sensitivity_Patient.png"),
]

FIXED_PLOTS = {
    "3M": [
        ("ROC Curve", "results/fixed_landmark_binary/ROC_3-Mo.png"),
        ("Model Comparison", "results/fixed_landmark_binary/Bar_3-Mo.png"),
        ("Calibration", "results/fixed_landmark_binary/Calibration_3-Mo.png"),
        ("Decision Curve Analysis", "results/fixed_landmark_binary/DCA_3-Mo.png"),
    ],
    "6M": [
        ("ROC Curve", "results/fixed_landmark_binary/ROC_6-Mo.png"),
        ("Model Comparison", "results/fixed_landmark_binary/Bar_6-Mo.png"),
        ("Calibration", "results/fixed_landmark_binary/Calibration_6-Mo.png"),
        ("Decision Curve Analysis", "results/fixed_landmark_binary/DCA_6-Mo.png"),
    ],
}

ADVANCED_SECTIONS = [
    {
        "title": "Recurrent-Survival Benchmark",
        "summary": "Display-only benchmark contrasting event-time formulations against the main rolling relapse task.",
        "plots": [
            ("Model Comparison", "results/recurrent_survival/Model_Comparison_Bar.png"),
            ("Next-Window Calibration", "results/recurrent_survival/Calibration_NextWindow.png"),
            ("Horizon Metrics", "results/recurrent_survival/Horizon_AUC_Brier.png"),
            ("Risk Stratification", "results/recurrent_survival/Patient_Risk_Q1Q4.png"),
        ],
        "tables": [
            "results/recurrent_survival/Best_Model_Summary.csv",
            "results/recurrent_survival/Model_Performance.csv",
            "results/recurrent_survival/Horizon_Metrics.csv",
        ],
    },
    {
        "title": "Two-Stage Physiology Benchmark",
        "summary": "Display-only benchmark that predicts future physiology first and relapse second.",
        "plots": [
            ("Stage-1 Metric Bar", "results/relapse_two_stage_physio/Stage1_Forecast_Bar.png"),
            ("Stage-1 Scatter", "results/relapse_two_stage_physio/RandomForest_Test_Scatter.png"),
            ("Stage-2 Group Comparison", "results/relapse_two_stage_physio/TwoStage_Group_Comparison.png"),
            ("Stage-2 Calibration", "results/relapse_two_stage_physio/Calibration_BestStage2.png"),
        ],
        "tables": [
            "results/relapse_two_stage_physio/Physio_Forecast_Metrics.csv",
            "results/relapse_two_stage_physio/TwoStage_Model_Comparison.csv",
        ],
    },
]

