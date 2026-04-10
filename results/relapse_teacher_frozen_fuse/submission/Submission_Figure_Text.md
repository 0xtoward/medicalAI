# Submission Figure Text

This submission package is based on a single reproducible model:
- Model file: `TeacherFrozenFuse_SubmissionModel.pkl`
- Backbone: `global_floor_040`
- Final fuse: `teacher_landmark_gate_raw__lr`

Core performance:
- Internal validation AUC: `0.869`
- Internal validation PR-AUC: `0.372`
- Temporal test AUC: `0.794`
- Temporal test PR-AUC: `0.349`
- Temporal test F1: `0.315`
- Temporal test recall: `0.341`
- Temporal test specificity: `0.928`
- Decision threshold for discrete test metrics: `0.20`

## Main-text figures

### Fig 1. Architecture
File: `Fig1_Architecture.png`

Caption:
The Teacher-Frozen Fuse model uses three input branches to encode static clinicopathologic features, recent surveillance features, and whole-course longitudinal features. A pretrained backbone is supervised by a 3-month landmark head, a legacy teacher-risk head, and a next-hyperthyroid-state head. The final deployed risk estimate is produced by a frozen-backbone logistic fuse that combines teacher risk, landmark signal, and branch-gating information.

### Fig 2. Discrimination
File: `Fig2_Discrimination.png`

Caption:
Discrimination of the finalized model across the fit, internal validation, and temporal test cohorts. The model achieved an internal validation AUC of 0.869 and PR-AUC of 0.372. On the temporal test cohort, discrimination remained acceptable with an AUC of 0.794 and PR-AUC of 0.349.

Suggested text:
The finalized model showed strong internal discrimination and preserved comparable ranking performance on the temporally held-out cohort, indicating stable generalization under temporal shift.

### Fig 3. Calibration
File: `Fig3_Calibration.png`

Caption:
Temporal test calibration curve of the finalized model. Predicted probabilities were concentrated in the clinically relevant low-risk range, with observed event rates generally tracking predicted risk in the populated probability region.

Suggested text:
Calibration was acceptable within the low-probability range where most postoperative surveillance decisions are concentrated.

### Fig 4. Decision Curve Analysis
File: `Fig4_DCA.png`

Caption:
Decision curve analysis on the temporal test cohort. Across low threshold probabilities, the model provided positive net benefit relative to treat-all and treat-none strategies.

Suggested text:
The model showed clinical utility over a low-threshold decision range, supporting its use as an early surveillance risk-stratification tool.

### Fig 5. Threshold Sensitivity
File: `Fig5_ThresholdSensitivity.png`

Caption:
Threshold sensitivity analysis for the temporal test cohort. The operating threshold of 0.20 balanced F1 with a high-specificity operating point.

Suggested text:
The selected threshold prioritized specificity while maintaining non-zero sensitivity, which is appropriate for a low-prevalence recurrence setting.

### Fig 6. Confusion Matrix
File: `Fig6_ConfusionMatrix.png`

Caption:
Temporal test confusion matrix of the finalized model at a threshold of 0.20.

Suggested text:
At the prespecified threshold, the model achieved high specificity with a conservative positive-calling pattern.

### Fig 7. Patient-level Risk Stratification
File: `Fig7_PatientRiskStrata.png`

Caption:
Patient-level risk stratification using quartiles defined from the training distribution. Predicted risk increased from Q1 to Q4, and the observed recurrence rate was highest in Q4.

Suggested text:
The finalized model separated a clinically meaningful high-risk subgroup, with the top quartile showing the greatest observed recurrence burden.

### Fig 8. End-to-End SHAP Summary
File: `Fig8_EndToEnd_SHAP.png`

Caption:
End-to-end SHAP summary plot for the finalized model, computed at the engineered-feature level rather than only at the fuse-feature level. Important contributors were dominated by whole-course longitudinal features and early-window interactions.

Suggested text:
Feature attribution showed that the model relied primarily on global trajectory descriptors, particularly Time_In_Normal, FT3/FT4-related measurements, interval width, and early-window interaction terms.

### Fig 9. End-to-End SHAP Importance
File: `Fig9_EndToEnd_SHAP_Bar.png`

Caption:
Mean absolute SHAP importance of the top engineered features. Whole-course longitudinal features contributed the largest share of model signal, whereas static background variables played a smaller role.

Suggested text:
The attribution profile suggests that dynamic surveillance features, rather than baseline static variables alone, drive most of the model's predictive signal.

## Supplementary figures

### Supp Fig 1. Coefficients of the Final Fuse
File: `Supp_Coefficients.png`

Caption:
Coefficients of the final logistic fuse. Teacher-derived risk remained the strongest contributor, followed by branch-gating information and the 3-month landmark signal.

### Supp Fig 2. High-risk Local Explanation
File: `Supp_EndToEnd_SHAP_HighRisk.png`

Caption:
Example end-to-end SHAP explanation for a high-risk temporal test patient. The prediction was driven upward by limited time spent in the normal state, landmark-positive information, and abnormal longitudinal thyroid-related features.

### Supp Fig 3. Low-risk Local Explanation
File: `Supp_EndToEnd_SHAP_LowRisk.png`

Caption:
Example end-to-end SHAP explanation for a low-risk temporal test patient. Lower risk was associated with longer time in the normal state, later surveillance timing, and more stable longitudinal biochemical signals.

## Main-text result paragraph

We developed a teacher-frozen fuse model that integrates static, recent, and whole-course surveillance features through a pretrained multi-branch backbone and a final logistic fusion layer. In internal validation, the finalized model achieved an AUC of 0.869 and a PR-AUC of 0.372. On the temporally held-out test cohort, the model retained an AUC of 0.794 and a PR-AUC of 0.349. Calibration was acceptable within the low-risk range, decision-curve analysis showed positive net benefit at low threshold probabilities, and quartile-based patient stratification identified a higher-risk subgroup in the top risk quartile. End-to-end SHAP analysis indicated that model predictions were driven primarily by longitudinal surveillance features and early-window interaction patterns.

## Wording to avoid

- Do not combine validation metrics from one model with test metrics from another model.
- Do not describe fuse-space SHAP as if it were feature-level SHAP.
- Do not call the temporal test set an internal validation set.
