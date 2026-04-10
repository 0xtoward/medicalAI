# Teacher Frozen Fuse Analysis

## Structure
- Formal winner backbone: `global_floor_040`
- Formal winner fuse: `teacher_landmark_gate_raw__lr`
- Fit AUC / PR-AUC: `0.781 / 0.392`
- Validation AUC / PR-AUC: `0.869 / 0.372`
- Temporal test AUC / PR-AUC: `0.794 / 0.349`

## Main Findings
- 当前 formal winner 在三段数据上都维持了可用判别力，且时间外推测试的 `PR-AUC = 0.349`，已经接近内部验证 `0.372`。
- 从 `AUC` 看，模型在内部验证集表现最强，但在时间外推集上仍保持稳定区分能力，没有出现明显崩塌。
- 从 `PR-AUC` 看，`Fit > Validation > TemporalTest`，这更符合常见训练后排序表现，也说明当前图文已经自洽。

## Gate Reading
- Winner backbone gate mode: `global_floor`
- Validation gate means for the winner were approximately static `0.139`, local `0.461`, global `0.400`
- 这说明最终 backbone 不是由单一分支主导，而是以 `local` 为主、`global` 保底、`static` 做补充

## Fuse-Space Reading
- Final fuse used `Teacher_Logit`, `Landmark_Signal`, `Gate_Static`, and `Gate_Local`
- LR coefficients were:
  - `Teacher_Logit = +0.551`
  - `Gate_Static = +0.268`
  - `Gate_Local = -0.268`
  - `Landmark_Signal = +0.160`
- 这说明最终融合层主要由旧 teacher 风险驱动，3M landmark 做增量补充，branch gate 提供方向性修正

## End-to-End SHAP
Top engineered features by mean absolute SHAP were:
- `Global::Time_In_Normal`
- `Global::FT3_Current_x_CoreWindow_1M->3M`
- `Global::FT4_Current`
- `Global::Interval_Width`
- `Global::FT3_Current`
- `Global::CoreWindow_1M->3M`
- `Global::Ever_Hypo_Before`
- `Local::Window_1M->3M`

## Interpretation
- 当前模型的主要有效信号来自全病程纵向特征和早期窗口交互，而不是单纯依赖静态背景变量。
- `Teacher_Logit` 仍是最终融合层的最强输入，但 end-to-end SHAP 表明，底层真正支撑预测的仍是病程状态、时间窗口和甲功相关轨迹。
- 从现有结果看，这条 teacher-frozen fuse 线已经具备“单模型、单图文、单套解释”的成稿条件。
