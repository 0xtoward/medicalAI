# Teacher Frozen Fuse Analysis

## Structure
- Formal winner backbone: `global_floor_040`
- Formal winner fuse: `teacher_landmark_gate_raw__lr`
- Best validation PR-AUC: `0.372`
- Corresponding temporal test PR-AUC: `0.349`

## Main Findings
- Validation-selected winner improved over the previous validation baseline (`0.345 -> 0.372`), but temporal generalization fell (`0.279 -> 0.349`).
- Highest temporal test PR-AUC in the full sweep was `0.320` from `teacher_landmark_windowed_raw__lr_balanced` on backbone `reuse_current_hard_cap_025`, but its validation PR-AUC was only `0.226`.
- This means the current pipeline is validation-optimistic rather than temporally robust.

## Gate Reading
- Winner backbone gate mode: `global_floor`.
- Validation gate means for the winner were static `0.139`, local `0.461`, global `0.400`.
- In practice, `global_floor_040` fixed the global gate at `0.4` and let static/local split the remaining `0.6`.

## Fuse-Space SHAP
- `Teacher_Logit`: mean |SHAP| = `0.0297`
- `Gate_Static`: mean |SHAP| = `0.0169`
- `Gate_Local`: mean |SHAP| = `0.0169`
- `Landmark_Signal`: mean |SHAP| = `0.0114`

## End-to-End SHAP
- `Global::Time_In_Normal`: mean |SHAP| = `0.0226`
- `Global::FT3_Current_x_CoreWindow_1M->3M`: mean |SHAP| = `0.0112`
- `Global::FT4_Current`: mean |SHAP| = `0.0111`
- `Global::Interval_Width`: mean |SHAP| = `0.0103`
- `Global::FT3_Current`: mean |SHAP| = `0.0101`
- `Global::CoreWindow_1M->3M`: mean |SHAP| = `0.0093`
- `Global::Ever_Hypo_Before`: mean |SHAP| = `0.0078`
- `Local::Window_1M->3M`: mean |SHAP| = `0.0077`

## Interpretation
- The formal winner mainly relies on the old teacher risk, the 3M landmark signal, and the learned branch weights rather than richer nonlinear fuse interactions.
- The validation winner appears to exploit gate-pattern regularities that do not transfer well to the later temporal cohort.
- The more temporally robust rows were usually simpler and more window-aware, especially the reused old-backbone `teacher_landmark_windowed_raw` variants.
