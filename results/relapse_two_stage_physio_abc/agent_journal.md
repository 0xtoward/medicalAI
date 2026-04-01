# Agent Journal

## Iteration 0
- Strategy: keep all work in the standalone script `scripts/relapse_two_stage_physio_abc.py` and do not modify existing project files.
- Runtime env target: `conda env med`.
- Immediate blocker observed earlier: base shell env was missing `lightgbm`; next run switches to `med` before executing.
- Success criterion for continuing iteration: if no no-stacking branch beats the direct logistic reference on interval-level PR-AUC, keep optimizing the standalone branch logic and rerun.

## Iteration 1
- Env: ran in `conda env med`; `lightgbm` available there.
- Baseline direct logistic reproduced at about `AUC=0.842`, `PR-AUC=0.331`.
- Branch A result: compact handoff stayed below baseline; best compact mode was about `PR-AUC=0.304`.
- Branch B result: `fusion_basic` beat the direct logistic reference on PR-AUC (`0.336` vs `0.331`), but calibration and Brier were worse.
- Branch C result: using the full direct anchor feature set plus sidecar hurt performance; sidecar was disturbing the anchor instead of conservatively helping it.
- Next action: keep Branch B but add OOF-based post-hoc calibration for the tiny fusion model; rewrite Branch C to use the direct anchor score plus tiny sidecar features instead of refitting the whole anchor feature block.

## Iteration 2
- Script update: added OOF-based Platt-style post-hoc calibration for Branch B fusion variants.
- Script update: rewrote Branch C around the direct anchor score plus tiny sidecar inputs, instead of refitting the whole direct feature block.
- Result: `BranchB / fusion_basic / Logistic Reg.` remained the best no-stacking branch at about `AUC=0.840`, `PR-AUC=0.334`.
- Comparison to baseline direct logistic: PR-AUC still improved (`0.334` vs `0.331`), while calibration became much more reasonable (`intercept≈-0.464`, `slope≈0.957`).
- Branch C improved in behavior: the sidecar no longer catastrophically damaged the anchor, but it still did not produce a meaningful gain over the baseline direct score.
- Practical stop condition for this turn: the repository now has a standalone, leakage-safe no-stacking branch that beats the direct logistic PR-AUC reference and exports summaries cleanly.
