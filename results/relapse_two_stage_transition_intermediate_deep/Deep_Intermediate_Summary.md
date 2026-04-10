# Deep Intermediate Transition Sweep

## Best Overall

- `fusion_committee_g234` with `Calibrated`.
- PR-AUC `0.351`, AUC `0.847`, Brier `0.063`.

## Top Results

- `fusion_committee_g234`: `PR-AUC 0.351`, `AUC 0.847`, `Brier 0.063`. members=Champion,Direct,committee_g234; best=deep_fusion_committee_g234_fs_cal_interval
- `fusion_sep_g2344`: `PR-AUC 0.344`, `AUC 0.850`, `Brier 0.062`. members=Champion,Direct,g2_nh,g3_n,g4_n,g4_nh; best=deep_fusion_sep_g2344_fs_raw
- `fusion_committee_g2344`: `PR-AUC 0.340`, `AUC 0.845`, `Brier 0.067`. members=Champion,Direct,committee_g2344; best=deep_fusion_committee_g2344_nofs_raw
- `fusion_sep_g34`: `PR-AUC 0.338`, `AUC 0.847`, `Brier 0.063`. members=Champion,Direct,g3_n,g4_nh; best=deep_fusion_sep_g34_fs_cal_interval
- `fusion_committee_g34`: `PR-AUC 0.336`, `AUC 0.841`, `Brier 0.068`. members=Champion,Direct,committee_g34; best=deep_fusion_committee_g34_nofs_raw
- `baseline_windowed_direct`: `PR-AUC 0.336`, `AUC 0.841`, `Brier 0.064`. windowed direct anchor baseline
- `committee_g234`: `PR-AUC 0.336`, `AUC 0.828`, `Brier 0.069`. members=g2_nh,g3_n,g4_nh; best=deep_committee_g234_fs_raw
- `committee_g2344`: `PR-AUC 0.326`, `AUC 0.828`, `Brier 0.063`. members=g2_nh,g3_n,g4_n,g4_nh; best=deep_committee_g2344_fs_cal_interval
- `fusion_sep_g234`: `PR-AUC 0.321`, `AUC 0.846`, `Brier 0.067`. members=Champion,Direct,g2_nh,g3_n,g4_nh; best=deep_fusion_sep_g234_nofs_raw
- `committee_g23`: `PR-AUC 0.309`, `AUC 0.831`, `Brier 0.064`. members=g2_nh,g3_n; best=deep_committee_g23_fs_raw
- `baseline_champion_best`: `PR-AUC 0.278`, `AUC 0.836`, `Brier 0.067`. best calibrated champion baseline
- `committee_g34`: `PR-AUC 0.257`, `AUC 0.802`, `Brier 0.068`. members=g3_n,g4_nh; best=deep_committee_g34_nofs_cal_phase

## Comparison

- vs `tuned_intermediate_best` / `nonhyper_g4_residual_pca_windowed_mainChampion_auxLatent`: `ΔPR-AUC -0.000`, `ΔAUC -0.001`.
- vs `transition_best` / `direct_windowed_core_plus_champion_calibrated`: `ΔPR-AUC -0.000`, `ΔAUC -0.002`.
