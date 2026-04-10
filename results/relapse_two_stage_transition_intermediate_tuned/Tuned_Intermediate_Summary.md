# Tuned Intermediate Transition Sweep

## Optimization Audit

- `direct_anchor_sweep`: used = `True`; detail = direct_ref, direct_windowed, direct_monotonic
- `sidecar_feature_selection`: used = `True`; detail = nofs and L1-selected sidecar branches
- `calibration_sweep`: used = `True`; detail = global, phase, interval intercept strategies
- `stage1_scope_sweep`: used = `True`; detail = normal and nonhyper stage-1 training scopes
- `auxiliary_sidecar_fusion`: used = `True`; detail = main-latent+aux-champion and main-champion+aux-latent

## Best Overall

- `nonhyper_g4_residual_pca_windowed_mainChampion_auxLatent` with `Elastic LR`.
- PR-AUC `0.351`, AUC `0.848`, Brier `0.063`.

## Best Sidecars

- `nonhyper_g2_pca_manifold_best_sidecar`: `Linear` with `PR-AUC 0.301`, `AUC 0.821`, `Brier 0.065`. best sidecar after fs/calibration sweep; candidate=nonhyper_g2_pca_manifold_nofs_raw
- `normal_g2_pca_manifold_best_sidecar`: `Linear` with `PR-AUC 0.290`, `AUC 0.818`, `Brier 0.065`. best sidecar after fs/calibration sweep; candidate=normal_g2_pca_manifold_nofs_raw
- `nonhyper_g3_cluster_soft_state_best_sidecar`: `Calibrated` with `PR-AUC 0.279`, `AUC 0.819`, `Brier 0.065`. best sidecar after fs/calibration sweep; candidate=nonhyper_g3_cluster_soft_state_fs_cal_interval
- `normal_g4_residual_pca_best_sidecar`: `Calibrated` with `PR-AUC 0.267`, `AUC 0.804`, `Brier 0.066`. best sidecar after fs/calibration sweep; candidate=normal_g4_residual_pca_nofs_cal_interval
- `nonhyper_g4_residual_pca_best_sidecar`: `Calibrated` with `PR-AUC 0.250`, `AUC 0.799`, `Brier 0.067`. best sidecar after fs/calibration sweep; candidate=nonhyper_g4_residual_pca_nofs_cal_phase
- `normal_g3_cluster_soft_state_best_sidecar`: `Calibrated` with `PR-AUC 0.231`, `AUC 0.784`, `Brier 0.067`. best sidecar after fs/calibration sweep; candidate=normal_g3_cluster_soft_state_nofs_cal_interval
- `normal_g5_residual_cluster_hybrid_best_sidecar`: `Calibrated` with `PR-AUC 0.219`, `AUC 0.789`, `Brier 0.067`. best sidecar after fs/calibration sweep; candidate=normal_g5_residual_cluster_hybrid_nofs_cal_interval
- `nonhyper_g5_residual_cluster_hybrid_best_sidecar`: `Calibrated` with `PR-AUC 0.182`, `AUC 0.784`, `Brier 0.069`. best sidecar after fs/calibration sweep; candidate=nonhyper_g5_residual_cluster_hybrid_nofs_cal_interval

## Best Fusions

- `nonhyper_g4_residual_pca_windowed_mainChampion_auxLatent`: `Elastic LR` with `PR-AUC 0.351`, `AUC 0.848`, `Brier 0.063`. windowed direct + champion main sidecar + latent auxiliary sidecar
- `normal_g4_residual_pca_windowed_mainChampion_auxLatent`: `Elastic LR` with `PR-AUC 0.351`, `AUC 0.849`, `Brier 0.063`. windowed direct + champion main sidecar + latent auxiliary sidecar
- `normal_g3_cluster_soft_state_windowed_mainChampion_auxLatent`: `Elastic LR` with `PR-AUC 0.351`, `AUC 0.849`, `Brier 0.063`. windowed direct + champion main sidecar + latent auxiliary sidecar
- `normal_g3_cluster_soft_state_direct_windowed_fusion`: `Elastic LR` with `PR-AUC 0.350`, `AUC 0.851`, `Brier 0.063`. direct=direct_windowed; latent_sidecar=g3_cluster_soft_state; scope=normal
- `normal_g3_cluster_soft_state_windowed_mainLatent_auxChampion`: `Elastic LR` with `PR-AUC 0.350`, `AUC 0.850`, `Brier 0.063`. windowed direct + latent main sidecar + champion auxiliary sidecar
- `normal_g3_cluster_soft_state_direct_ref_fusion`: `Elastic LR` with `PR-AUC 0.347`, `AUC 0.850`, `Brier 0.063`. direct=direct_ref; latent_sidecar=g3_cluster_soft_state; scope=normal
- `normal_g4_residual_pca_direct_ref_fusion`: `Elastic LR` with `PR-AUC 0.346`, `AUC 0.849`, `Brier 0.063`. direct=direct_ref; latent_sidecar=g4_residual_pca; scope=normal
- `nonhyper_g4_residual_pca_direct_ref_fusion`: `Elastic LR` with `PR-AUC 0.343`, `AUC 0.846`, `Brier 0.063`. direct=direct_ref; latent_sidecar=g4_residual_pca; scope=nonhyper
- `normal_g2_pca_manifold_windowed_mainChampion_auxLatent`: `Elastic LR` with `PR-AUC 0.343`, `AUC 0.849`, `Brier 0.063`. windowed direct + champion main sidecar + latent auxiliary sidecar
- `nonhyper_g4_residual_pca_windowed_mainLatent_auxChampion`: `Elastic LR` with `PR-AUC 0.342`, `AUC 0.848`, `Brier 0.063`. windowed direct + latent main sidecar + champion auxiliary sidecar
- `nonhyper_g4_residual_pca_direct_windowed_fusion`: `Elastic LR` with `PR-AUC 0.342`, `AUC 0.847`, `Brier 0.063`. direct=direct_windowed; latent_sidecar=g4_residual_pca; scope=nonhyper
- `normal_g2_pca_manifold_direct_ref_fusion`: `Elastic LR` with `PR-AUC 0.341`, `AUC 0.845`, `Brier 0.064`. direct=direct_ref; latent_sidecar=g2_pca_manifold; scope=normal

## Comparison

- vs `previous_intermediate_best` / `baseline_windowed_plus_champion_cal`: `ΔPR-AUC +0.000`, `ΔAUC -0.001`.
- vs `transition_best` / `direct_windowed_core_plus_champion_calibrated`: `ΔPR-AUC +0.000`, `ΔAUC -0.001`.
