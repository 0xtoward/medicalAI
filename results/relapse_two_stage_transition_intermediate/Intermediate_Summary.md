# Transition Intermediate Representation Sweep

## Goal

- Keep the two-stage transition setup, but make stage-1 output more intermediate and less directly aligned to relapse / hyper rule targets.
- All outputs here were generated in a brand-new experiment directory; no existing files were modified.

## Best Fusions

- `baseline_windowed_plus_champion_cal`: `Elastic LR` with `PR-AUC 0.351`, `AUC 0.849`, `Brier 0.063`.
- `g3_cluster_soft_state_windowed_plus_latent_cal`: `Elastic LR` with `PR-AUC 0.345`, `AUC 0.847`, `Brier 0.063`.
- `g5_residual_cluster_hybrid_windowed_plus_latent_cal`: `Elastic LR` with `PR-AUC 0.341`, `AUC 0.848`, `Brier 0.063`.
- `g4_residual_pca_windowed_plus_latent_cal`: `Elastic LR` with `PR-AUC 0.341`, `AUC 0.848`, `Brier 0.063`.
- `g2_pca_manifold_windowed_plus_latent_cal`: `Elastic LR` with `PR-AUC 0.333`, `AUC 0.844`, `Brier 0.064`.
- `g1_delta_latent_windowed_plus_latent_cal`: `Elastic LR` with `PR-AUC 0.320`, `AUC 0.848`, `Brier 0.064`.

## Best Sidecars

- `g2_pca_manifold_sidecar_ref`: `Elastic LR` with `PR-AUC 0.289`, `AUC 0.818`, `Brier 0.065`.
- `g2_pca_manifold_sidecar_cal`: `Calibrated` with `PR-AUC 0.287`, `AUC 0.812`, `Brier 0.065`.
- `baseline_champion_sidecar_cal`: `Calibrated` with `PR-AUC 0.278`, `AUC 0.836`, `Brier 0.067`.
- `baseline_champion_sidecar_ref`: `Logistic Reg.` with `PR-AUC 0.275`, `AUC 0.836`, `Brier 0.069`.
- `g3_cluster_soft_state_sidecar_cal`: `Calibrated` with `PR-AUC 0.251`, `AUC 0.796`, `Brier 0.067`.
- `g4_residual_pca_sidecar_cal`: `Calibrated` with `PR-AUC 0.250`, `AUC 0.798`, `Brier 0.067`.
- `g1_delta_latent_sidecar_ref`: `Elastic LR` with `PR-AUC 0.248`, `AUC 0.816`, `Brier 0.069`.
- `g1_delta_latent_sidecar_cal`: `Calibrated` with `PR-AUC 0.246`, `AUC 0.817`, `Brier 0.067`.

## Generation Notes

- `g1_delta_latent`: continuous physiology forecast geometry only; no state/rule target proxy
- `g2_pca_manifold`: compressed low-dimensional manifold of stage-1 continuous forecasts
- `g3_cluster_soft_state`: unsupervised physiology archetype memberships from predicted future labs
- `g4_residual_pca`: residualized future physiology after removing current-window context
- `g5_residual_cluster_hybrid`: residual manifold plus residual archetype memberships
