# QM Negative Result Package

## Method

We evaluated QM-informed rescoring as a post-processing stage after the docking pipeline.
The baseline used the docking system's `selection_score=energy` final-pose selection.
We then tested two GFN2-xTB rescoring variants on the exported top-k candidates:

1. ligand-only xTB rescoring (`results/qm_ready/...`)
2. pocket-aware protein-cluster + ligand complex xTB rescoring (`results/qm_ready_v2/...`, `--mode complex`)

All QM runs were executed on the local WSL Ubuntu CPU environment.

## Result

- Mean selected RMSD: 2.232 A
- Mean oracle-best RMSD: 1.959 A
- Mean ligand-only xTB recommended RMSD: 2.542 A
- Mean complex xTB recommended RMSD: 2.542 A

Neither QM variant improved the final pose ranking. In both tested targets, the xTB-recommended pose had higher RMSD than the baseline energy-selected pose.

### Target-level evidence

- `1gpk` seed 42: baseline 1.932 A, oracle 1.739 A, ligand-only xTB 2.339 A (delta +0.407 A), complex xTB 2.339 A (delta +0.407 A).
- `1b8o` seed 42: baseline 2.533 A, oracle 2.179 A, ligand-only xTB 2.745 A (delta +0.212 A), complex xTB 2.745 A (delta +0.212 A).

## Interpretation

This negative result indicates that the tested QM signal is not aligned with the pose-quality ranking objective.
Ligand-only xTB mainly captures internal ligand strain, while pose prediction depends on protein-ligand interaction quality.
The pocket-cluster approximation also failed to improve ranking, suggesting that this simplified xTB setup is still not the right scoring signal for the current docking task.
