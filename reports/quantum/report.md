# Quantum Report

## Scope
This report now contains two compact quantum tracks:

1. an H2 VQE bond-length scan used as a direct quantum-method evidence artifact
2. a QM-informed rescoring prototype using xTB on exported docking candidates

The VQE scan is the positive, controlled small-system result. The xTB rescoring workflow is the practical docking-side negative result.

## H2 VQE scan

The repository now includes a full H2 bond-length scan over `0.4–1.5 Å` with 10 points:

- `VQE`
- `exact diagonalization`
- `UFF`

This scan was executed with Qiskit + Qiskit Nature using a Jordan-Wigner mapping, `TwoLocal` ansatz, and `SLSQP` optimization. The output is intended for application materials and workshop-level evidence, not as a standalone quantum chemistry benchmark.

Key takeaway:

- VQE tracks the exact ground-state curve on a small molecule
- UFF captures only a rough geometric trend
- absolute energy scales remain fundamentally different between force fields and quantum chemistry

## Study scale
This is a pilot negative-result study, not a broad benchmark. It only covers 2 targets (`1gpk`, `1b8o`) and 1 seed each.

## Key result
The workflow runs end-to-end, but neither QM mode improved pose ranking on the tested cases.

## Why ligand-only and complex look identical here
In both tested cases, the final recommended candidate index was the same for ligand-only xTB and complex xTB.
The `ok_count` values differ because complex mode successfully scored fewer candidates, but among the candidates that did finish, the same pose remained the lowest-scoring recommendation.
So the identical RMSD values reflect the same final recommended index, not a failure to execute complex mode.

## Why the ranking headroom is small
- `1gpk`: selected 1.932 Å vs oracle 1.739 Å; only 0.193 Å gap.
- `1b8o`: selected 2.533 Å vs oracle 2.179 Å; only 0.354 Å gap.
This limited baseline-to-oracle gap already constrains how much any rescoring method could improve the final ranking on these two examples.

## Included files
- `h2_vqe_scan.png`
- `h2_vqe_scan.csv`
- `h2_vqe_scan.md`
- `application_paragraph.md`
- `qm_negative_result_rmsd.png`
- `qm_rmsd_comparison.png`
- `qm_delta_vs_selected.png`
- `qm_negative_result_summary.csv`
