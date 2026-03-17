# PoseRefineLab

PoseRefineLab is the working repository for our docking and pose-refinement project together with a small QM negative-result study.

The repository is organized around the code that is still relevant, the current report packages, and the scripts required to regenerate them.

## What This Repo Contains

- `src/`
  - main SAEB/PoseRefineLab implementation
  - refinement, scoring, MMFF safeguards, and benchmark logic
- `scripts/`
  - benchmark utilities
  - audit scripts
  - `build_reports.py` to rebuild the shared report packages
- `docs/`
  - current project status and technical notes
- `reports/`
  - final report packages for:
    - docking
    - quantum
- `quantum/`
  - QM/xTB-related scripts and lightweight notes used to build the QM report package

## Current Project Position

The docking project currently supports a stability-focused claim more strongly than an accuracy-superiority claim.

Supported by the current evidence:

- `SOCM` remains slightly better on aggregate docking accuracy
- `FK-SMC + SOCM` is materially more stable across seeds
- hard failures are now diagnosed as primarily `search-limited`
- MMFF outlier contamination is handled by auto-disable safeguards

The QM line is kept as a documented pilot negative result:

- the xTB rescoring workflow works end-to-end
- but ligand-only and pocket-cluster rescoring did not improve pose ranking on the tested cases

## Key Reports

Docking package:

- `reports/docking/report.md`
- `reports/docking/stability_comparison.png`
- `reports/docking/gap_audit_targets.png`
- `reports/docking/summary_metrics.png`

Quantum package:

- `reports/quantum/report.md`
- `reports/quantum/qm_rmsd_comparison.png`
- `reports/quantum/qm_delta_vs_selected.png`

## Rebuilding The Reports

From the repository root:

```bash
python scripts/build_reports.py
```

This regenerates:

- `reports/docking/`
- `reports/quantum/`

from the retained final CSV inputs and QM summary files.

## Benchmark Entry Points

Main benchmark CLI:

```bash
python src/run_benchmark.py --help
```

Astex-10 FK-SMC + SOCM wrapper:

```bash
python src/run_astex10_fksmc_socm.py --help
```

Search-vs-selection audit:

```bash
python scripts/search_selection_gap_audit.py --help
```

## Notes

- Intermediate Kaggle outputs, experimental scratch results, archives, and legacy submission folders are intentionally excluded from GitHub.
- The repository is meant to present the current working code and the current report packages, not every historical artifact.
