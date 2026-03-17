# Source Tree

This directory contains the runnable docking code, benchmark entry points, and the core `saeb` package.

## Key entry points

- `run_benchmark.py` — benchmark runner for explicit target lists
- `run_astex10_fksmc_socm.py` — FK-SMC + SOCM Astex-10 wrapper
- `paper_metrics.py` — aggregate benchmark tables
- `kaggle_paper_benchmark.py` — convenience benchmark wrapper for Kaggle-style runs

## Package layout

- `saeb/core/` — model and optimization components
- `saeb/experiment/` — refinement loop, configuration, and search logic
- `saeb/physics/` — force-field scoring and MMFF safeguards
- `saeb/reporting/` — plotting and report helpers
- `saeb/utils/` — shared geometry and I/O utilities

## Notes

- Root-level experiment logs and scratch outputs are intentionally excluded from version control.
- Benchmark evidence intended for sharing lives under `deliverables/workshop_evidence/` at the repository root.
