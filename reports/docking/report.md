# Docking Report

## Scope
Astex-9 final comparison between `fksmc_socm_final` and `socm_final` using the stable mainline with MMFF auto-disable protection.

## Key result
- `socm_final` remains better on aggregate accuracy (`SR@2`, median RMSD).
- `fksmc_socm_final` is materially more stable across seeds.
- Hard failures are now diagnosed as search-limited rather than scoring-limited.

## Per-seed evidence
- FK-SMC+SOCM median RMSD by seed: seed 42=3.281 Å, seed 43=3.176 Å, seed 44=3.039 Å.
- SOCM median RMSD by seed: seed 42=3.266 Å, seed 43=2.517 Å, seed 44=2.723 Å.
These per-seed values explain why SOCM has slightly better aggregate accuracy but much larger cross-seed spread.

## Non-search-limited view
- After removing search-limited targets, FK-SMC+SOCM mean selected RMSD = 2.818 Å.
- After removing search-limited targets, SOCM mean selected RMSD = 2.669 Å.
This filtered view is useful because search-limited targets are effectively insensitive to reranking and can dilute method differences.

## Included files
- `stability_comparison.png`
- `gap_audit_targets.png`
- `summary_metrics.png`
- `per_seed_median_rmsd.png`
- `main_table.csv`
- `stability_table.csv`
- `seed_metrics.csv`
- `non_search_limited_summary.csv`
- `target_gap_summary.csv`
