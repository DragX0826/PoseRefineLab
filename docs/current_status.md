# Current Status

## Mainline conclusion

The current PoseRefineLab prototype is stable enough to evaluate, but it does not yet support a strong "overall better than baseline" claim.

- `crash_rate` is effectively zero on the recent Astex runs.
- `selection_score=clash` is the most defensible default for current mainline runs.
- FK-SMC+SOCM shows some stability signal, but not a decisive accuracy win over SOCM on Astex-10.
- The main bottleneck on hard targets is now clearly search, not reranking.

## What is now established

### 1. Ranking is not the main blocker on hard targets

Search-vs-selection audit on `1cvu`, `1d3p`, and `1fpu` shows all three remain `search_limited` even after increasing search budget.

Representative means from the latest hard-target run:

| target | selected RMSD | oracle RMSD | class |
| --- | ---: | ---: | --- |
| `1cvu` | 4.385 | 4.043 | search_limited |
| `1d3p` | 4.217 | 4.175 | search_limited |
| `1fpu` | 5.030 | 4.699 | search_limited |

Interpretation:

- For these targets, the system does not reach near-native regions reliably.
- Further score tuning will not materially solve these cases.

### 2. MMFF scoring can be numerically unreliable on specific targets

`1d3p` produced extreme MMFF energy outliers (`1e8` to `1e9` scale) during final polish. This was contaminating final scoring.

This is now mitigated by auto-disabling MMFF final scoring when outliers dominate:

- commit: `c4c1823`
- output field: `mmff_disabled`

Validated on `1d3p`:

- `mmff_disabled = 1` for all three seeds
- `final_energy` returned to sane values around `-1200` to `-1470`
- result classification remained `search_limited`

Interpretation:

- The MMFF fix removes scoring contamination.
- It does not improve docking quality by itself.
- `1d3p` is genuinely hard because search is weak there, not because scoring was merely broken.

### 3. QM rescoring is not a current gain path

Both ligand-only xTB rescoring and pocket-cluster xTB rescoring were implemented and tested.

Observed result:

- workflow is operational
- pose ranking did not improve
- the recommended poses were often worse than the existing selected poses

Interpretation:

- QM rescoring is a valid negative result
- it should not be treated as a mainline improvement path for the current project state

See local package:

- `quantum/qm_negative_result_package/`

## Current best engineering position

Use the following as the stable working assumptions:

- default selection: `clash`
- keep MMFF auto-disable enabled on unstable targets
- treat hard targets separately from medium/easy targets

Do not spend more time on:

- more `selection_score` sweeps
- more toy QM/xTB rescue attempts
- more budget-only search scaling without changing the search mechanism

## Next experiments worth running

### A. Search-kernel improvement on hard targets

Targets:

- `1cvu`
- `1d3p`
- `1fpu`

Goal:

- improve `oracle_best_rmsd`
- not just `best_rmsd`

This is the only direct route to real improvement on the hard subset.

Latest status on this line:

- `d820ff9` (diversified ETKDG initialization) improved `1d3p`
  - `1d3p` moved from `search_limited` to `well_aligned`
  - representative audit mean: selected `3.684 A`, oracle `3.664 A`
- the same change did **not** improve `1cvu` or `1fpu`
- `f2a639d` (pocket-surface initialization) was tested next and showed no measurable gain on `1cvu` / `1fpu`

Interpretation:

- diversified initialization is worth keeping because it helps at least one hard target (`1d3p`)
- but `1cvu` and `1fpu` are unresolved and likely require a genuinely different search kernel, not more initialization variants

### B. Stability-focused claim package

Build a result package centered on:

- zero crash rate
- low fallback rate
- lower seed variance
- explicit diagnosis of search-limited vs ranking-limited behavior

This is currently more defensible than a broad accuracy-superiority claim.

### C. Medium-target audit for "where the method works"

Focus on targets that are already `well_aligned` or close to it, and summarize:

- selected RMSD
- oracle RMSD
- direct ranking hits
- seed variance

This supports a narrower but cleaner claim: the method is stable and interpretable on moderate cases, while hard cases remain search-limited.

## Paper / interview one-line summary

We separated ranking errors, MMFF scoring failures, and hard-target search failures, and the evidence now shows that the remaining limitation is primarily search quality rather than final scoring.

## Stable code baseline

Use these commits as reference points:

- `c4c1823`: keeps MMFF auto-disable on unstable targets
- `d820ff9`: keeps the only search initialization change that showed a real gain (`1d3p`)
- `f2a639d`: rejected; pocket-surface initialization added no value on `1cvu` / `1fpu`
