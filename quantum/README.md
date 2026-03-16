# Quantum Folder

This folder now serves two purposes:

- legacy proof artifacts for the small local VQE demo
- the current QM-rescoring workflow that plugs into the docking pipeline

Contents:

- `scripts/qm_rescore_xtb.py`
  Rescores exported docking candidates with `GFN-xTB` and writes `xtb_rescored.csv`.
- `scripts/run_xtb_rescore_wsl.ps1`
  Windows helper that forwards a candidate directory into WSL and runs `qm_rescore_xtb.py`.
- `scripts/quantum_h2_vqe_demo.py`
  Legacy local H2 VQE demo.
- `scripts/generate_two_track_proof_pdf.py`
  Legacy one-page PDF generator for the old proof sheet.
- `outputs/`
  Local artifacts from the older quantum demo route.

Current recommended workflow:

1. Run docking with candidate export enabled:
   `python -u src/run_benchmark.py ... --selection_score energy --dump_candidate_topk 5`
2. Locate exported candidates under:
   `<output_dir>/qm_candidates/<pdb_id>/seed_<seed>/`
3. Run xTB rescoring:
   `python quantum/scripts/qm_rescore_xtb.py --input <candidate_dir> --xtb_bin xtb`
   or on Windows:
   `powershell -ExecutionPolicy Bypass -File quantum/scripts/run_xtb_rescore_wsl.ps1 -InputPath <candidate_dir>`
4. Inspect:
   - `candidate_metadata.csv`
   - `candidate_topk.csv`
   - `xtb_rescored.csv`
   - `xtb_summary.txt`

What this QM step does:

- It does not replace docking.
- It rescales the final top-k ligand poses using a QM ligand-strain signal from `GFN-xTB`.
- The intended use is a lightweight, practical `QM-informed rescoring` stage after FK-SMC/SOCM.

Notes:

- This is the pragmatic first step. It corrects ligand strain, not full protein-ligand QM interaction.
- If this improves ranking, the next stage is a tighter pocket-cluster QM/MM style rescoring pass.
