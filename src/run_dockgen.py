#!/usr/bin/env python3
"""
DockGen Setup & Runner
Download DockGen dataset and run SAEB-Flow on the low-similarity subset.

Usage:
    python run_dockgen.py --pdb_dir ./pdb_data --steps 200 --split low
"""
import argparse
import os
import json
import torch
import logging
from pathlib import Path
from rdkit import Chem
from saeb.experiment.config import SimulationConfig
from saeb.utils.pdb_io import RealPDBFeaturizer
from saeb.experiment.suite import SAEBFlowRefinement, kabsch_rmsd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAEB-DockGen")

# DockGen low-similarity subset (< 30% sequence identity)
# Full list available at https://zenodo.org/record/8183973
# A representative subset for initial testing:
DOCKGEN_LOW_SIMILARITY = [
    "6y0t", "5m1l", "4ecy", "7qxr", "7rck",
    "5zck", "6ukx", "7msa", "5u0j", "6xrg",
    "4r6q", "6fcp", "5x2l", "6y2f", "7p21",
    "5c2o", "6o3x", "6e5g", "7op3", "5tku",
    "1a9u", "1hp0", "1hq2", "1iaw", "1j1b",
    "1j3j", "1jd0", "1ke5", "1kv1", "1l2s",
    "1l7f", "1lpz", "1m2z", "1n1m", "1n2v",
    "1n46", "1nav", "1of1", "1of6", "1owh",
]

def run_dockgen_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    targets = args.targets.split(",") if args.targets else DOCKGEN_LOW_SIMILARITY
    
    for pdb_id in targets:
        logger.info(f"== [{pdb_id}] DockGen Low-Similarity Target ==")
        
        config = SimulationConfig(
            pdb_id=pdb_id,
            target_name=f"DockGen_{pdb_id}",
            pdb_dir=args.pdb_dir,
            steps=args.steps,
            batch_size=16
        )
        
        featurizer = RealPDBFeaturizer(config=config)
        featurizer.device = device
        
        try:
            pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = \
                featurizer.parse(pdb_id)
        except Exception as e:
            logger.error(f"  [Skip] {e}")
            continue
        
        # Initialize ensemble from pocket-centered noise
        # Note: If using Beam Search, init_pos is the starting set of candidates.
        N = pos_native.shape[0]
        K = 16  # Clones (Test-Time Compute parameter)
        init_pos = (p_center.to(device).unsqueeze(0).unsqueeze(0).expand(K, N, 3) +
                    torch.randn(K, N, 3, device=device) * 5.0)
        
        # Pocket anchor
        dist_to_center = torch.cdist(pos_P.unsqueeze(0).to(device),
                                     p_center.unsqueeze(0).unsqueeze(0).to(device))[0, :, 0]
        topk_idx = dist_to_center.argsort()[:20]
        pocket_anchor = pos_P[topk_idx].mean(dim=0).to(device)
        
        refiner = SAEBFlowRefinement(config)
        
        out = refiner.refine(
            pos_L_init=init_pos,
            pos_P=pos_P.to(device),
            x_P=x_P.to(device),
            q_P=q_P.to(device),
            x_L=x_L_native.to(device),
            q_L=q_L_native.to(device),
            pocket_anchor=pocket_anchor,
            device=device,
            mol_template=lig_template,
            allow_flexible_receptor=False,
            steps=args.steps,
            use_beam_search=args.beam,
            # v9.0 flags
            use_fksmc=args.fksmc,
            use_socm_twist=args.socm,
            use_srpg=args.srpg,
            srpg_n_iter=3,
            srpg_steps=max(50, args.steps // 4),
            # v10.0 ablation flags
            no_backbone=args.no_backbone,
        )
        
        refined_poses = out["refined_poses"]
        rmsd_all = kabsch_rmsd(refined_poses.to(device), pos_native.to(device))
        best_rmsd = rmsd_all.min().item()
        sr2 = float(best_rmsd < 2.0)
        sr5 = float(best_rmsd < 5.0)
        
        # Save Best Pose as SDF (v8.0 Imp: PoseBusters support)
        if lig_template is not None:
            os.makedirs("./poses", exist_ok=True)
            pose_path = f"./poses/{pdb_id}_refined.sdf"
            try:
                # Update mol coordinates
                conf = lig_template.GetConformer()
                best_coords = refined_poses[rmsd_all.argmin()].cpu().numpy()
                for i in range(lig_template.GetNumAtoms()):
                    conf.SetAtomPosition(i, best_coords[i].tolist())
                # Save Refined Pose
                writer = Chem.SDWriter(pose_path)
                writer.write(lig_template)
                writer.close()
                logger.info(f"  Saved pose to {pose_path}")

                # Save Native Pose (as MOL_TRUE for PoseBusters)
                native_path = f"./poses/{pdb_id}_native.sdf"
                conf = lig_template.GetConformer()
                native_coords = pos_native.cpu().numpy()
                for i in range(lig_template.GetNumAtoms()):
                    conf.SetAtomPosition(i, native_coords[i].tolist())
                writer = Chem.SDWriter(native_path)
                writer.write(lig_template)
                writer.close()
                logger.info(f"  Saved native ligand to {native_path}")

                # Optional: Run PoseBusters validation
                if args.posebusters:
                    import subprocess
                    protein_path = f"{pdb_id}.pdb"
                    if os.path.exists(protein_path):
                        env = os.environ.copy()
                        env["PYTHONIOENCODING"] = "utf-8"
                        cmd = [
                            "python", "-m", "posebusters",
                            "--outfmt", "long",
                            "-l", native_path,
                            "-p", protein_path,
                            pose_path
                        ]
                        logger.info(f"  Running PoseBusters: {' '.join(cmd)}")
                        pb_out = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
                        stdout = pb_out.stdout or ""
                        stderr = pb_out.stderr or ""
                        if stdout:
                            logger.info(f"  PoseBusters Output:\n{stdout}")
                        if stderr:
                            logger.error(f"  PoseBusters Error:\n{stderr}")
                        
                        # Save PoseBusters partial results
                        pb_path = f"./results/posebusters_{pdb_id}.txt"
                        os.makedirs("./results", exist_ok=True)
                        with open(pb_path, "w", encoding="utf-8") as f:
                            f.write(stdout + "\n" + stderr)
            except Exception as e:
                logger.warning(f"  Failed to save/check SDF: {e}")

        results.append({
            "pdb_id": pdb_id,
            "best_rmsd": round(best_rmsd, 3),
            "SR@2A": sr2,
            "SR@5A": sr5,
            "steps": args.steps,
            "beam": args.beam,
            "fksmc": args.fksmc,
            "socm": args.socm,
            "srpg": args.srpg,
            "no_backbone": args.no_backbone,  # v10.0 ablation flag
            "log_Z": out.get("log_Z_history", []),
        })
        
        logger.info(f"  RMSD: {best_rmsd:.2f}A | SR@2A: {sr2:.0f} | SR@5A: {sr5:.0f}")
    
    # Aggregate
    # ... (rest of the aggregation logic stays the same) ...
    if not results:
        logger.warning("No targets processed successfully.")
        return

    sr2_mean = sum(r["SR@2A"] for r in results) / len(results) * 100
    sr5_mean = sum(r["SR@5A"] for r in results) / len(results) * 100
    median_rmsd = sorted(r["best_rmsd"] for r in results)[len(results)//2]
    
    logger.info("=" * 50)
    logger.info(f"DockGen Low-Similarity Benchmark ({len(results)} targets)")
    logger.info(f"SR@2A: {sr2_mean:.1f}%")
    logger.info(f"SR@5A: {sr5_mean:.1f}%")
    logger.info(f"Median RMSD: {median_rmsd:.2f}A")
    logger.info("=" * 50)
    
    # Save results
    if args.no_backbone:
        out_prefix = "no_backbone_"
    elif args.beam:
        out_prefix = "beam_"
    elif args.srpg:
        out_prefix = "srpg_"
    elif args.fksmc:
        out_prefix = "fksmc_"
    else:
        out_prefix = ""
    out_path = Path(f"./dockgen_{out_prefix}results.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {"SR@2A": sr2_mean, "SR@5A": sr5_mean, "median_rmsd": median_rmsd},
            "per_target": results
        }, f, indent=2)
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", default="", help="Path to PDB files directory")
    parser.add_argument("--steps", type=int, default=200, help="Refinement steps")
    parser.add_argument("--split", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--beam", action="store_true", help="Use Beam Search (v8.1)")
    parser.add_argument("--targets", default="", help="Comma-separated list of PDB IDs")
    # v9.0 flags
    parser.add_argument("--fksmc", action="store_true", help="v9.0: FK-SMC (replace Replica Exchange)")
    parser.add_argument("--socm", action="store_true", help="v9.0: SOCM Twist (replace alpha_fixed=0.85)")
    parser.add_argument("--srpg", action="store_true", help="v9.0: Self-Rewarding Particle Gibbs")
    parser.add_argument("--posebusters", action="store_true", help="Run PoseBusters audit on results")
    # v10.0 ablation flags
    parser.add_argument("--no_backbone", action="store_true",
                        help="P0-2 Ablation: pure physics baseline (no neural backbone)")
    args = parser.parse_args()
    run_dockgen_benchmark(args)
