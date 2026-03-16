#!/usr/bin/env python3
"""
Run Astex-10 with FK-SMC + SOCM enabled.

This is a thin wrapper around run_benchmark.py so local and Kaggle runs
use the same code path and output format.
"""
import argparse
import subprocess
import sys
from pathlib import Path


ASTEX10_DEFAULT = [
    "1aq1", "1b8o", "1cvu", "1d3p", "1eve",
    "1f0r", "1fc0", "1fpu", "1glh", "1gpk",
]


def parse_targets(raw: str):
    if not raw:
        return ASTEX10_DEFAULT
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser(description="Astex-10 FK-SMC+SOCM runner")
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(ASTEX10_DEFAULT),
        help="Comma-separated PDB IDs (default: first 10 from Astex-85).",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"])
    parser.add_argument("--pdb_dir", type=str, default="", help="Path to .pdb files directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 42,43")
    parser.add_argument("--num_gpus", type=int, default=None, help="Optional override")
    parser.add_argument("--kaggle", action="store_true", help="Pass-through flag")
    parser.add_argument("--high_fidelity", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP")
    parser.add_argument("--compile_backbone", action="store_true", help="Enable torch.compile backbone")
    parser.add_argument("--mmff_snap_fraction", type=float, default=0.50,
                        help="Fraction of worst clones to MMFF-snap in mid-run")
    parser.add_argument("--no_target_plots", action="store_true",
                        help="Skip per-target plots for faster benchmark runs")
    parser.add_argument("--no_aggregate_figures", action="store_true",
                        help="Skip aggregate figure generation")
    parser.add_argument("--final_mmff_topk", type=int, default=5,
                        help="Top-K low-energy clones for final MMFF polish")
    parser.add_argument("--final_mmff_max_iter", type=int, default=2000,
                        help="Final MMFF max iterations")
    parser.add_argument("--no_pose_dump", action="store_true",
                        help="Skip writing best pose PDB files")
    parser.add_argument("--adaptive_stop_thresh", type=float, default=0.05,
                        help="Adaptive early-stop threshold (lower = run longer)")
    parser.add_argument("--adaptive_min_step_frac", type=float, default=0.65,
                        help="Minimum step fraction before adaptive stop can trigger")
    parser.add_argument("--adaptive_patience_frac", type=float, default=0.12,
                        help="Patience fraction for adaptive stop")
    parser.add_argument("--rerank_polish_mult", type=int, default=2,
                        help="Second-stage rerank polish multiplier")
    parser.add_argument("--selection_score", type=str, default="clash",
                        choices=["hybrid", "logz", "energy", "clash", "energy_clash"],
                        help="Score used to select the final pose from refined particles")
    parser.add_argument("--search_rescue_min_step_frac", type=float, default=0.35,
                        help="Minimum step fraction before hard-target search rescue can trigger")
    parser.add_argument("--search_rescue_patience_frac", type=float, default=0.08,
                        help="Patience fraction before search rescue can trigger")
    parser.add_argument("--search_rescue_scale", type=float, default=2.5,
                        help="Strength of the stagnation-triggered search rescue move")
    parser.add_argument("--dump_candidate_topk", type=int, default=0,
                        help="If >0, dump top-k candidate poses and metadata for downstream rescoring")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce terminal output noise")
    parser.add_argument("--output_dir", type=str, default="results/astex10_fksmc_socm")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    if not targets:
        raise SystemExit("No valid targets were provided.")

    this_dir = Path(__file__).resolve().parent
    benchmark_script = this_dir / "run_benchmark.py"

    cmd = [
        sys.executable,
        str(benchmark_script),
        "--targets",
        ",".join(targets),
        "--steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--mode",
        args.mode,
        "--output_dir",
        args.output_dir,
        "--fksmc",
        "--socm",
        "--mmff_snap_fraction",
        str(args.mmff_snap_fraction),
        "--final_mmff_topk",
        str(args.final_mmff_topk),
        "--final_mmff_max_iter",
        str(args.final_mmff_max_iter),
        "--adaptive_stop_thresh",
        str(args.adaptive_stop_thresh),
        "--adaptive_min_step_frac",
        str(args.adaptive_min_step_frac),
        "--adaptive_patience_frac",
        str(args.adaptive_patience_frac),
        "--rerank_polish_mult",
        str(args.rerank_polish_mult),
        "--selection_score",
        str(args.selection_score),
        "--search_rescue_min_step_frac",
        str(args.search_rescue_min_step_frac),
        "--search_rescue_patience_frac",
        str(args.search_rescue_patience_frac),
        "--search_rescue_scale",
        str(args.search_rescue_scale),
        "--dump_candidate_topk",
        str(args.dump_candidate_topk),
    ]

    if args.pdb_dir:
        cmd.extend(["--pdb_dir", args.pdb_dir])
    if args.seeds:
        cmd.extend(["--seeds", args.seeds])
    else:
        cmd.extend(["--seed", str(args.seed)])
    if args.num_gpus is not None:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    if args.kaggle:
        cmd.append("--kaggle")
    if args.high_fidelity:
        cmd.append("--high_fidelity")
    if args.amp:
        cmd.append("--amp")
    if args.compile_backbone:
        cmd.append("--compile_backbone")
    if args.no_target_plots:
        cmd.append("--no_target_plots")
    if args.no_aggregate_figures:
        cmd.append("--no_aggregate_figures")
    if args.no_pose_dump:
        cmd.append("--no_pose_dump")
    if args.quiet:
        cmd.append("--quiet")

    print("Launching:", " ".join(cmd))
    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
