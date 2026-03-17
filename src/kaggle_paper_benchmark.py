"""
PoseRefineLab FK-SMC+SOCM — Kaggle T4x2 benchmark helper
=============================================================================
Optimizations applied:
  Speed-1: ESM embedding disk cache → saves 10-20s per (target × seed) on repeat runs
  Speed-2: Snapper fires only in 2nd half of training → ~40% MMFF time saved
  Speed-3: Seeds interleaved across GPUs → GPU-0 runs seed=42, GPU-1 runs seed=43 concurrently

Expected wall-clock (T4x2, 10 targets, 3 seeds):
  Config A (FK-SMC+SOCM): ~3-4 hr  (was ~7-8 hr)
  Config B (SOCM only):   ~3-4 hr
  Config C (FK-SMC only): ~3-4 hr
  Total: ~9-12 hr (3 configs) or ~6-8 hr (A+B only for paper)
"""

# ── Cell 1: GPU Check ─────────────────────────────────────────────────────────
import torch
import subprocess, sys, os, time, json
import pandas as pd
import numpy as np

print("=== GPU Environment ===")
print(f"torch:            {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"GPU count:        {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  VRAM={props.total_memory//2**20} MB")

assert torch.cuda.is_available(), "ABORT: No GPU. Enable T4x2 in Kaggle settings."
NUM_GPUS = min(torch.cuda.device_count(), 2)

# Also show ESM cache status
_cache_dir = os.path.expanduser("~/.saeb_cache/esm")
n_cached = len(os.listdir(_cache_dir)) if os.path.exists(_cache_dir) else 0
print(f"\nESM cache dir:    {_cache_dir}  ({n_cached} entries)")

# ── Cell 2: Setup ─────────────────────────────────────────────────────────────
# Adjust REPO_ROOT if you imported repo as dataset with a different name
REPO_ROOT = "/kaggle/input/poserefinelab/PoseRefineLab"
SRC_DIR   = f"{REPO_ROOT}/src"
OUT_ROOT  = "/kaggle/working/results"
os.makedirs(OUT_ROOT, exist_ok=True)
sys.path.insert(0, SRC_DIR)

ASTEX10 = "1aq1,1b8o,1cvu,1d3p,1eve,1f0r,1fc0,1fpu,1glh,1gpk"
SEEDS   = "42,43,44"
STEPS   = 300
BATCH   = 8            # Speed hint: 8 vs 16 → ~2x MMFF speedup, 3 seeds compensate coverage

print(f"Targets:          {ASTEX10}")
print(f"Seeds:            {SEEDS}")
print(f"Steps / Batch:    {STEPS} / {BATCH}")
print(f"num_gpus:         {NUM_GPUS}")
print(f"Speed-3 mode:     seed-first interleaving (GPU-0=seed42, GPU-1=seed43)")

# ── Cell 3: Install deps ──────────────────────────────────────────────────────
def shell(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.stdout.strip(): print(r.stdout[-3000:])
    if r.returncode != 0 and r.stderr.strip(): print("STDERR:", r.stderr[-500:])
    return r.returncode

shell("pip install -q posebusters rdkit-pypi biopython 2>/dev/null || true")

# ── Cell 4: Smoke Test (1 target, 1 seed, 100 steps) ─────────────────────────
print("\n" + "="*60)
print("  SMOKE TEST  (1aq1, seed=42, 100 steps, B=4)")
print("="*60)
t_smoke = time.time()
rc_smoke = shell(
    f"cd {SRC_DIR} && python run_benchmark.py "
    f"--targets 1aq1 --seed 42 --steps 100 --batch_size 4 "
    f"--num_gpus {NUM_GPUS} --fksmc --socm "
    f"--output_dir {OUT_ROOT}/smoke 2>&1 | tail -20"
)
print(f"\nSmoke test done in {(time.time()-t_smoke)/60:.1f} min  (exit={rc_smoke})")
assert rc_smoke == 0, "ABORT: smoke test failed. Check logs above."

# ── Cell 5: Experiment Runner ─────────────────────────────────────────────────
def run_exp(label, extra_flags, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_file = f"{out_dir}/run.log"
    t0 = time.time()

    cmd = (
        f"cd {SRC_DIR} && python run_benchmark.py "
        f"--targets {ASTEX10} "
        f"--seeds {SEEDS} "
        f"--steps {STEPS} "
        f"--batch_size {BATCH} "
        f"--num_gpus {NUM_GPUS} "
        f"{extra_flags} "
        f"--output_dir {out_dir} "
        f"2>&1 | tee {log_file}"
    )
    print(f"\n{'='*60}\n  [{label}] Starting ...\n{'='*60}")
    rc = shell(cmd)
    wall = time.time() - t0
    print(f"\n  [{label}] Finished in {wall/3600:.2f} hr  (exit={rc})")

    csv_agg = f"{out_dir}/benchmark_aggregated.csv"
    csv_raw = f"{out_dir}/benchmark_results.csv"
    if not os.path.exists(csv_agg):
        print(f"  [WARN] No aggregated CSV!")
        return None

    df     = pd.read_csv(csv_agg)
    rmsds  = df["best_rmsd"].values
    n      = len(df)
    n_fail = open(log_file).read().count("[FAIL]") if os.path.exists(log_file) else 0

    # Seed variance
    seed_var = 0.0
    if os.path.exists(csv_raw):
        dr = pd.read_csv(csv_raw)
        if "seed" in dr.columns:
            seed_sr2 = dr.groupby("seed").apply(lambda g: (g["best_rmsd"] < 2.0).mean() * 100)
            seed_var = seed_sr2.std()

    # Stability metrics
    ess_mu, rc_mu = "", ""
    if os.path.exists(csv_raw):
        dr = pd.read_csv(csv_raw)
        if "ess_min" in dr.columns:
            ess_mu = f"{pd.to_numeric(dr['ess_min'], errors='coerce').mean():.3f}"
        if "resample_count" in dr.columns:
            rc_mu = f"{pd.to_numeric(dr['resample_count'], errors='coerce').mean():.1f}"

    return {
        "Method":          label,
        "SR@2A (%)":       round((rmsds < 2.0).mean() * 100, 1),
        "SR@5A (%)":       round((rmsds < 5.0).mean() * 100, 1),
        "Median RMSD (A)": round(float(np.median(rmsds)), 2),
        "Mean RMSD (A)":   round(float(np.mean(rmsds)), 2),
        "Crash Rate (%)":  round(n_fail / (n + n_fail + 1e-8) * 100, 1),
        "Seed Var SR@2A":  round(seed_var, 1),
        "ESS_min (mean)":  ess_mu,
        "Resample/run":    rc_mu,
        "Wall Time (hr)":  round(wall / 3600, 2),
        "Time/target (s)": round(wall / max(n, 1), 0),
    }

# ── Cell 6: Run Experiments ───────────────────────────────────────────────────
# Prioritise A+B for paper. Add C if time permits.
EXPERIMENTS = [
    ("B_SOCM_baseline",   "--socm",         f"{OUT_ROOT}/astex10_socm_3seed"),
    ("A_FKSMC_SOCM_ours", "--fksmc --socm", f"{OUT_ROOT}/astex10_fksmc_socm_3seed"),
    # Uncomment if T4x2 session allows (~3-4 hr more):
    # ("C_FKSMC_only",    "--fksmc",        f"{OUT_ROOT}/astex10_fksmc_3seed"),
]

all_results = []
for label, flags, out in EXPERIMENTS:
    res = run_exp(label, flags, out)
    if res: all_results.append(res)

# ── Cell 7: Summary Table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  PAPER SUMMARY — 3-Claim Evidence Table")
print("="*70)
df_sum = pd.DataFrame(all_results)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)
print(df_sum.to_string(index=False))
df_sum.to_csv(f"{OUT_ROOT}/paper_summary.csv", index=False)

# ── Cell 8: Failure Cases ─────────────────────────────────────────────────────
print("\n=== Failure Cases ===")
for label, flags, out in EXPERIMENTS:
    log = f"{out}/run.log"
    if not os.path.exists(log): continue
    lines = open(log).readlines()
    fails = [(i, l) for i, l in enumerate(lines) if "[FAIL]" in l]
    if fails:
        print(f"\n{label}:")
        for i, l in fails:
            print(f"  {l.strip()}")
            if i + 1 < len(lines): print(f"    -> {lines[i+1].strip()[:100]}")
    else:
        print(f"  {label}: 0 failures ✓")

# ── Cell 9: Paper Figures ─────────────────────────────────────────────────────
FIG_DIR = f"{OUT_ROOT}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

try:
    from saeb.reporting.visualizer import PublicationVisualizer
    viz = PublicationVisualizer(output_dir=FIG_DIR)

    rmsd_by_method = {}
    for label, flags, out in EXPERIMENTS:
        f = f"{out}/benchmark_aggregated.csv"
        if os.path.exists(f):
            rmsd_by_method[label] = pd.read_csv(f)["best_rmsd"].values

    if rmsd_by_method:
        viz.plot_success_rate_curve(rmsd_by_method, "fig_sr_curve.pdf")
        viz.plot_rmsd_cdf(rmsd_by_method, "fig_rmsd_cdf.pdf")
        abl = {k: {"sr2": round((v < 2.0).mean()*100, 1),
                   "median_rmsd": round(float(np.median(v)), 2)}
               for k, v in rmsd_by_method.items()}
        viz.plot_ablation(abl, "fig_ablation.pdf")
        print(f"Figures saved → {FIG_DIR}")
except Exception as e:
    import traceback; traceback.print_exc()

# ── Cell 10: File listing ─────────────────────────────────────────────────────
print("\n=== Output Files ===")
for root, dirs, files in os.walk(OUT_ROOT):
    depth = root.replace(OUT_ROOT, "").count(os.sep)
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    for f in sorted(files):
        kb = os.path.getsize(os.path.join(root, f)) / 1024
        print(f"{indent}  {f}  ({kb:.0f} KB)")
