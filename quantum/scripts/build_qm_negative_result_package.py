import csv
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_QM = ROOT / "results" / "qm_ready"
RESULTS_QM_V2 = ROOT / "results" / "qm_ready_v2"
OUT_DIR = ROOT / "quantum" / "qm_negative_result_package"


def _load_recommended_row(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    rec = next((row for row in rows if str(row.get("xtb_recommended", "0")) == "1"), None)
    if rec is None and rows:
        ranked = [row for row in rows if str(row.get("xtb_rank", "")).isdigit()]
        if ranked:
            rec = sorted(ranked, key=lambda row: int(row["xtb_rank"]))[0]
    return rec, rows


def _f(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_summary():
    bench = pd.read_csv(RESULTS_QM / "benchmark_results.csv")
    bench = bench[(bench["seed"] == 42) & (bench["pdb_id"].isin(["1gpk", "1b8o"]))].copy()

    rows = []
    for _, row in bench.iterrows():
        pdb_id = str(row["pdb_id"])
        seed = int(row["seed"])
        ligand_csv = RESULTS_QM / "qm_candidates" / pdb_id / f"seed_{seed}" / "xtb_rescored.csv"
        complex_csv = RESULTS_QM_V2 / "qm_candidates" / pdb_id / f"seed_{seed}" / "xtb_rescored.csv"

        lig_rec, lig_rows = _load_recommended_row(ligand_csv)
        cmp_rec, cmp_rows = _load_recommended_row(complex_csv)

        selected = _f(row["best_rmsd"])
        oracle = _f(row["oracle_best_rmsd"])
        ligand_rmsd = _f(lig_rec["rmsd"]) if lig_rec else float("nan")
        complex_rmsd = _f(cmp_rec["rmsd"]) if cmp_rec else float("nan")

        rows.append({
            "pdb_id": pdb_id,
            "seed": seed,
            "selected_rmsd": selected,
            "oracle_best_rmsd": oracle,
            "ligand_only_rmsd": ligand_rmsd,
            "complex_rmsd": complex_rmsd,
            "ligand_only_delta_vs_selected": ligand_rmsd - selected,
            "complex_delta_vs_selected": complex_rmsd - selected,
            "ligand_only_ok_count": sum(int(r.get("xtb_ok", 0)) for r in lig_rows),
            "complex_ok_count": sum(int(r.get("xtb_ok", 0)) for r in cmp_rows),
            "ligand_only_recommended_idx": lig_rec.get("candidate_idx", "") if lig_rec else "",
            "complex_recommended_idx": cmp_rec.get("candidate_idx", "") if cmp_rec else "",
        })
    return pd.DataFrame(rows)


def build_plot(summary_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, len(summary_df), figsize=(10, 4), constrained_layout=True)
    if len(summary_df) == 1:
        axes = [axes]

    labels = ["Selected", "Oracle", "Ligand-xTB", "Complex-xTB"]
    colors = ["#4C78A8", "#54A24B", "#F58518", "#E45756"]

    for ax, (_, row) in zip(axes, summary_df.iterrows()):
        values = [
            row["selected_rmsd"],
            row["oracle_best_rmsd"],
            row["ligand_only_rmsd"],
            row["complex_rmsd"],
        ]
        ax.bar(labels, values, color=colors)
        ax.set_title(f"{row['pdb_id']} seed {int(row['seed'])}")
        ax.set_ylabel("RMSD (A)")
        ax.set_ylim(0, max(values) + 1.0)
        ax.tick_params(axis="x", rotation=20)

    out_path = OUT_DIR / "figures" / "qm_negative_result_rmsd.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_markdown(summary_df: pd.DataFrame):
    mean_selected = summary_df["selected_rmsd"].mean()
    mean_ligand = summary_df["ligand_only_rmsd"].mean()
    mean_complex = summary_df["complex_rmsd"].mean()
    mean_oracle = summary_df["oracle_best_rmsd"].mean()

    lines = [
        "# QM Negative Result Package",
        "",
        "## Method",
        "",
        "We evaluated QM-informed rescoring as a post-processing stage after the docking pipeline.",
        "The baseline used the docking system's `selection_score=energy` final-pose selection.",
        "We then tested two GFN2-xTB rescoring variants on the exported top-k candidates:",
        "",
        "1. ligand-only xTB rescoring (`results/qm_ready/...`)",
        "2. pocket-aware protein-cluster + ligand complex xTB rescoring (`results/qm_ready_v2/...`, `--mode complex`)",
        "",
        "All QM runs were executed on the local WSL Ubuntu CPU environment.",
        "",
        "## Result",
        "",
        f"- Mean selected RMSD: {mean_selected:.3f} A",
        f"- Mean oracle-best RMSD: {mean_oracle:.3f} A",
        f"- Mean ligand-only xTB recommended RMSD: {mean_ligand:.3f} A",
        f"- Mean complex xTB recommended RMSD: {mean_complex:.3f} A",
        "",
        "Neither QM variant improved the final pose ranking. In both tested targets, the xTB-recommended pose had higher RMSD than the baseline energy-selected pose.",
        "",
        "### Target-level evidence",
        "",
    ]

    for _, row in summary_df.iterrows():
        lines.extend([
            f"- `{row['pdb_id']}` seed {int(row['seed'])}: baseline {row['selected_rmsd']:.3f} A, "
            f"oracle {row['oracle_best_rmsd']:.3f} A, ligand-only xTB {row['ligand_only_rmsd']:.3f} A "
            f"(delta {row['ligand_only_delta_vs_selected']:+.3f} A), "
            f"complex xTB {row['complex_rmsd']:.3f} A (delta {row['complex_delta_vs_selected']:+.3f} A)."
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "This negative result indicates that the tested QM signal is not aligned with the pose-quality ranking objective.",
        "Ligand-only xTB mainly captures internal ligand strain, while pose prediction depends on protein-ligand interaction quality.",
        "The pocket-cluster approximation also failed to improve ranking, suggesting that this simplified xTB setup is still not the right scoring signal for the current docking task.",
    ])
    return "\n".join(lines) + "\n"


def build_next_steps():
    text = """# Mainline Refocus

The next three docking experiments worth running are:

1. Ranking ablation at larger scale
   - Compare `selection_score=energy` against `clash` on the strongest 5-10 Astex targets.
   - Goal: verify whether the small-subset `energy` win persists and whether `clash` helps Claim 3.

2. Search-vs-selection gap audit
   - Report `best_rmsd` vs `oracle_best_rmsd` on the medium targets where search already finds near-native poses.
   - Goal: identify targets where ranking is still the bottleneck and quantify the gap directly.

3. Hard-target search improvement run
   - Use targets like `1fpu` where oracle RMSD is still poor.
   - Goal: stop tuning ranking on targets that are search-limited and instead test broader search changes.
"""
    return text


def copy_evidence_files():
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        RESULTS_QM / "benchmark_results.csv": "benchmark_results_ligand_only.csv",
        RESULTS_QM / "qm_candidates" / "1gpk" / "seed_42" / "xtb_rescored.csv": "xtb_rescored_ligand_only_1gpk_seed42.csv",
        RESULTS_QM / "qm_candidates" / "1b8o" / "seed_42" / "xtb_rescored.csv": "xtb_rescored_ligand_only_1b8o_seed42.csv",
        RESULTS_QM_V2 / "benchmark_results.csv": "benchmark_results_complex_export.csv",
        RESULTS_QM_V2 / "qm_candidates" / "1gpk" / "seed_42" / "xtb_rescored.csv": "xtb_rescored_complex_1gpk_seed42.csv",
        RESULTS_QM_V2 / "qm_candidates" / "1b8o" / "seed_42" / "xtb_rescored.csv": "xtb_rescored_complex_1b8o_seed42.csv",
    }
    for src, dst_name in file_map.items():
        if src.exists():
            shutil.copy2(src, data_dir / dst_name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary()
    summary_df.to_csv(OUT_DIR / "qm_negative_result_summary.csv", index=False)
    plot_path = build_plot(summary_df)
    (OUT_DIR / "method_and_results.md").write_text(build_markdown(summary_df), encoding="utf-8")
    (OUT_DIR / "mainline_next_steps.md").write_text(build_next_steps(), encoding="utf-8")
    copy_evidence_files()

    readme_lines = [
        "# QM Negative Result Package",
        "",
        "Files:",
        "- `method_and_results.md`: application/paper-ready methods and results text",
        "- `qm_negative_result_summary.csv`: numeric evidence summary",
        "- `figures/qm_negative_result_rmsd.png`: selected vs oracle vs QM-rescored RMSD plot" if plot_path else "- `figures/`: plot generation skipped",
        "- `data/`: copied source CSV evidence",
        "- `mainline_next_steps.md`: recommended docking experiments after stopping the QM branch",
        "",
        "This package documents a negative result: both ligand-only xTB and pocket-cluster complex xTB failed to improve pose ranking over the baseline energy-selected docking result.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote package to {OUT_DIR}")


if __name__ == "__main__":
    main()
