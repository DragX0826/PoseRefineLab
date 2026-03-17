from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("reports")
DOCKING_DIR = ROOT / "docking"
QM_DIR = ROOT / "quantum"
FINAL_RESULTS = Path("results/results")
QM_RESULTS = Path("quantum/qm_negative_result_package")


def ensure_dirs() -> None:
    DOCKING_DIR.mkdir(parents=True, exist_ok=True)
    QM_DIR.mkdir(parents=True, exist_ok=True)


def build_docking_package() -> None:
    main_csv = FINAL_RESULTS / "astex10_final_tables/main_table.csv"
    stability_csv = FINAL_RESULTS / "astex10_final_tables/stability_table.csv"
    gap_summary_csv = FINAL_RESULTS / "astex10_final_gap_audit/target_gap_summary.csv"
    gap_audit_csv = FINAL_RESULTS / "astex10_final_gap_audit/target_gap_audit.csv"
    seed_metrics_csv = FINAL_RESULTS / "astex10_final_tables/seed_metrics.csv"

    for src in [main_csv, stability_csv, gap_summary_csv, seed_metrics_csv]:
        shutil.copy2(src, DOCKING_DIR / src.name)

    main_df = pd.read_csv(main_csv)
    stab_df = pd.read_csv(stability_csv)
    gap_df = pd.read_csv(gap_audit_csv)
    seed_df = pd.read_csv(seed_metrics_csv)
    non_search_df = gap_df[gap_df["target_class"] != "search_limited"].copy()
    non_search_summary = (
        non_search_df.groupby("method", as_index=False)
        .agg(
            n_targets=("pdb_id", "count"),
            mean_selected_rmsd=("selected_rmsd", "mean"),
            mean_oracle_rmsd=("oracle_rmsd", "mean"),
            mean_selection_gap=("selection_gap", "mean"),
            mean_fallback_rate=("fallback_rate", "mean"),
        )
    )
    non_search_summary.to_csv(DOCKING_DIR / "non_search_limited_summary.csv", index=False)

    labels = ["FK-SMC + SOCM", "SOCM"]
    order = ["fksmc_socm_final", "socm_final"]
    colors = ["#1f77b4", "#ff7f0e"]

    # Improved stability figure: show seed-level median RMSD as a strip plot and
    # overlay per-method standard deviation so the entire figure stays in Å.
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=180)
    x = np.arange(len(order))
    std_values = np.sqrt(stab_df.set_index("method").loc[order, "median_rmsd_var"].values)
    ax.bar(
        x,
        std_values,
        color=colors,
        width=0.56,
        alpha=0.28,
        edgecolor="black",
        linewidth=0.8,
        label="Median RMSD std",
    )
    for i, method in enumerate(order):
        method_seed = seed_df[seed_df["method"] == method]["median_rmsd"].to_numpy()
        offsets = np.linspace(-0.08, 0.08, len(method_seed))
        ax.scatter(
            np.full_like(method_seed, x[i], dtype=float) + offsets,
            method_seed,
            color=colors[i],
            edgecolors="black",
            linewidths=0.6,
            s=48,
            zorder=3,
            label="Seed median RMSD" if i == 0 else None,
        )
        ax.text(
            x[i],
            std_values[i] + 0.02,
            f"std={std_values[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Seed median RMSD / std (Å)")
    ax.set_title("Stability Comparison on Astex-9")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(DOCKING_DIR / "stability_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Target-level gap figure with Å in axis label.
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    method_labels = {"fksmc_socm_final": "FK-SMC + SOCM", "socm_final": "SOCM"}
    class_colors = {
        "well_aligned": "#2ca02c",
        "ranking_limited": "#d62728",
        "search_limited": "#9467bd",
        "mixed": "#8c564b",
    }
    markers = {"fksmc_socm_final": "o", "socm_final": "s"}
    targets = sorted(gap_df["pdb_id"].unique())
    base_x = np.arange(len(targets))
    offsets = {"fksmc_socm_final": -0.16, "socm_final": 0.16}
    for method in order:
        sub = gap_df[gap_df["method"] == method].set_index("pdb_id").loc[targets].reset_index()
        xs = base_x + offsets[method]
        ax.scatter(
            xs,
            sub["selected_rmsd"],
            s=70,
            marker=markers[method],
            c=[class_colors.get(c, "#7f7f7f") for c in sub["target_class"]],
            edgecolors="black",
            linewidths=0.6,
            label=method_labels[method],
            zorder=3,
        )
        ax.scatter(xs, sub["oracle_rmsd"], s=52, marker="_", c="black", linewidths=1.4, zorder=4)
        for x0, y_sel, y_oracle in zip(xs, sub["selected_rmsd"], sub["oracle_rmsd"]):
            ax.plot([x0, x0], [y_oracle, y_sel], color="gray", alpha=0.55, lw=1.0, zorder=2)
    ax.set_xticks(base_x, targets)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Target-Level Selected vs Oracle RMSD")
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], marker=markers[m], color="w", markerfacecolor="#bbbbbb",
               markeredgecolor="black", markersize=8, linestyle="None", label=method_labels[m])
        for m in order
    ]
    class_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black",
               markersize=8, linestyle="None", label=cls.replace("_", " "))
        for cls, color in class_colors.items() if cls in set(gap_df["target_class"])
    ]
    oracle_handle = Line2D([0], [0], marker="_", color="black", markersize=12, linestyle="None", label="Oracle RMSD")
    legend1 = ax.legend(handles=method_handles + [oracle_handle], loc="upper left", frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=class_handles, loc="upper right", frameon=True, title="Target class")
    fig.tight_layout()
    fig.savefig(DOCKING_DIR / "gap_audit_targets.png", bbox_inches="tight")
    plt.close(fig)

    # Summary figure with Å in title/axis.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180)
    width = 0.35
    x = np.arange(2)
    axes[0].bar(x - width / 2, main_df["sr2"], width, color="#4c72b0", label="SR@2Å")
    axes[0].bar(x + width / 2, main_df["sr5"], width, color="#55a868", label="SR@5Å")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Aggregate Success Rates")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, main_df["median_rmsd"], width, color="#c44e52", label="Selected median RMSD")
    axes[1].bar(x + width / 2, main_df["oracle_median_rmsd"], width, color="#8172b2", label="Oracle median RMSD")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("RMSD (Å)")
    axes[1].set_title("Selected vs Oracle Quality")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOCKING_DIR / "summary_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # Per-seed figure to support the stability claim directly.
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=180)
    seed_order = [42, 43, 44]
    x = np.arange(len(seed_order))
    width = 0.34
    fk = seed_df[seed_df["method"] == "fksmc_socm_final"].set_index("seed").loc[seed_order].reset_index()
    socm = seed_df[seed_df["method"] == "socm_final"].set_index("seed").loc[seed_order].reset_index()
    ax.bar(x - width / 2, fk["median_rmsd"], width, color="#1f77b4", label="FK-SMC + SOCM")
    ax.bar(x + width / 2, socm["median_rmsd"], width, color="#ff7f0e", label="SOCM")
    for i, v in enumerate(fk["median_rmsd"]):
        ax.text(i - width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(socm["median_rmsd"]):
        ax.text(i + width / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"seed {s}" for s in seed_order])
    ax.set_ylabel("Median RMSD (Å)")
    ax.set_title("Per-Seed Median RMSD")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOCKING_DIR / "per_seed_median_rmsd.png", bbox_inches="tight")
    plt.close(fig)

    (DOCKING_DIR / "report.md").write_text(
        "# Docking Report\n\n"
        "## Scope\n"
        "Astex-9 final comparison between `fksmc_socm_final` and `socm_final` using the stable mainline with MMFF auto-disable protection.\n\n"
        "## Key result\n"
        "- `socm_final` remains better on aggregate accuracy (`SR@2`, median RMSD).\n"
        "- `fksmc_socm_final` is materially more stable across seeds.\n"
        "- Hard failures are now diagnosed as search-limited rather than scoring-limited.\n\n"
        "## Per-seed evidence\n"
        f"- FK-SMC+SOCM median RMSD by seed: {', '.join(f'seed {int(r.seed)}={r.median_rmsd:.3f} Å' for _, r in fk.iterrows())}.\n"
        f"- SOCM median RMSD by seed: {', '.join(f'seed {int(r.seed)}={r.median_rmsd:.3f} Å' for _, r in socm.iterrows())}.\n"
        "These per-seed values explain why SOCM has slightly better aggregate accuracy but much larger cross-seed spread.\n\n"
        "## Non-search-limited view\n"
        f"- After removing search-limited targets, FK-SMC+SOCM mean selected RMSD = {non_search_summary.loc[non_search_summary['method']=='fksmc_socm_final', 'mean_selected_rmsd'].iloc[0]:.3f} Å.\n"
        f"- After removing search-limited targets, SOCM mean selected RMSD = {non_search_summary.loc[non_search_summary['method']=='socm_final', 'mean_selected_rmsd'].iloc[0]:.3f} Å.\n"
        "This filtered view is useful because search-limited targets are effectively insensitive to reranking and can dilute method differences.\n\n"
        "## Included files\n"
        "- `stability_comparison.png`\n"
        "- `gap_audit_targets.png`\n"
        "- `summary_metrics.png`\n"
        "- `per_seed_median_rmsd.png`\n"
        "- `main_table.csv`\n"
        "- `stability_table.csv`\n"
        "- `seed_metrics.csv`\n"
        "- `target_gap_summary.csv`\n",
        encoding="utf-8",
    )


def build_qm_package() -> None:
    summary_csv = QM_RESULTS / "qm_negative_result_summary.csv"
    method_md = QM_RESULTS / "method_and_results.md"
    original_fig = QM_RESULTS / "figures/qm_negative_result_rmsd.png"

    shutil.copy2(summary_csv, QM_DIR / summary_csv.name)
    shutil.copy2(method_md, QM_DIR / method_md.name)
    shutil.copy2(original_fig, QM_DIR / original_fig.name)

    qm_df = pd.read_csv(summary_csv)
    labels = [f"{pdb}\nseed {seed}" for pdb, seed in zip(qm_df["pdb_id"], qm_df["seed"])]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)
    width = 0.24
    ax.bar(x - width, qm_df["selected_rmsd"], width, color="#4c72b0", label="Selected")
    ax.bar(x, qm_df["ligand_only_rmsd"], width, color="#dd8452", label="Ligand-only xTB")
    ax.bar(x + width, qm_df["complex_rmsd"], width, color="#55a868", label="Complex xTB")
    ax.scatter(x, qm_df["oracle_best_rmsd"], color="black", marker="_", s=220, linewidths=2, label="Oracle")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("QM Rescoring Does Not Improve Top-Pose RMSD")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(QM_DIR / "qm_rmsd_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=180)
    width = 0.32
    ax.axhline(0.0, color="black", lw=1)
    ax.bar(x - width / 2, qm_df["ligand_only_delta_vs_selected"], width, color="#dd8452", label="Ligand-only xTB delta")
    ax.bar(x + width / 2, qm_df["complex_delta_vs_selected"], width, color="#55a868", label="Complex xTB delta")
    for i, v in enumerate(qm_df["ligand_only_delta_vs_selected"]):
        ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(qm_df["complex_delta_vs_selected"]):
        ax.text(i + width / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSD change vs selected pose (Å)")
    ax.set_title("QM Recommendations Are Worse Than Existing Selected Poses")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(QM_DIR / "qm_delta_vs_selected.png", bbox_inches="tight")
    plt.close(fig)

    (QM_DIR / "report.md").write_text(
        "# Quantum Report\n\n"
        "## Scope\n"
        "QM-informed rescoring prototype using xTB on exported docking candidates. Both ligand-only and pocket-cluster complex modes were tested.\n\n"
        "## Study scale\n"
        "This is a pilot negative-result study, not a broad benchmark. It only covers 2 targets (`1gpk`, `1b8o`) and 1 seed each.\n\n"
        "## Key result\n"
        "The workflow runs end-to-end, but neither QM mode improved pose ranking on the tested cases.\n\n"
        "## Why ligand-only and complex look identical here\n"
        "In both tested cases, the final recommended candidate index was the same for ligand-only xTB and complex xTB.\n"
        "The `ok_count` values differ because complex mode successfully scored fewer candidates, but among the candidates that did finish, the same pose remained the lowest-scoring recommendation.\n"
        "So the identical RMSD values reflect the same final recommended index, not a failure to execute complex mode.\n\n"
        "## Why the ranking headroom is small\n"
        f"- `1gpk`: selected {qm_df.loc[qm_df['pdb_id']=='1gpk','selected_rmsd'].iloc[0]:.3f} Å vs oracle {qm_df.loc[qm_df['pdb_id']=='1gpk','oracle_best_rmsd'].iloc[0]:.3f} Å; only {qm_df.loc[qm_df['pdb_id']=='1gpk','selected_rmsd'].iloc[0] - qm_df.loc[qm_df['pdb_id']=='1gpk','oracle_best_rmsd'].iloc[0]:.3f} Å gap.\n"
        f"- `1b8o`: selected {qm_df.loc[qm_df['pdb_id']=='1b8o','selected_rmsd'].iloc[0]:.3f} Å vs oracle {qm_df.loc[qm_df['pdb_id']=='1b8o','oracle_best_rmsd'].iloc[0]:.3f} Å; only {qm_df.loc[qm_df['pdb_id']=='1b8o','selected_rmsd'].iloc[0] - qm_df.loc[qm_df['pdb_id']=='1b8o','oracle_best_rmsd'].iloc[0]:.3f} Å gap.\n"
        "This limited baseline-to-oracle gap already constrains how much any rescoring method could improve the final ranking on these two examples.\n\n"
        "## Included files\n"
        "- `qm_negative_result_rmsd.png`\n"
        "- `qm_rmsd_comparison.png`\n"
        "- `qm_delta_vs_selected.png`\n"
        "- `qm_negative_result_summary.csv`\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    build_docking_package()
    build_qm_package()
    (ROOT / "README.md").write_text(
        "# Reports\n\n"
        "- `docking/`: docking figures, core tables, and report.\n"
        "- `quantum/`: QM negative-result figures, summary table, and report.\n",
        encoding="utf-8",
    )
    print(f"Updated report packages under: {ROOT.resolve()}")


if __name__ == "__main__":
    main()
