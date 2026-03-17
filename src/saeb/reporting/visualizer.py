"""
PublicationVisualizer — figure generation helpers for SAEB-Flow.

Produces the 6 core figures required for molecular docking papers:
  Fig 1: Success Rate Curve   (SR@threshold vs threshold)
  Fig 2: RMSD CDF             (cumulative distribution)
  Fig 3: Energy Convergence   (mean ± std across seeds)
  Fig 4: Ablation Study       (bar chart of method variants)
  Fig 5: Flow Vector Field    (2D PCA projection of v_pred)
  Fig 6: Benchmark Summary    (per-target bar chart, color-coded)
  Fig 7: Flow-Force Alignment (CosSim trend)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Kaggle-safe: no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA


# ── Style constants ────────────────────────────────────────────────────────────
PALETTE  = ["#1565C0", "#C62828", "#2E7D32", "#F57F17", "#6A1B9A", "#006064"]
LINEWIDTH = 2.2
FONTSIZE  = {"title": 15, "label": 13, "tick": 11, "legend": 11}

def _apply_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         FONTSIZE["tick"],
        "axes.labelsize":    FONTSIZE["label"],
        "axes.labelweight":  "bold",
        "axes.titlesize":    FONTSIZE["title"],
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.fontsize":   FONTSIZE["legend"],
        "figure.dpi":        150,
    })
    sns.set_style("whitegrid", {"axes.edgecolor": ".3"})


# ── Utilities ──────────────────────────────────────────────────────────────────

class LogTrendInterpreter:
    @staticmethod
    def interpret_rmsd_trend(history):
        if not history: return "No RMSD data."
        start, end = history[0], history[-1]
        delta = start - end
        if delta > 1.0:   return f"[Trend] RMSD: Improved ({start:.2f}A -> {end:.2f}A)."
        if delta > 0.1:   return f"[Trend] RMSD: Converging ({start:.2f}A -> {end:.2f}A)."
        return f"[Trend] RMSD: Stagnant at {end:.2f}A."

    @staticmethod
    def interpret_energy_trend(history):
        if not history: return "No energy data."
        start, end = history[0], history[-1]
        pct = (start - end) / (abs(start) + 1e-8) * 100
        if end < start: return f"[Trend] Energy: -{pct:.1f}% ({start:.1f} -> {end:.1f} kcal/mol)."
        return "[Trend] Energy: No improvement detected."


# ── Main Visualizer ────────────────────────────────────────────────────────────

class PublicationVisualizer:
    """Generates benchmark figures for molecular docking experiments."""

    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.palette     = PALETTE
        self.interpreter = LogTrendInterpreter()
        _apply_style()

    def _save(self, fig, filename):
        """Save as PDF + PNG (300 dpi), then close."""
        path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        png_path = path.replace(".pdf", ".png") if path.endswith(".pdf") else path + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def _verify(self, path):
        size_kb = os.path.getsize(path) / 1024
        if size_kb < 5:
            print(f"  [Verify] WARNING {os.path.basename(path)} is only {size_kb:.1f}KB")
        return size_kb >= 5

    # ── Fig 1: Success Rate Curve ──────────────────────────────────────────────

    def plot_success_rate_curve(self, rmsd_dict, filename="fig1_success_rate.pdf"):
        """
        Success Rate vs RMSD threshold (0–10 A).
        
        Args:
            rmsd_dict: dict {method_name: list_of_rmsds}  OR  list of rmsds (SAEB-Flow only)
        """
        if isinstance(rmsd_dict, (list, np.ndarray)):
            rmsd_dict = {"SAEB-Flow": rmsd_dict}

        # Published baselines (literature values for Astex Diverse Set)
        BASELINES = {
            "AutoDock Vina": 0.53,   # SR@2A from Trott & Olson 2009
            "GNINA":         0.62,
            "DiffDock":      0.38,   # Corso et al. 2023 NeurIPS
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        thresholds = np.linspace(0, 10, 200)

        colors = self.palette + ["#455A64", "#78909C", "#90A4AE"]
        for i, (method, rmsds) in enumerate(rmsd_dict.items()):
            arr = np.array(rmsds)
            rates = [(arr < t).mean() * 100 for t in thresholds]
            ax.plot(thresholds, rates, label=method, color=colors[i % len(colors)],
                    linewidth=LINEWIDTH + 0.5, zorder=5)
            # Mark SR@2A
            sr2 = (arr < 2.0).mean() * 100
            ax.scatter([2.0], [sr2], color=colors[i % len(colors)], s=80, zorder=6)

        # Baseline horizontal lines at SR@2A
        bl_colors = ["#9E9E9E", "#BDBDBD", "#E0E0E0"]
        for j, (baseline, sr) in enumerate(BASELINES.items()):
            ax.axhline(sr * 100, linestyle="--", color=bl_colors[j], linewidth=1.2,
                       label=f"{baseline} (SR@2A = {sr*100:.0f}%)")

        ax.axvline(2.0, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="2A threshold")
        ax.set_xlabel("RMSD Threshold (A)")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate Curve — Astex Diverse Set")
        ax.set_xlim(0, 10); ax.set_ylim(0, 100)
        ax.legend(loc="lower right")
        self._save(fig, filename)

    # ── Fig 2: RMSD CDF ────────────────────────────────────────────────────────

    def plot_rmsd_cdf(self, rmsd_dict, filename="fig2_rmsd_cdf.pdf"):
        """
        Cumulative Distribution Function of RMSD values.
        
        Args:
            rmsd_dict: dict {method: rmsds}  OR  list (SAEB-Flow only)
        """
        if isinstance(rmsd_dict, (list, np.ndarray)):
            rmsd_dict = {"SAEB-Flow": rmsd_dict}

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (method, rmsds) in enumerate(rmsd_dict.items()):
            arr = np.sort(np.array(rmsds))
            cdf = np.arange(1, len(arr) + 1) / len(arr) * 100
            ax.plot(arr, cdf, label=method, color=self.palette[i % len(self.palette)],
                    linewidth=LINEWIDTH)

        ax.axvline(2.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="2A")
        ax.axvline(5.0, color="orange", linestyle="--", linewidth=1.0, alpha=0.7, label="5A")
        ax.set_xlabel("RMSD (A)")
        ax.set_ylabel("Cumulative Fraction (%)")
        ax.set_title("RMSD Cumulative Distribution")
        ax.set_xlim(0, 15); ax.set_ylim(0, 100)
        ax.legend()
        self._save(fig, filename)

    # ── Fig 3: Energy Convergence (mean ± std) ─────────────────────────────────

    def plot_convergence_dynamics(self, history_E, filename="conv_energy.pdf",
                                  history_E_all=None):
        """
        Energy convergence with optional mean±std band across multiple seeds/targets.
        
        Args:
            history_E: list of floats (single run)
            history_E_all: list of lists (multiple seeds) — for std band
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        steps = np.arange(len(history_E))

        if history_E_all and len(history_E_all) > 1:
            max_len = max(len(h) for h in history_E_all)
            mat = np.full((len(history_E_all), max_len), np.nan)
            for i, h in enumerate(history_E_all):
                mat[i, :len(h)] = h
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0)
            ax.fill_between(np.arange(max_len), mean - std, mean + std,
                            color=self.palette[0], alpha=0.15, label="±1 std")
            ax.plot(np.arange(max_len), mean, color=self.palette[0], linewidth=LINEWIDTH,
                    label="Mean energy")
        else:
            ax.plot(steps, history_E, color=self.palette[0], linewidth=LINEWIDTH,
                    label="Binding Energy")
            ax.fill_between(steps, history_E, color=self.palette[0], alpha=0.08)

        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title("Energy Convergence — SAEB-Flow")
        ax.legend()
        print(self.interpreter.interpret_energy_trend(history_E))
        self._save(fig, filename)

    # ── Fig 4: Ablation Study ──────────────────────────────────────────────────

    def plot_ablation(self, ablation_data, filename="fig4_ablation.pdf"):
        """
        Horizontal bar chart for ablation study.
        
        Args:
            ablation_data: dict {variant_name: {"sr2": float, "median_rmsd": float}}
        Example:
            {
              "Full SAEB-Flow":       {"sr2": 45.0, "median_rmsd": 2.1},
              "w/o CBSF":             {"sr2": 32.0, "median_rmsd": 3.4},
              "w/o Physics":          {"sr2": 28.0, "median_rmsd": 3.9},
              "w/o Recycling":        {"sr2": 38.0, "median_rmsd": 2.8},
            }
        """
        if not ablation_data:
            return
        _apply_style()
        names    = list(ablation_data.keys())
        sr2vals  = [ablation_data[n].get("sr2", 0) for n in names]
        rmsdvals = [ablation_data[n].get("median_rmsd", 0) for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.7)))
        y = np.arange(len(names))
        colors = [self.palette[0] if "Full" in n else self.palette[1] for n in names]

        ax1.barh(y, sr2vals, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=10)
        ax1.set_xlabel("Success Rate @ 2A (%)"); ax1.set_title("SR@2A Ablation")
        ax1.axvline(sr2vals[0], color="green", linestyle="--", linewidth=1, alpha=0.6)

        ax2.barh(y, rmsdvals, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_yticks(y); ax2.set_yticklabels([])
        ax2.set_xlabel("Median RMSD (A)"); ax2.set_title("Median RMSD Ablation")
        ax2.axvline(rmsdvals[0], color="green", linestyle="--", linewidth=1, alpha=0.6)

        # Legend
        full_patch = mpatches.Patch(color=self.palette[0], label="Full Model")
        ab_patch   = mpatches.Patch(color=self.palette[1], label="Ablated")
        fig.legend(handles=[full_patch, ab_patch], loc="upper right")
        self._save(fig, filename)

    # ── Fig 5: Flow Vector Field (2D projection) ───────────────────────────────

    def plot_vector_field_2d(self, pos_L, v_pred, p_center, filename="fig5_flow_vectors.pdf"):
        """2D PCA projection of atom positions + flow velocity vectors."""
        _apply_style()
        try:
            def _np(x):
                if x is None: return None
                if hasattr(x, "detach"): x = x.detach().cpu().numpy()
                return x[0] if x.ndim == 3 else x

            p = _np(pos_L)
            v = _np(v_pred)
            c = _np(p_center).reshape(1, 3) if p_center is not None else None
            if p is None or len(p) < 3: return

            pca = PCA(n_components=2)
            pos_2d  = pca.fit_transform(p)
            v_2d    = np.dot(v, pca.components_.T)
            v_mag   = np.linalg.norm(v, axis=-1)
            v_2d_n  = v_2d / (np.linalg.norm(v_2d, axis=-1, keepdims=True) + 1e-6)
            v_plot  = v_2d_n * np.clip(v_mag, 0.5, 5.0).reshape(-1, 1)
            var1, var2 = pca.explained_variance_ratio_ * 100

            fig, ax = plt.subplots(figsize=(8, 7))
            q = ax.quiver(pos_2d[:, 0], pos_2d[:, 1],
                          v_plot[:, 0], v_plot[:, 1], v_mag,
                          cmap="magma", scale=40, alpha=0.88, width=0.005)
            ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c="gray", s=15, alpha=0.3)
            if c is not None:
                c2d = pca.transform(c)
                ax.scatter(c2d[0, 0], c2d[0, 1], c="red", marker="+",
                           s=250, linewidths=3, label="Pocket Centre", zorder=9)
            plt.colorbar(q, ax=ax, label="Velocity Magnitude (A/step)")
            ax.set_xlabel(f"PC1 ({var1:.1f}%)")
            ax.set_ylabel(f"PC2 ({var2:.1f}%)")
            ax.set_title("Neural Flow Geodesic — 2D PCA Projection")
            ax.legend()
            self._save(fig, filename)
        except Exception as e:
            pass   # Non-critical figure

    # ── Fig 6: Benchmark Summary Bar ──────────────────────────────────────────

    def plot_benchmark_summary(self, results_list, filename="fig6_benchmark_summary.pdf"):
        """
        Per-target best RMSD bar chart, color-coded by quality tier.
        Green: RMSD < 2A  |  Orange: 2-5A  |  Red: > 5A
        """
        if not results_list: return
        _apply_style()

        pdb_ids = [r["pdb_id"].upper() for r in results_list]
        rmsds   = [r["best_rmsd"] for r in results_list]
        # Sort by RMSD ascending
        order = np.argsort(rmsds)
        pdb_ids = [pdb_ids[i] for i in order]
        rmsds   = [rmsds[i] for i in order]

        colors = [self.palette[2] if r < 2.0 else self.palette[3] if r < 5.0 else self.palette[1]
                  for r in rmsds]

        fig, ax = plt.subplots(figsize=(max(10, len(pdb_ids) * 0.35), 5))
        bars = ax.bar(range(len(pdb_ids)), rmsds, color=colors,
                      edgecolor="black", linewidth=0.4)
        ax.axhline(2.0, color="green",  linestyle="--", linewidth=1.2, alpha=0.8, label="2A")
        ax.axhline(5.0, color="orange", linestyle="--", linewidth=1.0, alpha=0.6, label="5A")
        ax.set_xticks(range(len(pdb_ids)))
        ax.set_xticklabels(pdb_ids, rotation=90, fontsize=7)
        ax.set_ylabel("Best RMSD (A)")
        ax.set_title(f"SAEB-Flow Benchmark — Astex Diverse Set ({len(pdb_ids)} targets)")

        # Annotate SR@2A
        sr2 = sum(1 for r in rmsds if r < 2.0) / max(len(rmsds), 1) * 100
        ax.text(0.98, 0.96, f"SR@2A = {sr2:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

        patches = [mpatches.Patch(color=self.palette[2], label="< 2A (success)"),
                   mpatches.Patch(color=self.palette[3], label="2–5A"),
                   mpatches.Patch(color=self.palette[1], label="> 5A")]
        ax.legend(handles=patches, loc="upper left")
        self._save(fig, filename)

    # ── Fig 3b: RMSD Convergence ───────────────────────────────────────────────

    def plot_rmsd_convergence(self, history_rmsd, filename="rmsd_convergence.pdf"):
        """Per-target RMSD convergence (Kabsch-aligned min-RMSD over steps)."""
        if not history_rmsd or len(history_rmsd) < 2: return
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.arange(len(history_rmsd))
        ax.plot(steps, history_rmsd, color=self.palette[1], linewidth=LINEWIDTH,
                label="Min-RMSD (ensemble)")
        ax.fill_between(steps, history_rmsd, color=self.palette[1], alpha=0.08)
        ax.axhline(2.0, color="green", linestyle="--", linewidth=1.2, alpha=0.7, label="2A")
        ax.axhline(5.0, color="orange", linestyle="--", linewidth=1.0, alpha=0.5, label="5A")
        ax.set_xlabel("Checkpoint (every ~10% steps)")
        ax.set_ylabel("RMSD (A)")
        ax.set_title("RMSD Convergence")
        ax.legend()
        print(self.interpreter.interpret_rmsd_trend(history_rmsd))
        self._save(fig, filename)

    # ── Fig 7: Flow-Force Alignment ──────────────────────────────────────────
    
    def plot_alignment_trends(self, history_cos_sim, filename="fig7_alignment.pdf"):
        """Plot cosine similarity between flow and physical forces."""
        if not history_cos_sim: return
        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.arange(len(history_cos_sim))
        ax.plot(steps, history_cos_sim, color=self.palette[4], linewidth=LINEWIDTH,
                label="CosSim(v_pred, f_phys)")
        ax.fill_between(steps, history_cos_sim, 0, color=self.palette[4], alpha=0.1)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Neural Flow vs. Physical Force Alignment")
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
        self._save(fig, filename)

    # ── Legacy / Extra ─────────────────────────────────────────────────────────

    def plot_pareto_frontier(self, df_results, filename="fig_pareto.pdf"):
        try:
            import pandas as pd
            _apply_style()
            fig, ax = plt.subplots(figsize=(8, 6))
            eng_col = "Energy" if "Energy" in df_results.columns else "best_rmsd"
            rmsd_col = "RMSD" if "RMSD" in df_results.columns else "best_rmsd"
            energies = pd.to_numeric(df_results[eng_col], errors="coerce").dropna()
            rmsds    = pd.to_numeric(df_results[rmsd_col], errors="coerce").dropna()
            ax.scatter(rmsds, energies, color=self.palette[0], s=60, alpha=0.6)
            ax.set_xlabel("RMSD (A)"); ax.set_ylabel("Energy (kcal/mol)")
            ax.set_title("Pareto Frontier")
            self._save(fig, filename)
        except Exception: pass

    def plot_diversity_pareto(self, data, filename="fig_diversity.pdf"):
        pass   # Preserved for legacy calls

    def plot_convergence_cliff(self, cos_sim_history, energy_history=None,
                               filename="fig_convergence_cliff.pdf"):
        pass   # Preserved for legacy calls
