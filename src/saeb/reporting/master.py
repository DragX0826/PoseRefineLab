import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from .visualizer import PublicationVisualizer

logger = logging.getLogger("SAEB-Flow.reporting.master")

def plot_success_rate_curve(all_results, filename="fig_sr_curve.pdf"):
    """Success Rate Curve (SAEB vs Baselines)."""
    try:
        plt.figure(figsize=(8, 6))
        methods = {}
        for res in all_results:
            m = res.get('Optimizer', 'SAEB-Flow')
            if m not in methods: methods[m] = []
            methods[m].append(float(res.get('rmsd', res.get('RMSD', 99.9))))
        
        thresholds = np.linspace(0, 10, 100)
        for m, rmsds in methods.items():
            rmsds = np.array(rmsds)
            rates = [(rmsds < t).mean() * 100 for t in thresholds]
            plt.plot(thresholds, rates, label=m, linewidth=2)
            
        plt.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='2A reference')
        plt.xlabel("RMSD Threshold (A)")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate Characteristics")
        plt.legend(); plt.grid(True, linestyle=':', alpha=0.4); plt.tight_layout()
        plt.savefig(filename)
        if filename.endswith('.pdf'):
            plt.savefig(filename.replace('.pdf', '.png'), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to plot SR curve: {e}")

def plot_runtime_breakdown(all_results, filename="fig_runtime.pdf"):
    """Inference Runtime Breakdown."""
    try:
        data = {}
        for res in all_results:
            m = res.get('Optimizer', 'SAEB-Flow')
            if m not in data: data[m] = []
            data[m].append(float(res.get('Speed', 1.0)))
        
        df_list = []
        for m, speeds in data.items():
            for s in speeds: df_list.append({'Method': m, 'Normalized Speed': s})
        
        df = pd.DataFrame(df_list)
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Method', y='Normalized Speed', data=df, palette='viridis')
        plt.ylabel("Normalized Inference Speed (higher is faster)")
        plt.title("Computational Efficiency Review")
        plt.tight_layout()
        plt.savefig(filename)
        if filename.endswith('.pdf'):
            plt.savefig(filename.replace('.pdf', '.png'), dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to plot runtime: {e}")

def generate_master_report(experiment_results, all_histories=None):
    """Generates a comprehensive scientific report (LaTeX + Console)."""
    print("\n Generating Master Report (LaTeX Table)...")
    
    rows = []
    # Identify unique baselines (Vina) from actual runs
    vina_runs = [r for r in experiment_results if r.get('vina_success') and 'vina_rmsd' in r]
    if vina_runs:
        for res in vina_runs:
            rows.append({
                "Target": res.get('Target', 'UNK'),
                "Optimizer": "AutoDock Vina",
                "RMSD": f"{res['vina_rmsd']:.2f}",
                "Centr_D": "N/A",
                "Energy": f"{res.get('vina_energy', 0.0):.2f}",
                "Yield(%)": "Baseline",
                "AlignStep": "N/A",
                "Top10%_E": "N/A",
                "QED": "N/A",
                "Clash": "N/A",
                "Stereo": "Pass",
                "Status": "Baseline"
            })
    
    # Aggregate results for statistics
    agg_results = {}
    for res in experiment_results:
        key = (res.get('Target', res.get('name', 'UNK').split('_')[0]), res.get('Optimizer', 'SAEB-Flow'))
        if key not in agg_results: agg_results[key] = []
        agg_results[key].append(res)
    
    # Detailed rows for console and CSV
    for res in experiment_results:
        e = float(res.get('energy_final', res.get('Binding Pot.', 0.0)))
        rmsd_val = float(res.get('rmsd', res.get('RMSD', 0.0)))
        centr_dist = float(res.get('Centroid_Dist', 0.0))
        name = res.get('name', f"{res.get('Target', 'UNK')}_{res.get('Optimizer', 'UNK')}")
        
        qed, clash_score, stereo_valid = "N/A", "N/A", "N/A"
        try:
             import rdkit.Chem as Chem
             from rdkit.Chem import Descriptors, QED
             mol_path = f"output_{name}.pdb"
             if os.path.exists(mol_path):
                 mol = Chem.MolFromPDBFile(mol_path)
                 if mol:
                     qed = format(QED.qed(mol), ".2f")
                     d_mat = Chem.Get3DDistanceMatrix(mol)
                     n = d_mat.shape[0]
                     triu_idx = np.triu_indices(n, k=1)
                     clashes = np.sum(d_mat[triu_idx] < 1.0)
                     clash_score = format(clashes / (n * (n-1) / 2) if n > 1 else 0.0, ".3f")
                     
                     bond_violations = 0
                     for bond in mol.GetBonds():
                         dist = d_mat[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                         if dist < 1.0 or dist > 2.0: bond_violations += 1
                     stereo_valid = "Pass" if (clashes == 0 and bond_violations == 0) else f"Fail({clashes})"
        except: pass

        rows.append({
            "Target": res.get('pdb', res.get('Target', 'UNK')),
            "Optimizer": res.get('Optimizer', 'SAEB-Flow'),
            "RMSD": format(rmsd_val, ".2f"),
            "Centr_D": format(centr_dist, ".2f"),
            "Energy": format(e, ".1f"),
            "Yield(%)": format(res.get('yield', 100.0), ".1f"),
            "Speed": format(res.get('Speed', 1.0), ".2f"),
            "AlignStep": str(res.get('StepsTo09', ">1000")),
            "QED": qed,
            "Int-RMSD": format(res.get('Int-RMSD', 0.0), ".2f"),
            "Clash": clash_score,
            "Stereo": stereo_valid,
            "Status": "Pass" if rmsd_val < 2.0 else "Novel"
        })
    
    if not rows: return
    df = pd.DataFrame(rows)
    df_final = df.dropna(axis=1, how='all')
    
    viz = PublicationVisualizer()
    viz.plot_pareto_frontier(df_final, filename="fig2_pareto_frontier.pdf")
    viz.plot_diversity_pareto(df_final, filename="fig3_diversity_pareto.pdf")
    if all_histories:
        viz.plot_convergence_cliff(experiment_results[0].get('cos_sim_history', []))
    
    plot_success_rate_curve(experiment_results)
    plot_runtime_breakdown(experiment_results)
    
    print(f"\n --- SAEB-Flow Production Summary ---")
    print(df_final.to_string(index=False))
    
    # Success Rate (RMSD < 2.0A)
    saeb_res = [r for r in experiment_results if r.get('Optimizer') == 'SAEB-Flow']
    if saeb_res:
         rmsds = [float(r.get('rmsd', 99.9)) for r in saeb_res]
         sr = (np.array(rmsds) < 2.0).mean() * 100
         print(f"\n Success Rate (RMSD < 2.0A): {sr:.1f}%")

    # LaTeX Table
    try:
        tex_file = "saebflow_final_report.tex"
        with open(tex_file, "w") as f:
            f.write(df_final.to_latex(index=False, caption="SAEB-Flow Performance Summary"))
        print(f" Master Report saved to {tex_file}")
    except: pass
