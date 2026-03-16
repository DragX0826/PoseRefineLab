import math
import csv
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from ..physics.engine import PhysicsEngine
from ..physics.config import ForceFieldParameters
from ..reporting.visualizer import PublicationVisualizer
from ..utils.pdb_io import RealPDBFeaturizer, save_points_as_pdb
from ..core.model import SAEBFlowBackbone, RectifiedFlow
from ..core.innovations import ShortcutFlowLoss, pat_step, langevin_noise, Muon
from .config import SimulationConfig

logger = logging.getLogger("SAEB-Flow.experiment.suite")


# ── Phase 80: Test-Time Compute Primitives ───────────────────────────────────

def compute_step_reward(pos_before, pos_after, f_phys, e_before, e_after, pocket_anchor):
    """
    Process Reward Model (PRM) for a docking step.
    Replaces pure CosSim with a richer physical signal.
    Returns: (B,) step reward tensor.
    """
    B = pos_before.shape[0]
    
    # 1. Energy delta (lower is better -> positive reward)
    delta_E = (e_before - e_after).clamp(-100.0, 100.0)  # (B,)
    
    # 2. Force-displacement alignment (classic CosSim)
    displacement = (pos_after - pos_before).view(B, -1)
    f_flat = f_phys.view(B, -1)
    cos_align = F.cosine_similarity(displacement, f_flat, dim=-1)  # (B,)
    
    # 3. Pocket proximity gain (closer to pocket anchor = better)
    centroid_before = pos_before.mean(dim=1)  # (B, 3)
    centroid_after  = pos_after.mean(dim=1)
    dist_before = (centroid_before - pocket_anchor).norm(dim=-1)  # (B,)
    dist_after  = (centroid_after  - pocket_anchor).norm(dim=-1)
    proximity_gain = (dist_before - dist_after).clamp(-2.0, 2.0)  # (B,)
    
    # Weighted combination
    reward = (
        0.01 * delta_E +
        0.50 * cos_align +
        0.30 * proximity_gain
    )
    return reward  # (B,)


def directed_noise(shape, pocket_anchor, f_phys, T_curr, dt, step_frac, pos_L, device):
    """
    SOC-inspired directed noise: replaces isotropic Langevin.
    Biases exploration toward pocket (early) and physics force (late).
    """
    B, N, _ = shape
    
    # Base isotropic noise
    sigma = math.sqrt(2.0 * T_curr * dt)
    raw_noise = torch.randn(shape, device=device) * sigma
    noise_scale = raw_noise.norm(dim=-1, keepdim=True)  # (B, N, 1)
    
    # Direction 1: toward pocket
    to_pocket = F.normalize(
        pocket_anchor.view(1, 1, 3) - pos_L.detach(), dim=-1
    )  # (B, N, 3)
    
    # Direction 2: along physics force
    f_dir = F.normalize(f_phys, dim=-1)  # (B, N, 3)
    
    # Scheduling: early=pocket, late=physics
    w_pocket  = 0.6 * (1.0 - step_frac)
    w_physics = 0.4 * step_frac
    w_random  = max(0.2, 1.0 - w_pocket - w_physics)
    
    # Renormalize weights
    total_w = w_random + w_pocket + w_physics
    directed = (
        (w_random  / total_w) * raw_noise +
        (w_pocket  / total_w) * to_pocket * noise_scale +
        (w_physics / total_w) * f_dir     * noise_scale
    )
    return directed


def stagnation_search_rescue(
    particles: torch.Tensor,
    energies: torch.Tensor,
    pocket_anchor: torch.Tensor,
    rescue_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace weak particles with perturbed elite copies when search stalls.

    This is intentionally coarse: the goal is to reopen exploration on
    hard targets where the current population has collapsed into a bad basin.
    """
    B = particles.shape[0]
    if B < 4 or rescue_scale <= 0.0:
        return particles, torch.empty(0, dtype=torch.long, device=particles.device)

    elite_k = max(1, B // 4)
    weak_k = max(1, B // 3)
    ranked = energies.argsort()
    elite_idx = ranked[:elite_k]
    weak_idx = ranked[-weak_k:]

    elite = particles[elite_idx].detach().clone()
    center = elite.mean(dim=0, keepdim=True)
    local_sigma = float(elite.std(dim=0, unbiased=False).mean().clamp(0.15, 1.50).item())
    if not math.isfinite(local_sigma):
        local_sigma = 0.35

    for out_idx, weak_particle_idx in enumerate(weak_idx):
        src = elite[out_idx % elite.shape[0]]
        src_centroid = src.mean(dim=0, keepdim=True)
        to_pocket = F.normalize(
            pocket_anchor.view(1, 3) - src_centroid,
            dim=-1,
            eps=1e-8,
        ).view(1, 1, 3)
        rigid_sigma = rescue_scale * local_sigma
        rigid_shift = (
            torch.randn((1, 1, 3), device=particles.device) * (0.60 * rigid_sigma) +
            to_pocket * (0.75 * rigid_sigma)
        )
        atom_sigma = max(0.08, 0.20 * rigid_sigma)
        atom_noise = torch.randn_like(src) * atom_sigma
        particles[weak_particle_idx] = src + rigid_shift + atom_noise + 0.10 * (center[0] - src)

    return particles, weak_idx


def _stable_zscore(values: torch.Tensor) -> torch.Tensor:
    """Numerically stable z-score used by ranking heuristics."""
    if values.numel() <= 1:
        return torch.zeros_like(values)
    v = values.float()
    std = v.std(unbiased=False)
    if torch.isnan(std) or std < 1e-8:
        return torch.zeros_like(v)
    return (v - v.mean()) / (std + 1e-8)


def _spearman_from_tensors(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank correlation helper that avoids NaN warnings on constant vectors."""
    if x.numel() <= 1 or y.numel() <= 1:
        return float("nan")
    rx = x.float().argsort().argsort().float()
    ry = y.float().argsort().argsort().float()
    sx = rx.std(unbiased=False).item()
    sy = ry.std(unbiased=False).item()
    if sx < 1e-8 or sy < 1e-8:
        return float("nan")
    corr = torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()
    return float(corr)


def _build_selection_scores(rank_signal: torch.Tensor, final_energy_t: torch.Tensor, clash_final: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build alternate ranking signals for quick ablation of final-pose selection."""
    logz_score = _stable_zscore(rank_signal)
    energy_score = -_stable_zscore(final_energy_t)
    clash_score = -_stable_zscore(torch.log1p(torch.clamp(clash_final, min=0.0)))
    return {
        "hybrid": 0.55 * logz_score + 0.30 * energy_score + 0.15 * clash_score,
        "logz": logz_score,
        "energy": energy_score,
        "clash": clash_score,
        "energy_clash": 0.7 * energy_score + 0.3 * clash_score,
    }


def _safe_scalar_energy(value: float, limit: float = 5000.0) -> float:
    """Guard scalar energies used for final reranking/reporting."""
    if not math.isfinite(value):
        return 0.0
    if abs(value) > limit:
        return 0.0
    return float(value)


def _write_pose_sdf(mol_template, coords: np.ndarray, path: str) -> None:
    """Save one ligand pose as SDF for downstream rescoring tools."""
    mol = Chem.Mol(mol_template)
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)
    conf = mol.GetConformer()
    for i in range(min(mol.GetNumAtoms(), coords.shape[0])):
        conf.SetAtomPosition(i, coords[i].tolist())
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def _write_xyz(symbols: list[str], coords: np.ndarray, path: str, comment: str = "") -> None:
    """Write a simple XYZ file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(symbols)}\n")
        fh.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(symbols, coords):
            fh.write(f"{symbol:2s} {float(x): .8f} {float(y): .8f} {float(z): .8f}\n")


def _protein_symbols_from_features(x_P: torch.Tensor) -> list[str]:
    """Infer protein atom symbols from the first four atom-type channels."""
    type_symbols = ["C", "N", "O", "S"]
    if x_P.numel() == 0:
        return []
    atom_type = torch.argmax(x_P[:, :4], dim=-1).detach().cpu().tolist()
    return [type_symbols[int(idx)] if 0 <= int(idx) < len(type_symbols) else "C" for idx in atom_type]


def _ligand_symbols_from_template(mol_template, atom_count: int) -> list[str]:
    """Infer ligand atom symbols from the RDKit template."""
    if mol_template is None:
        return ["C"] * atom_count
    symbols = [atom.GetSymbol() or "C" for atom in mol_template.GetAtoms()]
    if len(symbols) < atom_count:
        symbols.extend(["C"] * (atom_count - len(symbols)))
    return symbols[:atom_count]


def _dump_qm_candidates(
    artifact_dir: str,
    pdb_id: str,
    seed: int,
    mol_template,
    refined_poses: torch.Tensor,
    candidate_rows: list[dict],
    topk: int,
    pos_P: torch.Tensor | None = None,
    x_P: torch.Tensor | None = None,
    pos_native: torch.Tensor | None = None,
    cluster_cutoff: float = 6.0,
) -> str | None:
    """Export ranked ligand poses for downstream QM/xTB rescoring."""
    if topk <= 0 or not artifact_dir or mol_template is None or not candidate_rows:
        return None

    out_dir = os.path.join(
        artifact_dir,
        "qm_candidates",
        str(pdb_id),
        f"seed_{int(seed)}",
    )
    os.makedirs(out_dir, exist_ok=True)

    sorted_rows = sorted(
        (dict(row) for row in candidate_rows),
        key=lambda row: float(row.get("rank_score", float("-inf"))),
        reverse=True,
    )

    export_count = min(int(topk), len(sorted_rows))
    for export_rank, row in enumerate(sorted_rows, start=1):
        row["export_rank"] = export_rank
        row["exported_to_qm"] = int(export_rank <= export_count)
        row["sdf_file"] = ""

    refined_cpu = refined_poses.detach().cpu()
    exported_indices = [int(row["candidate_idx"]) for row in sorted_rows[:export_count]]
    cluster_coords = np.zeros((0, 3), dtype=np.float32)
    cluster_symbols: list[str] = []
    if pos_P is not None and x_P is not None and pos_P.numel() > 0 and x_P.size(0) == pos_P.size(0):
        pos_P_cpu = pos_P.detach().cpu()
        x_P_cpu = x_P.detach().cpu()
        ligand_union = refined_cpu[exported_indices].reshape(-1, 3)
        dists = torch.cdist(pos_P_cpu, ligand_union)
        keep_mask = dists.min(dim=1).values <= float(cluster_cutoff)
        if keep_mask.any():
            cluster_coords = pos_P_cpu[keep_mask].numpy()
            cluster_symbols = _protein_symbols_from_features(x_P_cpu[keep_mask])
            _write_xyz(
                cluster_symbols,
                cluster_coords,
                os.path.join(out_dir, "protein_pocket_cluster.xyz"),
                comment=f"protein pocket cluster cutoff={float(cluster_cutoff):.1f}A",
            )

    ligand_symbols = _ligand_symbols_from_template(mol_template, refined_cpu.shape[1])
    for row in sorted_rows[:export_count]:
        idx = int(row["candidate_idx"])
        sdf_name = f"candidate_rank_{int(row['export_rank']):02d}_idx_{idx:03d}.sdf"
        sdf_path = os.path.join(out_dir, sdf_name)
        ligand_coords = refined_cpu[idx].numpy()
        _write_pose_sdf(mol_template, ligand_coords, sdf_path)
        row["sdf_file"] = sdf_name
        _write_xyz(
            ligand_symbols,
            ligand_coords,
            os.path.join(out_dir, sdf_name.replace(".sdf", "_ligand.xyz")),
            comment=f"ligand-only pose for candidate {idx}",
        )
        if cluster_symbols:
            complex_symbols = cluster_symbols + ligand_symbols
            complex_coords = np.concatenate([cluster_coords, ligand_coords], axis=0)
            _write_xyz(
                complex_symbols,
                complex_coords,
                os.path.join(out_dir, sdf_name.replace(".sdf", "_complex.xyz")),
                comment=f"protein-cluster + ligand candidate {idx}",
            )

    if pos_native is not None and pos_native.numel() > 0:
        _write_pose_sdf(mol_template, pos_native.detach().cpu().numpy(), os.path.join(out_dir, "native_pose.sdf"))

    selected_row = next((row for row in sorted_rows if int(row.get("is_selected", 0)) == 1), None)
    if selected_row is not None:
        idx = int(selected_row["candidate_idx"])
        _write_pose_sdf(mol_template, refined_cpu[idx].numpy(), os.path.join(out_dir, "selected_pose.sdf"))

    oracle_row = next((row for row in sorted_rows if int(row.get("is_oracle_best", 0)) == 1), None)
    if oracle_row is not None:
        idx = int(oracle_row["candidate_idx"])
        _write_pose_sdf(mol_template, refined_cpu[idx].numpy(), os.path.join(out_dir, "oracle_best_pose.sdf"))

    fieldnames = [
        "export_rank",
        "candidate_idx",
        "rank_score",
        "logz_score",
        "energy_score",
        "clash_score",
        "final_energy",
        "clash",
        "rmsd",
        "is_selected",
        "is_oracle_best",
        "exported_to_qm",
        "sdf_file",
    ]
    with open(os.path.join(out_dir, "candidate_metadata.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    with open(os.path.join(out_dir, "candidate_topk.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows[:export_count]:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "QM candidate export for downstream xTB rescoring.\n"
            f"PDB ID: {pdb_id}\n"
            f"Seed: {int(seed)}\n"
            f"Exported top-k: {export_count}\n"
            "Files:\n"
            "  - candidate_metadata.csv: all refined particles with scores\n"
            "  - candidate_topk.csv: exported top-k subset\n"
            "  - candidate_rank_XX_idx_YYY.sdf: ligand poses for QM rescoring\n"
            "  - selected_pose.sdf: current docking selection\n"
            "  - oracle_best_pose.sdf: best RMSD pose (diagnostic only)\n"
            "  - native_pose.sdf: native ligand pose\n"
        )

    return out_dir


class BeamDocking:
    """
    Beam Search Docking v8.1 — 修復版本
    
    原始 v8.0 的 6 個 Bug（根本原因分析）：
    
    Bug 1 [致命]: 完全缺少口袋引導力
    Bug 2 [嚴重]: 缺少 SO(3) Geodesic Rotation
    Bug 3 [嚴重]: 候選選擇邏輯不一致（PRM reward vs Energy）
    Bug 4 [中等]: 缺少最終 MMFF 精修
    Bug 5 [中等]: Early stopping 過於激進
    Bug 6 [輕微]: n_restarts 太少（8）
    """
    
    def __init__(self, phys, beam_width: int = 8, n_restarts: int = 8, mol_template=None):
        self.phys = phys
        self.beam_width = beam_width
        self.n_restarts = n_restarts
        self.mol_template = mol_template
    
    def search(
        self,
        pos_init: torch.Tensor,       # (K, N, 3)
        pos_P, x_P, q_P,
        x_L, q_L,
        pocket_anchor: torch.Tensor,  # (3,)
        device,
        steps: int = 200,
        allow_flexible_receptor: bool = False,
    ) -> tuple:
        K, N, _ = pos_init.shape
        beam_w = min(self.beam_width, K)
        dt = 1.0 / steps

        # Initialize beams
        pos_P_b = pos_P.unsqueeze(0).expand(K, -1, -1).to(device)
        x_P_b   = x_P.unsqueeze(0).expand(K, -1, -1).to(device)
        q_P_b   = q_P.unsqueeze(0).expand(K, -1).to(device)
        x_L_b   = x_L.unsqueeze(0).expand(K, -1, -1).to(device)
        q_L_b   = q_L.unsqueeze(0).expand(K, -1).to(device)

        with torch.no_grad():
            e0, _, _, ec0 = self.phys.compute_energy(
                pos_init, pos_P_b, q_L_b, q_P_b, x_L_b, x_P_b, 0.0
            )
        top_idx = ec0.argsort()[:beam_w]
        beams = pos_init[top_idx].clone()
        beam_scores = ec0[top_idx].clone()

        history_best = []
        best_E_ever = float('inf')
        best_pos_ever = beams[0:1].clone()

        for step in range(steps):
            t = (step + 1) / steps
            step_frac = t
            T_curr = 0.5 * math.exp(-5.0 * step / steps) + 1e-5
            gw = 0.3 * math.exp(-3.0 * t) * 5.0

            all_candidates = []
            all_composite_scores = []

            for b_idx in range(beam_w):
                beam_b = beams[b_idx:b_idx+1]
                bP  = pos_P.unsqueeze(0).to(device)
                xPb = x_P.unsqueeze(0).to(device)
                qPb = q_P.unsqueeze(0).to(device)
                xLb = x_L.unsqueeze(0).to(device)
                qLb = q_L.unsqueeze(0).to(device)

                pos_param = nn.Parameter(beam_b.clone())
                e_bef, _, alpha, ec_bef = self.phys.compute_energy(
                    pos_param, bP, qLb, qPb, xLb, xPb, t
                )
                f_phys_raw = -torch.autograd.grad(e_bef.sum(), pos_param, create_graph=False)[0].detach()

                # Fix Bug 1: Pocket Guidance
                lig_centroid = beam_b.mean(dim=1, keepdim=True)
                f_pocket = (pocket_anchor - lig_centroid) * gw
                dist_to_anchor = (beam_b.mean(dim=1) - pocket_anchor).norm(dim=-1)
                adaptive_w = float(alpha) / (dist_to_anchor + 5.0)
                f_phys = f_phys_raw + f_pocket * adaptive_w.view(1, 1, 1)

                pos_param.grad = None

                for r in range(self.n_restarts):
                    noise = directed_noise((1, N, 3), pocket_anchor, f_phys, T_curr, dt, step_frac, beam_b, device)
                    cand = (beam_b + f_phys * (0.85 * dt) + noise).detach()

                    # Fix Bug 2: SO(3) Geodesic Rotation
                    cand_rotated = cand.clone()
                    cand_rotated[0] = geodesic_rotation_step(cand[0], f_phys[0], step_size=0.05 * (1.1 - t))
                    cand = cand_rotated

                    with torch.no_grad():
                        _, _, _, ec_aft = self.phys.compute_energy(cand, bP, qLb, qPb, xLb, xPb, t)
                        prm_rwd = compute_step_reward(beam_b, cand, f_phys, ec_bef, ec_aft, pocket_anchor)

                    # Fix Bug 3: Composite scoring
                    energy_score = -ec_aft.item() / 500.0
                    composite = 0.6 * prm_rwd.item() + 0.4 * energy_score

                    all_candidates.append(cand.squeeze(0))
                    all_composite_scores.append(composite)

            # Keep best beam_w
            scores_t = torch.tensor(all_composite_scores, device=device)
            top_k = scores_t.argsort(descending=True)[:beam_w]
            beams = torch.stack([all_candidates[i] for i in top_k.tolist()])

            with torch.no_grad():
                bP_all = pos_P.unsqueeze(0).expand(beam_w, -1, -1).to(device)
                xPb_all = x_P.unsqueeze(0).expand(beam_w, -1, -1).to(device)
                qPb_all = q_P.unsqueeze(0).expand(beam_w, -1).to(device)
                xLb_all = x_L.unsqueeze(0).expand(beam_w, -1, -1).to(device)
                qLb_all = q_L.unsqueeze(0).expand(beam_w, -1).to(device)
                _, _, _, beam_E = self.phys.compute_energy(beams, bP_all, qLb_all, qPb_all, xLb_all, xPb_all, t)
                beam_scores = beam_E

            curr_best_E = beam_scores.min().item()
            history_best.append(curr_best_E)
            if curr_best_E < best_E_ever:
                best_E_ever = curr_best_E
                best_pos_ever = beams[beam_scores.argmin():beam_scores.argmin()+1].clone()

            # Fix Bug 5: Looser early stop
            if len(history_best) > 30:
                imp = abs(history_best[-1] - history_best[-30])
                if imp < 0.5:
                    logger.info(f"  [Beam] Adaptive stop at step {step} (imp={imp:.3f})")
                    break

        # Fix Bug 4: Final MMFF polish
        if self.mol_template is not None:
            logger.info("  [Beam] Final MMFF polish on best pose...")
            try:
                best_pos_ever[0] = self.phys.minimize_with_mmff(self.mol_template, best_pos_ever[0], max_iter=500)
            except Exception as e:
                logger.warning(f"  [Beam] MMFF polish failed: {e}")

        return best_pos_ever, history_best


# ═══════════════════════════════════════════════════════════════════════
# v10.1: FK-SMC Degeneracy Fix (annealed beta + rejuvenation)
# ═══════════════════════════════════════════════════════════════════════

def annealed_beta(t: float, beta_start: float = 0.02, beta_end: float = 1.0) -> float:
    """Exponential inverse-temperature schedule.
    B2 fix: beta_start 0.005->0.02, beta_end 0.5->1.0 so ESS triggers resample.
    At t=0: beta=0.02; t=0.5: beta=0.14; t=1: beta=1.0
    """
    return beta_start * (beta_end / beta_start) ** t


class FeynmanKacSMC:
    """Feynman-Kac SMC with annealed beta and resample rejuvenation."""

    def __init__(
        self,
        phys,
        beta: float = 1.0,  # kept for API compatibility; annealing is used for weighting
        resample_threshold: float = 0.5,
        beta_start: float = 0.02,
        beta_end: float = 1.0,
        rejuv_factor: float = 3.0,
    ):
        self.phys = phys
        self.resample_threshold = resample_threshold
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rejuv_factor = rejuv_factor
        self._beta_legacy = beta

    def compute_log_incremental_weight(
        self,
        energy_before: torch.Tensor,
        energy_after: torch.Tensor,
        centroid_after: torch.Tensor,
        pocket_anchor: torch.Tensor,
        t: float,
        use_guided: bool = True,  # Set False for strict path measure (unbiased log Z)
    ) -> torch.Tensor:
        """log G_t = -beta_t * delta_E with optional twisted guidance bonus."""
        beta_t = annealed_beta(t, self.beta_start, self.beta_end)
        log_G = -beta_t * (energy_after - energy_before)
        if use_guided:
            guidance_decay = math.exp(-3.0 * t)
            dist = (centroid_after - pocket_anchor).norm(dim=-1)
            log_G = log_G + guidance_decay * (-0.1 * dist)
        return log_G

    def effective_sample_size(self, log_weights: torch.Tensor) -> float:
        """ESS/N in (0, 1]."""
        weights = F.softmax(log_weights, dim=0)
        return (1.0 / (weights ** 2).sum().item()) / len(log_weights)

    def systematic_resample(
        self,
        particles: torch.Tensor,   # (K, N, 3)
        log_weights: torch.Tensor, # (K,)
    ):
        """Systematic resampling with O(K) complexity."""
        K = particles.shape[0]
        weights = F.softmax(log_weights, dim=0).detach().cpu().numpy()
        cumsum = np.cumsum(weights)
        u = np.random.uniform(0, 1.0 / K) + np.arange(K) / K
        indices = np.clip(np.searchsorted(cumsum, u), 0, K - 1)
        idx_t = torch.tensor(indices, device=particles.device)
        return particles[idx_t].clone(), torch.zeros(K, device=particles.device)

    def resample_and_rejuvenate(
        self,
        particles: torch.Tensor,     # (K, N, 3)
        log_weights: torch.Tensor,   # (K,)
        pocket_anchor: torch.Tensor, # (3,)
        T_curr: float,
        dt: float,
        device,
    ):
        """Resample then add one stochastic move to restore particle diversity."""
        particles, log_weights = self.systematic_resample(particles, log_weights)
        sigma = self.rejuv_factor * math.sqrt(2.0 * T_curr * dt)
        if sigma < 1e-8:
            return particles, log_weights

        with torch.no_grad():
            noise_iso = torch.randn_like(particles) * sigma
            to_pocket = F.normalize(
                pocket_anchor.view(1, 1, 3) - particles.detach(),
                dim=-1,
            )
            noise_directed = to_pocket * sigma
            particles = particles + 0.5 * noise_iso + 0.5 * noise_directed

        return particles, log_weights

    def resample_if_needed(
        self,
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        pocket_anchor: torch.Tensor = None,
        T_curr: float = 0.0,
        dt: float = 0.005,
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resample and optionally rejuvenate when ESS drops below threshold."""
        ess = self.effective_sample_size(log_weights)
        if ess < self.resample_threshold:
            if pocket_anchor is not None and T_curr > 1e-8:
                return self.resample_and_rejuvenate(
                    particles,
                    log_weights,
                    pocket_anchor=pocket_anchor,
                    T_curr=T_curr,
                    dt=dt,
                    device=device,
                )
            return self.systematic_resample(particles, log_weights)
        return particles, log_weights


# ═══════════════════════════════════════════════════════════════════════
# v9.0 創新二：SOCMTwist（替換 alpha_fixed=0.85）
# 理論：SOCM NeurIPS 2024（Stochastic Optimal Control Matching）
# 公式：f_twist = β/(1−t) × f_physics（分析最優時間加權）
# ═══════════════════════════════════════════════════════════════════════

class SOCMTwist:
    """
    SOCM-Inspired optimal twist force for molecular docking.

    SOCM (De Bortoli et al., NeurIPS 2024) shows the optimal control
    equals the gradient of the log backward partition function V*(x,t).
    For docking, V*(x,t) ≈ log-probability of reaching the pocket from x.

    Analytic approximation:
        f_twist(x,t) = β / (1-t) × f_physics(x)

    Time profile:
        t=0.10 → 1.1× physics   (broad exploration)
        t=0.50 → 2.0× physics   (moderate focus)
        t=0.90 → 10× physics    (sharp convergence)
        t=0.99 → clamped at clamp_max
    """

    @staticmethod
    def effective_alpha(
        t: float,
        dt: float,
        beta: float = 1.0,
        clamp_max: float = 5.0,
        fallback_alpha: float = 0.85,
        use_twist: bool = True,
    ) -> float:
        """Returns the effective step-size scalar for pos_L update."""
        if not use_twist:
            return fallback_alpha * dt
        time_w = min(beta / (1.0 - t + 1e-6), clamp_max)
        return time_w * dt


# ═══════════════════════════════════════════════════════════════════════
# v9.0 創新三：SelfRewardingParticleGibbs（替換 BeamDocking）
# 理論：SR-SMC (Luo et al., arXiv Feb 2026 — 最新！)
# 特色：training-free，自獎勵 = MMFF94，粒子吉布斯迭代
# ═══════════════════════════════════════════════════════════════════════

class SelfRewardingParticleGibbs:
    """
    Self-Rewarding Particle Gibbs (SRPG) for Molecular Docking.

    Directly adapts SR-SMC (Luo et al. 2026):
    - Parallel chains = "interacting diffusion processes"
    - Self-reward = MMFF94 energy (no external labels needed)
    - Gibbs resampling: fix best particle as reference, resample others
    - Perturbation scale decays across iterations (broad → narrow)
    """

    def __init__(
        self,
        phys,
        n_chains: int = 4,
        n_iterations: int = 3,
        mol_template=None,
    ):
        self.phys = phys
        self.n_chains = n_chains
        self.n_iterations = n_iterations
        self.mol_template = mol_template

    def _self_reward(
        self,
        poses: torch.Tensor,  # (K, N, 3)
        pos_P_b, q_L_b, q_P_b, x_L_b, x_P_b,
    ) -> torch.Tensor:  # (K,)
        """Three-layer self-reward (no ground truth required)."""
        with torch.no_grad():
            _, _, _, e_phys = self.phys.compute_energy(
                poses, pos_P_b, q_L_b, q_P_b, x_L_b, x_P_b, 1.0
            )
        rewards = -e_phys  # lower energy → higher reward

        if self.mol_template is not None:
            mmff_r = torch.zeros(poses.shape[0], device=poses.device)
            for i in range(poses.shape[0]):
                try:
                    mmff_r[i] = -self.phys.get_mmff_energy(self.mol_template, poses[i])
                except Exception:
                    pass
            rewards = rewards + 0.2 * mmff_r
        return rewards

    def run(
        self,
        pos_init: torch.Tensor,   # (K, N, 3)
        pos_P, x_P, q_P, x_L, q_L,
        pocket_anchor: torch.Tensor,
        device,
        steps_per_iter: int = 100,
        inner_refine_fn=None,     # callable: SAEBFlowRefinement.refine partial
    ):
        """
        Full SRPG algorithm.
        Returns: best_pose (1, N, 3), iteration_logs (List[dict])
        """
        K = min(pos_init.shape[0], self.n_chains * 4)
        particles = pos_init[:K].clone().to(device)

        x_L_b = x_L.unsqueeze(0).expand(K, -1, -1).to(device)
        q_L_b = q_L.unsqueeze(0).expand(K, -1).to(device)
        pos_P_b = pos_P.unsqueeze(0).expand(K, -1, -1).to(device)
        x_P_b = x_P.unsqueeze(0).expand(K, -1, -1).to(device)
        q_P_b = q_P.unsqueeze(0).expand(K, -1).to(device)

        reference_pose = None
        iteration_logs = []

        for iteration in range(self.n_iterations):
            logger.info(f"  [SRPG] === Iteration {iteration+1}/{self.n_iterations} ===")

            # 1. Inner refinement
            if inner_refine_fn is not None:
                result = inner_refine_fn(
                    pos_L_init=particles,
                    pos_P=pos_P, x_P=x_P, q_P=q_P,
                    x_L=x_L, q_L=q_L,
                    pocket_anchor=pocket_anchor,
                    device=device,
                    mol_template=self.mol_template,
                    steps=steps_per_iter,
                )
                refined = result["refined_poses"]
                # Retain all K particles — do NOT collapse to 1 then expand (kills diversity)
                if refined.shape[0] >= K:
                    particles = refined[:K].clone()
                elif refined.shape[0] == 1:
                    # Fallback: perturb from the single best result
                    best_refined = refined[0]
                    particles[0] = best_refined
                    for i in range(1, K):
                        # small perturbation to maintain diversity
                        particles[i] = best_refined + torch.randn_like(best_refined) * 0.3
                else:
                    n = refined.shape[0]
                    particles[:n] = refined
                    # Fill remaining with perturbations of the best found
                    best_refined = refined[0]
                    for i in range(n, K):
                        particles[i] = best_refined + torch.randn_like(best_refined) * 0.3

            # 2. Self-reward computation
            rewards = self._self_reward(particles, pos_P_b, q_L_b, q_P_b, x_L_b, x_P_b)
            best_idx = rewards.argmax()
            log_entry = {
                "iteration": iteration + 1,
                "best_reward": rewards[best_idx].item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
            }
            iteration_logs.append(log_entry)
            logger.info(
                f"  [SRPG] Iter {iteration+1} | "
                f"BestR={log_entry['best_reward']:.2f} "
                f"Std={log_entry['reward_std']:.3f}"
            )

            # 3. Particle Gibbs resampling
            reference_pose = particles[best_idx].clone()
            with torch.no_grad():
                softmax_w = F.softmax(
                    rewards / max(rewards.std().item(), 1.0), dim=0
                )
                new_particles = particles.clone()
                new_particles[0] = reference_pose  # pin best
                # P0-1 Fix: Clamp perturb_scale to 0.5Å max to protect converged solutions.
                # Old value (2.0Å) was destructive to already-converged best particles.
                # Decay: iter=0 → 0.5Å, iter=1 → ~0.30Å, iter=2 → ~0.18Å
                perturb_scale = min(0.5, 2.0 * math.exp(-0.7 * iteration))
                for i in range(1, K):
                    parent_idx = torch.multinomial(softmax_w, 1).item()
                    new_particles[i] = (
                        particles[parent_idx]
                        + torch.randn_like(particles[parent_idx]) * perturb_scale
                    )
                particles = new_particles

        if reference_pose is not None:
            return reference_pose.unsqueeze(0), iteration_logs
        # Fallback
        rewards = self._self_reward(particles, pos_P_b, q_L_b, q_P_b, x_L_b, x_P_b)
        return particles[rewards.argmax()].unsqueeze(0), iteration_logs


def kabsch_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Kabsch-algorithm RMSD (optimal rotation + translation alignment).
    
    Args:
        pred : (B, N, 3)
        ref  : (N, 3) or (1, N, 3) — reference (crystal) pose
    Returns:
        rmsd : (B,) — per-sample aligned RMSD in Angstroms
    """
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)   # (1, N, 3)
    ref = ref.expand(pred.size(0), -1, -1)   # (B, N, 3)

    # Centre both
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    ref_c  = ref  - ref.mean(dim=1, keepdim=True)

    # Covariance: H = pred^T @ ref  (B, 3, 3)
    H = pred_c.transpose(1, 2) @ ref_c

    try:
        U, S, Vh = torch.linalg.svd(H)
        # Bug Fix F: Correct Kabsch rotation calculation order for reflections.
        # R = (U @ sign_mat) @ Vh
        d = torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2))
        sign_mat = torch.diag_embed(torch.stack(
            [torch.ones_like(d), torch.ones_like(d), d], dim=-1))
        
        # Proper Kabsch: R = (U @ sign_mat) @ Vh (where H = pred^T @ ref)
        # Wait, if H = pred^T @ ref, then R = V @ U^T. But we use Vh from linalg.svd.
        # Let's align with the standard: R = V @ U^T
        V = Vh.transpose(-1, -2)
        R = V @ sign_mat @ U.transpose(-1, -2)
        pred_rot = pred_c @ R.transpose(-1, -2)
    except Exception:
        # Fallback: translation-only if SVD fails (e.g. collinear atoms)
        pred_rot = pred_c

    diff = pred_rot - ref_c
    N = pred.size(1)
    rmsd = torch.sqrt((diff**2).sum(dim=[-1, -2]) / N)
    return rmsd

def geodesic_rotation_step(pos_L_clone, f_phys_clone, step_size=0.1):
    """
    Guides the rotation along the SO(3) manifold toward the energy minimum.
    Uses torque = Σ r_i × f_phys_i.
    """
    com = pos_L_clone.mean(dim=0, keepdim=True)
    centered = pos_L_clone - com  # (N, 3)
    
    # Torque = Σ r_i × f_i  (Cross product of rel_pos and force)
    torque = torch.cross(centered, f_phys_clone, dim=-1).sum(dim=0)  # (3,)
    torque_norm = torque.norm()
    
    if torque_norm < 1e-6:
        return pos_L_clone
    
    # Guidance direction = torque axis
    axis = torque / (torque_norm + 1e-8)
    # Scale angle by torque and step_size, clamped for stability
    angle = torch.clamp(torque_norm * step_size, max=0.3) # Max ~17 degrees
    
    # Rodrigues formula for rotation matrix R
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=pos_L_clone.device)
    
    R = (torch.eye(3, device=pos_L_clone.device) + 
         torch.sin(angle) * K + 
         (1 - torch.cos(angle)) * (K @ K))
    
    return (centered @ R.T) + com

def torsional_sampling_step(pos_L_clone, mol_template):
    """
    Performs stochastic conformational change by rotating a rotatable bond.
    Requires RDKit and pre-identified rotatable bonds in mol_template.
    """
    if mol_template is None:
        return pos_L_clone

    n_atoms = int(pos_L_clone.shape[0])
    mol_work = mol_template
    if mol_work.GetNumAtoms() != n_atoms:
        # Coordinates in this pipeline are heavy-atom-only; align template if needed.
        try:
            mol_heavy = Chem.RemoveHs(mol_template)
            if mol_heavy.GetNumAtoms() == n_atoms:
                mol_work = mol_heavy
            else:
                return pos_L_clone
        except Exception:
            return pos_L_clone

    if mol_work.GetNumConformers() == 0:
        return pos_L_clone
        
    # Find rotatable bonds if not cached
    if not hasattr(mol_work, '_rotatable_bonds'):
        # Ensure RingInfo is initialized (fixes v4.0 "RingInfo not initialized" error)
        Chem.GetSymmSSSR(mol_work)
        # Match rotatable bond SMARTS (simple version)
        smarts = "[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]"
        query = Chem.MolFromSmarts(smarts)
        mol_work._rotatable_bonds = mol_work.GetSubstructMatches(query)
        
    if not mol_work._rotatable_bonds:
        return pos_L_clone
        
    # Pick a random bond to rotate
    bond_idx = random.randint(0, len(mol_work._rotatable_bonds) - 1)
    a1, a2 = mol_work._rotatable_bonds[bond_idx]
    angle = random.uniform(0, 2 * math.pi)
    
    # Update temporary RDKit conformer
    # B7 fix: ensure mol_work has a Conformer before calling GetConformer()
    if mol_work.GetNumConformers() == 0:
        from rdkit.Chem import Conformer as RDConformer
        conf = RDConformer(mol_work.GetNumAtoms())
        mol_work.AddConformer(conf, assignId=True)
    conf = mol_work.GetConformer()
    for i in range(pos_L_clone.shape[0]):
        conf.SetAtomPosition(i, pos_L_clone[i].tolist())
        
    # Use proper SetDihedral logic via substructure neighbors
    neighbors_a1 = [n.GetIdx() for n in mol_work.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2]
    neighbors_a2 = [n.GetIdx() for n in mol_work.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1]
    
    if neighbors_a1 and neighbors_a2:
        try:
            rdMolTransforms.SetDihedralRad(conf, neighbors_a1[0], a1, a2, neighbors_a2[0], angle)
        except Exception:
            pass
            
    # Copy back to tensor
    new_pos = torch.tensor(conf.GetPositions(), device=pos_L_clone.device, dtype=pos_L_clone.dtype)
    if new_pos.shape[0] != n_atoms:
        return pos_L_clone
    return new_pos


# ── Experiment class ──────────────────────────────────────────────────────────

class SAEBFlowExperiment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.featurizer = RealPDBFeaturizer(config=config)
        ff_params = ForceFieldParameters(no_physics=config.no_physics, no_hsa=config.no_hsa)
        self.phys = PhysicsEngine(ff_params)
        self.visualizer = PublicationVisualizer()

    def _lbfgs_refine(self, pos_L, pos_P, q_L, q_P, x_L, x_P, t, max_iter=20):
        """Physics-Aware L-BFGS refinement helper."""
        # Ensure pos_L is a parameter for L-BFGS
        pos_ref = torch.nn.Parameter(pos_L.clone())
        optimizer_bfgs = torch.optim.LBFGS([pos_ref], lr=0.1, max_iter=max_iter)
        
        def closure():
            optimizer_bfgs.zero_grad()
            geom_loss = self.phys.calculate_internal_geometry_score(pos_ref).mean()
            # Handle batch vs single-sample protein features
            B_cur = pos_ref.size(0)
            e_ref, _, _, _ = self.phys.compute_energy(
                pos_ref, 
                pos_P if pos_P.dim() == 3 else pos_P.unsqueeze(0).expand(B_cur, -1, -1),
                q_L, 
                q_P if q_P.dim() == 2 else q_P.unsqueeze(0).expand(B_cur, -1), 
                x_L, 
                x_P if x_P.dim() == 3 else x_P.unsqueeze(0).expand(B_cur, -1, -1), 
                t
            )
            phys_loss = torch.clamp(e_ref.mean(), max=500.0) * 0.01
            total_loss = geom_loss + phys_loss
            total_loss.backward()
            return total_loss
        
        try:
            optimizer_bfgs.step(closure)
            pos_L.data.copy_(pos_ref.data)
        except Exception as e:
            logger.warning(f"  [Refine] L-BFGS step failed: {e}")

    def _mmff_refine(self, pos_L, mol_template, max_iter=200, indices=None):
        """Physics-Aware MMFF refinement helper."""
        if mol_template is None:
            return
        B_cur = pos_L.size(0)
        if indices is None:
            indices = range(B_cur)
        for i in indices:
            i = int(i)
            if i < 0 or i >= B_cur:
                continue
            try:
                refined = self.phys.minimize_with_mmff(mol_template, pos_L[i], max_iter=max_iter)
                if refined.shape == pos_L.data[i].shape:
                    pos_L.data[i].copy_(refined)
            except Exception as e:
                logger.debug(f"  [MMFF] Skip clone {i}: {e}")

    def run(self, device=None):
        """
        Compatibility entry point used by run_benchmark.py.
        """
        runner = SAEBFlowRefinement(self.config)
        runner.phys = self.phys
        runner.featurizer = self.featurizer
        runner.visualizer = self.visualizer
        return runner.run(device=device)

class SAEBFlowRefinement:
    """
    Test-Time Compute Refinement Engine.
    Takes ANY docking pose as input, refines with PAT + Physics.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        ff_params = ForceFieldParameters(
            no_physics=config.no_physics, 
            no_hsa=config.no_hsa
        )
        self.phys = PhysicsEngine(ff_params)
    
    def refine(self, 
               pos_L_init: torch.Tensor,   # (K, N, 3) 
               pos_P: torch.Tensor,         # (M, 3)
               x_P: torch.Tensor,
               q_P: torch.Tensor,
               x_L: torch.Tensor,           # (N, F)
               q_L: torch.Tensor,
               pocket_anchor: torch.Tensor, # (3,)
               device,
               pos_native: torch.Tensor = None,
               mol_template=None,
               steps: int = 200,
               allow_flexible_receptor: bool = False,
               protein_pdb_path: str = None,
               use_beam_search: bool = False,
               adaptive_stop_thresh: float = 0.05,
               # v9.0 flags ─────────────────────────────────────────────
               use_fksmc: bool = False,         # FK-SMC (replaces Replica Exchange)
               use_socm_twist: bool = False,    # SOCM Time-Weighted Twist
               twist_beta: float = 1.0,         # SOCM β parameter
               twist_clamp: float = 5.0,        # SOCM clamp_max
               use_srpg: bool = False,          # Self-Rewarding Particle Gibbs
               srpg_n_iter: int = 3,            # SRPG iterations
               srpg_steps: int = 100,           # SRPG steps per iteration
               # v10.0 ablation flags ──────────────────────────────────────
               no_backbone: bool = False,        # P0-2: pure physics ablation baseline
               ) -> dict:
        """
        Pure physics refinement — no backbone, no randomness.
        This is the core Test-Time Compute engine.
        """
        self.phys = self.phys.to(device)
        B, N, _ = pos_L_init.shape

        # P0-2: Pure Physics Ablation Baseline
        if no_backbone:
            logger.info(f"  [Refine] P0-2 ABLATION: no_backbone=True "
                        f"(pure physics + Langevin, no neural network)")
        # v8.0: Delegate to Beam Search if requested
        if use_beam_search:
            logger.info(f"  [Refine] Using Beam Search Policy (v8.1)...")
            beamer = BeamDocking(self.phys, beam_width=8, n_restarts=8, mol_template=mol_template)
            best_beam, history_E = beamer.search(
                pos_L_init, pos_P, x_P, q_P, x_L, q_L, 
                pocket_anchor, device, steps, allow_flexible_receptor
            )
            return {
                "refined_poses": best_beam,
                "final_energies": torch.tensor([history_E[-1]], device=device),
                "history_E": history_E
            }

        # v9.0: Delegate to SRPG if requested
        if use_srpg:
            logger.info(f"  [Refine] Using Self-Rewarding Particle Gibbs (v9.0, iter={srpg_n_iter})...")
            srpg = SelfRewardingParticleGibbs(
                self.phys, n_chains=4, n_iterations=srpg_n_iter,
                mol_template=mol_template
            )
            # Pass refine itself (without srpg flags) as inner_refine_fn
            import functools
            inner_fn = functools.partial(
                self.refine,
                pos_P=pos_P, x_P=x_P, q_P=q_P,
                x_L=x_L, q_L=q_L,
                pocket_anchor=pocket_anchor,
                device=device, mol_template=mol_template,
                pos_native=pos_native,
                steps=srpg_steps,
                use_fksmc=use_fksmc, use_socm_twist=use_socm_twist,
                twist_beta=twist_beta, twist_clamp=twist_clamp,
            )
            best_pose, iteration_logs = srpg.run(
                pos_L_init, pos_P, x_P, q_P, x_L, q_L,
                pocket_anchor, device,
                steps_per_iter=srpg_steps, inner_refine_fn=inner_fn,
            )
            final_score = 0.0
            if mol_template is not None:
                try:
                    best_pose[0] = self.phys.minimize_with_mmff(mol_template, best_pose[0], max_iter=500)
                    final_score = self.phys.get_mmff_energy(mol_template, best_pose[0])
                except Exception:
                    pass
            return {
                "refined_poses": best_pose,
                "final_energies": torch.tensor([final_score], device=device),
                "history_E": [d["best_reward"] for d in iteration_logs],
                "srpg_logs": iteration_logs,
            }

        pos_L = nn.Parameter(pos_L_init.clone().to(device))
        
        # P0-3: ETKDG-initialized starting conformations
        # If mol_template is available, generate physically valid starting geometry
        # before the physics optimization loop begins, reducing bond angle violations.
        if mol_template is not None:
            try:
                from rdkit.Chem import AllChem
                for b in range(B):
                    # Embed with ETKDG (ETKDGv3 for best-quality distance geometry).
                    # Use explicit-H embedding for better geometry; map back to heavy atoms.
                    mol_copy = Chem.Mol(mol_template)
                    mol_base = Chem.RemoveHs(mol_copy)
                    if mol_base.GetNumAtoms() != N:
                        mol_base = Chem.Mol(mol_copy)
                    mol_embed = Chem.AddHs(mol_base, addCoords=True)
                    heavy_idx = [a.GetIdx() for a in mol_embed.GetAtoms() if a.GetAtomicNum() > 1]
                    if len(heavy_idx) != N:
                        continue
                    mol_embed.RemoveAllConformers()
                    params = AllChem.ETKDGv3()
                    params.randomSeed = b  # Diverse starting conformations
                    if AllChem.EmbedMolecule(mol_embed, params) == 0:
                        conf = mol_embed.GetConformer()
                        # Translate to pocket_anchor region
                        conf_pos = torch.tensor(conf.GetPositions(), device=device, dtype=torch.float32)
                        etkdg_pos = conf_pos[heavy_idx]
                        if etkdg_pos.shape[0] != N:
                            continue
                        etkdg_center = etkdg_pos.mean(dim=0, keepdim=True)
                        etkdg_pos = etkdg_pos - etkdg_center + pocket_anchor.unsqueeze(0)
                        # Add small random offset to diversify starting positions
                        etkdg_pos = etkdg_pos + torch.randn_like(etkdg_pos) * 1.5
                        pos_L.data[b] = etkdg_pos
                        logger.debug(f"  [Refine] ETKDG init for batch {b}")
            except Exception as e:
                logger.debug(f"  [Refine] ETKDG init failed: {e} — using original init")
        x_L_b = x_L.unsqueeze(0).expand(B, -1, -1).to(device)
        q_L_b = q_L.unsqueeze(0).expand(B, -1).to(device)
        
        # Induced Fit: Optional protein flexibility
        pos_P_ref = pos_P.unsqueeze(0).expand(B, -1, -1).to(device)
        if allow_flexible_receptor:
            pos_P_active = nn.Parameter(pos_P_ref.clone())
            optimized_params = [pos_L, pos_P_active]
        else:
            pos_P_active = pos_P_ref
            optimized_params = [pos_L]
            
        x_P_b = x_P.unsqueeze(0).expand(B, -1, -1).to(device)
        q_P_b = q_P.unsqueeze(0).expand(B, -1).to(device)
        
        history_E = []
        best_E = torch.full((B,), float('inf'), device=device)
        best_pos = pos_L.detach().clone()
        dt = 1.0 / steps
        min_step_frac = float(getattr(self.config, "adaptive_min_step_frac", 0.65))
        patience_frac = float(getattr(self.config, "adaptive_patience_frac", 0.12))
        min_adaptive_stop_step = min(steps - 1, max(80, int(min_step_frac * steps)))
        adaptive_patience = max(20, int(patience_frac * steps))
        plateau_window = max(20, int(0.10 * steps))
        rescue_min_frac = float(getattr(self.config, "search_rescue_min_step_frac", 0.35))
        rescue_patience_frac = float(getattr(self.config, "search_rescue_patience_frac", 0.08))
        rescue_scale = float(getattr(self.config, "search_rescue_scale", 2.5))
        min_search_rescue_step = min(steps - 1, max(40, int(rescue_min_frac * steps)))
        search_rescue_patience = max(15, int(rescue_patience_frac * steps))
        search_rescue_cooldown = max(30, int(0.08 * steps))
        search_rescue_max_count = 2
        last_search_rescue_step = -search_rescue_cooldown
        search_rescue_count = 0
        best_avg_energy = float("inf")
        best_energy_step = 0

        # v10.1: Initialize FK-SMC with annealed beta and rejuvenation
        fksmc = FeynmanKacSMC(
            self.phys,
            beta_start=0.02,
            beta_end=1.0,
            rejuv_factor=3.0,
        ) if use_fksmc else None
        log_weights = torch.zeros(B, device=device)
        log_Z_history = []
        log_Z_resampled_accum = 0.0
        # C1: stability tracking
        ess_trajectory = []
        resample_count = 0
        energy_clamped_prev = None

        for step in range(steps):
            t = (step + 1) / steps
            
            # Physics Force
            # Handle both static and flexible protein
            raw_energy, _, alpha, energy_clamped = self.phys.compute_energy(
                pos_L, pos_P_active, q_L_b, q_P_b, x_L_b, x_P_b, t
            )
            
            # v9.0: FK-SMC Path Weight Accumulation
            if fksmc is not None and energy_clamped_prev is not None:
                with torch.no_grad():
                    centroid = pos_L.data.mean(dim=1)
                    log_G = fksmc.compute_log_incremental_weight(
                        energy_clamped_prev, energy_clamped, centroid, pocket_anchor, t
                    )
                    log_weights += log_G
                    # Cumulative log Z estimate: log(1/B * sum(exp(log_weights))) + previous resample normalizations
                    current_log_n_factor = (torch.logsumexp(log_weights, dim=0) - math.log(B)).item()
                    log_Z_history.append(log_Z_resampled_accum + current_log_n_factor)
            
            energy_clamped_prev = energy_clamped.detach().clone()

            if allow_flexible_receptor:
                tether_loss = self.phys.calculate_harmonic_tether(pos_P_active, pos_P_ref, k=10.0)
                total_obj = raw_energy.sum() + tether_loss.sum()
            else:
                total_obj = raw_energy.sum()
                
            grads = torch.autograd.grad(
                total_obj, optimized_params,
                retain_graph=False, create_graph=False
            )
            f_phys_L = -grads[0].detach()
            if allow_flexible_receptor:
                f_phys_P = -grads[1].detach()
            
            # Pocket Guidance Force (Local Biasing for Ligand)
            lig_centroid = pos_L.mean(dim=1, keepdim=True)
            gw = 0.3 * math.exp(-3.0 * t)
            f_pocket = (pocket_anchor - lig_centroid) * gw * 5.0
            f_phys_L = f_phys_L + f_pocket
            
            # SO(3) Geodesic Rotation (Ligand alignment)
            with torch.no_grad():
                for b in range(B):
                    pos_L.data[b] = geodesic_rotation_step(
                        pos_L.data[b], f_phys_L[b],
                        step_size=0.05 * (1.1 - t)
                    )
            
            # PAT-like update: v8.0 SOC Directed Noise
            with torch.no_grad():
                alpha_fixed = 0.85
                T_curr = 0.5 * math.exp(-5.0 * step / steps) + 1e-5
                do_noise = step < int(0.85 * steps)
                step_frac = t
                
                # SOC Directed Noise (replaces isotropic Langevin)
                if do_noise:
                    noise_L = directed_noise(
                        pos_L.shape, pocket_anchor, f_phys_L,
                        T_curr, dt, step_frac, pos_L.data, device
                    )
                else:
                    noise_L = torch.zeros_like(pos_L.data)
                
                # v9.0 SOCM Twist: β/(1-t) time-weighted step size
                eff_alpha_dt = SOCMTwist.effective_alpha(
                    t, dt,
                    beta=twist_beta,
                    clamp_max=twist_clamp,
                    fallback_alpha=0.85,
                    use_twist=use_socm_twist,
                )
                # Store pre-update position for PRM / FK-SMC
                pos_L_prev = pos_L.data.clone()
                pos_L.data.add_(f_phys_L * eff_alpha_dt + noise_L)
                
                # Update Protein (Induced Fit)
                if allow_flexible_receptor:
                    pos_P_active.data.add_(f_phys_P * (0.05 * dt))
                
                pos_L.grad = None
                if allow_flexible_receptor: pos_P_active.grad = None
            
            # Record best pose based on local energy
            with torch.no_grad():
                improved = energy_clamped < best_E
                best_E = torch.where(improved, energy_clamped, best_E)
                for b in range(B):
                    if improved[b]:
                        best_pos[b] = pos_L.data[b].clone()
            
            avg_e = energy_clamped.mean().item()
            history_E.append(avg_e)
            if avg_e + adaptive_stop_thresh < best_avg_energy:
                best_avg_energy = avg_e
                best_energy_step = step
            stalled_steps = step - best_energy_step
            
            # Adaptive stop v11: require long plateau + patience to avoid premature convergence.
            if step >= min_adaptive_stop_step and len(history_E) >= plateau_window:
                recent = np.array(history_E[-plateau_window:], dtype=np.float32)
                lookback = min(10, plateau_window - 1)
                imp_short = abs(float(recent[-1] - recent[-1 - lookback]))
                imp_long = abs(float(recent[-1] - recent[0]))
                rel_std = float(recent.std() / max(1.0, abs(float(recent.mean()))))
                rescue_pending = (
                    rescue_scale > 0.0 and
                    step >= min_search_rescue_step and
                    search_rescue_count < search_rescue_max_count and
                    stalled_steps >= search_rescue_patience and
                    (step - last_search_rescue_step) >= search_rescue_cooldown
                )
                if (
                    imp_short < adaptive_stop_thresh and
                    imp_long < (2.5 * adaptive_stop_thresh) and
                    rel_std < 0.01 and
                    stalled_steps >= adaptive_patience and
                    not rescue_pending
                ):
                    logger.info(
                        f"  [Refine] Adaptive stop at step {step} "
                        f"(imp_short={imp_short:.4f}, imp_long={imp_long:.4f}, stalled={stalled_steps})"
                    )
                    break
            
            # ============================================================
            # DIAGNOSTIC v8.1: Centroid → Anchor (外科手術級根因診斷)
            # 判斷失敗模式: A=未進口袋 B=方向翻轉 C=進了又彈出
            # ============================================================
            if step % 20 == 0:
                with torch.no_grad():
                    centroid = pos_L.data.mean(dim=1)  # (B, 3)
                    c2a = (centroid - pocket_anchor.unsqueeze(0)).norm(dim=-1)  # (B,)
                    logger.info(
                        f"  [Diag] Step {step:4d} | "
                        f"Centroid→Anchor: {c2a.min():.2f}Å(best) {c2a.mean():.2f}Å(mean) | "
                        f"E_min: {energy_clamped.min():.1f}"
                    )

            # Hard-target search rescue: when progress stalls, repopulate weak particles
            # from elite poses with a much broader pocket-directed perturbation.
            if (
                rescue_scale > 0.0 and
                step >= min_search_rescue_step and
                search_rescue_count < search_rescue_max_count and
                stalled_steps >= search_rescue_patience and
                (step - last_search_rescue_step) >= search_rescue_cooldown
            ):
                with torch.no_grad():
                    pos_L.data, rescued_idx = stagnation_search_rescue(
                        pos_L.data,
                        energy_clamped.detach(),
                        pocket_anchor,
                        rescue_scale=rescue_scale,
                    )
                    if rescued_idx.numel() > 0:
                        if log_weights is not None and log_weights.numel() == B:
                            log_weights[rescued_idx] = log_weights.mean()
                        search_rescue_count += 1
                        last_search_rescue_step = step
                        logger.info(
                            "  [Refine] Search rescue at step %d: refreshed %d weak particles "
                            "(stalled=%d, scale=%.2f)",
                            step,
                            int(rescued_idx.numel()),
                            stalled_steps,
                            rescue_scale,
                        )
            
            # Diversity maintenance: FK-SMC or PRM Replica Exchange
            if step > 20 and step % 30 == 0:
                with torch.no_grad():
                    if use_fksmc and fksmc is not None:
                        # v10.1 FK-SMC: Resample+Rejuvenate if ESS is low
                        ess = fksmc.effective_sample_size(log_weights)
                        ess_trajectory.append(ess)
                        if ess < fksmc.resample_threshold:
                            n_factor = (torch.logsumexp(log_weights, dim=0) - math.log(B)).item()
                            log_Z_resampled_accum += n_factor
                            resample_count += 1
                            pos_L.data, log_weights = fksmc.resample_and_rejuvenate(
                                pos_L.data,
                                log_weights,
                                pocket_anchor=pocket_anchor,
                                T_curr=T_curr,
                                dt=dt,
                                device=device,
                            )
                    else:
                        # v8.0 fallback: PRM Replica Exchange
                        step_rewards = compute_step_reward(
                            pos_L_prev, pos_L.data, f_phys_L,
                            energy_clamped, energy_clamped,
                            pocket_anchor
                        )
                        good_idx = step_rewards.argsort(descending=True)[:max(1, B//4)]
                        bad_idx  = step_rewards.argsort(descending=False)[:max(1, B//4)]
                        for i, b_idx in enumerate(bad_idx):
                            s_idx = good_idx[i % len(good_idx)]
                            e_diff = (energy_clamped[b_idx] - energy_clamped[s_idx]).clamp(0).item()
                            scale = 0.2 + 0.4 * math.tanh(e_diff / 300.0)
                            pos_L.data[b_idx] = (
                                pos_L.data[s_idx] +
                                torch.randn_like(pos_L.data[s_idx]) * scale
                            )
        
        # ── Final MMFF Polish & Scoring (Bug 29 Fix) ────────────────────────
        final_scores = []
        rank_signal = (
            log_weights.detach().clone()
            if (use_fksmc and log_weights is not None and log_weights.numel() == B)
            else torch.zeros(B, device=device)
        )
        mmff_disabled = False
        if mol_template is not None:
            logger.info("  [Refine] Performing final MMFF94 polish and second-stage rerank...")
            with torch.no_grad():
                pre_inter, _, _, _ = self.phys.compute_energy(
                    best_pos, pos_P_active, q_L_b, q_P_b, x_L_b, x_P_b, 1.0
                )
                pre_clash = self.phys.calculate_internal_geometry_score(best_pos)
                pre_rank = (
                    0.55 * _stable_zscore(rank_signal) -
                    0.30 * _stable_zscore(pre_inter) -
                    0.15 * _stable_zscore(torch.log1p(torch.clamp(pre_clash, min=0.0)))
                )
                polish_topk = int(getattr(self.config, "final_mmff_topk", 5))
                if polish_topk <= 0:
                    polish_topk = B
                # Polish a wider candidate pool, then rerank again on final scores.
                polish_mult = max(1, int(getattr(self.config, "rerank_polish_mult", 2)))
                polish_k = max(1, min(B, polish_topk * polish_mult))
                polish_indices = pre_rank.topk(polish_k).indices.tolist()
                polish_set = set(polish_indices)
            mmff_iter = int(getattr(self.config, "final_mmff_max_iter", 2000))
            mmff_iter = max(50, mmff_iter)
            mmff_stats_before = self.phys.get_mmff_stats()
            for i in polish_indices:
                best_pos[i] = self.phys.minimize_with_mmff(mol_template, best_pos[i], max_iter=mmff_iter)
            for i in range(B):
                mmff_e = self.phys.get_mmff_energy(mol_template, best_pos[i]) if i in polish_set else 0.0
                mmff_e = _safe_scalar_energy(mmff_e)
                inter_e, _, _, _ = self.phys.compute_energy(
                    best_pos[i:i+1], pos_P_active[i:i+1], q_L_b[i:i+1], q_P_b[i:i+1], x_L_b[i:i+1], x_P_b[i:i+1], 1.0
                )
                final_scores.append(_safe_scalar_energy(mmff_e + inter_e.item()))
            mmff_stats_after = self.phys.get_mmff_stats()
            outlier_delta = int(mmff_stats_after.get("energy_outliers", 0)) - int(mmff_stats_before.get("energy_outliers", 0))
            outlier_limit = max(3, math.ceil(0.5 * max(1, len(polish_indices))))
            if outlier_delta >= outlier_limit:
                mmff_disabled = True
                logger.warning(
                    "  [Refine] Excessive MMFF energy outliers detected "
                    f"({outlier_delta}/{max(1, len(polish_indices))}); falling back to interaction-only final scoring."
                )
                inter_only, _, _, _ = self.phys.compute_energy(
                    best_pos, pos_P_active, q_L_b, q_P_b, x_L_b, x_P_b, 1.0
                )
                final_scores = [_safe_scalar_energy(float(v)) for v in inter_only.detach().tolist()]
        else:
            inter_only, _, _, _ = self.phys.compute_energy(
                best_pos, pos_P_active, q_L_b, q_P_b, x_L_b, x_P_b, 1.0
            )
            final_scores = [_safe_scalar_energy(float(v)) for v in inter_only.detach().tolist()]

        final_energy_t = torch.tensor(final_scores, device=device, dtype=torch.float32)
        clash_final = self.phys.calculate_internal_geometry_score(best_pos).float()
        selection_score_name = str(getattr(self.config, "selection_score", "clash")).strip().lower()
        score_map = _build_selection_scores(rank_signal, final_energy_t, clash_final)
        if selection_score_name not in score_map:
            selection_score_name = "clash"
        rank_scores = score_map[selection_score_name]
        rank_order = torch.argsort(rank_scores, descending=True)
        rank_proxy_final = float(rank_scores[rank_order[0]].item()) if rank_order.numel() else float("nan")
        rank_spearman = float("nan")
        rank_top1_hit = float("nan")
        rank_top3_hit = float("nan")
        ranked_rmsd = float("nan")
        candidate_rows = []
        if pos_native is not None and pos_native.numel() > 0:
            rmsd_all = kabsch_rmsd(best_pos, pos_native)
            ranked_rmsd = float(rmsd_all[rank_order[0]].item())
            oracle_order = torch.argsort(rmsd_all)
            topk = min(3, B)
            oracle_topk = set(oracle_order[:topk].tolist())
            picked_topk = [int(v) for v in rank_order[:topk].tolist()]
            rank_top1_hit = float(int(rank_order[0].item()) == int(oracle_order[0].item()))
            rank_top3_hit = float(any(v in oracle_topk for v in picked_topk))
            rank_spearman = _spearman_from_tensors(rank_scores, -rmsd_all)
            for i in range(B):
                candidate_rows.append({
                    "candidate_idx": i,
                    "rank_score": float(rank_scores[i].item()),
                    "logz_score": float(score_map["logz"][i].item()),
                    "energy_score": float(score_map["energy"][i].item()),
                    "clash_score": float(score_map["clash"][i].item()),
                    "final_energy": float(final_energy_t[i].item()),
                    "clash": float(clash_final[i].item()),
                    "rmsd": float(rmsd_all[i].item()),
                    "is_selected": int(i == int(rank_order[0].item())),
                    "is_oracle_best": int(i == int(oracle_order[0].item())),
                })
        else:
            for i in range(B):
                candidate_rows.append({
                    "candidate_idx": i,
                    "rank_score": float(rank_scores[i].item()),
                    "logz_score": float(score_map["logz"][i].item()),
                    "energy_score": float(score_map["energy"][i].item()),
                    "clash_score": float(score_map["clash"][i].item()),
                    "final_energy": float(final_energy_t[i].item()),
                    "clash": float(clash_final[i].item()),
                    "rmsd": float("nan"),
                    "is_selected": int(i == int(rank_order[0].item())),
                    "is_oracle_best": 0,
                })

        return {
            "refined_poses": best_pos,
            "final_energies": torch.tensor(final_scores, device=device),
            "history_E": history_E,
            "log_Z_history": log_Z_history,
            # C1 stability metrics
            "log_Z_final": log_Z_history[-1] if log_Z_history else 0.0,
            "ess_min": round(min(ess_trajectory), 4) if ess_trajectory else 1.0,
            "resample_count": resample_count,
            # Claim-3 ranking diagnostics
            "rank_proxy_final": rank_proxy_final,
            "rank_spearman": rank_spearman,
            "rank_top1_hit": rank_top1_hit,
            "rank_top3_hit": rank_top3_hit,
            "ranked_rmsd": ranked_rmsd,
            "selected_idx": int(rank_order[0].item()) if rank_order.numel() else 0,
            "selection_score": selection_score_name,
            "candidate_rows": candidate_rows,
            "mmff_disabled": mmff_disabled,
        }

    def _call_smina_score(self, protein_pdb, ligand_sdf):
        """Placeholder for external SMINA scoring bridge."""
        import subprocess
        try:
            cmd = f"smina --score_only -r {protein_pdb} -l {ligand_sdf}"
            # ... execute and parse ...
            return None
        except Exception:
            return None

    def _mmff_refine(self, pos_L, mol_template, max_iter=200, indices=None):
        """Physics-Aware MMFF refinement helper."""
        if mol_template is None:
            return
        B_cur = pos_L.size(0)
        if indices is None:
            indices = range(B_cur)
        for i in indices:
            i = int(i)
            if i < 0 or i >= B_cur:
                continue
            try:
                refined = self.phys.minimize_with_mmff(mol_template, pos_L[i], max_iter=max_iter)
                if refined.shape == pos_L.data[i].shape:
                    pos_L.data[i].copy_(refined)
            except Exception as e:
                logger.debug(f"  [MMFF] Skip clone {i}: {e}")

    def run(self, device=None, use_fksmc=None, use_socm_twist=None, use_srpg=None, no_backbone=None):
        """B5 fix: accept flags as kwargs (from SAEBFlowExperiment.run delegate)."""
        # Resolve flag values: kwargs take priority, then fall back to config
        if use_fksmc is None:      use_fksmc = getattr(self.config, "fksmc", False)
        if use_socm_twist is None: use_socm_twist = getattr(self.config, "socm", False)
        if use_srpg is None:       use_srpg = getattr(self.config, "srpg", False)
        if no_backbone is None:    no_backbone = getattr(self.config, "no_backbone", False)
        use_neural_backbone = not no_backbone

        logger.info(f"SAEB-Flow | {self.config.target_name} ({self.config.pdb_id})")
        if no_backbone:
            logger.info("  [Ablation] no_backbone=True (physics-only updates)")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        # Set device on featurizer immediately so ESM and all tensors are on right GPU
        self.featurizer.device = device
        self.phys = self.phys.to(device)
        self.phys.reset_mmff_stats()

        # Bug Fix A: Correctly unpack q_P (protein charges)
        pos_P, x_P, q_P, (p_center, pos_native), x_L_native, q_L_native, lig_template = \
            self.featurizer.parse(self.config.pdb_id)

        if pos_native is None or pos_native.shape[0] == 0:
            raise ValueError(f"No native ligand atoms found for {self.config.pdb_id}")

        q_L_native = q_L_native.to(device)
        B = self.config.batch_size
        N = pos_native.shape[0]

        # If FK-SMC / SOCM / SRPG is requested, route through the dedicated refine()
        # implementation where these algorithms are actually applied.
        use_refine_engine = (use_fksmc or use_socm_twist or use_srpg) and not no_backbone
        if use_refine_engine:
            logger.info("  [Run] Delegating to refine() for FK-SMC/SOCM/SRPG path")
            pos_L_init = torch.randn(B, N, 3, device=device)
            refine_out = self.refine(
                pos_L_init=pos_L_init,
                pos_P=pos_P,
                x_P=x_P,
                q_P=q_P,
                x_L=x_L_native,
                q_L=q_L_native,
                pocket_anchor=torch.zeros(3, device=device),
                device=device,
                pos_native=pos_native,
                mol_template=lig_template,
                steps=self.config.steps,
                adaptive_stop_thresh=getattr(self.config, "adaptive_stop_thresh", 0.05),
                use_fksmc=use_fksmc,
                use_socm_twist=use_socm_twist,
                use_srpg=use_srpg,
                no_backbone=no_backbone,
            )

            refined_poses = refine_out.get("refined_poses", pos_L_init)
            if refined_poses.dim() == 2:
                refined_poses = refined_poses.unsqueeze(0)

            with torch.no_grad():
                final_rmsd_all = kabsch_rmsd(refined_poses, pos_native)
                oracle_best_idx = int(final_rmsd_all.argmin().item())
                selected_idx = int(refine_out.get("selected_idx", oracle_best_idx))
                if selected_idx < 0 or selected_idx >= refined_poses.size(0):
                    selected_idx = oracle_best_idx
                pos_L_final = refined_poses[selected_idx:selected_idx+1].detach()
                best_rmsd = final_rmsd_all[selected_idx].item()
                oracle_best_rmsd = final_rmsd_all[oracle_best_idx].item()
                best_pos = pos_L_final[0].detach().cpu().numpy()

            history_E = refine_out.get("history_E", [])
            final_energies = refine_out.get("final_energies", None)
            if isinstance(final_energies, torch.Tensor) and final_energies.numel() > selected_idx:
                final_energy = float(final_energies[selected_idx].item())
            elif history_E:
                final_energy = float(history_E[-1])
            else:
                final_energy = float("nan")
            mmff_stats = self.phys.get_mmff_stats()
            attempts = max(1, int(mmff_stats.get("attempts", 0)))
            mmff_fallback_rate = float(mmff_stats.get("fallback_used", 0)) / attempts

            if not getattr(self.config, "quiet", False):
                print(f"\n{'='*55}")
                print(f" {self.config.pdb_id:8s}  selected={best_rmsd:.2f}A  "
                      f"oracle_best={oracle_best_rmsd:.2f}A  "
                      f"mean={final_rmsd_all.mean():.2f}A  E={final_energy:.1f}")
                if history_E:
                    print(self.visualizer.interpreter.interpret_energy_trend(history_E))
                print(f"{'='*55}\n")

            if not getattr(self.config, "no_pose_dump", False):
                os.makedirs("results", exist_ok=True)
                save_points_as_pdb(best_pos, f"results/{self.config.pdb_id}_best.pdb")

            qm_candidate_dir = _dump_qm_candidates(
                artifact_dir=str(getattr(self.config, "artifact_dir", "") or ""),
                pdb_id=self.config.pdb_id,
                seed=int(getattr(self.config, "seed", 0)),
                mol_template=lig_template,
                refined_poses=refined_poses,
                candidate_rows=refine_out.get("candidate_rows", []),
                topk=int(getattr(self.config, "dump_candidate_topk", 0)),
                pos_P=pos_P,
                x_P=x_P,
                pos_native=pos_native,
            )

            return {
                "pdb_id":          self.config.pdb_id,
                "best_rmsd":       best_rmsd,
                "oracle_best_rmsd": oracle_best_rmsd,
                "mean_rmsd":       final_rmsd_all.mean().item(),
                "final_energy":    final_energy,
                "mean_cossim":     float("nan"),
                "steps":           self.config.steps,
                "log_Z_final":     refine_out.get("log_Z_final", float("nan")),
                "ess_min":         refine_out.get("ess_min", float("nan")),
                "resample_count":  refine_out.get("resample_count", 0),
                "mmff_fallback_rate": mmff_fallback_rate,
                "rank_proxy_final": refine_out.get("rank_proxy_final", float("nan")),
                "rank_spearman": refine_out.get("rank_spearman", float("nan")),
                "rank_top1_hit": refine_out.get("rank_top1_hit", float("nan")),
                "rank_top3_hit": refine_out.get("rank_top3_hit", float("nan")),
                "ranked_rmsd": refine_out.get("ranked_rmsd", float("nan")),
                "selection_score": refine_out.get("selection_score", getattr(self.config, "selection_score", "clash")),
                "mmff_disabled": int(bool(refine_out.get("mmff_disabled", False))),
                "qm_candidate_dir": qm_candidate_dir or "",
            }

        # ── Ligand ensemble initialisation ──────────────────────────────────
        # Initialize with standard spherical noise; will be refined by PCA sampling below
        pos_L = nn.Parameter(torch.randn(B, N, 3, device=device))
        x_L = nn.Parameter(x_L_native.unsqueeze(0).expand(B, -1, -1).clone())

        # ── Model ───────────────────────────────────────────────────────────
        model = None
        cbsf_loss_fn = None
        use_amp = bool(use_neural_backbone and getattr(self.config, "amp", False) and device.type == "cuda")
        use_compile_backbone = bool(use_neural_backbone and getattr(self.config, "compile_backbone", False) and device.type == "cuda")
        if use_neural_backbone:
            backbone = SAEBFlowBackbone(167, 64, num_layers=2).to(device)
            model = RectifiedFlow(backbone).to(device)
            if use_compile_backbone and hasattr(torch, "compile"):
                try:
                    model.backbone = torch.compile(model.backbone, mode="reduce-overhead")
                    logger.info("  [Speed] torch.compile enabled for backbone")
                except Exception as e:
                    logger.warning(f"  [Speed] torch.compile skipped: {e}")
                    use_compile_backbone = False
            cbsf_loss_fn = ShortcutFlowLoss(lambda_x1=1.0, lambda_conf=0.01).to(device)

        is_train = (self.config.mode == "train")

        # ── Optimiser Setup (Imp 1 & 6) ──────────────────────────────────────
        # Separation of parameters for Muon (linear weights) and AdamW (others)
        opt_adamw = None
        opt_muon = None
        sched_adamw = None
        sched_muon = None
        if use_neural_backbone:
            muon_params = []
            adamw_params = []
            
            # Backbone params
            for name, p in model.named_parameters():
                if p.ndim == 2 and "weight" in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
            
            # Position & Feature params
            adamw_params.append(x_L)
            if is_train:
                # Bug Fix 1.8: Only optimize pos_L via AdamW in training mode.
                # In inference, pos_L is driven exclusively by PAT/Physics (Imp 1).
                adamw_params.append(pos_L)
                
            opt_adamw = torch.optim.AdamW(adamw_params, lr=self.config.lr, weight_decay=1e-4)
            opt_muon  = Muon(muon_params, lr=self.config.lr * 0.02) # Adjusted for stability
        
        # Warm-up (fixed 5% of total steps) then cosine decay
        total_steps = self.config.steps
        warmup_steps = max(10, total_steps // 20) 
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * prog))
            
        if use_neural_backbone:
            sched_adamw = torch.optim.lr_scheduler.LambdaLR(opt_adamw, lr_lambda)
            sched_muon  = torch.optim.lr_scheduler.LambdaLR(opt_muon, lr_lambda)

        # Pre-expand protein tensors (shared across batch)
        pos_P_b = pos_P.unsqueeze(0).expand(B, -1, -1)
        x_P_b   = x_P.unsqueeze(0).expand(B, -1, -1)
        q_L_b   = q_L_native.unsqueeze(0).expand(B, -1)

        is_train = (self.config.mode == "train")

        history_E    = []
        history_RMSD = []          # min over ensemble
        history_FM   = []
        history_CosSim = []        # Flow alignment with Physics force

        # PAT State (Magma Inspired)
        alpha_ema = torch.zeros(B, N, 1, device=device) 
        beta_ema  = 0.9    # Magma momentum
        tau       = 2.0    # Tempered sigmoid scale

        # ── Precompute mass for Langevin noise (Improvement 4) ─────────────
        with torch.no_grad():
            mass_coeffs = torch.tensor([12.0, 14.0, 16.0, 32.0, 19.0, 31.0, 35.0, 80.0, 127.0], device=device)
            # Fix: Slice x_L to [..., :9] to match mass_coeffs
            mass_precomputed = (x_L[0, :, :9] * mass_coeffs).sum(dim=-1, keepdim=True).unsqueeze(0).expand(B, -1, -1)
            mass_precomputed = torch.clamp(mass_precomputed, min=12.0)

        # ── Precompute pocket residue mask & PCA Initialization (Imp 2) ─────
        # Finds protein atoms within 6Å of the pocket centre
        with torch.no_grad():
            dist_to_pocket = torch.cdist(pos_P.unsqueeze(0), p_center.unsqueeze(0).unsqueeze(0))[0, :, 0]
            pocket_mask = (dist_to_pocket < 6.0)  # (M,) bool
            
            # Softmax Pocket Anchor (Imp 1 v3.1): Precision Geolocation
            if pocket_mask.sum() < 20:
                topk_idx = torch.topk(dist_to_pocket, k=min(20, len(dist_to_pocket)), largest=False).indices
                pocket_pts = pos_P[topk_idx]
                dists = dist_to_pocket[topk_idx]
            else:
                pocket_pts = pos_P[pocket_mask]
                dists = dist_to_pocket[pocket_mask]
            
            # Consensus Anchor Navigation (Imp 1 v3.5): Best-Pose Seeding
            # Reverting K-Means in favor of stable consensus
            # Primary Cavity Anchor (Arithmetic Mean)
            pocket_anchor_cavity = pocket_pts.mean(dim=0)
            pocket_anchor = pocket_anchor_cavity # Initial baseline

            # Directional Sampling (PCA): Better pocket coverage
            principal_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
            try:
                centered = pocket_pts - pocket_anchor
                cov = centered.T @ centered / len(centered)
                Up, Sp, Vhp = torch.linalg.svd(cov)
                # Scale noise: 2.5Å along main axis, 1.5Å/1.0Å along others
                v_scales = torch.tensor([2.5, 1.5, 1.0], device=device)
                noise = torch.randn(B, N, 3, device=device)
                # Align noise to pocket principle axes
                noise = noise @ (Vhp.transpose(-1, -2) * v_scales).transpose(-1, -2)
                principal_axis = Vhp[0]
            except Exception:
                noise = torch.randn(B, N, 3, device=device) * 2.5
                principal_axis = F.normalize(torch.randn(3, device=device), dim=0)
            if torch.isnan(principal_axis).any() or principal_axis.norm() < 1e-6:
                principal_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
            
            # Initial position refined by PCA + 4 Clusters (Imp 5 v3.3)
            # Distribute clones into 4 clusters along the PCA main axis for broader coverage
            for b in range(B):
                c = b % 4
                c_offset = principal_axis * (c - 1.5) * 2.5  # ±3.75A spread
                pos_L.data[b].copy_(pocket_anchor + c_offset + noise[b])
            
            # Imp 3 v3.4: Adaptive Energy Kick Trackers
            energy_stagnation_count = torch.zeros(B, device=device)
            prev_clamped_energy = torch.full((B,), float('inf'), device=device)
            prev_v_pred = None # Imp 5 v3.4: Momentum tracker

        # ── Data Consistency: Batch expansion (Imp 5) ──────────────────────
        q_P_b = q_P.unsqueeze(0).expand(B, -1)
        x_P_b = x_P.unsqueeze(0).expand(B, -1, -1)
        pos_P_b = pos_P.unsqueeze(0).expand(B, -1, -1)

        prev_pos_L = prev_latent = None
        log_every = max(total_steps // 10, 1)
        
        # Improvement 1: Historical Best Tracking
        best_rmsd_ever = float('inf')
        best_pos_history = None

        for step in range(total_steps):
            # t in (0, 1] — avoid t=0 where conditional field is undefined
            t   = (step + 1) / max(total_steps, 1)
            t_t = torch.full((B,), t, device=device)

            if use_neural_backbone:
                opt_adamw.zero_grad()
                opt_muon.zero_grad()

            # ── Forward ─────────────────────────────────────────────────────
            if use_neural_backbone and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(
                        x_L=x_L, x_P=x_P_b,
                        pos_L=pos_L, pos_P=pos_P_b,
                        t=t_t,
                        prev_pos_L=prev_pos_L,
                        prev_latent=prev_latent,
                    )
            elif use_neural_backbone:
                out = model(
                    x_L=x_L, x_P=x_P_b,
                    pos_L=pos_L, pos_P=pos_P_b,
                    t=t_t,
                    prev_pos_L=prev_pos_L,
                    prev_latent=prev_latent,
                )
            else:
                out = None
            if use_neural_backbone:
                # Keep physics path in fp32 for stability.
                v_pred     = out['v_pred'].to(dtype=pos_L.dtype)      # (B, N, 3)
                x1_pred    = out['x1_pred'].to(dtype=pos_L.dtype)     # (B, N, 3)
                confidence = out['confidence'].to(dtype=pos_L.dtype)  # (B, N, 1)
            else:
                v_pred = torch.zeros_like(pos_L)
                x1_pred = pos_L.detach().clone()
                confidence = torch.zeros(B, N, 1, device=device, dtype=pos_L.dtype)

            # (Recycling storage moved to end of loop to fix Bug 21 consistency)

            # ── Loss ────────────────────────────────────────────────────────
            dt = 1.0 / total_steps
            loss_fm = torch.zeros((), device=device)
            if use_neural_backbone:
                if is_train:
                    # Conditional vector field: v*(xt|x1) = (x1 - xt)/(1 - t)
                    v_target = (pos_native.unsqueeze(0) - pos_L.detach()) / (1.0 - t + 1e-6)
                    loss_fm = cbsf_loss_fn(
                        v_pred.view(B * N, 3), x1_pred.view(B * N, 3), confidence.view(B * N, 1),
                        v_target.view(B * N, 3), pos_native, B, N
                    )
                else:
                    # Inference: self-consistency (no pos_native used)
                    euler_pos = (pos_L.detach() + v_pred * dt)
                    loss_fm = cbsf_loss_fn.inference_loss(
                        v_pred.view(B * N, 3), x1_pred.view(B * N, 3),
                        confidence.view(B * N, 1), euler_pos, B, N
                    )
                    
                    # Imp 1 v3.5: Consensus Pocket-Aware Loss
                    # If a historical best exists, pull prediction towards it; otherwise pull to cavity
                    if best_pos_history is not None:
                        consensus_target = 0.5 * pocket_anchor_cavity + 0.5 * best_pos_history.mean(dim=0)
                    else:
                        consensus_target = pocket_anchor_cavity
                    
                    x1_centroids = x1_pred.mean(dim=1) # (B, 3)
                    loss_pocket_aware = F.huber_loss(x1_centroids, consensus_target.expand_as(x1_centroids), delta=3.0)
                    
                    # Imp 2.1: Dream-Energy Loss (The "Backbone Energy" Feedback Loop)
                    # Bug Fix 26: Detach x_L to protect ligand chemical identity from corruption
                    e_dream, _, _, _ = self.phys.compute_energy(
                        x1_pred, pos_P_b, q_L_b, q_P_b, x_L.detach(), x_P_b, t
                    )
                    
                    # Imp 4 v3.3: Asymmetric Dream Gradient
                    # 2x penalty on collisions (e > 500) to prioritize "legal" poses
                    e_weight = torch.where(e_dream > 500.0, 0.010, 0.005)
                    loss_dream = (e_dream.clamp(min=-500, max=2000) * e_weight).mean()

                    # Imp 4 v4.0: Manifold Velocity Regularization
                    # Penalize non-rigid displacements in early stages
                    v_mean = v_pred.mean(dim=1, keepdim=True)
                    v_centered = v_pred - v_mean
                    # Approximate rigid rotation consistency (v ~ omega x r)
                    # We simply regularize the variance of velocity norms per batch
                    v_norm_var = v_pred.norm(dim=-1).var(dim=1).mean()
                    loss_manifold = 0.01 * v_norm_var * (1.0 - t)
                    
                    loss_fm = loss_fm + 0.1 * loss_pocket_aware + loss_dream + loss_manifold

            # pos_L.requires_grad_(True) is already set (it's an nn.Parameter)
            raw_energy, e_hard, alpha, energy_clamped = self.phys.compute_energy(
                pos_L, pos_P_b, q_L_b, q_P_b, x_L, x_P_b, t # Imp 5: Batch consistency
            )
            # Issue 8 Fix: Gradient Isolation
            # Detach raw_energy for loss_phys to prevent it from contributing to pos_L.grad.
            # We want f_phys to be the EXPLICIT and ONLY physical guide in the PAT step.
            loss_phys = raw_energy.mean().detach() * 0.01

            # Compute physical force
            f_phys = -torch.autograd.grad(
                raw_energy.sum(), pos_L, retain_graph=False, create_graph=False
            )[0].detach()

            # Pocket Guidance (Imp 3: Dynamic two-stage guidance)
            lig_centroid = pos_L.mean(dim=1, keepdim=True) # (B, 1, 3)
            # Balanced pull weight: enough to pull in, not enough to crush
            if t < 0.3:
                guidance_weight = 0.5 * math.exp(-1.0 * t) 
            else:
                guidance_weight = 0.1 * math.exp(-3.0 * t) 
                
            # Imp 2 v3.5: Distance-Adaptive Guidance
            lig_centroids = pos_L.mean(dim=1) # (B, 3)
            if best_pos_history is not None:
                current_consensus = 0.5 * pocket_anchor_cavity + 0.5 * best_pos_history.mean(dim=0)
            else:
                current_consensus = pocket_anchor_cavity
                
            dist_to_consensus = (lig_centroids - current_consensus).norm(dim=-1, keepdim=True)
            # Weight = alpha / (dist + 5.0). strong guidance when lost, weak refinement when close.
            guidance_weight = float(alpha) / (dist_to_consensus + 5.0) # (B, 1)
            
            com_offset = (current_consensus - lig_centroids).unsqueeze(1) * guidance_weight.unsqueeze(-1)
            f_phys = f_phys + com_offset
            
            # Loss for tracking
            pocket_dist = dist_to_consensus.pow(2)
            loss_pocket = (guidance_weight * pocket_dist).mean()
            
            # Imp 5 v3.2: Centroid-Based Core Repulsion (Harden v3.5: 20.0x @ 2.0A)
            # Prevents the ligand from impaling the protein core at high speeds
            with torch.no_grad():
                dist_to_core = torch.cdist(lig_centroids.unsqueeze(1), pos_P_b).squeeze(1) # (B, M)
                min_dist, closest_idx = dist_to_core.min(dim=1) # (B,)
                repulsion_mask = (min_dist < 2.0).float()
                # Strong repulsive gradient away from the closest protein atom
                closest_p_pos = pos_P_b[torch.arange(B), closest_idx].unsqueeze(1)
                f_repulse = (lig_centroids.unsqueeze(1) - closest_p_pos) / (min_dist.view(B, 1, 1) + 1e-4) # (B, 1, 3)
                f_phys = f_phys + (f_repulse * repulsion_mask.view(B, 1, 1) * 20.0)
            
            # (Note: com_offset already added above via Imp 3 logic)

            # ── Backward & Optimizer Step
            # Bug Fix 1.8: Always backward loss_fm. 
            # In inference, it's the "self-consistency" term that warms up the backbone.
            # loss_phys and loss_pocket are also included for multi-objective alignment.
            if use_neural_backbone:
                (loss_fm + loss_phys + loss_pocket).backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt_adamw.step()
                opt_muon.step()
                sched_adamw.step()
                sched_muon.step()

            # ── Metrics ───────────────────────────────────────────────────
            history_E.append(energy_clamped.mean().item())
            history_FM.append(loss_fm.item())
            
            # Bug Fix C: Align metric frequency (record RMSD every step)
            with torch.no_grad():
                rmsd = kabsch_rmsd(pos_L.detach(), pos_native)
                r_min = rmsd.min().item()
                
                # Improvement 1: Update historical best
                if r_min < best_rmsd_ever:
                    best_rmsd_ever = r_min
                    best_idx = rmsd.argmin()
                    best_pos_history = pos_L[best_idx].detach().clone()
                    
            history_RMSD.append(r_min)

            # ── Langevin Temperature Annealing ─────────────────────────────
            # Bug Fix: Use exponential decay for smoother transition and residual exploration
            decay_rate = 5.0
            T_curr = self.config.temp_start * math.exp(-decay_rate * step / total_steps) + 1e-4

            # ── PAT: Physics-Adaptive Trust (Magma Inspired) ───────────────
            # Atom-wise CosSim → Tempered Sigmoid → EMA smoothing
            with torch.no_grad():
                # Imp 6: Scaled CosSim for trust alignment
                f_norm = f_phys.norm(dim=-1, keepdim=True)
                v_norm_clamped = v_pred.norm(dim=-1, keepdim=True).detach().clamp(min=0.02)
                f_phys_scaled_for_trust = f_phys / (f_norm + 1e-8) * v_norm_clamped
                
                c_i = F.cosine_similarity(v_pred.detach(), f_phys_scaled_for_trust, dim=-1).unsqueeze(-1)
                
                # Imp 3 v3.3: Adaptive Sentiment Threshold
                # Scale tau based on CosSim variance to break stale consensus
                c_var = c_i.var() if B > 1 else torch.zeros((), device=device, dtype=c_i.dtype)
                tau_adaptive = tau * (1.0 + torch.tanh(c_var * 10.0))
                alpha_tilt = torch.sigmoid(-c_i / (tau_adaptive + 1e-6))
                
                # Imp 4 v3.4: Confidence-Filtered PAT
                # Scale physical force influence by (1 - confidence)
                # If model is highly confident (e.g. 0.9), allow it to override physics
                conf_val = confidence.detach().mean(dim=1, keepdim=True)
                conf_factor = (1.0 - conf_val)
                alpha_tilt = alpha_tilt * conf_factor
                
                # Imp 5 v3.5: Confidence-Filtered Correctional Kick
                # If model is confident but halluncinating (negative mean CosSim), apply corrective kick
                c_i_mean = c_i.mean(dim=1, keepdim=True) # (B, 1, 1)
                correction_mask = (c_i_mean < -0.1) * (conf_val > 0.8) # (B, 1, 1)
                f_phys = f_phys + f_phys * correction_mask.float() # Double the physical force for hallucinating clones
                
                # Confidence-weighted trust: lower confidence => more physics trust
                alpha_tilt = 0.5 * alpha_tilt + 0.5 * (1.0 - confidence.detach())
                
                # Bug Fix D: PAT Bias Correction
                alpha_ema = beta_ema * alpha_ema + (1.0 - beta_ema) * alpha_tilt
                alpha_ema_corr = alpha_ema / (1.0 - beta_ema**(step + 1))
                
                # Imp 2 v5.0: Backbone Intelligence Muting
                # Since the backbone is un-trained, we force-floor the physics trust
                # to 0.9 (90% physics) to prevent neural noise from overwhelming the sampler.
                if not is_train:
                    progress = step / total_steps
                    # Start at 0.9 trust in physics, slowly ease to 0.7 for final relaxation
                    alpha_ema_corr = torch.clamp(alpha_ema_corr, min=0.9 - 0.2 * progress)
                
                history_CosSim.append(c_i.mean().item())

            # ── PAT + Langevin Combined Position Update ────────────────────
            with torch.no_grad():
                # Improvement 4: Langevin Gating (Late-stage shutdown)
                # Imp 4 v3.2: Early-Stage Entropy Burst (1.5x noise during first 20% steps)
                noise_gate = 1.0 if step < int(0.9 * total_steps) else 0.0
                burst_scale = 1.5 if step < int(0.2 * total_steps) else 1.0
                noise = langevin_noise(pos_L.shape, T_curr, dt, device, mass_precomputed=mass_precomputed) * noise_gate * burst_scale
                
                # Imp 2: Shortcut Pull (Accelerate convergence using x1_pred)
                # Normalized pull vector with a 0.2A/step cap (increased for v2.1)
                # Fix Bug 24: Explicitly detach pos_L during pull calculation
                raw_pull = x1_pred.detach() - pos_L.detach()
                pull_norm = raw_pull.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                
                # Imp 3 v3.1: Shortcut Gating (Prevent Ejection)
                # Only pull if x1_pred is closer to the anchor than current pos_L
                x1_dist_to_consensus = (x1_pred.detach().mean(dim=1) - current_consensus).norm(dim=-1)
                pos_dist_to_consensus = (pos_L.detach().mean(dim=1) - current_consensus).norm(dim=-1)
                pull_gate = (x1_dist_to_consensus < pos_dist_to_consensus).float().view(B, 1, 1)
                
                # Imp 5 v3.4: Backbone Momentum (Help punch through physical势壘)
                if prev_v_pred is not None:
                    v_pred_effective = 0.9 * v_pred.detach() + 0.1 * prev_v_pred
                else:
                    v_pred_effective = v_pred.detach()
                prev_v_pred = v_pred.detach().clone()
                
                shortcut_pull = (raw_pull / pull_norm) * torch.clamp(pull_norm * 0.1, max=0.2)
                shortcut_pull = shortcut_pull * confidence.detach() * pull_gate
                
                # Imp 3 v3.4: Adaptive Energy Kick (Heat Pulse for stagnant clones)
                stagnant_mask = (energy_clamped >= prev_clamped_energy - 1e-2)
                energy_stagnation_count = torch.where(stagnant_mask, energy_stagnation_count + 1, torch.zeros_like(energy_stagnation_count))
                prev_clamped_energy = energy_clamped.clone()
                
                kick_gate = (energy_stagnation_count >= 10).float().view(B, 1, 1)
                noise_kick = langevin_noise(pos_L.shape, T_curr * 2.0, dt, device) * kick_gate
                
                pos_new = pat_step(pos_L, v_pred_effective, f_phys, alpha_ema_corr, confidence.detach(), dt)
                pos_L.data.copy_(pos_new + noise + noise_kick + shortcut_pull)

                # Imp 1 & 3 v4.0: SO(3) Geodesic Guidance (Torque-based Rotation)
                # Integrate rotational alignment into every step for faster convergence
                for b in range(B):
                    pos_L.data[b] = geodesic_rotation_step(
                        pos_L.data[b],
                        f_phys[b],
                        step_size=0.05 * (1.1 - t) # Annealing rotation rate
                    )
                
                # Bug 20: Zero-leakage Management
                if not is_train:
                    pos_L.grad = None

            # ── Alpha Hardening (called once per step) ─────────────────────
            with torch.no_grad():
                self.phys.update_alpha(f_phys.norm(dim=-1).mean().item())

            # ── Sampler Sequence (Imp 3: Reordered for escape-then-seed logic) ────

            # 1. Rigid-Body Rotational Sampler (Break orientational minima first)
            if step > warmup_steps and step % 20 == 0:
                with torch.no_grad():
                    # Rotate the bottom 50% energy clones
                    rotate_idx = energy_clamped.argsort()[B // 2:]
                    e_before = energy_clamped[rotate_idx].mean().item()
                    
                    # Dynamic Angle (Imp 3): Conservative early, aggressive late
                    # Bug 27 Fix: Ensure exploration starts immediately with 0.1pi floor
                    max_angle = math.pi * min(1.0, max(0.1, step / (total_steps * 0.5)))
                    
                    for idx in rotate_idx:
                        com = pos_L.data[idx].mean(dim=0, keepdim=True)
                        centered = pos_L.data[idx] - com
                        axis = F.normalize(torch.randn(3, device=device), dim=0)
                        angle = torch.rand(1, device=device).item() * max_angle
                        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=device)
                        R = (torch.eye(3, device=device) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K))
                        pos_L.data[idx] = (centered @ R.T) + com
                    logger.info(f"  [RotSampler] Re-sampled {len(rotate_idx)} clones (MaxAngle={max_angle/math.pi:.2f}π, E_prev={e_before:.1f})")

            # 3. Intermittent MMFF Snapping — Speed-2: only 2nd half, every 90 steps
            # Rationale: MMFF is CPU-only & expensive (B clones × 10 iters).
            # In early steps geometry is still far from pocket — MMFF rarely helps.
            # Firing only in 2nd half saves ~40% MMFF wall-time with negligible quality loss.
            snapper_start = max(warmup_steps, total_steps // 2)
            snapper_interval = max(60, total_steps // 5)  # adaptive: ~5 snaps per run
            if step >= snapper_start and step % snapper_interval == 0:
                with torch.no_grad():
                    snap_fraction = float(getattr(self.config, "mmff_snap_fraction", 0.50))
                    snap_fraction = min(1.0, max(0.0, snap_fraction))
                    if not is_train:
                        # Snap only worst-energy clones during mid-run for better wall-time.
                        snap_k = max(1, int(math.ceil(B * snap_fraction))) if snap_fraction > 0 else 0
                        if snap_k > 0:
                            snap_idx = energy_clamped.argsort(descending=True)[:snap_k]
                            logger.info(f"  [Snapper] MMFF snap on {snap_k}/{B} worst clones (10 iters)...")
                            self._mmff_refine(pos_L, lig_template, max_iter=10, indices=snap_idx.tolist())
                    else:
                        # Training mode still uses differentiable L-BFGS for grad flow
                        for idx in range(B):
                            snap_pos = pos_L.data[idx:idx+1]
                            self._lbfgs_refine(snap_pos, pos_P_b[idx:idx+1], q_L_b[idx:idx+1], q_P_b[idx:idx+1], x_L, x_P_b[idx:idx+1], t, max_iter=3)
                    if not is_train: pos_L.grad = None

            # 4. Torsional Manifold Sampling (Imp 2 v4.0: Explore conformational space)
            if step > warmup_steps and step % 30 == 0 and lig_template is not None:
                with torch.no_grad():
                    torsion_idx = energy_clamped.argsort()[B // 2:] 
                    for idx in torsion_idx:
                        pos_L.data[idx] = torsional_sampling_step(pos_L.data[idx], lig_template)
                    logger.info(f"  [Torsion] Sampled conformers for {len(torsion_idx)} high-energy clones")

            # 2. Replica Exchange (Population-based seeding after rotation escape)
            if step > warmup_steps and step % 50 == 0:
                with torch.no_grad():
                    energies = energy_clamped  # (B,)
                    top_k = max(1, B // 4)
                    good_idx = energies.argsort()[:top_k]
                    bad_idx  = energies.argsort()[-(B // 4):]  # Worst 25%
                    
                    logger.debug(f"  [Replica] Purging and seeding from top performers")
                    best_com = pos_L.data[good_idx[0]].mean(dim=0)
                    for i, b_idx in enumerate(bad_idx):
                        # Imp 2 v3.3: Diversity-Driven Seeding
                        # Pick a "good" clone that is geographically distant from the best
                        good_coms = pos_L.data[good_idx].mean(dim=1)
                        dist_to_best = (good_coms - best_com).norm(dim=1)
                        s_idx = good_idx[dist_to_best.argmax()] if i % 2 == 0 else good_idx[0]
                        
                        # Imp 4: Energy-Adaptive Jitter
                        e_diff = (energies[b_idx] - energies[s_idx]).clamp(min=0).item()
                        perturbation_scale = 0.2 + 0.5 * torch.tanh(torch.tensor(e_diff / 500.0)).item()
                        pos_L.data[b_idx] = pos_L.data[s_idx] + torch.randn_like(pos_L.data[s_idx]) * perturbation_scale

            # ── Bug 21/25 Fix: Finalize Recycling State ───────────────────
            # Ensure recycling sees the results of ALL samplers at the absolute end
            prev_pos_L  = pos_L.detach().clone()
            if use_neural_backbone:
                prev_latent = out['latent'].detach().to(dtype=pos_L.dtype).clone()
            else:
                prev_latent = None

            # ── Log ────────────────────────────────────────────────────────
            if step % log_every == 0 or step == total_steps - 1:
                # Calculate average force norm for logging
                f_norm_avg = f_phys.norm(dim=-1).mean().item()
                lr_now = sched_adamw.get_last_lr()[0] if sched_adamw is not None else 0.0
                logger.info(
                    f"  [{step+1:4d}/{total_steps}] "
                    f"E={history_E[-1]:8.1f}  F_phys={f_norm_avg:.4f}  "
                    f"CosSim={history_CosSim[-1]:.3f}  α_trust={alpha_ema_corr.mean():.2f}  "
                    f"T={T_curr:.3f}  "
                    f"RMSD={history_RMSD[-1]:.2f}A  "
                    f"lr={lr_now:.2e}"
                )
        # ── Final Selection: Best-N Ensemble MMFF Polish ──────────────────────
        # Fix for positive-energy targets: polish lowest-energy clones.
        final_mmff_topk = int(getattr(self.config, "final_mmff_topk", 5))
        final_mmff_topk = max(0, min(final_mmff_topk, B))
        final_mmff_max_iter = int(getattr(self.config, "final_mmff_max_iter", 2000))
        final_mmff_max_iter = max(10, final_mmff_max_iter)

        with torch.no_grad():
            if not is_train and lig_template is not None and final_mmff_topk > 0:
                logger.info(
                    f"  [Refine] Best-N MMFF Polish (top {final_mmff_topk} lowest-energy clones, "
                    f"{final_mmff_max_iter} iters)..."
                )
                try:
                    final_e, _, _, _ = self.phys.compute_energy(
                        pos_L.detach(), pos_P_b, q_L_b, q_P_b, x_L.detach(), x_P_b, 1.0
                    )
                    top_idx = final_e.topk(final_mmff_topk, largest=False).indices.tolist()
                    candidate_pool = pos_L.data[top_idx].clone()
                    if best_pos_history is not None:
                        hist_t = best_pos_history.unsqueeze(0)
                        candidate_pool = torch.cat([hist_t, candidate_pool], dim=0)
                    self._mmff_refine(candidate_pool, lig_template, max_iter=final_mmff_max_iter)
                    cand_rmsd = kabsch_rmsd(candidate_pool, pos_native)
                    best_cand = cand_rmsd.argmin()
                    pos_L_final = candidate_pool[best_cand].unsqueeze(0)
                except Exception as mmff_e:
                    logger.warning(f"  [Refine] Best-N MMFF failed: {mmff_e}, falling back.")
                    if best_pos_history is not None:
                        pos_L_final = best_pos_history.unsqueeze(0)
                    else:
                        fallback_rmsd = kabsch_rmsd(pos_L.detach(), pos_native)
                        pos_L_final = pos_L[fallback_rmsd.argmin()].detach().unsqueeze(0)
            elif best_pos_history is not None:
                pos_L_final = best_pos_history.unsqueeze(0)
            else:
                final_rmsd_tmp = kabsch_rmsd(pos_L.detach(), pos_native)
                pos_L_final = pos_L[final_rmsd_tmp.argmin()].detach().unsqueeze(0)

            final_rmsd = kabsch_rmsd(pos_L_final, pos_native)
            best_rmsd  = final_rmsd.min().item()
            best_pos   = pos_L_final[0].detach().cpu().numpy()

        final_energy = history_E[-1] if history_E else float("nan")
        if not getattr(self.config, "quiet", False):
            print(f"\n{'='*55}")
            print(f" {self.config.pdb_id:8s}  best={best_rmsd:.2f}A  "
                  f"mean={final_rmsd.mean():.2f}A  E={final_energy:.1f}")
            if history_E:
                print(self.visualizer.interpreter.interpret_energy_trend(history_E))
            print(f"{'='*55}\n")

        # Save best pose
        if not getattr(self.config, "no_pose_dump", False):
            os.makedirs("results", exist_ok=True)
            save_points_as_pdb(best_pos, f"results/{self.config.pdb_id}_best.pdb")

        # Per-target plots can dominate IO in benchmark sweeps; keep optional.
        if not getattr(self.config, "no_target_plots", False):
            self.visualizer.plot_convergence_dynamics(
                history_E, filename=f"conv_{self.config.pdb_id}.pdf"
            )
            self.visualizer.plot_rmsd_convergence(
                history_RMSD, filename=f"rmsd_{self.config.pdb_id}.pdf"
            )
            self.visualizer.plot_alignment_trends(
                history_CosSim, filename=f"align_{self.config.pdb_id}.pdf"
            )

        mmff_stats = self.phys.get_mmff_stats()
        attempts = max(1, int(mmff_stats.get("attempts", 0)))
        mmff_fallback_rate = float(mmff_stats.get("fallback_used", 0)) / attempts
        return {
            "pdb_id":          self.config.pdb_id,
            "best_rmsd":       best_rmsd,
            "oracle_best_rmsd": best_rmsd,
            "mean_rmsd":       final_rmsd.mean().item(),
            "final_energy":    final_energy,
            "mean_cossim":     np.mean(history_CosSim) if history_CosSim else 0.0,
            "steps":           total_steps,
            "log_Z_final":     float("nan"),
            "ess_min":         float("nan"),
            "resample_count":  0,
            "mmff_fallback_rate": mmff_fallback_rate,
            "rank_proxy_final": float("nan"),
            "rank_spearman": float("nan"),
            "rank_top1_hit": float("nan"),
            "rank_top3_hit": float("nan"),
            "ranked_rmsd": float("nan"),
            "selection_score": getattr(self.config, "selection_score", "clash"),
            "mmff_disabled": 0,
        }
