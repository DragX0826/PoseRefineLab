from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import torch

@dataclass
class SimulationConfig:
    """
    Central Configuration for SAEB-Flow Experiments.
    Standardizes all hyperparameters for ICLR-grade benchmarking.
    """
    pdb_id: str
    target_name: str
    steps: int = 300 
    lr: float = 1e-3
    temp_start: float = 1.0
    temp_end: float = 0.1
    softness_start: float = 5.0
    softness_end: float = 0.0
    use_muon: bool = True
    use_grpo: bool = True
    use_kl: bool = True
    kl_beta: float = 0.05
    checkpoint_freq: int = 5
    
    no_rgf: bool = False
    no_physics: bool = False
    no_grpo: bool = False
    mode: str = "train" 
    redocking: bool = True 
    blind_docking: bool = False 
    mutation_rate: float = 0.0 
    
    batch_size: int = 16 
    mcmc_steps: int = 4000 
    accum_steps: int = 4 
    
    b_mcmc: int = 64
    fp32: bool = False
    vina: bool = False
    target_pdb_path: str = "" 
    pdb_dir: str = ""           # Kaggle: path to pre-downloaded PDB files directory
    seed: int = 42              # Random seed for reproducibility
    
    no_hsa: bool = False
    no_adaptive_mcmc: bool = False
    no_jiggling: bool = False
    jiggle_scale: float = 1.0 
    
    no_fse3: bool = False
    no_cbsf: bool = False
    no_pidrift: bool = False
    
    # v9.0 Mathematical Foundations
    fksmc: bool = False
    socm: bool = False
    srpg: bool = False
    # v10.0 Ablation flags (B4 fix)
    no_backbone: bool = False

    # v10.2 Runtime acceleration knobs
    amp: bool = False
    compile_backbone: bool = False
    mmff_snap_fraction: float = 0.50
    no_target_plots: bool = False
    final_mmff_topk: int = 5
    final_mmff_max_iter: int = 2000
    no_pose_dump: bool = False
    adaptive_stop_thresh: float = 0.05
    adaptive_min_step_frac: float = 0.65
    adaptive_patience_frac: float = 0.12
    rerank_polish_mult: int = 2
    selection_score: str = "clash"
    search_rescue_min_step_frac: float = 0.35
    search_rescue_patience_frac: float = 0.08
    search_rescue_scale: float = 2.5
    dump_candidate_topk: int = 0
    artifact_dir: str = ""
    quiet: bool = False
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
