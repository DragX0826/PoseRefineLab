#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

HARTREE_TO_KCAL_MOL = 627.509474


def build_h2_mol(distance_angstrom: float) -> Chem.Mol:
    mol = Chem.MolFromSmiles("[H][H]")
    if mol is None:
        raise RuntimeError("Failed to build H2 molecule from SMILES.")
    mol = Chem.AddHs(Chem.RemoveHs(mol), addCoords=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (0.0, 0.0, float(distance_angstrom)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def compute_uff_energy(distance_angstrom: float) -> float:
    mol = build_h2_mol(distance_angstrom)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    if ff is None:
        raise RuntimeError("UFF is unavailable for H2 in the current RDKit build.")
    return float(ff.CalcEnergy())


def compute_vqe_and_exact_energy(distance_angstrom: float, maxiter: int) -> tuple[float, float]:
    from qiskit.primitives import Estimator
    from qiskit.circuit.library import TwoLocal
    from qiskit_algorithms import NumPyMinimumEigensolver, VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance_angstrom}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    mapper = JordanWignerMapper()

    solver_exact = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    result_exact = solver_exact.solve(problem)
    exact_energy = float(result_exact.total_energies[0].real)

    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    ansatz = TwoLocal(
        qubit_op.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cz",
        entanglement="full",
        reps=2,
    )
    optimizer = SLSQP(maxiter=maxiter)
    solver_vqe = GroundStateEigensolver(
        mapper,
        VQE(
            estimator=Estimator(),
            ansatz=ansatz,
            optimizer=optimizer,
        ),
    )
    result_vqe = solver_vqe.solve(problem)
    vqe_energy = float(result_vqe.total_energies[0].real)
    return vqe_energy, exact_energy


def build_application_paragraph(df: pd.DataFrame) -> str:
    best_idx = df["exact_energy_hartree"].idxmin()
    best_row = df.loc[best_idx]
    max_abs_err = float(df["abs_vqe_minus_exact_hartree"].max())
    mean_abs_err = float(df["abs_vqe_minus_exact_hartree"].mean())

    return (
        "為了更直接理解傳統力場與量子計算方法在能量評估上的差距，我用 Qiskit 對氫分子（H2）進行了 VQE 鍵長掃描實驗，"
        "並與精確對角化（exact diagonalization）及 UFF 力場的結果並排比較。"
        "實驗使用 Jordan-Wigner 映射、TwoLocal ansatz 與 SLSQP 最佳化，在 "
        f"{df['bond_length_angstrom'].min():.1f}–{df['bond_length_angstrom'].max():.1f} Å 範圍內掃描 {len(df)} 個鍵長點。"
        f"VQE 與精確解的最大誤差為 {max_abs_err:.2e} Hartree，平均誤差為 {mean_abs_err:.2e} Hartree；"
        f"最低能量點出現在約 {best_row['bond_length_angstrom']:.3f} Å。"
        "結果顯示，VQE 能重現小體系的量子力學基態能量曲線，而 UFF 雖能大致捕捉極小值附近的幾何趨勢，"
        "但能量尺度與量子化學結果不同，無法直接作為量子化學能量的替代。"
        "這個實驗讓我更具體理解了傳統力場在精細能量評估上的限制，也讓我更清楚未來量子化學方法在分子建模中的角色。"
    )


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    best_idx = df["exact_energy_hartree"].idxmin()
    best_row = df.loc[best_idx]
    max_abs_err = float(df["abs_vqe_minus_exact_hartree"].max())
    mean_abs_err = float(df["abs_vqe_minus_exact_hartree"].mean())
    md = f"""# H2 VQE Scan

This run compares hydrogen bond-length energy curves from:

- VQE with Qiskit Nature (`sto3g`, Jordan-Wigner, `TwoLocal` ansatz)
- Exact diagonalization on the same qubit Hamiltonian
- RDKit UFF as a classical force-field baseline

Scan setup:

- bond range: {df['bond_length_angstrom'].min():.3f}–{df['bond_length_angstrom'].max():.3f} Å
- number of points: {len(df)}
- optimizer: SLSQP

Best exact-energy point:

- bond_length_angstrom: {best_row["bond_length_angstrom"]:.3f}
- vqe_energy_hartree: {best_row["vqe_energy_hartree"]:.8f}
- exact_energy_hartree: {best_row["exact_energy_hartree"]:.8f}
- uff_energy_kcal_mol: {best_row["uff_energy_kcal_mol"]:.8f}

VQE agreement with exact diagonalization:

- max_abs_error_hartree: {max_abs_err:.6e}
- mean_abs_error_hartree: {mean_abs_err:.6e}

Interpretation:

- VQE and exact diagonalization live on the same quantum chemistry scale (Hartree).
- UFF is an empirical force field, so its absolute values are not directly comparable to the quantum energies.
- The useful comparison is that UFF captures only a rough geometric trend, while VQE reproduces the quantum ground-state curve itself.
"""
    (out_dir / "h2_vqe_scan.md").write_text(md, encoding="utf-8")
    (out_dir / "application_paragraph.md").write_text(build_application_paragraph(df), encoding="utf-8")


def plot_scan(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax_left = plt.subplots(figsize=(8.8, 5.2), dpi=220)
    ax_right = ax_left.twinx()

    ax_left.plot(
        df["bond_length_angstrom"],
        df["vqe_energy_hartree"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="VQE (Hartree)",
    )
    ax_left.plot(
        df["bond_length_angstrom"],
        df["exact_energy_hartree"],
        marker="s",
        linewidth=2.0,
        color="#ff7f0e",
        label="Exact diagonalization (Hartree)",
    )
    ax_right.plot(
        df["bond_length_angstrom"],
        df["uff_energy_kcal_mol"],
        marker="^",
        linewidth=1.8,
        color="#2ca02c",
        label="UFF (kcal/mol)",
    )

    ax_left.set_xlabel("H-H bond length (Å)")
    ax_left.set_ylabel("Quantum energy (Hartree)")
    ax_right.set_ylabel("UFF energy (kcal/mol)")
    ax_left.set_title("H2 bond-length scan: VQE vs exact diagonalization vs UFF")
    ax_left.grid(alpha=0.25)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "h2_vqe_scan.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an H2 VQE / exact / UFF bond scan.")
    parser.add_argument("--out_dir", type=Path, default=Path("reports/quantum"))
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--min_bond", type=float, default=0.4)
    parser.add_argument("--max_bond", type=float, default=1.5)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bond_lengths = np.linspace(args.min_bond, args.max_bond, args.points)
    rows = []
    for bond_length in bond_lengths:
        print(f"[run] bond_length={bond_length:.3f} Å", flush=True)
        vqe_energy, exact_energy = compute_vqe_and_exact_energy(float(bond_length), args.maxiter)
        uff_energy = compute_uff_energy(float(bond_length))
        rows.append(
            {
                "bond_length_angstrom": float(bond_length),
                "vqe_energy_hartree": vqe_energy,
                "exact_energy_hartree": exact_energy,
                "uff_energy_kcal_mol": uff_energy,
                "abs_vqe_minus_exact_hartree": abs(vqe_energy - exact_energy),
                "uff_energy_hartree_equiv": uff_energy / HARTREE_TO_KCAL_MOL,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "h2_vqe_scan.csv", index=False)
    plot_scan(df, args.out_dir)
    write_summary(df, args.out_dir)
    print(f"[done] outputs written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
