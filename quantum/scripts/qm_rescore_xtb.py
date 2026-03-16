import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem


HARTREE_TO_KCAL_MOL = 627.509474


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _zscore(values):
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) <= 1:
        return [0.0 for _ in values]
    mean_v = sum(finite) / len(finite)
    var = sum((v - mean_v) ** 2 for v in finite) / len(finite)
    std = math.sqrt(max(var, 1e-12))
    out = []
    for value in values:
        if math.isfinite(value):
            out.append((value - mean_v) / std)
        else:
            out.append(0.0)
    return out


def _discover_candidate_dirs(input_path: Path):
    if input_path.is_file():
        if input_path.name != "candidate_topk.csv":
            raise FileNotFoundError(f"Expected candidate_topk.csv, got: {input_path}")
        return [input_path.parent]
    candidate_files = sorted(input_path.rglob("candidate_topk.csv"))
    return [path.parent for path in candidate_files]


def _write_xyz_from_sdf(sdf_path: Path, xyz_path: Path):
    mols = [mol for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if mol is not None]
    if not mols:
        raise ValueError(f"Failed to read SDF: {sdf_path}")
    mol = mols[0]
    if mol.GetNumAtoms() == mol.GetNumHeavyAtoms():
        mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), sdf_path.stem]
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        lines.append(f"{atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_total_energy(stdout_text: str):
    patterns = [
        r"TOTAL ENERGY\s+(-?\d+\.\d+)",
        r"total energy\s+(-?\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError("Could not parse TOTAL ENERGY from xTB output")


def _run_xtb(sdf_path: Path, xtb_bin: str, charge: int, uhf: int, gfn: int, alpb: str, opt: bool, keep_workdir: bool):
    workdir_obj = tempfile.TemporaryDirectory(prefix="xtb_rescore_")
    workdir = Path(workdir_obj.name)
    xyz_path = workdir / "input.xyz"
    _write_xyz_from_sdf(sdf_path, xyz_path)

    cmd = [
        xtb_bin,
        str(xyz_path.name),
        "--gfn",
        str(gfn),
        "--chrg",
        str(charge),
        "--uhf",
        str(uhf),
    ]
    if alpb:
        cmd.extend(["--alpb", alpb])
    if opt:
        cmd.append("--opt")

    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_text = proc.stdout + "\n" + proc.stderr
    energy = float("nan")
    if proc.returncode == 0:
        try:
            energy = _parse_total_energy(stdout_text)
        except ValueError:
            pass

    saved_workdir = ""
    if keep_workdir:
        saved_root = sdf_path.parent / "xtb_workdirs"
        saved_root.mkdir(exist_ok=True)
        saved_dir = saved_root / sdf_path.stem
        if saved_dir.exists():
            shutil.rmtree(saved_dir)
        shutil.copytree(workdir, saved_dir)
        saved_workdir = str(saved_dir)

    workdir_obj.cleanup()
    return {
        "returncode": proc.returncode,
        "stdout": stdout_text,
        "energy_eh": energy,
        "saved_workdir": saved_workdir,
    }


def _rescore_directory(candidate_dir: Path, args):
    topk_csv = candidate_dir / "candidate_topk.csv"
    rows = list(csv.DictReader(topk_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows in {topk_csv}")

    results = []
    for row in rows:
        sdf_file = row.get("sdf_file", "").strip()
        if not sdf_file:
            continue
        sdf_path = candidate_dir / sdf_file
        run_info = _run_xtb(
            sdf_path=sdf_path,
            xtb_bin=args.xtb_bin,
            charge=args.charge,
            uhf=args.uhf,
            gfn=args.gfn,
            alpb=args.alpb,
            opt=args.opt,
            keep_workdir=args.keep_workdir,
        )
        result = dict(row)
        result["xtb_ok"] = int(run_info["returncode"] == 0 and math.isfinite(run_info["energy_eh"]))
        result["xtb_returncode"] = run_info["returncode"]
        result["xtb_energy_eh"] = run_info["energy_eh"]
        result["xtb_workdir"] = run_info["saved_workdir"]
        results.append(result)

    energies = [_safe_float(row.get("xtb_energy_eh")) for row in results]
    finite_energies = [energy for energy in energies if math.isfinite(energy)]
    min_energy = min(finite_energies) if finite_energies else float("nan")
    xtb_delta_kcal = [
        (energy - min_energy) * HARTREE_TO_KCAL_MOL if math.isfinite(energy) and math.isfinite(min_energy) else float("nan")
        for energy in energies
    ]
    docking_energies = [_safe_float(row.get("final_energy")) for row in results]
    docking_z = [-value for value in _zscore(docking_energies)]
    xtb_z = [-value for value in _zscore(xtb_delta_kcal)]

    blend_scores = []
    for dock_component, xtb_component, delta in zip(docking_z, xtb_z, xtb_delta_kcal):
        if math.isfinite(delta):
            blend_scores.append(dock_component + args.strain_weight * xtb_component)
        else:
            blend_scores.append(float("nan"))

    finite_blend = [(idx, score) for idx, score in enumerate(blend_scores) if math.isfinite(score)]
    ranked_indices = [idx for idx, _ in sorted(finite_blend, key=lambda item: item[1], reverse=True)]
    rank_map = {idx: rank + 1 for rank, idx in enumerate(ranked_indices)}
    best_idx = ranked_indices[0] if ranked_indices else None

    for idx, row in enumerate(results):
        row["xtb_delta_kcal"] = xtb_delta_kcal[idx]
        row["dock_z"] = docking_z[idx]
        row["xtb_z"] = xtb_z[idx]
        row["xtb_blend_score"] = blend_scores[idx]
        row["xtb_rank"] = rank_map.get(idx, "")
        row["xtb_recommended"] = int(best_idx == idx)

    out_csv = candidate_dir / "xtb_rescored.csv"
    fieldnames = list(results[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary_path = candidate_dir / "xtb_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("xTB rescoring summary\n")
        fh.write(f"Candidate directory: {candidate_dir}\n")
        fh.write(f"Method: GFN{args.gfn}-xTB\n")
        fh.write(f"Charge: {args.charge}\n")
        fh.write(f"UHF: {args.uhf}\n")
        fh.write(f"ALPB: {args.alpb or 'none'}\n")
        fh.write(f"Optimization: {int(args.opt)}\n")
        fh.write(f"Strain weight: {args.strain_weight}\n")
        if best_idx is not None:
            best_row = results[best_idx]
            fh.write(
                "Recommended candidate: "
                f"rank={best_row.get('xtb_rank')} "
                f"sdf={best_row.get('sdf_file')} "
                f"blend={best_row.get('xtb_blend_score')}\n"
            )
        else:
            fh.write("No valid xTB result was produced.\n")

    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Rescore exported docking poses with GFN-xTB.")
    parser.add_argument("--input", required=True, help="Candidate directory or root containing candidate_topk.csv files")
    parser.add_argument("--xtb_bin", default="xtb", help="Path to xtb executable")
    parser.add_argument("--charge", type=int, default=0, help="Total charge for xTB")
    parser.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons for xTB")
    parser.add_argument("--gfn", type=int, default=2, choices=[0, 1, 2], help="GFN level")
    parser.add_argument("--alpb", default="", help="Optional implicit solvent, e.g. water")
    parser.add_argument("--opt", action="store_true", help="Run xTB geometry optimization before reading total energy")
    parser.add_argument("--strain_weight", type=float, default=0.5, help="Weight of xTB strain correction in blend score")
    parser.add_argument("--keep_workdir", action="store_true", help="Keep per-candidate xTB workdirs for inspection")
    args = parser.parse_args()

    if shutil.which(args.xtb_bin) is None and not Path(args.xtb_bin).exists():
        raise FileNotFoundError(f"xTB executable not found: {args.xtb_bin}")

    input_path = Path(args.input)
    candidate_dirs = _discover_candidate_dirs(input_path)
    if not candidate_dirs:
        raise FileNotFoundError(f"No candidate_topk.csv files found under: {input_path}")

    for candidate_dir in candidate_dirs:
        out_csv = _rescore_directory(candidate_dir, args)
        print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
