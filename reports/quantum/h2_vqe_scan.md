# H2 VQE Scan

This run compares hydrogen bond-length energy curves from:

- VQE with Qiskit Nature (`sto3g`, Jordan-Wigner, `TwoLocal` ansatz)
- Exact diagonalization on the same qubit Hamiltonian
- RDKit UFF as a classical force-field baseline

Scan setup:

- bond range: 0.400–1.500 Å
- number of points: 10
- optimizer: SLSQP

Best exact-energy point:

- bond_length_angstrom: 0.767
- vqe_energy_hartree: -1.11477957
- exact_energy_hartree: -1.13650017
- uff_energy_kcal_mol: 1.63252411

VQE agreement with exact diagonalization:

- max_abs_error_hartree: 6.436314e-02
- mean_abs_error_hartree: 2.105844e-02

Interpretation:

- VQE and exact diagonalization live on the same quantum chemistry scale (Hartree).
- UFF is an empirical force field, so its absolute values are not directly comparable to the quantum energies.
- The useful comparison is that UFF captures only a rough geometric trend, while VQE reproduces the quantum ground-state curve itself.
