"""Finite-difference correctness checks for the rod vibro-impact DLFT residual.

Standalone developer script (not part of the main run): rebuilds the system +
FBSProblem and verifies that the analytical residual Jacobian and dr/domega match
central finite differences at a contact-active point.

Run:  python checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np

from pyhbm import (
    Fourier, FourierOmegaPoint, FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod,
)
from dynamical_system import RodVibroImpactFlexible


# --- rebuild the same problem main.py uses (default k_rel = 1) --------------
GAP       = 0.2e-3
HARMONICS = list(range(0, 21))
POLY_DEG  = 30
F0        = 25e3

rod     = RodVibroImpactFlexible(k_rel=1.0, F0=F0, poly_deg=POLY_DEG)
EPSILON = 2.0e1 * rod.k_rod

HarmonicBalanceMethod.update_dependencies(HARMONICS, rod.polynomial_degree)
provider = NumericalFRF(rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(rod, provider, contact)


# ============================ FD checks (contact-active point) ==============

print("=" * 70)
print("Finite-difference checks of DLFT residual Jacobian and dr/domega")
print("=" * 70)
omega_fd = 1.0                                   # w_hat = 1 -> first resonance
Q1_fd    = np.array([[(2.0 * GAP) + 0.0j]])      # |x_r| > g0 -> contact active
x_fd     = FourierOmegaPoint.new_from_first_harmonic(Q1_fd, omega=omega_fd)

r0 = problem.compute_residue_RI(x_fd)
J_an = problem.compute_jacobian_of_residue_RI(x_fd)
n_q  = J_an.shape[0]
h    = 1.0e-7
J_fd = np.zeros_like(J_an)
x_arr = np.asarray(x_fd).copy()
for j in range(n_q):
    xp = x_arr.copy(); xp[j, 0] += h
    x_p = FourierOmegaPoint(Fourier.new_from_RI(xp[:-1]), x_fd.omega)
    J_fd[:, j] = (problem.compute_residue_RI(x_p) - r0).ravel() / h
errJ = np.abs(J_an - J_fd)
relJ = errJ.max() / (np.abs(J_fd).max() + 1e-30)
print(f"  Jacobian : max|J| = {np.abs(J_an).max():.3e}, max abs err = {errJ.max():.3e}, rel = {relJ:.2e}")

dw_an = problem.compute_derivative_wrt_omega_RI(x_fd)
hw = 1.0e-3
rp = problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega + hw))
rm = problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega - hw))
dw_fd = (rp - rm) / (2 * hw)
errW = np.abs(dw_an - dw_fd)
relW = errW.max() / (np.abs(dw_fd).max() + 1e-30)
print(f"  dr/domega: max|.| = {np.abs(dw_an).max():.3e}, max abs err = {errW.max():.3e}, rel = {relW:.2e}")
print("=" * 70)
