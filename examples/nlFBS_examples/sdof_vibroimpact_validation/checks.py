"""Developer checks for the SDOF vibro-impact example (standalone).

  1. Finite-difference verification of the DLFT residual Jacobian and dr/domega at
     a contact-active point (not a solution -- only the analytical vs FD match
     matters).
  2. Continuation diagnostics: omega-bin counts, the last 10 continuation points,
     and residual norms across the contact regime.

Rebuilds the same problem as main.py.  Run:  python checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

from pyhbm import (
    Fourier, FourierOmegaPoint, FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod,
)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from dynamical_system import SDOFVibroImpact


# --- rebuild the same problem main.py uses ---------------------------------
EPSILON   = 1.0
PARAMS    = dict(m=1.0, c=0.05, k=1.0, F0=0.02)
GAP       = 0.1
HARMONICS = list(range(0, 30))
OMEGA_START, OMEGA_END = 0.5, 2.5

system   = SDOFVibroImpact(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)
provider = NumericalFRF(system.mass_matrix, system.damping_matrix, system.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(system, provider, contact)


# ============================ FD Jacobian check =============================
# Exercise the mask machinery: pick a state with |Q1| > g0 so contact is active at
# some time samples. This need NOT be a solution.
print("=" * 70)
print("Finite-difference check of DLFT residual Jacobian")
print("=" * 70)
omega_fd = 1.0
Q1_fd    = np.array([[0.5 + 0.0j]])     # amp > g0 = 0.1 -> contact active
x_fd     = FourierOmegaPoint.new_from_first_harmonic(Q1_fd, omega=omega_fd)

r0 = problem.compute_residue_RI(x_fd)
print(f"  Test point: ω = {omega_fd}, |Q1| = {abs(Q1_fd[0,0]):.3f}, g0 = {GAP}")
print(f"  ||r0|| = {np.linalg.norm(r0):.3e}   (not a solution -- irrelevant for FD test)")

J_analytical = problem.compute_jacobian_of_residue_RI(x_fd)
n_q          = J_analytical.shape[0]      # 2 * Nh * d_int  (omega NOT included)

h     = 1.0e-7
J_fd  = np.zeros_like(J_analytical)
x_arr = np.asarray(x_fd).copy()
for j in range(n_q):
    x_pert = x_arr.copy()
    x_pert[j, 0] += h
    fourier_pert = Fourier.new_from_RI(x_pert[:-1])      # exclude omega slot
    x_p          = FourierOmegaPoint(fourier_pert, x_fd.omega)
    r_p          = problem.compute_residue_RI(x_p)
    J_fd[:, j]   = (r_p - r0).ravel() / h

abs_err = np.abs(J_analytical - J_fd)
print(f"\n  shape J            = {J_analytical.shape}")
print(f"  max |J_analytic|   = {np.abs(J_analytical).max():.3e}")
print(f"  max |J_fd|         = {np.abs(J_fd).max():.3e}")
print(f"  max abs error      = {abs_err.max():.3e}")
mask_big = abs_err > 1e-8
if mask_big.any():
    rel = abs_err[mask_big] / (np.abs(J_fd[mask_big]) + 1e-12)
    print(f"  max rel error      = {rel.max():.3e}   (where abs error > 1e-8)")
print("\n  top 5 disagreements:")
flat_top = np.argsort(abs_err.ravel())[::-1][:5]
for k in flat_top:
    i, jj = np.unravel_index(k, abs_err.shape)
    print(f"    [{i:3d},{jj:3d}]  analytic = {J_analytical[i, jj]:+.4e}   "
          f"fd = {J_fd[i, jj]:+.4e}   err = {abs_err[i, jj]:.2e}")
print("=" * 70)


# ============================ FD check of dr/domega =========================
print("Finite-difference check of ∂r/∂ω")
print("=" * 70)

dr_dw_analytical = problem.compute_derivative_wrt_omega_RI(x_fd)

h_w = 1.0e-6
x_omega_p = FourierOmegaPoint(x_fd.fourier, x_fd.omega + h_w)
x_omega_m = FourierOmegaPoint(x_fd.fourier, x_fd.omega - h_w)
r_plus  = problem.compute_residue_RI(x_omega_p)
r_minus = problem.compute_residue_RI(x_omega_m)
dr_dw_fd = (r_plus - r_minus) / (2 * h_w)

abs_err_w = np.abs(dr_dw_analytical - dr_dw_fd)
print(f"  shape           = {dr_dw_analytical.shape}")
print(f"  max |∂r/∂ω_an|  = {np.abs(dr_dw_analytical).max():.3e}")
print(f"  max |∂r/∂ω_fd|  = {np.abs(dr_dw_fd).max():.3e}")
print(f"  max abs error   = {abs_err_w.max():.3e}")
big_w = abs_err_w > 1e-8
if big_w.any():
    rel_w = abs_err_w[big_w] / (np.abs(dr_dw_fd[big_w]) + 1e-12)
    print(f"  max rel error   = {rel_w.max():.3e}   (where abs error > 1e-8)")
print("\n  top 5 disagreements:")
flat_top_w = np.argsort(abs_err_w.ravel())[::-1][:5]
for k in flat_top_w:
    i = int(k)
    print(f"    [{i:3d}]  analytic = {dr_dw_analytical.ravel()[i]:+.4e}   "
          f"fd = {dr_dw_fd.ravel()[i]:+.4e}   err = {abs_err_w.ravel()[i]:.2e}")
print("=" * 70)


# ============================ continuation diagnostics =====================
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization)
initial_guess = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=OMEGA_START)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(
    np.zeros((1, 1), complex), omega=1.0)
solution_set = solver.solve_and_continue(
    initial_guess                 = initial_guess,
    initial_reference_direction   = initial_reference_direction,
    maximum_number_of_solutions   = 10000,
    angular_frequency_range       = [OMEGA_START, OMEGA_END],
    solver_kwargs                 = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs = {"base": 2.0, "initial_step_length": 0.005,
                                     "maximum_step_length": 1.0, "minimum_step_length": 1e-6,
                                     "goal_number_of_iterations": 4},
    jacobian_update_frequency     = 1,
)

omegas_arr = np.array(solution_set.omega)
it_arr     = np.array(solution_set.iterations)
sl_arr     = np.array(solution_set.step_length)

print(f"\nSolution set: {len(omegas_arr)} points")
print(f"ω range: [{omegas_arr.min():.4f}, {omegas_arr.max():.4f}]")
print(f"  ω in [0.5, 1.0): {int(((omegas_arr >= 0.5) & (omegas_arr < 1.0)).sum())}")
print(f"  ω in [1.0, 1.5): {int(((omegas_arr >= 1.0) & (omegas_arr < 1.5)).sum())}")
print(f"  ω >= 1.5       : {int((omegas_arr >= 1.5).sum())}")

print(f"\nLast 10 continuation points:")
print(f"  ω             iter   step_length")
for i in range(max(0, len(omegas_arr) - 10), len(omegas_arr)):
    print(f"  {omegas_arr[i]:12.6f}   {it_arr[i]:3d}   {sl_arr[i]:.2e}")

plateau_idx = np.where((omegas_arr > 0.95) & (omegas_arr < 1.5))[0]
print(f"\nResidual norms in contact regime "
      f"({len(plateau_idx)} points, showing up to 10):")
if len(plateau_idx) > 0:
    step = max(1, len(plateau_idx) // 10)
    for i in plateau_idx[::step]:
        om  = solution_set.omega[i]
        x_i = FourierOmegaPoint(solution_set.fourier[i], om)
        r   = problem.compute_residue_RI(x_i)
        print(f"  ω = {om:.4f}   ||r|| = {np.linalg.norm(r):.3e}")
else:
    print("  (no points found in [0.95, 1.5] -- continuation halted at contact onset)")
