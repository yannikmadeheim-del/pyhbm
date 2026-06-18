"""Developer checks for the dry-friction-damper beam example (standalone).

  1. Finite-difference verification of the DLFTFriction residual Jacobian and
     dr/domega at a partial-slip point.  The analytical tangent drops the friction
     history coupling, so the Jacobian rel-err is expected to be MODERATE (not
     ~1e-6); dr/domega should be small.
  2. epsilon-independence: re-solve at the resonance peak for eps/10, eps, 10*eps
     and confirm the converged peak amplitude is unchanged.

Rebuilds the same problem as main.py.  Run:  python checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTFriction, HarmonicBalanceMethod,
)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from dynamical_system import BeamFrictionDamper


# --- rebuild the same problem main.py uses ---------------------------------
MU       = 0.1
XI       = 0.01
K_T      = 2.4e7
K_N      = 2.4e3
PRELOAD  = 1500.0
EPSILON  = 5.0e7
HARMONICS = list(range(0, 21))
POLY_DEG  = 20
AMP_FACTOR = 3.0
SLIP_THRESHOLD = MU * PRELOAD / K_T

HarmonicBalanceMethod.update_dependencies(HARMONICS, POLY_DEG)
beam = BeamFrictionDamper(k_T=K_T, k_N=K_N, preload=PRELOAD, F0=1.0,
                          xi=XI, poly_deg=POLY_DEG)
OMEGA_REF = beam.omega_ref
M, C, K = beam.mass_matrix, beam.damping_matrix, beam.stiffness_matrix


def stuck_response(omega_hat, F0=1.0):
    """Linear stuck+in-contact FRF (damper springs grounding the contact node)."""
    Z = -omega_hat**2 * M + 1j * omega_hat * C + K
    Z[beam.beam_trans_damper, beam.beam_trans_damper] += K_T
    Z[beam.beam_axial_damper, beam.beam_axial_damper] += K_N
    F = zeros((beam.total_dimension, 1), complex); F[beam.beam_trans_tip, 0] = F0
    return np.linalg.solve(Z, F)


# size F0 so the stuck peak motion at 0.3L = AMP_FACTOR * slip threshold
wh_scan = np.linspace(0.6, 1.4, 200)
stuck_peak = max(abs(stuck_response(w)[beam.beam_trans_damper, 0]) for w in wh_scan)
beam.F0 = AMP_FACTOR * SLIP_THRESHOLD / stuck_peak

provider = NumericalFRF(M, C, K)
contact  = DLFTFriction(epsilon_N=EPSILON, epsilon_T=EPSILON, mu=MU,
                        g_zero=0.0, n_tangential=1, n_sweep=1)
problem  = FBSProblem(beam, provider, contact)
h0 = HARMONICS.index(0); h1 = HARMONICS.index(1)


def linear_relative(omega_hat):
    """Open-contact linear relative response x_r = B u (continuation seed)."""
    Z = -omega_hat**2 * M + 1j * omega_hat * C + K
    F = zeros((beam.total_dimension, 1), complex); F[beam.beam_trans_tip, 0] = beam.F0
    return (beam.B_coupling @ np.linalg.solve(Z, F))[:, 0]


# ============================ FD checks =====================================
print("=" * 70)
print("Finite-difference checks of DLFT-friction residual Jacobian and dr/domega")
print("=" * 70)
Q = zeros((Fourier.number_of_harmonics, beam.dimension, 1), complex)
Q[h0, 0, 0] = -SLIP_THRESHOLD * 0.5          # DC normal (pressed in)
Q[h1, 1, 0] =  SLIP_THRESHOLD * 0.3          # 1st-harmonic tangential (partial slip)
x_fd = FourierOmegaPoint(Fourier(Q), omega=1.0)
r0 = problem.compute_residue_RI(x_fd)
J_an = problem.compute_jacobian_of_residue_RI(x_fd)
n_q = J_an.shape[0]; h = 1.0e-10
J_fd = zeros((n_q, n_q)); x_arr = np.asarray(x_fd).copy()
for j in range(n_q):
    xp = x_arr.copy(); xp[j, 0] += h
    J_fd[:, j] = (problem.compute_residue_RI(
        FourierOmegaPoint(Fourier.new_from_RI(xp[:-1]), x_fd.omega)) - r0).ravel() / h
relJ = np.abs(J_an - J_fd).max() / (np.abs(J_fd).max() + 1e-30)
print(f"  Jacobian : rel err = {relJ:.2e}  (history-dropped tangent)")
dw_an = problem.compute_derivative_wrt_omega_RI(x_fd)
hw = 1e-4
dw_fd = (problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega + hw))
         - problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega - hw))) / (2*hw)
relW = np.abs(dw_an - dw_fd).max() / (np.abs(dw_fd).max() + 1e-30)
print(f"  dr/domega: rel err = {relW:.2e}")
print("=" * 70)


# ============================ continuation (to reach the peak) ==============
solver = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=problem,
    corrector_parameterization=ArcLengthParameterization,
    predictor=TangentPredictorBordered)
OMEGA_START, OMEGA_END = 0.70, 1.30
xr0 = linear_relative(OMEGA_START)
Q_ig = zeros((Fourier.number_of_harmonics, beam.dimension, 1), complex)
Q_ig[h1, 0, 0] = xr0[0]; Q_ig[h1, 1, 0] = xr0[1]
ig = FourierOmegaPoint(Fourier(Q_ig), omega=OMEGA_START)
rd = FourierOmegaPoint.new_from_first_harmonic(
    zeros((beam.dimension, 1), complex), omega=1.0)
ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=2000,
    angular_frequency_range=[OMEGA_START, OMEGA_END],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={
        "base": 4.0, "initial_step_length": 0.008, "maximum_step_length": 0.015,
        "minimum_step_length": 1e-7, "goal_number_of_iterations": 10},
    jacobian_update_frequency=1)
wh = np.array(ss.omega)
peak_inf = np.zeros(len(wh))
for idx, (f, w) in enumerate(zip(ss.fourier, wh)):
    full = problem.compute_full_response(f, w)
    Fourier_Real.compute_time_series(full)
    peak_inf[idx] = np.abs(full.time_series[:, beam.beam_trans_damper, 0]).max()
i_peak = int(peak_inf.argmax())


# ============================ epsilon-independence ==========================
print("epsilon-independence (peak |u(0.3L)| at resonance, +/- 1 decade):")
for eps in (EPSILON / 10, EPSILON, EPSILON * 10):
    c = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=MU, g_zero=0.0,
                     n_tangential=1, n_sweep=1)
    p = FBSProblem(beam, NumericalFRF(M, C, K), c)
    s = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=p,
                              corrector_parameterization=ArcLengthParameterization,
                              predictor=TangentPredictorBordered)
    sol, _, ok, _ = s.solve_fixed_frequency(
        FourierOmegaPoint(ss.fourier[i_peak], wh[i_peak]),
        maximum_iterations=80, absolute_tolerance=1e-7)
    if ok:
        full = p.compute_full_response(sol.fourier, sol.omega)
        Fourier_Real.compute_time_series(full)
        amp = np.abs(full.time_series[:, beam.beam_trans_damper, 0]).max()
        print(f"  eps = {eps:.1e}: peak = {amp*1e6:.3f} um  (converged)")
    else:
        print(f"  eps = {eps:.1e}: did not converge")
