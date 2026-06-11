"""Scratch: AFT and DLFT continuation over the thesis frequency window."""
import sys
from pathlib import Path
from time import time
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, AFT, DLFTFriction, HarmonicBalanceMethod)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from systems import BarBeamParams, BarBeamFrictionDLFT, BarBeamFrictionAFT

HARM = list(range(0, 16))
POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
p = BarBeamParams(poly_deg=POLY)
h0 = HARM.index(0); h1 = HARM.index(1)

M = assemble = None
sysA = BarBeamFrictionAFT(p)
M, C, K = sysA.mass_matrix, sysA.damping_matrix, sysA.stiffness_matrix


def linear_guess(system, omega):
    """Open-contact linear relative response x_r = B u to the harmonic part of f_ext."""
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((6, 1), complex); F[0, 0] = p.P1          # P1 on q1 (1st harmonic)
    u = np.linalg.solve(Z, F)
    return (system.B_coupling @ u)[:, 0]


def run(system, method, label, omega_start=0.85, omega_end=1.09,
        init_step=0.01, max_step=0.02):
    prob = FBSProblem(system, NumericalFRF(M, C, K), method)
    solver = HarmonicBalanceMethod(
        harmonics=HARM, freq_domain_ode=prob,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)
    xr0 = linear_guess(system, omega_start)
    Q = zeros((Fourier.number_of_harmonics, 2, 1), complex)
    Q[h0, 0, 0] = p.eps                 # normal DC ~ gap (clamped in contact)
    Q[h1, 1, 0] = xr0[1]                # tangential 1st harmonic
    Q[h1, 0, 0] = xr0[0]
    ig = FourierOmegaPoint(Fourier(Q), omega=omega_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2, 1), complex), omega=1.0)
    t0 = time()
    ss = solver.solve_and_continue(
        initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=3000,
        angular_frequency_range=[omega_start, omega_end],
        solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
        step_length_adaptation_kwargs={
            "base": 3.0, "initial_step_length": init_step,
            "maximum_step_length": max_step, "minimum_step_length": 1e-8,
            "goal_number_of_iterations": 4},
        jacobian_update_frequency=1, verbose=False)
    wh = np.array(ss.omega)
    q1pk = np.zeros(len(wh)); q4pk = np.zeros(len(wh))
    for i, (f, w) in enumerate(zip(ss.fourier, wh)):
        full = prob.compute_full_response(f, w)
        Fourier_Real.compute_time_series(full)
        q1pk[i] = np.abs(full.time_series[:, 0, 0]).max()
        q4pk[i] = np.abs(full.time_series[:, 3, 0]).max()
    ip = q1pk.argmax()
    print(f"[{label}] {len(wh)} pts in {time()-t0:.1f}s, "
          f"w in [{wh.min():.3f},{wh.max():.3f}], "
          f"peak q1={q1pk.max():.3f} q4={q4pk[ip]:.3f} @ w={wh[ip]:.4f}")
    return ss, wh, q1pk, q4pk


print("=" * 60)
sysA = BarBeamFrictionAFT(p)
run(sysA, AFT(), "AFT-reg")

print("=" * 60)
sysD = BarBeamFrictionDLFT(p)
for eps in (1e2, 1e3, 1e4):
    c = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps,
                     n_tangential=1, n_sweep=1)
    try:
        run(sysD, c, f"DLFT eps={eps:.0e}")
    except Exception as e:
        print(f"[DLFT eps={eps:.0e}] FAILED: {e}")
