"""Cold-start DLFT continuation, sweeping epsilon. NO warm starts.

Run A: thesis params (gross slip, mu*P5 = 0.04 < P1 = 0.1).
Run B: raised preload P5 (partial slip, mu*P5 > P1) -> full-branch DLFT.
Each branch is compared against the AFT-regularized reference at the SAME params.
"""
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

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1)
W0, W1 = 0.85, 1.09


def linear_guess(system, p, omega):
    M, C, K = system.mass_matrix, system.damping_matrix, system.stiffness_matrix
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((6, 1), complex); F[0, 0] = p.P1
    return (system.B_coupling @ np.linalg.solve(Z, F))[:, 0]


def cold_continuation(system, method, p, label):
    M, C, K = system.mass_matrix, system.damping_matrix, system.stiffness_matrix
    prob = FBSProblem(system, NumericalFRF(M, C, K), method)
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
            corrector_parameterization=ArcLengthParameterization,
            predictor=TangentPredictorBordered)
    xr0 = linear_guess(system, p, W0)
    Q = zeros((Fourier.number_of_harmonics, 2, 1), complex)
    Q[h0, 0, 0] = p.eps; Q[h1, 0, 0] = xr0[0]; Q[h1, 1, 0] = xr0[1]
    ig = FourierOmegaPoint(Fourier(Q), omega=W0)
    rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2, 1), complex), omega=1.0)
    t0 = time()
    ss = solver.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=3000, angular_frequency_range=[W0, W1],
        solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
        step_length_adaptation_kwargs={"base": 3.0, "initial_step_length": 0.01,
            "maximum_step_length": 0.02, "minimum_step_length": 1e-8,
            "goal_number_of_iterations": 4}, jacobian_update_frequency=1, verbose=False)
    wh = np.array(ss.omega)
    cov = (wh.max() - wh.min()) / (W1 - W0) if len(wh) else 0.0
    q1pk = 0.0
    if len(wh):
        for f, w in zip(ss.fourier, wh):
            full = prob.compute_full_response(f, w); Fourier_Real.compute_time_series(full)
            q1pk = max(q1pk, np.abs(full.time_series[:, 0, 0]).max())
    print(f"  [{label}] {len(wh):4d} pts, coverage {cov*100:5.1f}%, "
          f"w in [{wh.min():.3f},{wh.max():.3f}], peak q1={q1pk:.3f}  ({time()-t0:.1f}s)"
          if len(wh) else f"  [{label}] 0 pts (failed at start)")
    return ss, prob


def run_case(P5, tag):
    print("=" * 64)
    print(f"{tag}:  P5={P5}  (mu*P5={0.1*P5:.3f} vs P1=0.1 -> "
          f"{'PARTIAL' if 0.1*P5 > 0.1 else 'GROSS'} slip)")
    print("=" * 64)
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    print(" AFT-regularized reference:")
    cold_continuation(BarBeamFrictionAFT(p), AFT(), p, "AFT")
    print(" DLFT (cold, epsilon sweep):")
    sysD = BarBeamFrictionDLFT(p)
    for eps in (1e3, 1e4, 1e5, 1e6, 1e7):
        c = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps,
                         n_tangential=1, n_sweep=1)
        cold_continuation(sysD, c, p, f"DLFT eps={eps:.0e}")


run_case(0.4, "RUN A  thesis gross slip")
run_case(2.0, "RUN B  raised preload, partial slip")
