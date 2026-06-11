"""Start far below resonance (fully stuck) and continue up through the peak.
Partial-slip variant (P5 raised so mu*N > P1). Cold, NO warm start."""
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
h0 = HARM.index(0); h1 = HARM.index(1); Nh = Fourier.number_of_harmonics


def stuck_guess(p, omega):
    Q = zeros((Nh, 2, 1), complex); Q[h0, 0, 0] = p.eps   # contact closed, no AC
    return FourierOmegaPoint(Fourier(Q), omega=omega)


def slip_fraction(x, method):
    method._get_lambda_corrected(x); J = x.contact_mask; Nt = J.shape[0]
    stick = sum(abs(J[k,1,1]-1) < 1e-9 for k in range(Nt))
    return 1.0 - stick / Nt


P5, EPS = 2.0, 1e4
p = BarBeamParams(P5=P5, poly_deg=POLY)
sysD = BarBeamFrictionDLFT(p)
M, C, K = sysD.mass_matrix, sysD.damping_matrix, sysD.stiffness_matrix

print("First-point cold solve (stuck guess) vs start frequency:")
for w0 in (0.3, 0.4, 0.5, 0.6, 0.7):
    prob = FBSProblem(sysD, NumericalFRF(M, C, K),
            DLFTFriction(epsilon_N=EPS, epsilon_T=EPS, mu=p.mu, g_zero=p.eps,
                         n_tangential=1, n_sweep=1))
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob)
    sol, it, ok, _ = solver.solve_fixed_frequency(
        stuck_guess(p, w0), maximum_iterations=200, absolute_tolerance=1e-7)
    sf = slip_fraction(sol, prob.method) if ok else float('nan')
    print(f"  w0={w0}: converged={ok} ({it} it), slip frac={sf:.2f}")

print("\nCold continuation from w0=0.4 up through resonance (eps sweep):")
for EPS in (1e3, 1e4, 1e5):
    prob = FBSProblem(sysD, NumericalFRF(M, C, K),
            DLFTFriction(epsilon_N=EPS, epsilon_T=EPS, mu=p.mu, g_zero=p.eps,
                         n_tangential=1, n_sweep=1))
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
            corrector_parameterization=ArcLengthParameterization,
            predictor=TangentPredictorBordered)
    rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2,1), complex), omega=-1.0)
    t0 = time()
    ss = solver.solve_and_continue(initial_guess=stuck_guess(p, 0.40),
        initial_reference_direction=rd, maximum_number_of_solutions=4000,
        angular_frequency_range=[0.40, 1.20],
        solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
        step_length_adaptation_kwargs={"base": 3.0, "initial_step_length": 0.005,
            "maximum_step_length": 0.01, "minimum_step_length": 1e-9,
            "goal_number_of_iterations": 4}, jacobian_update_frequency=1, verbose=False)
    wh = np.array(ss.omega)
    if len(wh):
        q1pk = max(np.abs(prob.compute_full_response(f, w).coefficients[:, 0, 0]).sum()
                   for f, w in zip(ss.fourier, wh))  # rough
        peak = 0.0; wpk = 0.0
        for f, w in zip(ss.fourier, wh):
            full = prob.compute_full_response(f, w); Fourier_Real.compute_time_series(full)
            a = np.abs(full.time_series[:, 0, 0]).max()
            if a > peak: peak, wpk = a, w
        print(f"  eps={EPS:.0e}: {len(wh)} pts, w in [{wh.min():.3f},{wh.max():.3f}], "
              f"peak q1={peak:.3f} @ w={wpk:.3f}  ({time()-t0:.1f}s)")
    else:
        print(f"  eps={EPS:.0e}: 0 pts")
