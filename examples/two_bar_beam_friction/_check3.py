"""Scratch: warm-start DLFT from the converged AFT branch, point by point."""
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
p = BarBeamParams(poly_deg=POLY)
h0 = HARM.index(0); h1 = HARM.index(1)
sysA = BarBeamFrictionAFT(p); sysD = BarBeamFrictionDLFT(p)
M, C, K = sysA.mass_matrix, sysA.damping_matrix, sysA.stiffness_matrix


def linear_guess(system, omega):
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((6, 1), complex); F[0, 0] = p.P1
    return (system.B_coupling @ np.linalg.solve(Z, F))[:, 0]


# --- AFT continuation (reference branch) ---
probA = FBSProblem(sysA, NumericalFRF(M, C, K), AFT())
solA = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=probA,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)
xr0 = linear_guess(sysA, 0.85)
Q = zeros((Fourier.number_of_harmonics, 2, 1), complex)
Q[h0, 0, 0] = p.eps; Q[h1, 1, 0] = xr0[1]; Q[h1, 0, 0] = xr0[0]
ig = FourierOmegaPoint(Fourier(Q), omega=0.85)
rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2, 1), complex), omega=1.0)
ssA = solA.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=3000, angular_frequency_range=[0.85, 1.09],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={"base": 3.0, "initial_step_length": 0.01,
        "maximum_step_length": 0.02, "minimum_step_length": 1e-8,
        "goal_number_of_iterations": 4}, jacobian_update_frequency=1, verbose=False)
whA = np.array(ssA.omega)
print(f"AFT branch: {len(whA)} pts")

# --- DLFT point-by-point, warm-started from AFT at same frequency ---
for eps in (1e2, 1e3, 1e4):
    c = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps,
                     n_tangential=1, n_sweep=1)
    probD = FBSProblem(sysD, NumericalFRF(M, C, K), c)
    solD = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=probD)
    # sample a few frequencies across the branch
    nok = 0; ntot = 0; errs = []
    idxs = np.linspace(0, len(whA)-1, 25).astype(int)
    for i in idxs:
        ntot += 1
        ig = FourierOmegaPoint(ssA.fourier[i], whA[i])
        sol, it, ok, _ = solD.solve_fixed_frequency(
            ig, maximum_iterations=300, absolute_tolerance=1e-7)
        if ok:
            nok += 1
            fullA = probA.compute_full_response(ssA.fourier[i], whA[i])
            fullD = probD.compute_full_response(sol.fourier, sol.omega)
            Fourier_Real.compute_time_series(fullA); Fourier_Real.compute_time_series(fullD)
            a = np.abs(fullA.time_series[:, 0, 0]).max()
            d = np.abs(fullD.time_series[:, 0, 0]).max()
            errs.append(abs(a-d)/a)
    me = (np.mean(errs)*100 if errs else float('nan'))
    print(f"DLFT eps={eps:.0e}: converged {nok}/{ntot} pts, "
          f"mean |q1| rel diff vs AFT = {me:.2f}%")
