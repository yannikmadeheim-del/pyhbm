"""Moderate epsilon + load-ramp homotopy (P1: 0 -> thesis) at a low frequency.
If the first point lands, run frequency continuation. Thesis gross slip P5=0.4."""
import sys
from pathlib import Path
from time import time
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, DLFTFriction, HarmonicBalanceMethod)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from systems import BarBeamParams, BarBeamFrictionDLFT

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1); Nh = Fourier.number_of_harmonics


def slip_frac(x, meth):
    meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
    return 1 - sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))/Nt


def load_ramp(prob, sysD, meth, w0, P1_target, nsteps=20):
    """Solve DLFT at w0, ramping P1 from 0 to target, warm-stepping in P1 only."""
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob)
    Q = zeros((Nh,2,1),complex); Q[h0,0,0]=sysD.p.eps
    x = FourierOmegaPoint(Fourier(Q), w0)
    for s in range(1, nsteps+1):
        sysD.p.P1 = P1_target * s/nsteps
        prob.external_term = Fourier_Real.new_from_time_series(
            sysD.external_term(Fourier.adimensional_time_samples))
        prob.F_ext_full = np.vstack(prob.external_term.coefficients)
        prob.F_ext_full_RI = np.vstack((prob.F_ext_full.real, prob.F_ext_full.imag))
        sol, it, ok, _ = solver.solve_fixed_frequency(x, maximum_iterations=200, absolute_tolerance=1e-8)
        if not ok:
            return None, s
        x = sol
    return x, nsteps


for EPS in (10.0, 30.0, 100.0, 300.0):
    p = BarBeamParams(P5=0.4, poly_deg=POLY)
    sysD = BarBeamFrictionDLFT(p)
    M,C,K = sysD.mass_matrix, sysD.damping_matrix, sysD.stiffness_matrix
    meth = DLFTFriction(epsilon_N=EPS, epsilon_T=EPS, mu=p.mu, g_zero=p.eps, n_tangential=1, n_sweep=1)
    prob = FBSProblem(sysD, NumericalFRF(M,C,K), meth)
    x0, reached = load_ramp(prob, sysD, meth, w0=0.5, P1_target=0.1, nsteps=20)
    if x0 is None:
        print(f"eps={EPS:6.0f}: load-ramp FAILED at step {reached}/20")
        continue
    sf = slip_frac(x0, meth)
    print(f"eps={EPS:6.0f}: load-ramp OK, w=0.5 slip={sf:.2f}, |r|="
          f"{np.linalg.norm(prob.compute_residue_RI(x0)):.1e}", end="")
    # frequency continuation up + down from this seed
    cov_total = 0.0; npts = 0; peak=0.0; wpk=0.0
    for direction in (+1.0, -1.0):
        solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
                corrector_parameterization=ArcLengthParameterization, predictor=TangentPredictorBordered)
        rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex), omega=direction)
        ss = solver.solve_and_continue(initial_guess=x0, initial_reference_direction=rd,
            maximum_number_of_solutions=2000, angular_frequency_range=[0.85,1.09],
            solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
            step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.005,
                "maximum_step_length":0.01,"minimum_step_length":1e-9,"goal_number_of_iterations":4},
            jacobian_update_frequency=1, verbose=False)
        wh=np.array(ss.omega); npts += len(wh)
        for f,w in zip(ss.fourier, wh):
            full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
            a=np.abs(full.time_series[:,0,0]).max()
            if a>peak: peak,wpk=a,w
    print(f"  -> continuation {npts} pts, peak q1={peak:.3f} @ w={wpk:.3f}")
