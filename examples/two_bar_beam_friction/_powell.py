"""Does the DLFT gross-slip solution exist & is it reachable WITHOUT the exact
analytic tangent?  Solve compute_residue_RI with scipy Powell-hybrid (MINPACK
hybr, Nacivet's solver) and Broyden, started from the AFT state. Compare to
analytic-Newton (which we know stalls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros
from scipy.optimize import root

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, AFT, DLFTFriction, HarmonicBalanceMethod)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from systems import BarBeamParams, BarBeamFrictionDLFT, BarBeamFrictionAFT

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1); Nh = Fourier.number_of_harmonics


def aft_branch(p):
    sysA = BarBeamFrictionAFT(p); M,C,K = sysA.mass_matrix, sysA.damping_matrix, sysA.stiffness_matrix
    prob = FBSProblem(sysA, NumericalFRF(M,C,K), AFT())
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
            corrector_parameterization=ArcLengthParameterization, predictor=TangentPredictorBordered)
    Z=-0.85**2*M+1j*0.85*C+K; F=zeros((6,1),complex); F[0,0]=p.P1
    xr0=(sysA.B_coupling@np.linalg.solve(Z,F))[:,0]
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps; Q[h1,0,0]=xr0[0]; Q[h1,1,0]=xr0[1]
    ig=FourierOmegaPoint(Fourier(Q),0.85); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=3000, angular_frequency_range=[0.85,1.09],
        solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.01,
            "maximum_step_length":0.02,"minimum_step_length":1e-8,"goal_number_of_iterations":4},
        jacobian_update_frequency=1, verbose=False)
    return prob, np.array(ss.omega), ss.fourier


def test(P5, eps=1e4):
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    probA, wh, fours = aft_branch(p)
    sysD = BarBeamFrictionDLFT(p); M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    meth = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps, n_tangential=1, n_sweep=1)
    probD = FBSProblem(sysD, NumericalFRF(M,C,K), meth)
    print(f"\n=== P5={P5} (mu*P5={p.mu*P5:.2f} vs P1={p.P1}) ===")
    for wt in (0.86, 0.93, 0.97, 0.99, 1.02):
        i = int(np.argmin(np.abs(wh-wt))); w = wh[i]
        x0 = FourierOmegaPoint(fours[i], w)
        rstart = np.linalg.norm(probD.compute_residue_RI(x0))
        x0v = np.asarray(x0)[:-1, 0].copy()
        def fun(v):
            return probD.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(v.reshape(-1,1)), w)).ravel()
        out = {}
        for method in ("hybr", "broyden1"):
            try:
                sol = root(fun, x0v, method=method,
                           options=({"maxfev":4000} if method=="hybr" else {"maxiter":2000}))
                rn = np.linalg.norm(sol.fun)
                out[method] = ("OK" if rn < 1e-6 else "no") + f"({rn:.0e})"
            except Exception as e:
                out[method] = f"err"
        # slip frac of AFT state
        meth._get_lambda_corrected(x0); J=x0.contact_mask; Nt=J.shape[0]
        stick=sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt)); slipf=1-stick/Nt
        print(f"  w={w:.3f} slip={slipf:.2f} |r0|={rstart:.1e}  "
              f"hybr={out['hybr']:14s} broyden1={out['broyden1']}")


test(0.4)
test(2.0)
