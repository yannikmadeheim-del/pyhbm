"""Verify a clean partial-slip DLFT config: moderate eps, cold STUCK start far
below resonance, arc-length continuation up through the peak. Compare to AFT."""
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
W0, W1 = 0.40, 1.15


def aft_branch(p):
    s = BarBeamFrictionAFT(p); M,C,K=s.mass_matrix,s.damping_matrix,s.stiffness_matrix
    prob = FBSProblem(s, NumericalFRF(M,C,K), AFT())
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
            corrector_parameterization=ArcLengthParameterization, predictor=TangentPredictorBordered)
    Z=-W0**2*M+1j*W0*C+K; F=zeros((6,1),complex); F[0,0]=p.P1
    xr0=(s.B_coupling@np.linalg.solve(Z,F))[:,0]
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps; Q[h1,0,0]=xr0[0]; Q[h1,1,0]=xr0[1]
    ig=FourierOmegaPoint(Fourier(Q),W0); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=4000, angular_frequency_range=[W0,W1],
        solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.01,
            "maximum_step_length":0.02,"minimum_step_length":1e-8,"goal_number_of_iterations":4},
        jacobian_update_frequency=1, verbose=False)
    return prob, np.array(ss.omega), ss.fourier


def peakcurve(prob, wh, fours):
    pk=np.zeros(len(wh))
    for i,(f,w) in enumerate(zip(fours,wh)):
        full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
        pk[i]=np.abs(full.time_series[:,0,0]).max()
    return pk


for P5 in (1.5, 2.0, 3.0):
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    probA, whA, fA = aft_branch(p); pkA = peakcurve(probA, whA, fA)
    iA = pkA.argmax()
    print(f"\nP5={P5} (mu*P5={0.1*P5:.2f}): AFT peak q1={pkA.max():.3f} @ w={whA[iA]:.3f}")
    sysD = BarBeamFrictionDLFT(p); M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    for EPS in (10.0, 30.0, 100.0):
        meth = DLFTFriction(epsilon_N=EPS, epsilon_T=EPS, mu=p.mu, g_zero=p.eps, n_tangential=1, n_sweep=1)
        prob = FBSProblem(sysD, NumericalFRF(M,C,K), meth)
        solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
                corrector_parameterization=ArcLengthParameterization, predictor=TangentPredictorBordered)
        Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps        # cold stuck seed
        ig=FourierOmegaPoint(Fourier(Q),W0); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
        t0=time()
        ss=solver.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
            maximum_number_of_solutions=4000, angular_frequency_range=[W0,W1],
            solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
            step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.005,
                "maximum_step_length":0.01,"minimum_step_length":1e-9,"goal_number_of_iterations":4},
            jacobian_update_frequency=1, verbose=False)
        wh=np.array(ss.omega)
        if len(wh) < 2:
            print(f"  eps={EPS:5.0f}: FAILED ({len(wh)} pts)"); continue
        pk=peakcurve(prob, wh, ss.fourier)
        cov=(wh.max()-wh.min())/(W1-W0)
        # compare peak to AFT (interp AFT onto DLFT freqs)
        order=np.argsort(whA)
        pkA_i=np.interp(wh, whA[order], pkA[order])
        rel=np.abs(pk-pkA_i)/(np.abs(pkA_i)+1e-12)
        print(f"  eps={EPS:5.0f}: {len(wh):4d} pts cov={cov*100:5.1f}% peak q1={pk.max():.3f} "
              f"@w={wh[pk.argmax()]:.3f}  vs AFT diff mean={rel.mean()*100:.2f}% max={rel.max()*100:.1f}%  ({time()-t0:.0f}s)")
