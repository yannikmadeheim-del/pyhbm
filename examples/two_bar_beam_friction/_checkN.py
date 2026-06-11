"""Confirm DLFT normal force & slip fraction scale with preload P5 (no bug)."""
import sys
from pathlib import Path
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
h0=HARM.index(0); Nh=Fourier.number_of_harmonics

for P5 in (1.5, 2.0, 3.0):
    p=BarBeamParams(P5=P5, poly_deg=POLY); sysD=BarBeamFrictionDLFT(p)
    M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    meth=DLFTFriction(epsilon_N=30.0,epsilon_T=30.0,mu=p.mu,g_zero=p.eps,n_tangential=1,n_sweep=1)
    prob=FBSProblem(sysD,NumericalFRF(M,C,K),meth)
    solver=HarmonicBalanceMethod(harmonics=HARM,freq_domain_ode=prob,
        corrector_parameterization=ArcLengthParameterization,predictor=TangentPredictorBordered)
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps
    ig=FourierOmegaPoint(Fourier(Q),0.40); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig,initial_reference_direction=rd,
        maximum_number_of_solutions=4000,angular_frequency_range=[0.40,1.15],
        solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.005,
            "maximum_step_length":0.01,"minimum_step_length":1e-9,"goal_number_of_iterations":4},
        jacobian_update_frequency=1,verbose=False)
    wh=np.array(ss.omega)
    # find peak q1
    pk=np.zeros(len(wh))
    for i,(f,w) in enumerate(zip(ss.fourier,wh)):
        full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
        pk[i]=np.abs(full.time_series[:,0,0]).max()
    ip=pk.argmax(); x=FourierOmegaPoint(ss.fourier[ip],wh[ip])
    lam=meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
    lf=Fourier(lam.reshape(Nh,2,1)); Fourier_Real.compute_time_series(lf)
    N=lf.time_series[:,0,0]; fT=lf.time_series[:,1,0]
    stick=sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))
    print(f"P5={P5}: peak q1={pk.max():.3f}@w={wh[ip]:.3f}  N_mean={N.mean():.3f} "
          f"muN={p.mu*N.mean():.3f}  |fT|max={np.abs(fT).max():.3f}  slip={1-stick/Nt:.2f}")
