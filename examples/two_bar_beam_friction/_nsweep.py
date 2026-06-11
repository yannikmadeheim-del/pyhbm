"""Does n_sweep>1 (corrector periodicity passes) let DLFT cross into partial slip?
Also: does adding a small tangential contact compliance k_T (Nacivet-style)
restore the slip-regime tangent and let the FULL thesis gross-slip converge?"""
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

HARM=list(range(0,16)); POLY=20
HarmonicBalanceMethod.update_dependencies(HARM,POLY)
h0=HARM.index(0); Nh=Fourier.number_of_harmonics
W0,W1=0.40,1.15

def run(P5,P1,nsweep,EPS=30.0,istep=0.003,mstep=0.006):
    p=BarBeamParams(P5=P5,P1=P1,poly_deg=POLY); sysD=BarBeamFrictionDLFT(p)
    M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    meth=DLFTFriction(epsilon_N=EPS,epsilon_T=EPS,mu=p.mu,g_zero=p.eps,n_tangential=1,n_sweep=nsweep)
    prob=FBSProblem(sysD,NumericalFRF(M,C,K),meth)
    solver=HarmonicBalanceMethod(harmonics=HARM,freq_domain_ode=prob,
        corrector_parameterization=ArcLengthParameterization,predictor=TangentPredictorBordered)
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps
    ig=FourierOmegaPoint(Fourier(Q),W0); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig,initial_reference_direction=rd,
        maximum_number_of_solutions=6000,angular_frequency_range=[W0,W1],
        solver_kwargs={"maximum_iterations":300,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":istep,
            "maximum_step_length":mstep,"minimum_step_length":1e-10,"goal_number_of_iterations":4},
        jacobian_update_frequency=1,verbose=False)
    wh=np.array(ss.omega)
    if len(wh)<2: return None
    pk=np.zeros(len(wh)); sf=np.zeros(len(wh))
    for i,(f,w) in enumerate(zip(ss.fourier,wh)):
        full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
        pk[i]=np.abs(full.time_series[:,0,0]).max()
        x=FourierOmegaPoint(f,w); meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
        sf[i]=1-sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))/Nt
    ip=pk.argmax()
    return len(wh),(wh.max()-wh.min())/(W1-W0),pk.max(),wh[ip],sf[ip],sf.max()

print("n_sweep test (P1=0.1, partial-slip-edge P5):")
for P5 in (0.6,0.7):
    for ns in (1,3,8,15):
        r=run(P5,0.1,ns)
        s=(f"{r[0]:4d}pts cov={r[1]*100:5.1f}% peak={r[2]:.3f} slip@pk={r[4]:.2f} slipmax={r[5]:.2f}"
           if r else "FAILED")
        print(f"  P5={P5} n_sweep={ns:2d}: {s}")
