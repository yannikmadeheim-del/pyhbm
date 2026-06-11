"""Scan preload P5 for the partial-slip sweet spot: DLFT converges full branch
AND shows moderate slip at the peak. Also try raising forcing P1."""
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

def run(P5,P1,EPS=30.0):
    p=BarBeamParams(P5=P5,P1=P1,poly_deg=POLY); sysD=BarBeamFrictionDLFT(p)
    M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    meth=DLFTFriction(epsilon_N=EPS,epsilon_T=EPS,mu=p.mu,g_zero=p.eps,n_tangential=1,n_sweep=1)
    prob=FBSProblem(sysD,NumericalFRF(M,C,K),meth)
    solver=HarmonicBalanceMethod(harmonics=HARM,freq_domain_ode=prob,
        corrector_parameterization=ArcLengthParameterization,predictor=TangentPredictorBordered)
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps
    ig=FourierOmegaPoint(Fourier(Q),W0); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig,initial_reference_direction=rd,
        maximum_number_of_solutions=5000,angular_frequency_range=[W0,W1],
        solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.004,
            "maximum_step_length":0.008,"minimum_step_length":1e-9,"goal_number_of_iterations":4},
        jacobian_update_frequency=1,verbose=False)
    wh=np.array(ss.omega)
    if len(wh)<2: return None
    pk=np.zeros(len(wh)); sf=np.zeros(len(wh))
    for i,(f,w) in enumerate(zip(ss.fourier,wh)):
        full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
        pk[i]=np.abs(full.time_series[:,0,0]).max()
        x=FourierOmegaPoint(f,w); meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
        sf[i]=1-sum(abs(J[k,1,1]-1)<1e-9 and not np.all(np.abs(J[k])<1e-30) for k in range(Nt))/Nt
    ip=pk.argmax(); cov=(wh.max()-wh.min())/(W1-W0)
    return len(wh),cov,pk.max(),wh[ip],sf[ip],sf.max()

print("P1=0.1 (thesis), scan P5:")
for P5 in (0.5,0.6,0.7,0.8,1.0,1.2):
    r=run(0.0 if False else P5,0.1)
    if r: print(f"  P5={P5} muN={0.1*P5:.3f}: {r[0]:4d}pts cov={r[1]*100:5.1f}% peak q1={r[2]:.3f}@w={r[3]:.3f} slip@peak={r[4]:.2f} slipmax={r[5]:.2f}")
    else: print(f"  P5={P5}: FAILED")

print("Raise forcing P1=0.2, scan P5 (wider slip):")
for P5 in (0.8,1.0,1.2,1.5):
    r=run(P5,0.2)
    if r: print(f"  P5={P5} muN={0.1*P5:.3f}: {r[0]:4d}pts cov={r[1]*100:5.1f}% peak q1={r[2]:.3f}@w={r[3]:.3f} slip@peak={r[4]:.2f} slipmax={r[5]:.2f}")
    else: print(f"  P5={P5}: FAILED")
