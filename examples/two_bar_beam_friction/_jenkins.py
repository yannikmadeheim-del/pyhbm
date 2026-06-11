"""Does a tangential contact compliance k_T (Jenkins element) let DLFT converge
in genuine slip for the thesis system? Add a massless asperity node tied to q1
by k_T; friction acts between asperity and q4. Rigid Coulomb is k_T->inf."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from numpy import zeros, cos
from pyhbm import (FBS_System, Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, DLFTFriction, HarmonicBalanceMethod)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from systems import BarBeamParams, assemble, Q1,Q2,Q4,Q5

HARM=list(range(0,16)); POLY=20
HarmonicBalanceMethod.update_dependencies(HARM,POLY)
h0=HARM.index(0); Nh=Fourier.number_of_harmonics
W0,W1=0.40,1.15

class CompliantDLFT(FBS_System):
    is_real_valued=True
    def __init__(self,p,k_T):
        M6,C6,K6=assemble(p)
        d=7; A=6                       # asperity node index
        M=zeros((d,d)); C=zeros((d,d)); K=zeros((d,d))
        M[:6,:6]=M6; C[:6,:6]=C6; K[:6,:6]=K6
        # tangential spring k_T between q1 and asperity A
        K[Q1,Q1]+=k_T; K[A,A]+=k_T; K[Q1,A]-=k_T; K[A,Q1]-=k_T
        self.p=p; self.A=A
        self.omega_ref=1.0
        self.mass_matrix=M; self.damping_matrix=C; self.stiffness_matrix=K
        B=zeros((2,d))
        B[0,Q5]=+1.0; B[0,Q2]=-1.0     # normal x_r^N = q5-q2
        B[1,A]=+1.0;  B[1,Q4]=-1.0     # tangential x_r^T = u_asperity - q4
        self.B_coupling=B
        self.dimension=2; self.total_dimension=d; self.polynomial_degree=p.poly_deg
    def external_term(self,tau):
        f=zeros((len(tau),self.total_dimension,1))
        f[:,Q1,0]=self.p.P1*cos(tau); f[:,Q2,0]=-self.p.P5; f[:,Q5,0]=+self.p.P5
        return f
    interface_force=lambda self,u,ud,tau: zeros((len(tau),2,1))
    jacobian_interface_force=lambda self,u,ud,tau: zeros((len(tau),2,2))
    jacobian_interface_force_qdot=lambda self,u,ud,tau: zeros((len(tau),2,2))

def run(P5,P1,k_T,EPS=30.0):
    p=BarBeamParams(P5=P5,P1=P1,poly_deg=POLY); sysD=CompliantDLFT(p,k_T)
    M,C,K=sysD.mass_matrix,sysD.damping_matrix,sysD.stiffness_matrix
    meth=DLFTFriction(epsilon_N=EPS,epsilon_T=EPS,mu=p.mu,g_zero=p.eps,n_tangential=1,n_sweep=1)
    prob=FBSProblem(sysD,NumericalFRF(M,C,K),meth)
    solver=HarmonicBalanceMethod(harmonics=HARM,freq_domain_ode=prob,
        corrector_parameterization=ArcLengthParameterization,predictor=TangentPredictorBordered)
    Q=zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps
    ig=FourierOmegaPoint(Fourier(Q),W0); rd=FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex),1.0)
    ss=solver.solve_and_continue(initial_guess=ig,initial_reference_direction=rd,
        maximum_number_of_solutions=6000,angular_frequency_range=[W0,W1],
        solver_kwargs={"maximum_iterations":300,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.003,
            "maximum_step_length":0.006,"minimum_step_length":1e-10,"goal_number_of_iterations":4},
        jacobian_update_frequency=1,verbose=False)
    wh=np.array(ss.omega)
    if len(wh)<2: return None
    pk=np.zeros(len(wh)); sf=np.zeros(len(wh))
    for i,(f,w) in enumerate(zip(ss.fourier,wh)):
        full=prob.compute_full_response(f,w); Fourier_Real.compute_time_series(full)
        pk[i]=np.abs(full.time_series[:,Q1,0]).max()
        x=FourierOmegaPoint(f,w); meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
        sf[i]=1-sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))/Nt
    ip=pk.argmax()
    return len(wh),(wh.max()-wh.min())/(W1-W0),pk.max(),wh[ip],sf[ip],sf.max()

print("Compliant DLFT (Jenkins k_T), thesis gross slip P5=0.4 P1=0.1:")
for k_T in (5.0,20.0,100.0,500.0):
    r=run(0.4,0.1,k_T)
    s=(f"{r[0]:4d}pts cov={r[1]*100:5.1f}% peak q1={r[2]:.3f}@w={r[3]:.3f} slip@pk={r[4]:.2f} slipmax={r[5]:.2f}"
       if r else "FAILED")
    print(f"  k_T={k_T:6.1f}: {s}")
