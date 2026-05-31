"""Test the theory-doc prescription: LARGE epsilon + relative/stagnation stop.

doc fbs_dlft_admittance.tex sec 5.1: eps >~ 1e6 (robust 1e8); the absolute
residual floor scales with eps, so accept on relative OR stagnation.
Warm-started downward omega-sweep; compare A1 / A_peak to the NLvib CSV.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np, pandas as pd
from pyhbm import (FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
                   FBSProblem, NumericalFRF, DLFTContact, HarmonicBalanceMethod)

class SDOF(FBS_System):
    is_real_valued = True
    def __init__(self,m=1.0,c=0.05,k=1.0,F0=0.02,poly_deg=100):
        self.mass_matrix=np.array([[m]]);self.damping_matrix=np.array([[c]])
        self.stiffness_matrix=np.array([[k]]);self.B_coupling=np.array([[1.0]])
        self.total_dimension=1;self.dimension=1;self.polynomial_degree=poly_deg;self.F0=F0
    def external_term(self,tau):
        f=np.zeros((len(tau),1,1));f[:,0,0]=self.F0*np.cos(tau);return f
    def interface_force(self,u,ud,t):return np.zeros((len(t),1,1))
    def jacobian_interface_force(self,u,ud,t):return np.zeros((len(t),1,1))
    def jacobian_interface_force_qdot(self,u,ud,t):return np.zeros((len(t),1,1))

PARAMS=dict(m=1.0,c=0.05,k=1.0,F0=0.02);GAP=0.1

def newton(problem,x0,max_iter=200,rtol=1e-4,stag=1e-8,abstol=1e-9):
    x=np.asarray(x0).copy();omega=x0.omega;r0=None
    for it in range(max_iter):
        xp=FourierOmegaPoint(Fourier.new_from_RI(x[:-1]),omega)
        r=problem.compute_residue_RI(xp);rn=np.linalg.norm(r)
        if r0 is None:r0=rn
        if rn<abstol:return xp,it,True,rn/r0
        if rn<rtol*r0:return xp,it,True,rn/r0
        J=problem.compute_jacobian_of_residue_RI(xp)
        delta=np.linalg.solve(J,r)
        if np.linalg.norm(delta)<stag*max(np.linalg.norm(x[:-1]),1e-12):
            return xp,it,True,rn/r0
        # backtracking
        alpha=1.0;ok=False
        for _ in range(20):
            xt=x.copy();xt[:-1]=x[:-1]-alpha*delta
            rt=np.linalg.norm(problem.compute_residue_RI(
                FourierOmegaPoint(Fourier.new_from_RI(xt[:-1]),omega)))
            if rt<rn:x=xt;ok=True;break
            alpha*=0.5
        if not ok:x[:-1]=x[:-1]-delta
    return xp,max_iter,False,rn/r0

def peak_amp(problem,xp):
    full=problem.compute_full_response(xp.fourier,xp.omega)
    Nt=Fourier.number_of_time_samples
    A1=2.0/Nt*abs(full.coefficients[1,0,0])
    Fourier_Real.compute_time_series(full)
    return A1,float(np.max(np.abs(full.time_series[:,0,0])))

df=pd.read_csv(Path(__file__).parent/"nlvib_sdof_vibroimpact_shooting_frc.csv")

def n_active(problem,xp):
    problem.method._get_lambda_corrected(xp)
    return int(xp.contact_mask.sum())

for Nh in [9, 30]:
  for eps in [1.0, 1e6, 1e8]:
    harmonics=list(range(0,Nh))
    system=SDOF(**PARAMS)
    HarmonicBalanceMethod.update_dependencies(harmonics,system.polynomial_degree)
    provider=NumericalFRF(system.mass_matrix,system.damping_matrix,system.stiffness_matrix)
    problem=FBSProblem(system,provider,DLFTContact(epsilon=eps,g_zero=GAP))
    # sweep UP along resonance, like the real continuation
    omegas=np.arange(0.50,1.52,0.02)
    Z=-omegas[0]**2+1j*omegas[0]*PARAMS['c']+PARAMS['k']
    x=FourierOmegaPoint.new_from_first_harmonic(np.array([[PARAMS['F0']/Z]]),omega=omegas[0])
    nok=0;rows=[]
    for om in omegas:
        x=FourierOmegaPoint(x.fourier,om)
        xs,it,ok,relr=newton(problem,x)
        if ok:
            nok+=1
            A1,Ap=peak_amp(problem,xs)
            na=n_active(problem,xs)
            ref=df.iloc[(df.omega-om).abs().idxmin()]
            rows.append((om,it,na,A1,Ap,ref.A1,ref.A_peak))
            x=xs
        else:
            rows.append((om,it,-1,np.nan,np.nan,np.nan,np.nan))
    print(f"\n=== Nh={Nh} eps={eps:.0e}: {nok}/{len(omegas)} converged ===")
    print("  omega   it  active   A1      A_peak  | nlvib A1  A_peak")
    for om,it,na,A1,Ap,rA1,rAp in rows[::2]:
        print(f"  {om:.3f}  {it:3d}  {na:5d}  {A1:.4f}  {Ap:.4f}  | {rA1:.4f}   {rAp:.4f}")
