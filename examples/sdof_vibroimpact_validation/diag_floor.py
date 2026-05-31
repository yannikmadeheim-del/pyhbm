"""Isolate (1) convergence-criterion floor and (2) the omega~1.25 wall, at eps=1.

Up-sweep along resonance. Report ABSOLUTE ||r|| at the accepted point, the
relative drop, and active-set size. Vary Nh. Accept on relative+stagnation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import numpy as np, pandas as pd
from pyhbm import (FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
                   FBSProblem, NumericalFRF, DLFTContact, HarmonicBalanceMethod)

class SDOF(FBS_System):
    is_real_valued=True
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

def newton(problem,x0,max_iter=300,rtol=1e-6,stag=1e-10,abstol=1e-9):
    x=np.asarray(x0).copy();omega=x0.omega;r0=None;why=""
    for it in range(max_iter):
        xp=FourierOmegaPoint(Fourier.new_from_RI(x[:-1]),omega)
        r=problem.compute_residue_RI(xp);rn=np.linalg.norm(r)
        if r0 is None:r0=rn
        if rn<abstol:return xp,it,True,rn,rn/r0,"abs"
        if rn<rtol*r0:return xp,it,True,rn,rn/r0,"rel"
        J=problem.compute_jacobian_of_residue_RI(xp)
        delta=np.linalg.solve(J,r)
        if np.linalg.norm(delta)<stag*max(np.linalg.norm(x[:-1]),1e-12):
            return xp,it,True,rn,rn/r0,"stag"
        alpha=1.0;ok=False
        for _ in range(25):
            xt=x.copy();xt[:-1]=x[:-1]-alpha*delta
            rt=np.linalg.norm(problem.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(xt[:-1]),omega)))
            if rt<rn:x=xt;ok=True;break
            alpha*=0.5
        if not ok:x[:-1]=x[:-1]-delta
    return xp,max_iter,False,rn,rn/r0,"FAIL"

def peak(problem,xp):
    full=problem.compute_full_response(xp.fourier,xp.omega);Nt=Fourier.number_of_time_samples
    A1=2.0/Nt*abs(full.coefficients[1,0,0]);Fourier_Real.compute_time_series(full)
    return A1,float(np.max(np.abs(full.time_series[:,0,0])))
df=pd.read_csv(Path(__file__).parent/"nlvib_sdof_vibroimpact_shooting_frc.csv")

for Nh in [9,15,30]:
    harmonics=list(range(0,Nh));system=SDOF(**PARAMS)
    HarmonicBalanceMethod.update_dependencies(harmonics,system.polynomial_degree)
    provider=NumericalFRF(system.mass_matrix,system.damping_matrix,system.stiffness_matrix)
    problem=FBSProblem(system,provider,DLFTContact(epsilon=1.0,g_zero=GAP))
    omegas=np.arange(0.86,1.45,0.01)
    Z=-omegas[0]**2+1j*omegas[0]*PARAMS['c']+PARAMS['k']
    x=FourierOmegaPoint.new_from_first_harmonic(np.array([[PARAMS['F0']/Z]]),omega=omegas[0])
    print(f"\n=== Nh={Nh}, eps=1.0 ===")
    print("  omega   it  why    |r|_abs  rel_drop  active   A_peak | nlvib")
    last_ok=None
    for om in omegas:
        x=FourierOmegaPoint(x.fourier,om)
        xs,it,ok,absr,relr,why=newton(problem,x)
        if ok:
            na=int(xs.contact_mask.sum());A1,Ap=peak(problem,xs)
            ref=df.iloc[(df.omega-om).abs().idxmin()]
            print(f"  {om:.3f}  {it:3d}  {why:4s}  {absr:.2e}  {relr:.1e}  {na:5d}  {Ap:.4f} | {ref.A_peak:.4f}")
            x=xs;last_ok=om
        else:
            print(f"  {om:.3f}  {it:3d}  FAIL  {absr:.2e}  {relr:.1e}   ---")
            break
    print(f"  --> last converged omega = {last_ok}")
