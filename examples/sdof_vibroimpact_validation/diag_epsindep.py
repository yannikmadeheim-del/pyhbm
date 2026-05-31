"""eps-independence check for the single-field DLFT (rigid one-sided wall).

Reference is a RIGID stop at u <= g0: NLvib max(u) is clamped at ~+0.099 and
A_peak is the free NEGATIVE swing. So there is no contact stiffness to match;
eps is purely a numerical penalty. Claim: at convergence the response is
eps-independent (eps only changes the Newton path/conditioning).

Warm-start a fixed-frequency solve at several omegas, for each eps, and report
the CONVERGED max(u), min(u), A_peak versus NLvib.
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

def newton(problem,x0,max_iter=400,rtol=1e-5,stag=1e-11,abstol=1e-10):
    x=np.asarray(x0).copy();omega=x0.omega;r0=None
    for it in range(max_iter):
        xp=FourierOmegaPoint(Fourier.new_from_RI(x[:-1]),omega)
        r=problem.compute_residue_RI(xp);rn=np.linalg.norm(r)
        if r0 is None:r0=rn
        if rn<abstol or rn<rtol*r0:return xp,it,True
        J=problem.compute_jacobian_of_residue_RI(xp)
        delta=np.linalg.solve(J,r)
        if np.linalg.norm(delta)<stag*max(np.linalg.norm(x[:-1]),1e-12):
            return xp,it,True
        alpha=1.0;ok=False
        for _ in range(30):
            xt=x.copy();xt[:-1]=x[:-1]-alpha*delta
            rt=np.linalg.norm(problem.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(xt[:-1]),omega)))
            if rt<rn:x=xt;ok=True;break
            alpha*=0.5
        if not ok:x[:-1]=x[:-1]-delta
    return xp,max_iter,False

def response(problem,xp):
    full=problem.compute_full_response(xp.fourier,xp.omega)
    Fourier_Real.compute_time_series(full)
    u=full.time_series[:,0,0]
    return u.max(),u.min(),float(np.max(np.abs(u)))

df=pd.read_csv(Path(__file__).parent/"nlvib_sdof_vibroimpact_shooting_frc.csv")
Nh=15
harmonics=list(range(0,Nh));system=SDOF(**PARAMS)
HarmonicBalanceMethod.update_dependencies(harmonics,system.polynomial_degree)
provider=NumericalFRF(system.mass_matrix,system.damping_matrix,system.stiffness_matrix)

test_omegas=[1.05,1.20,1.30,1.40]
print(f"Nh={Nh}  rigid wall g0={GAP}.  Compare CONVERGED response across eps.\n")
for om_target in test_omegas:
    ref=df.iloc[(df.omega-om_target).abs().idxmin()]
    print(f"omega~{ref.omega:.3f}   NLvib: max(u)=+{0.1:.3f}(wall)  A_peak={ref.A_peak:.4f}")
    print(f"    {'eps':>8}  {'conv':>5} {'it':>3}   {'max(u)':>8} {'min(u)':>8} {'A_peak':>8}")
    for eps in [0.5,1.0,5.0,20.0]:
        problem=FBSProblem(system,provider,DLFTContact(epsilon=eps,g_zero=GAP))
        # warm-start: up-sweep from 0.86 to target along resonance
        omegas=np.arange(0.86,om_target+1e-9,0.01)
        Z=-omegas[0]**2+1j*omegas[0]*PARAMS['c']+PARAMS['k']
        x=FourierOmegaPoint.new_from_first_harmonic(np.array([[PARAMS['F0']/Z]]),omega=omegas[0])
        ok=True
        for om in omegas:
            x=FourierOmegaPoint(x.fourier,om)
            xs,it,ok=newton(problem,x)
            if not ok:break
            x=xs
        if ok:
            mx,mn,ap=response(problem,x)
            print(f"    {eps:8.1f}  {'yes':>5} {it:3d}   {mx:+8.4f} {mn:+8.4f} {ap:8.4f}")
        else:
            print(f"    {eps:8.1f}  {'NO':>5} {it:3d}   (failed at omega={om:.3f})")
    print()
