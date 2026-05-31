"""Warm-started fixed-frequency omega-sweep to isolate Newton behaviour in contact.

Marches omega downward from above resonance into the contact band, using the
previous converged solution as the next guess (poor-man's continuation, no
arc-length geometry involved). Reports per-omega: converged?, iters, peak amp.

Tests the effect of (a) number of harmonics, (b) epsilon, (c) smoothed mask.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from pyhbm import (
    FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact, HarmonicBalanceMethod,
)

class SDOF(FBS_System):
    is_real_valued = True
    def __init__(self, m=1.0, c=0.05, k=1.0, F0=0.02, poly_deg=100):
        self.mass_matrix=np.array([[m]]); self.damping_matrix=np.array([[c]])
        self.stiffness_matrix=np.array([[k]]); self.B_coupling=np.array([[1.0]])
        self.total_dimension=1; self.dimension=1; self.polynomial_degree=poly_deg; self.F0=F0
    def external_term(self, tau):
        f=np.zeros((len(tau),1,1)); f[:,0,0]=self.F0*np.cos(tau); return f
    def interface_force(self,u,ud,t): return np.zeros((len(t),1,1))
    def jacobian_interface_force(self,u,ud,t): return np.zeros((len(t),1,1))
    def jacobian_interface_force_qdot(self,u,ud,t): return np.zeros((len(t),1,1))

PARAMS=dict(m=1.0,c=0.05,k=1.0,F0=0.02); GAP=0.1

def newton(problem, x0, max_iter=80, tol=1e-6):
    x=np.asarray(x0).copy(); omega=x0.omega
    for it in range(max_iter):
        xp=FourierOmegaPoint(Fourier.new_from_RI(x[:-1]),omega)
        r=problem.compute_residue_RI(xp); rn=np.linalg.norm(r)
        if rn<tol: return xp,it,True
        J=problem.compute_jacobian_of_residue_RI(xp)
        # simple backtracking
        delta=np.linalg.solve(J,r); alpha=1.0; ok=False
        for _ in range(15):
            xt=x.copy(); xt[:-1]=x[:-1]-alpha*delta
            rt=problem.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(xt[:-1]),omega))
            if np.linalg.norm(rt)<rn: x=xt; ok=True; break
            alpha*=0.5
        if not ok: x[:-1]=x[:-1]-delta
    return xp,max_iter,False

def sweep(Nh, eps, smooth=False, amp0=0.45):
    harmonics=list(range(0,Nh))
    system=SDOF(**PARAMS)
    HarmonicBalanceMethod.update_dependencies(harmonics, system.polynomial_degree)
    provider=NumericalFRF(system.mass_matrix,system.damping_matrix,system.stiffness_matrix)
    contact=DLFTContact(epsilon=eps,g_zero=GAP)
    if smooth:
        contact = _SmoothDLFT(epsilon=eps,g_zero=GAP)
    problem=FBSProblem(system,provider,contact)
    # start above resonance where amplitude is high (on bent peak), sweep down
    omegas=np.arange(1.45,0.95,-0.01)
    x=FourierOmegaPoint.new_from_first_harmonic(np.array([[amp0+0j]]),omega=omegas[0])
    n_ok=0; n_fail=0; first_fail=None
    for om in omegas:
        x=FourierOmegaPoint(x.fourier,om)
        xs,it,ok=newton(problem,x)
        if ok:
            n_ok+=1; x=xs
        else:
            n_fail+=1
            if first_fail is None: first_fail=om
    print(f"  Nh={Nh:2d} eps={eps:7.1f} smooth={smooth!s:5s}: {n_ok} ok / {n_fail} fail"
          + (f"  first fail at omega={first_fail:.3f}" if first_fail else "  ALL OK"))

# smoothed-mask variant: differentiable max via 0.5*lambda*(1+tanh(alpha*lambda))
class _SmoothDLFT(DLFTContact):
    ALPHA=20.0
    def _get_lambda_corrected(self,x):
        if x.lambda_corrected is None:
            n_int=self._problem.ode.dimension
            zr=Fourier(self._get_Zr_rhs(x).reshape(Fourier.number_of_harmonics,n_int,1))
            Fourier_Real.compute_time_series(zr)
            Fourier_Real.compute_time_series(x.fourier)
            lam_p=zr.time_series+self.epsilon*(x.fourier.time_series-self.g_zero)
            s=0.5*(1.0+np.tanh(self.ALPHA*lam_p))
            x.contact_mask=s            # smooth derivative of softplus-like correction
            lam=s*lam_p
            x.lambda_corrected=np.vstack(Fourier_Real.new_from_time_series(lam).coefficients)
        return x.lambda_corrected

print("="*70); print("Warm-started omega-sweep DOWN through contact band (amp0=0.45)"); print("="*70)
for Nh in [5, 9, 15, 30]:
    sweep(Nh, eps=1.0)
print("-- vary epsilon at Nh=9 --")
for eps in [0.5, 5.0, 20.0, 50.0]:
    sweep(9, eps=eps)
print("-- smoothed mask (tanh) --")
for Nh in [9, 30]:
    sweep(Nh, eps=1.0, smooth=True)
