"""Find a practical robust config: vary (Nh, eps), report max omega reached and
RMS A_peak error vs NLvib over the contact band."""
import sys; from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent.parent/"src"))
import numpy as np, pandas as pd
from pyhbm import (FBS_System,Fourier,Fourier_Real,FourierOmegaPoint,
                   FBSProblem,NumericalFRF,DLFTContact,HarmonicBalanceMethod)
class SDOF(FBS_System):
    is_real_valued=True
    def __init__(s,m=1.,c=.05,k=1.,F0=.02,poly_deg=100):
        s.mass_matrix=np.array([[m]]);s.damping_matrix=np.array([[c]]);s.stiffness_matrix=np.array([[k]])
        s.B_coupling=np.array([[1.]]);s.total_dimension=1;s.dimension=1;s.polynomial_degree=poly_deg;s.F0=F0
    def external_term(s,t):f=np.zeros((len(t),1,1));f[:,0,0]=s.F0*np.cos(t);return f
    def interface_force(s,u,d,t):return np.zeros((len(t),1,1))
    def jacobian_interface_force(s,u,d,t):return np.zeros((len(t),1,1))
    def jacobian_interface_force_qdot(s,u,d,t):return np.zeros((len(t),1,1))
P=dict(m=1.,c=.05,k=1.,F0=.02);GAP=.1
def newton(pr,x0,mi=300,rtol=1e-6,stag=1e-10,ab=1e-9):
    x=np.asarray(x0).copy();om=x0.omega;r0=None
    for it in range(mi):
        xp=FourierOmegaPoint(Fourier.new_from_RI(x[:-1]),om);r=pr.compute_residue_RI(xp);rn=np.linalg.norm(r)
        if r0 is None:r0=rn
        if rn<ab or rn<rtol*r0:return xp,it,True
        J=pr.compute_jacobian_of_residue_RI(xp);d=np.linalg.solve(J,r)
        if np.linalg.norm(d)<stag*max(np.linalg.norm(x[:-1]),1e-12):return xp,it,True
        a=1.;ok=False
        for _ in range(25):
            xt=x.copy();xt[:-1]=x[:-1]-a*d
            if np.linalg.norm(pr.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(xt[:-1]),om)))<rn:x=xt;ok=True;break
            a*=.5
        if not ok:x[:-1]=x[:-1]-d
    return xp,mi,False
def peak(pr,xp):
    f=pr.compute_full_response(xp.fourier,xp.omega);Fourier_Real.compute_time_series(f)
    return float(np.max(np.abs(f.time_series[:,0,0])))
df=pd.read_csv(Path(__file__).parent/"nlvib_sdof_vibroimpact_shooting_frc.csv")
print(f"{'Nh':>3} {'eps':>8} {'maxOmega':>9} {'#conv':>6} {'RMS_Apeak_err':>14}")
for Nh in [5,7,9,13]:
  for eps in [1.,10.,100.,1000.]:
    h=list(range(0,Nh));sy=SDOF(**P);HarmonicBalanceMethod.update_dependencies(h,sy.polynomial_degree)
    pv=NumericalFRF(sy.mass_matrix,sy.damping_matrix,sy.stiffness_matrix)
    pr=FBSProblem(sy,pv,DLFTContact(epsilon=eps,g_zero=GAP))
    oms=np.arange(0.86,1.50,0.01);Z=-oms[0]**2+1j*oms[0]*P['c']+P['k']
    x=FourierOmegaPoint.new_from_first_harmonic(np.array([[P['F0']/Z]]),omega=oms[0])
    mx=oms[0];n=0;errs=[]
    for om in oms:
        x=FourierOmegaPoint(x.fourier,om);xs,it,ok=newton(pr,x)
        if not ok:break
        n+=1;mx=om;x=xs
        if om>=0.9:
            Ap=peak(pr,xs);ref=df.iloc[(df.omega-om).abs().idxmin()].A_peak;errs.append(Ap-ref)
    rms=np.sqrt(np.mean(np.array(errs)**2)) if errs else float('nan')
    print(f"{Nh:>3} {eps:>8.0f} {mx:>9.3f} {n:>6} {rms:>14.4f}")
