"""Decisive: can a Powell (hybr) corrector push past where Newton stalls,
from a WELL-SCALED moderate-eps seed? Load-ramp P1 0->0.1 at w=0.5, Newton with
Powell fallback. Then frequency continuation with Powell corrector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros
from scipy.optimize import root

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, DLFTFriction, HarmonicBalanceMethod)
from systems import BarBeamParams, BarBeamFrictionDLFT

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1); Nh = Fourier.number_of_harmonics


def set_P1(prob, sysD, P1):
    sysD.p.P1 = P1
    prob.external_term = Fourier_Real.new_from_time_series(
        sysD.external_term(Fourier.adimensional_time_samples))
    prob.F_ext_full = np.vstack(prob.external_term.coefficients)
    prob.F_ext_full_RI = np.vstack((prob.F_ext_full.real, prob.F_ext_full.imag))


def powell_solve(prob, x_seed, w):
    v0 = np.asarray(x_seed)[:-1, 0].copy()
    def fun(v):
        return prob.compute_residue_RI(
            FourierOmegaPoint(Fourier.new_from_RI(v.reshape(-1,1)), w)).ravel()
    sol = root(fun, v0, method="hybr", options={"maxfev": 6000, "xtol": 1e-10})
    rn = np.linalg.norm(sol.fun)
    x = FourierOmegaPoint(Fourier.new_from_RI(sol.x.reshape(-1,1)), w)
    return x, rn


def slip_frac(x, meth):
    meth._get_lambda_corrected(x); J=x.contact_mask; Nt=J.shape[0]
    return 1 - sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))/Nt


EPS = 30.0
p = BarBeamParams(P5=0.4, poly_deg=POLY)
sysD = BarBeamFrictionDLFT(p)
M,C,K = sysD.mass_matrix, sysD.damping_matrix, sysD.stiffness_matrix
meth = DLFTFriction(epsilon_N=EPS, epsilon_T=EPS, mu=p.mu, g_zero=p.eps, n_tangential=1, n_sweep=1)
prob = FBSProblem(sysD, NumericalFRF(M,C,K), meth)
solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob)

print(f"Load ramp P1: 0 -> 0.1 at w=0.5, eps={EPS}, Newton + Powell fallback")
Q = zeros((Nh,2,1),complex); Q[h0,0,0]=p.eps
x = FourierOmegaPoint(Fourier(Q), 0.5)
nsteps = 40
for s in range(1, nsteps+1):
    P1 = 0.1 * s/nsteps
    set_P1(prob, sysD, P1)
    sol, it, ok, _ = solver.solve_fixed_frequency(x, maximum_iterations=150, absolute_tolerance=1e-8)
    how = "newton"
    if not ok:
        sol, rn = powell_solve(prob, x, 0.5)
        ok = rn < 1e-6; how = f"POWELL({rn:.0e})"
    if not ok:
        print(f"  P1={P1:.4f}: FAILED ({how}), slip={slip_frac(x,meth):.2f}")
        break
    x = sol
    if s % 4 == 0 or not ok:
        print(f"  P1={P1:.4f}: OK [{how}], slip={slip_frac(x,meth):.2f}")
else:
    print(f"FULL THESIS LOAD reached at w=0.5! slip={slip_frac(x,meth):.2f}")
    # now sweep frequency with Powell corrector
    print("Frequency sweep with Powell corrector (no continuation, direct):")
    for w in np.arange(0.86, 1.10, 0.02):
        xw, rn = powell_solve(prob, x, w)
        if rn < 1e-6:
            full=prob.compute_full_response(xw.fourier, w); Fourier_Real.compute_time_series(full)
            q1=np.abs(full.time_series[:,0,0]).max()
            print(f"  w={w:.2f}: OK slip={slip_frac(xw,meth):.2f} peak q1={q1:.3f}")
            x = xw
        else:
            print(f"  w={w:.2f}: Powell |r|={rn:.0e}")
