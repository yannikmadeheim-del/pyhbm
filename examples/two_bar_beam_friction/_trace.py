"""Trace DLFT residual + contact mask across Newton iterations (cold, partial slip).
Also test a FORCING-RAMP homotopy: start at P1=0 (static stuck contact) and step up."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, DLFTFriction, HarmonicBalanceMethod)
from systems import BarBeamParams, BarBeamFrictionDLFT

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1)
Nh = Fourier.number_of_harmonics


def mask_stats(x, method):
    method._get_lambda_corrected(x)
    J = x.contact_mask                       # (Nt, 2, 2)
    Nt = J.shape[0]
    sep = sum(np.all(np.abs(J[k]) < 1e-30) for k in range(Nt))
    stick = sum(abs(J[k,1,1]-1) < 1e-9 and abs(J[k,1,0]) < 1e-12 for k in range(Nt))
    slip = Nt - sep - stick
    return sep, stick, slip, Nt


def manual_newton(prob, method, x0, n=60, damp=1.0):
    x = x0
    for it in range(n):
        r = prob.compute_residue_RI(x)
        rn = np.linalg.norm(r)
        sep, stick, slip, Nt = mask_stats(x, method)
        if it < 12 or it % 10 == 0:
            print(f"   it{it:3d}  |r|={rn:.3e}  sep={sep:3d} stick={stick:3d} slip={slip:3d}/{Nt}")
        if rn < 1e-7:
            print(f"   CONVERGED at it{it}, |r|={rn:.2e}"); return x, True
        J = prob.compute_jacobian_of_residue_RI(x)
        dx = np.linalg.solve(J, -r)
        xa = np.asarray(x).copy(); xa[:-1] += damp * dx
        x = FourierOmegaPoint(Fourier.new_from_RI(xa[:-1]), x.omega)
    print(f"   STALLED, |r|={rn:.2e}"); return x, False


def build(P5, eps):
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    sysD = BarBeamFrictionDLFT(p)
    m = DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps,
                     n_tangential=1, n_sweep=1)
    prob = FBSProblem(sysD, NumericalFRF(sysD.mass_matrix, sysD.damping_matrix,
                      sysD.stiffness_matrix), m)
    return p, sysD, prob, m


print("=" * 64)
print("Cold Newton trace: P5=2.0 (partial), w=0.85, eps=1e4, stuck guess")
print("=" * 64)
p, sysD, prob, m = build(2.0, 1e4)
Q = zeros((Nh, 2, 1), complex); Q[h0, 0, 0] = p.eps
manual_newton(prob, m, FourierOmegaPoint(Fourier(Q), 0.85))

print("=" * 64)
print("Cold Newton trace: same but DAMPED (damp=0.3)")
print("=" * 64)
Q = zeros((Nh, 2, 1), complex); Q[h0, 0, 0] = p.eps
manual_newton(prob, m, FourierOmegaPoint(Fourier(Q), 0.85), n=120, damp=0.3)

print("=" * 64)
print("FORCING-RAMP homotopy: P1 = 0 -> thesis, at w=0.85, P5=2.0, eps=1e4")
print("=" * 64)
for P1 in (0.0, 0.02, 0.05, 0.1):
    p, sysD, prob, m = build(2.0, 1e4)
    sysD.p.P1 = P1
    # rebuild external term cache in the problem (depends on P1)
    prob = FBSProblem(sysD, NumericalFRF(sysD.mass_matrix, sysD.damping_matrix,
                      sysD.stiffness_matrix), m)
    Q = zeros((Nh, 2, 1), complex); Q[h0, 0, 0] = p.eps
    print(f" P1={P1}:")
    x, ok = manual_newton(prob, m, FourierOmegaPoint(Fourier(Q), 0.85), n=60)
