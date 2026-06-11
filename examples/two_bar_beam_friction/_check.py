"""Sanity checks: FD Jacobian of both residuals + single-frequency solve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, AFT, DLFTFriction, HarmonicBalanceMethod)
from systems import BarBeamParams, BarBeamFrictionDLFT, BarBeamFrictionAFT

HARM = list(range(0, 16))
POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
p = BarBeamParams(poly_deg=POLY)
h0 = HARM.index(0); h1 = HARM.index(1)


def fd_jac(problem, x, h=1e-7):
    r0 = problem.compute_residue_RI(x)
    n = r0.shape[0]
    J = zeros((n, n)); xa = np.asarray(x).copy()
    for j in range(n):
        xp = xa.copy(); xp[j, 0] += h
        rp = problem.compute_residue_RI(
            FourierOmegaPoint(Fourier.new_from_RI(xp[:-1]), x.omega))
        J[:, j] = (rp - r0).ravel() / h
    return J


def check(name, system, method, Q):
    prob = FBSProblem(system, NumericalFRF(system.mass_matrix,
                      system.damping_matrix, system.stiffness_matrix), method)
    x = FourierOmegaPoint(Fourier(Q), omega=1.0)
    Ja = prob.compute_jacobian_of_residue_RI(x)
    Jf = fd_jac(prob, x)
    rel = np.abs(Ja - Jf).max() / (np.abs(Jf).max() + 1e-30)
    dwa = prob.compute_derivative_wrt_omega_RI(x)
    hw = 1e-5
    dwf = (prob.compute_residue_RI(FourierOmegaPoint(x.fourier, x.omega + hw))
           - prob.compute_residue_RI(FourierOmegaPoint(x.fourier, x.omega - hw))) / (2 * hw)
    relw = np.abs(dwa - dwf).max() / (np.abs(dwf).max() + 1e-30)
    print(f"  [{name}] Jacobian rel err = {rel:.2e},  dr/dw rel err = {relw:.2e}")
    return prob


# small partial-slip-ish state
Q = zeros((Fourier.number_of_harmonics, 2, 1), complex)
Q[h0, 0, 0] = p.eps + 1e-3          # normal DC: slightly in contact
Q[h1, 1, 0] = 5e-3                  # tangential 1st harmonic
Q[h1, 0, 0] = 1e-4

print("=" * 60)
print("Finite-difference Jacobian checks")
print("=" * 60)
aft = BarBeamFrictionAFT(p)
check("AFT-reg", aft, AFT(), Q.copy())

dlft = BarBeamFrictionDLFT(p)
contact = DLFTFriction(epsilon_N=1e3, epsilon_T=1e3, mu=p.mu, g_zero=p.eps,
                       n_tangential=1, n_sweep=1)
prob_dlft = check("DLFT", dlft, contact, Q.copy())

print("=" * 60)
print("Single-frequency solve at omega = 1.0 (DLFT)")
print("=" * 60)
solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob_dlft)
# cold guess: small tangential 1st harmonic, normal pressed to gap
Q0 = zeros((Fourier.number_of_harmonics, 2, 1), complex)
Q0[h0, 0, 0] = p.eps
Q0[h1, 1, 0] = 1e-3
ig = FourierOmegaPoint(Fourier(Q0), omega=1.0)
sol, iters, ok, _ = solver.solve_fixed_frequency(
    ig, maximum_iterations=200, absolute_tolerance=1e-7)
print(f"  converged={ok} in {iters} iters")
if ok:
    full = prob_dlft.compute_full_response(sol.fourier, sol.omega)
    Fourier_Real.compute_time_series(full)
    q1 = full.time_series[:, 0, 0]; q4 = full.time_series[:, 3, 0]
    lam = contact._get_lambda_corrected(sol)
    lf = Fourier(lam.reshape(Fourier.number_of_harmonics, 2, 1))
    Fourier_Real.compute_time_series(lf)
    N = lf.time_series[:, 0, 0]; fT = lf.time_series[:, 1, 0]
    print(f"  q1 amplitude = {np.abs(q1).max():.4f}, q4 amplitude = {np.abs(q4).max():.4f}")
    print(f"  N mean = {N.mean():.4f} (thesis ~0.3826), fT max = {np.abs(fT).max():.4f}")
    print(f"  mu*N mean = {p.mu*N.mean():.4f} (slip bound)")
