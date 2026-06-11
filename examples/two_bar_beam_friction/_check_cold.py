"""Cold first-point DLFT solve: analytic guesses x epsilon x start frequency.
NO warm starts (no nonlinear solution reused as guess)."""
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


def linear_open(system, p, omega):
    M, C, K = system.mass_matrix, system.damping_matrix, system.stiffness_matrix
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((6, 1), complex); F[0, 0] = p.P1
    return (system.B_coupling @ np.linalg.solve(Z, F))[:, 0]


def guess(kind, system, p, omega):
    Q = zeros((Fourier.number_of_harmonics, 2, 1), complex)
    Q[h0, 0, 0] = p.eps                          # normal DC ~ gap (in contact)
    if kind == "stuck":
        pass                                     # zero tangential oscillation
    elif kind == "open":
        xr = linear_open(system, p, omega)
        Q[h1, 0, 0] = xr[0]; Q[h1, 1, 0] = xr[1]
    return FourierOmegaPoint(Fourier(Q), omega=omega)


def trial(P5, omega, kind, eps):
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    sysD = BarBeamFrictionDLFT(p)
    prob = FBSProblem(sysD, NumericalFRF(sysD.mass_matrix, sysD.damping_matrix,
                      sysD.stiffness_matrix),
                      DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu,
                                   g_zero=p.eps, n_tangential=1, n_sweep=1))
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob)
    sol, it, ok, _ = solver.solve_fixed_frequency(
        guess(kind, sysD, p, omega), maximum_iterations=300, absolute_tolerance=1e-7)
    return ok, it


for P5, tag in ((2.0, "partial"), (0.4, "gross")):
    print("=" * 60)
    print(f"P5={P5} ({tag} slip)   first-point cold solve")
    print("=" * 60)
    for omega in (0.70, 0.85):
        for kind in ("stuck", "open"):
            row = []
            for eps in (1e3, 1e4, 1e5, 1e6):
                ok, it = trial(P5, omega, kind, eps)
                row.append(f"e{int(np.log10(eps))}:{'OK' if ok else '--'}({it})")
            print(f"  w={omega:.2f} {kind:5s}: " + "  ".join(row))
