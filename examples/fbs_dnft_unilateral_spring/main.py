# Minimal example: 2-DOF chain with a unilateral (compression-only) contact on
# DOF 0, solved with the corrected DLFT-in-admittance scheme (FBS form).
#
#   * residual / Jacobian: see docs/fbs_dlft_admittance.tex
#   * full validation (AFT comparison, analytic-Jacobian-vs-FD check, epsilon
#     study): see validation_dlft_vs_aft.py
#
# This script sweeps the frequency response with a *warm-started fixed-frequency*
# Newton solve (robust for the non-smooth contact), rather than the arc-length
# continuation: the arc step is dominated by the large (unnormalised-rfft)
# displacement coefficients on the steep resonance flank, so it crawls here.
# The corrected DLFT residual/Jacobian is identical either way - they plug into
# `solve_and_continue` unchanged.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; remove for interactive plots
import matplotlib.pyplot as plt

from pyhbm.frequency_domain import FBS_DLFT_numerical, Fourier_Real, FourierOmegaPoint
from pyhbm import HarmonicBalanceMethod
from dynamical_system import dlft_unilateral

# --- discretisation ---------------------------------------------------------
HARMONICS = list(range(0, 22))
EPSILON   = 1.0e6              # penalty: must be large (>~ 1e6 * stiffness). The
                              # converged solution is epsilon-independent (Vadcard 2022).
GAP       = 1.0               # wall offset g0 (contact when q_rel > g0)

system = dlft_unilateral(P=0.1)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)
ode    = FBS_DLFT_numerical(system, epsilon=EPSILON, g_zero=GAP)
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=ode)

# Convergence criteria: an absolute tolerance is unreachable for rigid contact
# (the truncated contact force has a Gibbs floor that scales with epsilon), so we
# add a relative + step-stagnation criterion.
SOLVER_KWARGS = dict(maximum_iterations=80, absolute_tolerance=1e-6,
                     relative_tolerance=1e-4, stagnation_tolerance=1e-9)


def warm_sweep(omegas):
    """Warm-started fixed-frequency sweep.

    Returns rows of (omega, ok, peak DOF0, peak DOF1, ||Q||_DOF0, ||Q||_DOF1),
    where ||Q|| is the L2 norm over harmonics of the Fourier coefficients (the
    same amplitude measure used in the duffing_FBS_2DoF example).
    """
    x = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=float(omegas[0]))
    rows = []
    for w in omegas:
        sol, iters, ok, _ = solver.solve_fixed_frequency(
            FourierOmegaPoint(x.fourier, float(w)), **SOLVER_KWARGS)
        x = sol                                # warm start for the next frequency
        full = ode.compute_full_response(sol.fourier, float(w))
        Fourier_Real.compute_time_series(full)
        rows.append((float(w), ok,
                     np.abs(full.time_series[:, 0, 0]).max(),
                     np.abs(full.time_series[:, 1, 0]).max(),
                     np.linalg.norm(full.coefficients[:, 0, 0]),
                     np.linalg.norm(full.coefficients[:, 1, 0])))
    return rows


# downward branch (from the high-frequency linear regime into contact) and upward
print("--- DLFT-HBM unilateral spring: warm-started fixed-frequency sweep ---")
down = warm_sweep(np.round(np.arange(0.80, 0.495, -0.0025), 4))
up   = warm_sweep(np.round(np.arange(0.50, 0.805,  0.0025), 4))
n_ok = sum(r[1] for r in down) + sum(r[1] for r in up)
print(f"converged {n_ok}/{len(down) + len(up)} points "
      f"(non-contact: 1 iter; deep-contact zone is truncation-limited).")

# --- plot: peak |q| ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"DLFT-HBM, 2-DOF unilateral spring (g0={GAP}, eps={EPSILON:.0e})")
for ax, dof, title in zip(axes, (0, 1), ("DOF 0 (contact)", "DOF 1 (excited)")):
    for rows, color, lbl in ((down, "C0", "down"), (up, "C1", "up")):
        A = np.array([(w, q0, q1) for w, ok, q0, q1, n0, n1 in rows if ok])
        if len(A):
            ax.plot(A[:, 0], A[:, 1 + dof], color + ".", ms=3, label=lbl)
    if dof == 0:
        ax.axhline(GAP, color="k", ls="--", lw=0.8, label="wall g0")
    ax.set_xlabel("omega"); ax.set_ylabel("peak |q|"); ax.set_title(title)
    ax.legend(); ax.grid(True)
fig.tight_layout()
out = Path(__file__).parent / "frc_dlft.png"
fig.savefig(out, dpi=130)
print(f"FRC figure saved to {out.name}")

# --- plot: ||Q|| Fourier-coefficient norm per DOF (cf. duffing_FBS_2DoF) --------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(f"DLFT-HBM, 2-DOF unilateral spring — ||Q|| per DOF "
              f"(g0={GAP}, eps={EPSILON:.0e})")
for ax, dof, title in zip(axes2, (0, 1), ("DOF 0 (contact)", "DOF 1 (excited)")):
    for rows, color, lbl in ((down, "C0", "down"), (up, "C1", "up")):
        A = np.array([(w, n0, n1) for w, ok, q0, q1, n0, n1 in rows if ok])
        if len(A):
            ax.plot(A[:, 0], A[:, 1 + dof], color + ".", ms=3, label=lbl)
    ax.set_xlabel("omega"); ax.set_ylabel("||Q||"); ax.set_title(title)
    ax.legend(); ax.grid(True)
fig2.tight_layout()
out2 = Path(__file__).parent / "frc_dlft_coeffnorm.png"
fig2.savefig(out2, dpi=130)
print(f"||Q|| figure saved to {out2.name}")
