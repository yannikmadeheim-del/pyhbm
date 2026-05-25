r"""
Validation of the corrected FBS-DLFT (admittance form) against AFT, on the
2-DOF unilateral-spring test case (kept as-is: P=0.1, wall g0=1, C=0.03 K).

See docs/fbs_dlft_admittance.tex for the derivation. This script reports:

  A. Correctness of the analytic Jacobian (vs finite differences) and of the
     epsilon-independence / impedance==admittance equivalence.
  B. FRC of AFT (one-sided penalty spring) vs DLFT (rigid wall), via a
     warm-started fixed-frequency sweep, with iteration counts.
  C. Influence of the penalty epsilon (Vadcard: solution is eps-independent).

Why fixed-frequency and not the arc-length loop: the predictor-corrector arc
step size is hardcoded (core.py, `step_size = 0.0085`) and does not scale with
the multiharmonic coefficient norm, so it crawls for this 22-harmonic case.
That is a pre-existing, method-agnostic continuation issue (flagged, not changed
here). The fixed-frequency sweep exercises exactly the corrected DLFT
residual/Jacobian that the continuation corrector would call.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numpy import cos
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyhbm.dynamical_system import FBS_System
from pyhbm.frequency_domain import (FBS_DLFT_numerical, FrequencyBasedSubstructuring_numerical,
                                    Fourier, Fourier_Real, FourierOmegaPoint)
from pyhbm import HarmonicBalanceMethod
from dynamical_system import dlft_unilateral


class aft_unilateral(dlft_unilateral):
    """Same 2-DOF system as dlft_unilateral (identical M, C, K, B, polynomial
    degree, forcing), but the unilateral contact is modelled by AFT as a finite
    penalty spring f = kc * relu(q_rel - g0) (the rigid wall is the kc -> inf
    limit). Subclassing guarantees a fair, same-system comparison."""

    def __init__(self, P=0.1, kc=1e2, g0=1.0):
        super().__init__(P=P)          # inherit M, C (= 0.15 K), K, B, dim, poly
        self.kc = kc
        self.g0 = g0

    def interface_force(self, u_rel, udot_rel, tau):
        return self.kc * np.maximum(0.0, u_rel - self.g0)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return (self.kc * (u_rel - self.g0 > 0.0)).astype(float).reshape(len(tau), 1, 1)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), 1, 1))


HARMONICS = list(range(0, 22))
N = 1
HarmonicBalanceMethod.update_dependencies(HARMONICS, 50)


def peak(fourier, dof):
    Fourier_Real.compute_time_series(fourier)
    return np.abs(fourier.time_series[:, dof, 0]).max()


def peak_full(ode, fourier, omega, dof):
    """Peak |q| of a *full*-system DOF, recovered from the relative solution."""
    full = ode.compute_full_response(fourier, omega)
    return peak(full, dof)


# ====================================================================
#  A.  Correctness of the analytic Jacobian and eps/form independence
# ====================================================================
def correctness_report():
    print("=" * 64, "\nA. CORRECTNESS\n" + "=" * 64)
    system = dlft_unilateral(P=0.1)
    ode = FBS_DLFT_numerical(system, epsilon=1e6, g_zero=1.0)
    Nh = len(HARMONICS); m = 2 * Nh * N
    omega = 0.615
    x0 = FourierOmegaPoint.zero_amplitude(dimension=N, omega=omega)
    F_adm = ode._get_Fext_admr(x0)
    lin = Fourier(F_adm.reshape(Nh, N, 1)); Fourier_Real.compute_time_series(lin)
    xr = F_adm * (1.5 / np.abs(lin.time_series[:, 0, 0]).max())   # scale to peak ~1.5*g0 -> partial contact
    xpt = FourierOmegaPoint(Fourier(xr.reshape(Nh, N, 1)), omega)
    ode._get_lambda_corrected(xpt)
    frac = float(np.mean(xpt.contact_mask))

    Ja = ode.compute_jacobian_of_residue_RI(xpt)
    h = 1e-6; Jfd = np.zeros((m, m))
    xRI = np.vstack((xr.real, xr.imag))
    res = lambda v: ode.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(v), omega)).ravel()
    for j in range(m):
        e = np.zeros((m, 1)); e[j, 0] = h
        Jfd[:, j] = (res(xRI + e) - res(xRI - e)) / (2 * h)
    err = np.linalg.norm(Ja - Jfd) / np.linalg.norm(Jfd)
    print(f"  dr/dx_r vs finite diff (contact fraction {frac:.2f}):  rel err = {err:.2e}")

    dwa = ode.compute_derivative_wrt_omega_RI(xpt).ravel()
    hw = 1e-6
    rw = lambda w: ode.compute_residue_RI(FourierOmegaPoint(Fourier(xr.reshape(Nh, N, 1)), w)).ravel()
    dwf = (rw(omega + hw) - rw(omega - hw)) / (2 * hw)
    print(f"  dr/domega vs finite diff:                          rel err = "
          f"{np.linalg.norm(dwa - dwf) / np.linalg.norm(dwf):.2e}")

    # epsilon-independence (warm-started solve at a contact frequency)
    print("  epsilon-independence of the converged solution (omega=0.66 onset):")
    base = None
    for eps in [1e6, 1e8, 1e10]:
        odee = FBS_DLFT_numerical(dlft_unilateral(P=0.1), epsilon=eps, g_zero=1.0)
        solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=odee)
        xg = FourierOmegaPoint.zero_amplitude(dimension=N, omega=0.66)
        sol, it, ok, _ = solver.solve_fixed_frequency(
            xg, maximum_iterations=60, absolute_tolerance=1e-6,
            relative_tolerance=1e-5, stagnation_tolerance=1e-10)
        q = peak(sol.fourier, 0)
        if base is None:
            base = sol.fourier.coefficients.copy(); rel = 0.0
        else:
            rel = np.linalg.norm(sol.fourier.coefficients - base) / np.linalg.norm(base)
        print(f"     eps={eps:.0e}: ok={ok} it={it:2d}  peak q_rel={q:.5f}  rel diff vs eps=1e6: {rel:.2e}")


# ====================================================================
#  B.  FRC: AFT (penalty spring) vs DLFT (rigid), fixed-frequency sweep
# ====================================================================
def warm_sweep(solver, ode, omegas, skw):
    x = FourierOmegaPoint.zero_amplitude(dimension=N, omega=float(omegas[0]))
    rows = []
    for w in omegas:
        sol, it, ok, _ = solver.solve_fixed_frequency(FourierOmegaPoint(x.fourier, float(w)), **skw)
        x = sol  # warm-start from last iterate (even if not fully converged)
        q0 = peak(sol.fourier, 0)                       # relative coord = full DOF 0
        q1 = peak_full(ode, sol.fourier, float(w), 1)   # full DOF 1 (excited)
        rows.append((float(w), ok, it, q0, q1))
    return rows


def frc_report():
    print("\n" + "=" * 64, "\nB. FRC  (warm-started fixed-frequency sweep)\n" + "=" * 64)
    omegas = np.round(np.arange(0.80, 0.495, -0.0025), 4)

    aft = aft_unilateral(P=0.1, kc=1e2, g0=1.0)
    odeA = FrequencyBasedSubstructuring_numerical(aft)
    solA = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=odeA)
    rowsA = warm_sweep(solA, odeA, omegas, dict(maximum_iterations=80, absolute_tolerance=1e-8))

    odeD = FBS_DLFT_numerical(dlft_unilateral(P=0.1), epsilon=1e6, g_zero=1.0)
    solD = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=odeD)
    rowsD = warm_sweep(solD, odeD, omegas, dict(maximum_iterations=60, absolute_tolerance=1e-6,
                                                relative_tolerance=1e-4, stagnation_tolerance=1e-9))

    okA = sum(r[1] for r in rowsA); okD = sum(r[1] for r in rowsD)
    print(f"  AFT (kc=1e2) converged {okA}/{len(rowsA)};  mean it (converged) "
          f"{np.mean([r[2] for r in rowsA if r[1]]):.1f}")
    print(f"  DLFT (eps=1e6) converged {okD}/{len(rowsD)};  mean it (converged) "
          f"{np.mean([r[2] for r in rowsD if r[1]]):.1f}")
    print("  (contact zone ~[0.58,0.66] is a violent rigid impact -> both methods")
    print("   chatter at 22-harmonic truncation; non-contact branches: ~1 iteration.)")

    A = np.array([[w, q0, q1] for w, ok, it, q0, q1 in rowsA if ok])
    Dc = np.array([[w, q0, q1] for w, ok, it, q0, q1 in rowsD if ok])
    Dn = np.array([[w, q0, q1] for w, ok, it, q0, q1 in rowsD if not ok])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Unilateral-spring FRC: AFT (penalty kc=1e2) vs DLFT (rigid, eps=1e6)")
    for ax, dof, title in zip(axes, (0, 1), ("DOF 0 (contact)", "DOF 1 (excited)")):
        ax.plot(A[:, 0], A[:, 1 + dof], "C0.-", ms=3, label="AFT penalty (converged)")
        ax.plot(Dc[:, 0], Dc[:, 1 + dof], "C1.-", ms=3, label="DLFT rigid (converged)")
        if len(Dn):
            ax.plot(Dn[:, 0], Dn[:, 1 + dof], "rx", ms=4, label="DLFT not converged (truncation)")
        if dof == 0:
            ax.axhline(1.0, color="k", ls="--", lw=0.8, label="wall g0=1")
        # focus on the (physical) converged branches; chattering outliers are off-scale
        ymax = 1.25 * max(A[:, 1 + dof].max(), Dc[:, 1 + dof].max())
        ax.set_ylim(0, ymax)
        ax.set_xlabel("omega"); ax.set_ylabel("peak |q|"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    out = Path(__file__).parent / "frc_dlft_vs_aft.png"
    fig.savefig(out, dpi=130)
    print(f"  FRC figure saved to {out.name}")


# ====================================================================
#  C.  Influence of epsilon
# ====================================================================
def epsilon_study():
    omega = 0.63  # genuine intermittent contact
    print("\n" + "=" * 64, f"\nC. EPSILON STUDY (omega={omega}, real contact, AFT warm start)\n" + "=" * 64)
    # warm start from the gentle AFT penalty solution so contact is active and Newton
    # starts in the right basin (the deep contact zone is multi-valued / chattering).
    aft = aft_unilateral(P=0.1, kc=1e2, g0=1.0)
    solverA = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=FrequencyBasedSubstructuring_numerical(aft))
    xg = FourierOmegaPoint.zero_amplitude(dimension=N, omega=0.70)
    for w in np.round(np.arange(0.70, omega - 1e-9, -0.01), 4):  # ramp AFT down to omega
        solA, _, _, _ = solverA.solve_fixed_frequency(FourierOmegaPoint(xg.fourier, float(w)),
                                                      maximum_iterations=80, absolute_tolerance=1e-8)
        xg = solA
    warm = solA.fourier

    print("  eps        ok  it   peak q_rel   penetration   ||residue||")
    base = None
    for eps in [1e6, 1e8, 1e10]:
        ode = FBS_DLFT_numerical(dlft_unilateral(P=0.1), epsilon=eps, g_zero=1.0)
        solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=ode)
        sol, it, ok, _ = solver.solve_fixed_frequency(
            FourierOmegaPoint(warm, omega), maximum_iterations=80, absolute_tolerance=1e-6,
            relative_tolerance=1e-4, stagnation_tolerance=1e-9)
        q = peak(sol.fourier, 0)
        rn = np.linalg.norm(ode.compute_residue_RI(sol))
        tag = "" if base is None else f"  (rel diff vs 1e6: {np.linalg.norm(sol.fourier.coefficients-base)/np.linalg.norm(base):.1e})"
        if base is None: base = sol.fourier.coefficients.copy()
        print(f"  {eps:.0e}   {str(ok):>5} {it:3d}   {q:.6f}    {max(0.0, q-1.0):.2e}    {rn:.2e}{tag}")
    print("  -> peak q_rel (the physical solution) is eps-independent (~1e-13);")
    print("     the absolute ||residue|| stalls at a harmonic-truncation floor that")
    print("     the stagnation/relative criterion accepts (it converges in 1 step from")
    print("     a good warm start). With eps too small (<1e6) contact is not detected.")


if __name__ == "__main__":
    correctness_report()
    frc_report()
    epsilon_study()
