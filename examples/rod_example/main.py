"""
Clamped-free FE rod impacting a rigid wall  (Colaitis & Batailly, JSV 502, 2021, Sec. 3.1).

Reproduces the paper's academic test case with the RL-HBM scheme:
DLFT normal contact + regularized contact law (gamma) + Lanczos filter (m, C_H).

Model (Fig. 6, Eq. 43):
    - bar of length L = 1 m, E = 21 GPa, rho = 7500 kg/m^3, section A = 0.05 m^2
    - n = 100 two-node bar elements; left end A clamped, right end B free
    - element matrices  M_e = rho*A*l/6 [[2,1],[1,2]],  K_e = E*A/l [[1,-1],[-1,1]]
    - modal damping xi = 7.5e-3 for every free-vibration mode
    - harmonic forcing f_ext(t) = F0 cos(w t), F0 = 25 kN, applied at node B
    - rigid wall a distance d = 6e-5 m from B at rest -> contact when u_B > d
"""
import sys
from pathlib import Path

from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Live progress (no block buffering) + UTF-8 so the continuation's "Δω" print
# does not crash under Windows' default cp1252 console encoding.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from time import time
import numpy as np
from numpy import zeros, cos, eye
from scipy.linalg import eigh

import matplotlib.pyplot as plt

from pyhbm import (
    FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod, SolutionSet,
)


# ============================ system definition =============================

class RodVibroImpact(FBS_System):
    """Clamped-free axial bar, free end impacting a rigid wall, in FBS form."""
    is_real_valued = True

    def __init__(self, n_elem=100, L=1.0, E=21e9, rho=7500.0, A=0.05,
                 F0=25_000.0, xi=7.5e-3, poly_deg=33):
        l = L / n_elem
        n = n_elem                       # free DOF count (node 0 clamped)

        # --- assemble global M, K, then drop the clamped DOF (node 0) ---
        Me = (rho * A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        Ke = (E * A / l)         * np.array([[1.0, -1.0], [-1.0, 1.0]])
        M = zeros((n + 1, n + 1))
        K = zeros((n + 1, n + 1))
        for e in range(n_elem):
            M[e:e + 2, e:e + 2] += Me
            K[e:e + 2, e:e + 2] += Ke
        M = M[1:, 1:]                    # remove clamped node 0
        K = K[1:, 1:]

        # --- modal damping: C = M Phi diag(2 xi w_i) Phi^T M  (Phi mass-normalized) ---
        w2, Phi = eigh(K, M)             # Phi^T M Phi = I
        omega_modes = np.sqrt(np.clip(w2, 0.0, None))
        C = M @ Phi @ np.diag(2.0 * xi * omega_modes) @ Phi.T @ M

        # --- nondimensionalize frequency by omega_1: sweep w_hat = w/w_1 ~ O(1).
        # With M' = w1^2 M, C' = w1 C, K' = K the FRF in w_hat equals the physical
        # FRF in w, so the arc-length stepper sees an O(1) frequency axis. ---
        omega_ref = omega_modes[0]
        self.omega_ref        = omega_ref
        self.mass_matrix      = M
        self.damping_matrix   = C
        self.stiffness_matrix = K

        B = zeros((1, n))                # select free end (last DOF) as interface
        B[0, -1] = 1.0
        self.B_coupling = B

        self.dimension        = 1        # n_int: one contact DOF
        self.total_dimension  = n
        self.polynomial_degree = poly_deg
        self.F0 = F0
        self.omega_modes = omega_modes

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, -1, 0] = self.F0 * cos(tau)     # forcing at the free end B
        return f

    # DLFT supplies the contact force; AFT stubs kept for interface completeness.
    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))


# ============================ parameters ====================================
# Paper values for the rod (RL-HBM): kappa -> epsilon (penalty stiffness, N/m),
# gamma smoothing (N), H = 30 harmonics, N ~ 1024 time samples.
EPSILON  = 1.3e11        # penalty / Lagrangian factor (paper's kappa)
GAMMA    = 6.0e4         # regularized contact-law smoothing parameter [N]
LANCZOS_M       = 1.0    # "unitary" Lanczos filter
LANCZOS_CUTOFF  = 1      # C_H

GAP       = 6.0e-5       # d: wall offset
HARMONICS = list(range(0, 31))   # 0..30  -> H = 30
POLY_DEG  = 33                   # N = (POLY_DEG+1)*30 + 1 = 1021 time samples

rod = RodVibroImpact(poly_deg=POLY_DEG)
OMEGA_1 = rod.omega_modes[0]
print(f"first axial mode: omega_1 = {OMEGA_1:.1f} rad/s  ({OMEGA_1/2/np.pi:.1f} Hz)")

# continuation runs in the nondimensional frequency w_hat = w / omega_1
OMEGA_START = 1900        # w_hat
OMEGA_END   = 3400        # w_hat


# ============================ build problem =================================

HarmonicBalanceMethod.update_dependencies(HARMONICS, rod.polynomial_degree)
provider = NumericalFRF(rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP,
                       gamma=GAMMA, lanczos_m=LANCZOS_M, lanczos_cutoff=LANCZOS_CUTOFF)
problem  = FBSProblem(rod, provider, contact)
print(f"DOFs: total = {rod.total_dimension}, interface = {rod.dimension}, "
      f"N_time = {Fourier.number_of_time_samples}, N_h = {Fourier.number_of_harmonics}")


# ============================ FD checks (contact-active point) ==============

print("=" * 70)
print("Finite-difference checks of DLFT residual Jacobian and dr/domega")
print("=" * 70)
omega_fd = 1.0                                   # w_hat = 1 -> first resonance
Q1_fd    = np.array([[(2.0 * GAP) + 0.0j]])      # |u_B| > d -> contact active
x_fd     = FourierOmegaPoint.new_from_first_harmonic(Q1_fd, omega=omega_fd)

r0 = problem.compute_residue_RI(x_fd)
J_an = problem.compute_jacobian_of_residue_RI(x_fd)
n_q  = J_an.shape[0]
h    = 1.0e-7
J_fd = np.zeros_like(J_an)
x_arr = np.asarray(x_fd).copy()
for j in range(n_q):
    xp = x_arr.copy(); xp[j, 0] += h
    x_p = FourierOmegaPoint(Fourier.new_from_RI(xp[:-1]), x_fd.omega)
    J_fd[:, j] = (problem.compute_residue_RI(x_p) - r0).ravel() / h
errJ = np.abs(J_an - J_fd)
relJ = errJ.max() / (np.abs(J_fd).max() + 1e-30)
print(f"  Jacobian : max|J| = {np.abs(J_an).max():.3e}, max abs err = {errJ.max():.3e}, rel = {relJ:.2e}")

dw_an = problem.compute_derivative_wrt_omega_RI(x_fd)
hw = 1.0e-3
rp = problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega + hw))
rm = problem.compute_residue_RI(FourierOmegaPoint(x_fd.fourier, x_fd.omega - hw))
dw_fd = (rp - rm) / (2 * hw)
errW = np.abs(dw_an - dw_fd)
relW = errW.max() / (np.abs(dw_fd).max() + 1e-30)
print(f"  dr/domega: max|.| = {np.abs(dw_an).max():.3e}, max abs err = {errW.max():.3e}, rel = {relW:.2e}")
print("=" * 70)


# ============================ initial guess (linear) ========================

def linear_free_end(omega):
    M, C, K = rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((rod.total_dimension, 1)); F[-1, 0] = rod.F0
    u = np.linalg.solve(Z, F)
    return u[-1, 0]                      # complex free-end amplitude (1st harmonic)

Q1_start = abs(linear_free_end(OMEGA_START))
print(f"linear free-end amplitude at omega_start: {Q1_start:.3e} m  (gap d = {GAP:.1e} m)")


# ============================ continuation ==================================

solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": 1e-6}
step_kwargs = {
    "base":                      2.0,
    "initial_step_length":       0.005,
    "maximum_step_length":       5.0,
    "minimum_step_length":       1e-6,
    "goal_number_of_iterations": 4,
}

def run_frc(lanczos_m, label):
    """One continuation sweep; returns (omega_phys, peak_uB) along the branch.

    peak_uB = ||x_100(t)||_inf, the infinity-norm (max over a period) of the
    free-end displacement -- the quantity plotted in the paper's Fig. 9.
    """
    contact_l = DLFTContact(epsilon=EPSILON, g_zero=GAP, gamma=GAMMA,
                            lanczos_m=lanczos_m, lanczos_cutoff=LANCZOS_CUTOFF)
    prob_l   = FBSProblem(rod, provider, contact_l)
    solver_l = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=prob_l, corrector_parameterization=ArcLengthParameterization)

    Q1_l = np.array([[linear_free_end(OMEGA_START)]])
    ig   = FourierOmegaPoint.new_from_first_harmonic(Q1_l, omega=OMEGA_START)
    rd   = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=1.0)

    print(f"\nContinuation ({label})")
    t0 = time()
    ss = solver_l.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = 5000,
        angular_frequency_range       = [OMEGA_START, OMEGA_END],
        solver_kwargs                 = solver_kwargs,
        step_length_adaptation_kwargs = step_kwargs,
        jacobian_update_frequency     = 1,
    )
    omega_hat = np.array(ss.omega)
    omega_phys = omega_hat * OMEGA_1
    peak = np.zeros_like(omega_phys)
    for i, (four, o_hat) in enumerate(zip(ss.fourier, omega_hat)):
        full = prob_l.compute_full_response(four, o_hat)
        Fourier_Real.compute_time_series(full)
        peak[i] = float(np.max(np.abs(full.time_series[:, -1, 0])))
        if i == 0:
            Nt_dbg = Fourier.number_of_time_samples
            a_iface = 2.0 / Nt_dbg * np.abs(four.coefficients[1, 0, 0])
            a_full1 = 2.0 / Nt_dbg * np.abs(full.coefficients[1, -1, 0])
            dc_full = 1.0 / Nt_dbg * np.abs(full.coefficients[0, -1, 0])
            har_amp = 2.0 / Nt_dbg * np.abs(full.coefficients[:, -1, 0])
            print(f"   [dbg i=0 omega={omega_phys[0]:.1f}] iface_a1={a_iface:.3e}  "
                  f"full_a1={a_full1:.3e}  full_DC={dc_full:.3e}  ts_peak={peak[0]:.3e}")
            print(f"   [dbg] free-end harmonic amps (2/Nt|c_n|), n=0..6: "
                  + ", ".join(f"{v:.2e}" for v in har_amp[:7]))
    print(f"-> {len(omega_phys)} points, omega in "
          f"[{omega_phys.min():.1f}, {omega_phys.max():.1f}] rad/s, {time() - t0:.1f} s")
    print(f"   peak ||x_100||_inf: min={peak.min():.3e}  max={peak.max():.3e}  "
          f"(scaled /1e-4: {peak.min()/1e-4:.2f} .. {peak.max()/1e-4:.2f})")
    for o, p in zip(omega_phys, peak):
        print(f"     omega={o:8.1f}  peak={p:.4e}  (={p/1e-4:7.3f} x1e-4)")
    # branch order is preserved (do NOT sort: that would break the fold)
    return omega_phys, peak


RUN_RHBM = True   # set False to skip the (light-orange) R-HBM sweep and halve runtime

om_RL, peak_RL = run_frc(LANCZOS_M, "RL-HBM: regularized DLFT + Lanczos")
if RUN_RHBM:
    om_R, peak_R = run_frc(0.0, "R-HBM: regularized DLFT, no Lanczos")


# ============================ linear FRF (reference) ========================
# Pure-harmonic linear response: the inf-norm equals the complex amplitude
# magnitude. Diverges at omega_1, so it is clipped by the y-limit on the plot.
wh_lin   = np.linspace(OMEGA_START, OMEGA_END, 600)
peak_lin = np.array([abs(linear_free_end(w)) for w in wh_lin])
om_lin   = wh_lin * OMEGA_1


# ============================ plot (recreates Fig. 9) =======================

SCALE = 1.0e-4   # y-axis in units of 1e-4 m, as in the paper

fig, ax = plt.subplots(figsize=(9.0, 4.2))

ax.plot(om_lin, peak_lin / SCALE, '-', color="0.78", lw=1.3, label="linear FRF")
ax.axhline(GAP / SCALE, color="0.35", lw=1.6, label="contact threshold")
if RUN_RHBM:
    ax.plot(om_R, peak_R / SCALE, '-', color="#F6B26B", lw=1.7, label="R-HBM")
ax.plot(om_RL, peak_RL / SCALE, '-', color="#E06C00", lw=2.7, label="RL-HBM")

ax.axvline(OMEGA_1, color="0.5", ls="--", lw=0.8)

ax.set_xlim(OMEGA_START * OMEGA_1, OMEGA_END * OMEGA_1)
ax.set_ylim(0.0, 5.0)
ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
ax.set_ylabel(r"$\|\mathbf{x}_{100}(t)\|_\infty$   [$\times 10^{-4}$ m]")

# mark omega_1 on the x-axis (add a tick + label, paper-style)
ticks = list(ax.get_xticks())
ax.set_xticks(sorted(ticks + [OMEGA_1]))
ax.set_xticklabels([(r"$\omega_1$" if abs(t - OMEGA_1) < 1.0 else f"{t:.0f}")
                    for t in sorted(ticks + [OMEGA_1])])

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
plt.tight_layout()

out = Path(__file__).parent / "rod_vibroimpact_frc.png"
fig.savefig(out, dpi=150)
print(f"\nFigure saved: {out}")
plt.show()
