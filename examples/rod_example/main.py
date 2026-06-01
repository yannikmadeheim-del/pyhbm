"""
Clamped-free FE rod impacting a FLEXIBLE obstacle (Vadcard, Batailly & Thouverez,
J. Sound Vib. 531 (2022) 116950, Fig. 6/17 + Table 1).

The flexible wall is modelled as a SECOND SUBSTRUCTURE -- a grounded linear spring
k_obs -- coupled to the rod through Frequency-Based Substructuring (FBS).  The
unmodified, *rigid* DLFT contact then acts on the RELATIVE interface DOF

        x_r(t) = u_B(t) - u_w(t),        contact when  x_r > g0,

where u_B is the rod free-end displacement and u_w the obstacle-spring node.  The
obstacle compliance 1/k_obs enters the interface admittance Y_r = B Y B^T
automatically, so at convergence the contact force satisfies the obstacle law
        lambda = k_obs * (u_B - g0)     (penetration resisted by finite stiffness)
and the rigid wall is recovered as k_obs -> infinity.  No change to the DLFT
nonlinear-method code is needed: the flexibility lives entirely in the FRF.

Model (Vadcard Table 1):
    A = 15.6 cm^2,  E = 210 GPa,  rho = 7800 kg/m^3,  L = 13 cm,
    g0 = 0.2 mm,  n = 20 two-node bar elements (left end A clamped, right end B free).
    element matrices  M_e = rho*A*l/6 [[2,1],[1,2]],  K_e = E*A/l [[1,-1],[-1,1]].
    harmonic forcing f_ex = 25e3 N at node B;  k_rod = E*A/L ~ 2.5e9 N/m.

Obstacle (flexible wall):  k_obs = k_rel * k_rod,  k_rel = k_obs / k_rod the
relative obstacle stiffness.  This script reproduces Vadcard Fig. 17: the NFRC for
k_rel = 0.4, 4, 20, 40 (weakly -> strongly nonlinear), as a 2x2 panel grid.
"""
import sys
from pathlib import Path

from pyhbm.numerical_continuation.corrector_step import (
    ArcLengthParameterization, OrthogonalParameterization,
)
from pyhbm.numerical_continuation.predictor_step import (
    TangentPredictorOne, TangentPredictorBordered,
)

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

class RodVibroImpactFlexible(FBS_System):
    """Clamped-free axial bar whose free end B impacts a FLEXIBLE wall.

    The wall is a grounded spring k_obs appended as one extra DOF (the obstacle
    node u_w).  The assembled system is block-diagonal -- rod and obstacle are
    linearly UNcoupled; they interact only through the unilateral DLFT contact on
    the relative interface DOF  x_r = u_B - u_w.  Hence:

        M = blkdiag(M_rod, 0),  C = blkdiag(C_rod, 0),  K = blkdiag(K_rod, k_obs)
        B_coupling = [ ... +1 (rod tip) ... -1 (obstacle node) ]   ->  x_r = u_B - u_w

    The obstacle stiffness is k_obs = k_rel * k_rod with k_rod = E*A/L.
    """
    is_real_valued = True

    def __init__(self, n_elem=20, L=0.13, E=210e9, rho=7800.0, A=15.6e-4,
                 F0=1.0e4, xi=7.5e-3, k_rel=1.0, poly_deg=33):
        l = L / n_elem
        n = n_elem                       # rod free DOF count (node 0 clamped)

        # --- assemble global rod M, K, then drop the clamped DOF (node 0) ---
        Me = (rho * A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        Ke = (E * A / l)         * np.array([[1.0, -1.0], [-1.0, 1.0]])
        M_rod = zeros((n + 1, n + 1))
        K_rod = zeros((n + 1, n + 1))
        for e in range(n_elem):
            M_rod[e:e + 2, e:e + 2] += Me
            K_rod[e:e + 2, e:e + 2] += Ke
        M_rod = M_rod[1:, 1:]            # remove clamped node 0
        K_rod = K_rod[1:, 1:]

        # --- modal damping: C = M Phi diag(2 xi w_i) Phi^T M  (Phi mass-normalized) ---
        w2, Phi = eigh(K_rod, M_rod)     # Phi^T M Phi = I
        omega_modes = np.sqrt(np.clip(w2, 0.0, None))
        C_rod = M_rod @ Phi @ np.diag(2.0 * xi * omega_modes) @ Phi.T @ M_rod

        # --- flexible wall: grounded spring k_obs = k_rel * k_rod ---
        # The rigid wall is the limit k_rel -> infinity; a finite k_rel is a finite
        # obstacle stiffness.  k_rel must be > 0 (k_rel = 0 would mean no obstacle).
        if k_rel <= 0.0:
            raise ValueError("k_rel must be > 0 (k_obs = k_rel * k_rod).")
        k_rod = E * A / L                # rod static axial stiffness  [N/m]
        k_obs = k_rel * k_rod
        self.k_rod = k_rod
        self.k_ref = k_rod               # alias kept for back-compat
        self.k_obs = k_obs
        self.k_rel = k_rel

        # --- augment with the obstacle node (zero mass, zero damping, k_obs to ground) ---
        d_tot = n + 1                    # rod DOFs (n) + obstacle node (1)
        M = zeros((d_tot, d_tot)); M[:n, :n] = M_rod
        C = zeros((d_tot, d_tot)); C[:n, :n] = C_rod
        K = zeros((d_tot, d_tot)); K[:n, :n] = K_rod
        K[n, n] = k_obs                  # obstacle spring to ground

        # --- nondimensionalize frequency by omega_1 of the ROD: sweep w_hat = w/w_1.
        # With M' = w1^2 M, C' = w1 C, K' = K the FRF in w_hat equals the physical
        # FRF in w (the constant obstacle block k_obs is frequency-independent, so
        # it is unaffected), so the arc-length stepper sees an O(1) frequency axis. ---
        omega_ref = omega_modes[0]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * M
        self.damping_matrix   = omega_ref      * C
        self.stiffness_matrix = K

        # interface = relative DOF x_r = u_B - u_w  (rod tip minus obstacle node)
        self.rod_tip_idx  = n - 1
        self.obstacle_idx = n
        B = zeros((1, d_tot))
        B[0, self.rod_tip_idx]  = +1.0
        B[0, self.obstacle_idx] = -1.0
        self.B_coupling = B

        self.dimension        = 1        # n_int: one contact DOF
        self.total_dimension  = d_tot
        self.polynomial_degree = poly_deg
        self.F0 = F0
        self.omega_modes = omega_modes

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)   # forcing at the free end B
        return f

    # DLFT supplies the contact force; AFT stubs kept for interface completeness.
    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))


# ============================ parameters ====================================
GAP       = 0.2e-3       # g0: wall offset [m]  (Vadcard Table 1)
HARMONICS = list(range(0, 21))   # 0..20  -> H = 20
POLY_DEG  = 30                   # N = (POLY_DEG+1)*20 + 1 = 421 time samples
F0        = 25e3                 # f_ex: harmonic forcing at node B [N]  (Vadcard)

# Obstacle stiffnesses to sweep, as multiples of k_rod = E*A/L
# (Vadcard Fig. 17 a-d:  k_obs = 0.4, 4, 20, 40 x k_rod).
K_REL_VALUES = [0.4, 4.0, 20.0, 40.0]

# default system (k_rel = 1) used for the omega_1 reference, FD checks and the
# linear (no-contact) FRF -- all of which are independent of k_obs.
rod = RodVibroImpactFlexible(k_rel=1.0, F0=F0, poly_deg=POLY_DEG)
OMEGA_1 = rod.omega_modes[0]
# DLFT penalty: stiffness units; large vs the interface dynamic stiffness ~ k_rod.
# Converged solution is epsilon-independent (Vadcard 2022), so any large value works.
EPSILON   = 1.0e2 * rod.k_rod

print(f"first axial mode: omega_1 = {OMEGA_1:.1f} rad/s  ({OMEGA_1/2/np.pi:.1f} Hz)")
print(f"rod static stiffness k_rod = E*A/L = {rod.k_rod:.3e} N/m")
print(f"sweeping k_rel = k_obs/k_rod in {K_REL_VALUES}")
print(f"DLFT penalty epsilon = {EPSILON:.3e}")

# continuation runs in the nondimensional frequency w_hat = w / omega_1.
# Plot window matches Vadcard Fig. 17.
OMEGA_START = 1.20              # w_hat
OMEGA_END   = 0.90               # w_hat  (room for the contact stiffening overhang)


# ============================ build problem (default) =======================

HarmonicBalanceMethod.update_dependencies(HARMONICS, rod.polynomial_degree)
provider = NumericalFRF(rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(rod, provider, contact)
print(f"DOFs: total = {rod.total_dimension} (rod {rod.total_dimension-1} + obstacle 1), "
      f"interface = {rod.dimension}, "
      f"N_time = {Fourier.number_of_time_samples}, N_h = {Fourier.number_of_harmonics}")


# ============================ FD checks (contact-active point) ==============

print("=" * 70)
print("Finite-difference checks of DLFT residual Jacobian and dr/domega")
print("=" * 70)
omega_fd = 1.0                                   # w_hat = 1 -> first resonance
Q1_fd    = np.array([[(2.0 * GAP) + 0.0j]])      # |x_r| > g0 -> contact active
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

def linear_relative(omega):
    """Linear (no-contact) relative interface response x_r = B u at frequency omega.

    In the linear regime the obstacle carries no force (u_w = 0), so x_r = u_B; the
    result is therefore independent of k_obs.
    """
    M, C, K = rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((rod.total_dimension, 1)); F[rod.rod_tip_idx, 0] = rod.F0
    u = np.linalg.solve(Z, F)
    return (rod.B_coupling @ u)[0, 0]    # complex relative amplitude (1st harmonic)

Q1_start = abs(linear_relative(OMEGA_START))
print(f"linear free-end amplitude at omega_start: {Q1_start:.3e} m  (gap g0 = {GAP:.1e} m)")


# ============================ continuation ==================================

solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
step_kwargs = {
    "base":                      4.0,
    "initial_step_length":       0.002,
    "maximum_step_length":       0.001,   # narrow w_hat window -> keep steps small
    "minimum_step_length":       1e-7,
    "goal_number_of_iterations": 3,
}

def run_frc(k_rel_value,
            parameterization=ArcLengthParameterization,
            predictor=TangentPredictorBordered):
    """Continuation sweep for one obstacle stiffness  k_obs = k_rel_value * k_rod.

    Rebuilds the rod + grounded-spring substructure system for this k_obs, runs the
    DLFT-HBM continuation, and returns (omega_phys, peak_uB) along the branch.
    peak_uB = ||u_B(t)||_inf (max over a period of the rod free-end displacement).
    Branch order is preserved -- do NOT sort, that would break the fold.
    """
    sys_k  = RodVibroImpactFlexible(k_rel=k_rel_value, F0=F0, poly_deg=POLY_DEG)
    prov_k = NumericalFRF(sys_k.mass_matrix, sys_k.damping_matrix, sys_k.stiffness_matrix)
    cont_k = DLFTContact(epsilon=EPSILON, g_zero=GAP)
    prob_k = FBSProblem(sys_k, prov_k, cont_k)
    solver_k = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=prob_k,
                                     corrector_parameterization=parameterization,
                                     predictor=predictor)

    Q1_l = np.array([[linear_relative(OMEGA_START)]])
    ig   = FourierOmegaPoint.new_from_first_harmonic(Q1_l, omega=OMEGA_START)
    rd   = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=-1.0)

    print(f"\nContinuation: k_obs = {k_rel_value:g} k_rod  (k_obs = {sys_k.k_obs:.3e} N/m)")
    t0 = time()
    ss = solver_k.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = 5000,
        angular_frequency_range       = [OMEGA_START, OMEGA_END],
        solver_kwargs                 = solver_kwargs,
        step_length_adaptation_kwargs = step_kwargs,
        jacobian_update_frequency     = 1,
    )
    omega_hat  = np.array(ss.omega)
    omega_phys = omega_hat * OMEGA_1
    peak = np.zeros_like(omega_phys)
    for i, (four, o_hat) in enumerate(zip(ss.fourier, omega_hat)):
        full = prob_k.compute_full_response(four, o_hat)
        Fourier_Real.compute_time_series(full)
        peak[i] = float(np.max(np.abs(full.time_series[:, sys_k.rod_tip_idx, 0])))
    print(f"-> {len(omega_phys)} points, omega in "
          f"[{omega_phys.min():.1f}, {omega_phys.max():.1f}] rad/s, "
          f"peak/1e-4 in [{peak.min()/1e-4:.2f}, {peak.max()/1e-4:.2f}], {time()-t0:.1f} s")
    # return branch + raw solution set (needed for warm-starting the rigid case)
    return omega_phys, peak, ss, sys_k


# one continuation branch per obstacle stiffness
raw = {kv: run_frc(kv) for kv in K_REL_VALUES}
results = {kv: (raw[kv][0], raw[kv][1]) for kv in K_REL_VALUES}


# ============================ linear FRF (reference) ========================
# Pure-harmonic linear (no-contact) relative response; same for every k_obs (the
# obstacle is decoupled until contact). Diverges at omega_1 -> clipped by y-limit.
wh_lin   = np.linspace(OMEGA_START, OMEGA_END, 600)
peak_lin = np.array([abs(linear_relative(w)) for w in wh_lin])
om_lin   = wh_lin * OMEGA_1


# ============================ plot (Vadcard Fig. 17 style) ==================
# 2x2 grid, one panel per k_obs.  Each panel: linear FRF (dotted), contact
# threshold g0 (red dashed), and the DLFT-HBM flexible-obstacle NFRC (orange).

SCALE = 1.0e-4              # y-axis in units of 1e-4 m
XLIM  = (5.8e4, 7.4e4)      # Vadcard Fig. 17 frequency window [rad/s]
YLIM  = (0.0, 5.5)
PANEL = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
for ax, lbl, kv in zip(axes.ravel(), PANEL, K_REL_VALUES):
    om_k, peak_k = results[kv]
    ax.plot(om_lin, peak_lin / SCALE, ':',  color="k", lw=1.0, label="linear")
    ax.axhline(GAP / SCALE,                  color="red", ls="--", lw=1.2, label="$g_0$")
    ax.plot(om_k, peak_k / SCALE, '-', color="#E8820C", lw=1.8,
            label="DLFT-HBM (flexible)")
    ax.set_title(rf"{lbl}  $k_\mathrm{{obs}} = {kv:g}\,k_\mathrm{{rod}}$", fontsize=11)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.25)

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\|x(t)\|_\infty$  [$\times 10^{-4}$ m]")
for ax in axes[1, :]:
    ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)

# ============================ NLvib CSV overlay =============================
# Load the four CSVs produced by NLvib/validation/nlvib_vibroimpact_rod_flexible.m
# and add them as black dot markers on each panel for cross-validation.
import pandas as pd
NLVIB_DIR = (Path(__file__).parent.parent.parent.parent.parent
             / "NLvib" / "validation")
for ax, kv in zip(axes.ravel(), K_REL_VALUES):
    # NLvib writes e.g. nlvib_rod_flexible_kobs_0.4_krod.csv
    csv = NLVIB_DIR / f"nlvib_rod_flexible_kobs_{kv:g}_krod.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        ax.plot(df["omega"], df["A_peak"] / SCALE, 'k.',
                ms=2.5, label="NLvib HB", zorder=5)
    else:
        print(f"  [overlay] CSV not found: {csv.name}")

# refresh legend only on first panel (now has NLvib entry if CSV present)
axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle("NFRC vs. obstacle stiffness  (rod + flexible wall as 2 substructures, "
             "DLFT-HBM  vs  NLvib HBM)  -- cf. Vadcard Fig. 17", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out = Path(__file__).parent / "rod_vibroimpact_frc.png"
fig.savefig(out, dpi=150)
print(f"\nFigure saved: {out}")


# ============================ rigid wall: warm-start from k_rel=40 ==========
# The rigid wall is approached as k_rel → ∞.  We use k_rel = 1000 as a proxy
# (k_obs = 2.52e12 N/m ≫ rod dynamic stiffness) and warm-start Newton from
# the last converged point of the k_rel=40 branch.  Because the DOF structure
# is identical (same B_coupling, same obstacle DOF), the Fourier coefficients
# transfer directly; only the FRF/impedance changes.

print("\n" + "=" * 70)
print("Rigid wall (k_rel=1000, warm-start from k_rel=40 branch)")
print("=" * 70)

K_REL_RIGID = 1000.0
ss_40  = raw[40.0][2]          # raw solution set from k_rel=40
sys_40 = raw[40.0][3]          # system object from k_rel=40

sys_rigid  = RodVibroImpactFlexible(k_rel=K_REL_RIGID, F0=F0, poly_deg=POLY_DEG)
prov_rigid = NumericalFRF(sys_rigid.mass_matrix,
                          sys_rigid.damping_matrix,
                          sys_rigid.stiffness_matrix)
cont_rigid = DLFTContact(epsilon=EPSILON, g_zero=GAP)
prob_rigid = FBSProblem(sys_rigid, prov_rigid, cont_rigid)

solver_rigid = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=prob_rigid,
    corrector_parameterization=OrthogonalParameterization,
    predictor=TangentPredictorBordered,
)

# Use the last solution of the 40*k_rod branch as initial guess.
# Both systems share the same Newton unknown (relative DOF x_r = u_B - u_w),
# so the Fourier coefficients transfer directly.
ig_rigid = FourierOmegaPoint(ss_40.fourier[-1], ss_40.omega[-1])
rd_rigid = FourierOmegaPoint.new_from_first_harmonic(
    np.zeros((1, 1), complex), omega=1.0)

step_kwargs_rigid = {
    "base":                      2.0,
    "initial_step_length":       0.001,
    "maximum_step_length":       0.003,
    "minimum_step_length":       1e-7,
    "goal_number_of_iterations": 3,
}

t0 = time()
ss_rigid = solver_rigid.solve_and_continue(
    initial_guess                 = ig_rigid,
    initial_reference_direction   = rd_rigid,
    maximum_number_of_solutions   = 5000,
    angular_frequency_range       = [OMEGA_START, OMEGA_END],
    solver_kwargs                 = solver_kwargs,
    step_length_adaptation_kwargs = step_kwargs_rigid,
    jacobian_update_frequency     = 1,
)

omega_hat_r  = np.array(ss_rigid.omega)
omega_phys_r = omega_hat_r * OMEGA_1
peak_r = np.zeros_like(omega_phys_r)
for i, (four, o_hat) in enumerate(zip(ss_rigid.fourier, omega_hat_r)):
    full = prob_rigid.compute_full_response(four, o_hat)
    Fourier_Real.compute_time_series(full)
    peak_r[i] = float(np.max(np.abs(full.time_series[:, sys_rigid.rod_tip_idx, 0])))

print(f"-> {len(omega_phys_r)} points, omega in "
      f"[{omega_phys_r.min():.1f}, {omega_phys_r.max():.1f}] rad/s, "
      f"peak/1e-4 in [{peak_r.min()/1e-4:.2f}, {peak_r.max()/1e-4:.2f}], {time()-t0:.1f} s")


# ============================ plot: rigid vs flexible comparison =============

fig2, ax2 = plt.subplots(figsize=(9.0, 4.5))

ax2.plot(om_lin, peak_lin / SCALE, ':',  color="k", lw=1.0, label="linear FRF")
ax2.axhline(GAP / SCALE,                  color="red", ls="--", lw=1.2, label="$g_0$")

# flexible branches (faded orange, labeled by k_rel)
colors_fl = ["#FACC8E", "#F0A046", "#E8820C", "#C05800"]
for (kv, col) in zip(K_REL_VALUES, colors_fl):
    om_k, pk_k = results[kv]
    ax2.plot(om_k, pk_k / SCALE, '-', color=col, lw=1.4,
             label=f"flexible $k_\\mathrm{{obs}}={kv:g}k_\\mathrm{{rod}}$")

# rigid branch (thick blue)
ax2.plot(omega_phys_r, peak_r / SCALE, '-', color="#1F77B4", lw=2.4,
         label=f"near-rigid ($k_\\mathrm{{obs}}={K_REL_RIGID:.0f}k_\\mathrm{{rod}}$)")

ax2.set_xlim(*XLIM); ax2.set_ylim(*YLIM)
ax2.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
ax2.set_ylabel(r"$\|x(t)\|_\infty$  [$\times 10^{-4}$ m]")
ax2.set_title("Flexible → rigid obstacle: stiffness sweep + warm-started rigid branch")
ax2.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.25)
fig2.tight_layout()

out2 = Path(__file__).parent / "rod_vibroimpact_rigid_vs_flexible.png"
fig2.savefig(out2, dpi=150)
print(f"Figure saved: {out2}")

plt.show()
