"""
SDOF vibro-impact validation: pyhbm DLFT vs. NLvib shooting (rigid-wall reference).

Solves the same SDOF mass-spring-damper-wall as the NLvib shooting script
using the new FBSProblem + NumericalFRF + DLFTContact API, then overlays the
two FRCs.
"""
import sys
from pathlib import Path

from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve as lin_solve

from pyhbm import (
    FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod, SolutionSet, TangentPredictorRobust,
)


# ============================ system definition =============================

class SDOFVibroImpact(FBS_System):
    """
    SDOF mass-spring-damper-wall in FBS form.

        m q'' + c q' + k q = F0 cos(tau)
        subject to  q <= g0  (rigid wall, enforced by DLFTContact)
    """
    is_real_valued = True

    def __init__(self, m=1.0, c=0.01, k=1.0, F0=0.02, poly_deg=100, k_rel=30):
        self.mass_matrix       = np.array([[m, 0],[0, 0]])
        self.damping_matrix    = np.array([[c, 0],[0, 0]])
        self.stiffness_matrix  = np.array([[k, 0],[0, k_rel*k]])
        self.B_coupling        = np.array([[1.0, -1.0]])      # 1 interface DOF
        self.total_dimension   = 2
        self.dimension         = 1                       # n_int
        self.polynomial_degree = poly_deg
        self.F0 = F0

    def external_term(self, tau):
        f = np.zeros((len(tau), self.total_dimension, 1))
        f[:, 0, 0] = self.F0 * np.cos(tau)
        return f

    # DLFT computes contact internally; stubs kept so AFT could be plugged in.
    def interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, 1))
    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))


# ============================ parameters ====================================
EPSILON          = 1.0   # first entry; FD check uses this
MAX_POINTS_BETWEEN_PHASES = 5000          # downsample FRC before each warm-start
k_rel_shedule = [100, 150]
PARAMS = dict(m=1.0, c=0.05, k=1.0, F0=0.02)   # c=0.1 -> linear amp at res ~2*g0
GAP       = 0.1
HARMONICS = list(range(0, 30))

OMEGA_START = 0.5
OMEGA_END   = 2.5


# ============================ build problem (NEW API) ======================

system   = SDOFVibroImpact(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)

provider = NumericalFRF(system.mass_matrix, system.damping_matrix, system.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(system, provider, contact)


# ============================ FD Jacobian check =============================
# Exercise the mask machinery: pick a state with |Q1| > g0 so contact is
# active at some time samples. This need NOT be a solution -- we are only
# checking that the analytical Jacobian matches finite differences.

print("=" * 70)
print("Finite-difference check of DLFT residual Jacobian")
print("=" * 70)
omega_fd = 1.0
Q1_fd    = np.array([[0.5 + 0.0j]])     # amp = 0.15 > g0 = 0.1 -> contact active
x_fd     = FourierOmegaPoint.new_from_first_harmonic(Q1_fd, omega=omega_fd)

r0 = problem.compute_residue_RI(x_fd)
print(f"  Test point: ω = {omega_fd}, |Q1| = {abs(Q1_fd[0,0]):.3f}, g0 = {GAP}")
print(f"  ||r0|| = {np.linalg.norm(r0):.3e}   (not a solution -- irrelevant for FD test)")

J_analytical = problem.compute_jacobian_of_residue_RI(x_fd)
n_q          = J_analytical.shape[0]      # 2 * Nh * d_int  (omega NOT included)

h     = 1.0e-7
J_fd  = np.zeros_like(J_analytical)
x_arr = np.asarray(x_fd).copy()
for j in range(n_q):
    x_pert = x_arr.copy()
    x_pert[j, 0] += h
    fourier_pert = Fourier.new_from_RI(x_pert[:-1])      # exclude omega slot
    x_p          = FourierOmegaPoint(fourier_pert, x_fd.omega)
    r_p          = problem.compute_residue_RI(x_p)
    J_fd[:, j]   = (r_p - r0).ravel() / h

abs_err = np.abs(J_analytical - J_fd)
print(f"\n  shape J            = {J_analytical.shape}")
print(f"  max |J_analytic|   = {np.abs(J_analytical).max():.3e}")
print(f"  max |J_fd|         = {np.abs(J_fd).max():.3e}")
print(f"  max abs error      = {abs_err.max():.3e}")
mask_big = abs_err > 1e-8
if mask_big.any():
    rel = abs_err[mask_big] / (np.abs(J_fd[mask_big]) + 1e-12)
    print(f"  max rel error      = {rel.max():.3e}   (where abs error > 1e-8)")
print("\n  top 5 disagreements:")
flat_top = np.argsort(abs_err.ravel())[::-1][:5]
for k in flat_top:
    i, jj = np.unravel_index(k, abs_err.shape)
    print(f"    [{i:3d},{jj:3d}]  analytic = {J_analytical[i, jj]:+.4e}   "
          f"fd = {J_fd[i, jj]:+.4e}   err = {abs_err[i, jj]:.2e}")
print("=" * 70)

# To stop here and inspect the FD output only, uncomment:
# import sys; sys.exit(0)

# ============================ FD check of ∂r/∂ω =============================
# Same test point (ω = omega_fd, |Q1| = 0.15 — contact-active).
# Perturb only omega; coefficients stay fixed. Comparison against
# problem.compute_derivative_wrt_omega_RI(x_fd).

print("Finite-difference check of ∂r/∂ω")
print("=" * 70)

dr_dw_analytical = problem.compute_derivative_wrt_omega_RI(x_fd)

h_w = 1.0e-6
x_omega_p = FourierOmegaPoint(x_fd.fourier, x_fd.omega + h_w)
x_omega_m = FourierOmegaPoint(x_fd.fourier, x_fd.omega - h_w)
r_plus  = problem.compute_residue_RI(x_omega_p)
r_minus = problem.compute_residue_RI(x_omega_m)
dr_dw_fd = (r_plus - r_minus) / (2 * h_w)

abs_err_w = np.abs(dr_dw_analytical - dr_dw_fd)
print(f"  shape           = {dr_dw_analytical.shape}")
print(f"  max |∂r/∂ω_an|  = {np.abs(dr_dw_analytical).max():.3e}")
print(f"  max |∂r/∂ω_fd|  = {np.abs(dr_dw_fd).max():.3e}")
print(f"  max abs error   = {abs_err_w.max():.3e}")
big_w = abs_err_w > 1e-8
if big_w.any():
    rel_w = abs_err_w[big_w] / (np.abs(dr_dw_fd[big_w]) + 1e-12)
    print(f"  max rel error   = {rel_w.max():.3e}   (where abs error > 1e-8)")
print("\n  top 5 disagreements:")
flat_top_w = np.argsort(abs_err_w.ravel())[::-1][:5]
for k in flat_top_w:
    i = int(k)
    print(f"    [{i:3d}]  analytic = {dr_dw_analytical.ravel()[i]:+.4e}   "
          f"fd = {dr_dw_fd.ravel()[i]:+.4e}   err = {abs_err_w.ravel()[i]:.2e}")
print("=" * 70)

# ============================ initial guess =================================

Z  = -OMEGA_START**2 * system.mass_matrix \
   + 1j * OMEGA_START * system.damping_matrix \
   +                    system.stiffness_matrix


initial_guess               = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=OMEGA_START)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=1.0)


# ============================ continuation: ε-sweep =========================
# Phase 1: cold-start continuation at the softest ε (problem already built with
#          EPSILON = EPSILON_SCHEDULE[0]). Soft penalty -> easy convergence.
# Phase 2: walk ε up by 10x per step. At each step, Newton-correct every point
#          of the previous FRC at the new ε via solve_fixed_frequency. Each
#          step is a small perturbation, so Newton stays in the basin.

solver_kwargs = {
    "maximum_iterations": 300,
    "absolute_tolerance": 1e-6,
}
step_kwargs = {
    "base":                      2.0,    # gentler growth (was 2.0)
    "initial_step_length":       0.005,
    "maximum_step_length":       1.0,   # was 0.005
    "minimum_step_length":       1e-6,   # was 1e-7 -- bail earlier instead of micro-stepping
    "goal_number_of_iterations": 4,      # was 3 -- accept more iters before growing
}

print("\n" + "=" * 70)
print("ε-sweep continuation")
print("=" * 70)

# --- Phase 1: cold start at softest ε ----------------------------------------
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem, corrector_parameterization=ArcLengthParameterization)
print(f"\nPhase 1 (cold start) at ε = {EPSILON:.1e}")
t0 = time()
solution_set = solver.solve_and_continue(
    initial_guess                 = initial_guess,
    initial_reference_direction   = initial_reference_direction,
    maximum_number_of_solutions   = 10000,
    angular_frequency_range       = [OMEGA_START, OMEGA_END],
    solver_kwargs                 = solver_kwargs,
    step_length_adaptation_kwargs = step_kwargs,
    jacobian_update_frequency     = 1,    # full Newton -- no Jacobian reuse
)
om_now = np.array(solution_set.omega)
print(f"  -> {len(om_now)} points, "
      f"ω range [{om_now.min():.4f}, {om_now.max():.4f}]")

# Downsample FRC before warm-start phase (keeps Phase 2 tractable)
if len(solution_set.omega) > MAX_POINTS_BETWEEN_PHASES:
    idx = np.linspace(0, len(solution_set.omega) - 1,
                      MAX_POINTS_BETWEEN_PHASES, dtype=int)
    sub = SolutionSet(
        FourierOmegaPoint(solution_set.fourier[idx[0]],
                          solution_set.omega[idx[0]]),
        solution_set.iterations[idx[0]],
        solution_set.step_length[idx[0]],
    )
    for i in idx[1:]:
        sub.append(
            FourierOmegaPoint(solution_set.fourier[i], solution_set.omega[i]),
            solution_set.iterations[i],
            solution_set.step_length[i],
        )
    print(f"  subsampled {len(om_now)} -> {len(sub.omega)} points for warm-start")
    solution_set = sub

# ============================ diagnostics (always run) =====================

omegas_arr = np.array(solution_set.omega)
it_arr     = np.array(solution_set.iterations)
sl_arr     = np.array(solution_set.step_length)

print(f"\nSolution set: {len(omegas_arr)} points")
print(f"ω range: [{omegas_arr.min():.4f}, {omegas_arr.max():.4f}]")
print(f"  ω in [0.5, 1.0): {int(((omegas_arr >= 0.5) & (omegas_arr < 1.0)).sum())}")
print(f"  ω in [1.0, 1.5): {int(((omegas_arr >= 1.0) & (omegas_arr < 1.5)).sum())}")
print(f"  ω >= 1.5       : {int((omegas_arr >= 1.5).sum())}")

print(f"\nLast 10 continuation points:")
print(f"  ω             iter   step_length")
for i in range(max(0, len(omegas_arr) - 10), len(omegas_arr)):
    print(f"  {omegas_arr[i]:12.6f}   {it_arr[i]:3d}   {sl_arr[i]:.2e}")

plateau_idx = np.where((omegas_arr > 0.95) & (omegas_arr < 1.5))[0]
print(f"\nResidual norms in contact regime "
      f"({len(plateau_idx)} points, showing up to 10):")
if len(plateau_idx) > 0:
    step = max(1, len(plateau_idx) // 10)
    for i in plateau_idx[::step]:
        om  = solution_set.omega[i]
        x_i = FourierOmegaPoint(solution_set.fourier[i], om)
        r   = problem.compute_residue_RI(x_i)
        print(f"  ω = {om:.4f}   ||r|| = {np.linalg.norm(r):.3e}")
else:
    print("  (no points found in [0.95, 1.5] -- continuation halted at contact onset)")


# ============================ post-process amplitudes =====================

Nt     = Fourier.number_of_time_samples
omegas = np.array(solution_set.omega)
A1     = np.zeros_like(omegas)
Apeak  = np.zeros_like(omegas)

for i, (four, om) in enumerate(zip(solution_set.fourier, omegas)):
    full = problem.compute_full_response(four, om)
    A1[i] = 2.0 / Nt * np.abs(full.coefficients[1, 0, 0])
    Fourier_Real.compute_time_series(full)
    Apeak[i] = float(np.max(np.abs(full.time_series[:, 0, 0])))


# ============================ NLvib reference =============================

ref_csv = Path(__file__).parent / "nlvib_sdof_vibroimpact_shooting_frc.csv"
df = pd.read_csv(ref_csv) if ref_csv.exists() else None
if df is None:
    print(f"\nWARNING: NLvib CSV not found at {ref_csv}. Overlay skipped.")


# ============================ plot =========================================

fig, ax = plt.subplots(figsize=(8.5, 5.5))
if df is not None:
    ax.plot(df["omega"], df["A1"],     'o-', color="C1", ms=3, lw=1.0,
            label="NLvib shooting -- 1st harmonic")
    ax.plot(df["omega"], df["A_peak"], 'o-', color="C3", ms=3, lw=1.0,
            label="NLvib shooting -- peak |q(t)|")
ax.plot(omegas, A1,    '-',  color="C0", lw=1.6, label="pyhbm DLFT -- 1st harmonic")
ax.plot(omegas, Apeak, '--', color="C2", lw=1.4, label="pyhbm DLFT -- peak |q(t)|")
ax.axhline(GAP, color="k", ls=":", lw=0.8, label=f"gap $g_0$={GAP}")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("amplitude")
ax.set_title(f"SDOF vibro-impact -- DLFT (ε={EPSILON:.0e}, ε-sweep) vs NLvib shooting")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.4)
plt.tight_layout()

out = Path(__file__).parent / "comparison_dlft_vs_nlvib.png"
fig.savefig(out, dpi=130)
print(f"\nFigure saved: {out}")
plt.show()
