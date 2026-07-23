"""Branch runner + metrics for the two-rod example.

`run_branch` wires one (system, FRF provider, nonlinear method) triple into an
FBSProblem, runs the arc-length DLFT/AFT-HBM continuation across the frequency
window, post-processes TWO response signals per branch point --

    "tipA" : the driven tip displacement u_A(t)           (rod A free end)
    "xr"   : the relative approach x_r(t) = u_A + u_B     (interface DOF)

-- and bundles the branch with the performance metrics into a
:class:`BranchResult`.  Both signals get a peak amplitude ||.||_inf and a full
multiharmonic spectrum, so the trajectory error can be evaluated per signal.
"""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy import zeros

from pyhbm import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, HarmonicBalanceMethod,
)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered

SIGNALS = ("tipA", "xr")

# response-signal labels, shared by main.py (quicklook) and studies/plotting.py
SIGNAL_TEX  = {"tipA": r"u_A", "xr": r"x_r"}
SIGNAL_DESC = {"tipA": "driven tip $u_A$",
               "xr":   "relative approach $x_r = u_A + u_B$"}


# ============================ result container ==============================

@dataclass
class BranchResult:
    label:   str                    # human-readable variant name
    method:  str                    # "dlft" | "aft"
    frf:     str                    # "numerical" | "experimental"
    params:  dict                   # {k_rel, alpha, density, n_freq, ...}

    omega_phys: np.ndarray          # physical angular frequency [rad/s]
    omega_hat:  np.ndarray          # nondimensional omega / omega_1
    peak:       dict = None         # {signal: ||y(t)||_inf along the branch [m]}
    harmonics:  dict = None         # {signal: (n_points, Nh) complex spectrum}

    # metrics
    solve_time:   float = 0.0       # wall-clock continuation time [s]
    n_points:     int   = 0         # converged continuation points
    total_iters:  int   = 0         # summed Newton iterations
    mean_iters:   float = 0.0
    median_step:  float = 0.0       # median arc-length step length
    mean_step:    float = 0.0
    coverage:     float = 0.0       # fraction of the omega window actually traced

    solution_set: object = field(default=None, repr=False)


# ============================ linear reference =============================

def linear_relative(system, omega):
    """Linear (no-contact) relative interface response x_r = B u at frequency omega.

    Without contact the rods are uncoupled and rod B is unforced (u_B = 0), so
    x_r,lin = u_A,lin -- one linear backdrop serves both plot windows.  ``omega``
    is nondimensional.
    """
    M, C, K = system.mass_matrix, system.damping_matrix, system.stiffness_matrix
    Z = -omega ** 2 * M + 1j * omega * C + K
    F = zeros((system.total_dimension, 1))
    F[system.rod_tip_idx, 0] = system.F0
    u = np.linalg.solve(Z, F)
    return (system.B_coupling @ u)[0, 0]


# ============================ branch runner ================================

def run_branch(system, provider, method, *, harmonics, omega_range,
               solver_kwargs, step_kwargs, max_solutions,
               label="", method_name="", frf_name="", params=None,
               parameterization=ArcLengthParameterization,
               predictor=TangentPredictorBordered, verbose=True):
    """Run one continuation branch and return a populated :class:`BranchResult`."""
    omega_start, omega_end = omega_range

    # Fourier class variables (number_of_harmonics, time samples) must be set
    # BEFORE building the FBSProblem, which captures them at construction.
    HarmonicBalanceMethod.update_dependencies(harmonics, system.polynomial_degree)
    problem = FBSProblem(system, provider, method)
    solver = HarmonicBalanceMethod(
        harmonics=harmonics, freq_domain_ode=problem,
        corrector_parameterization=parameterization, predictor=predictor,
    )

    Q1 = np.array([[linear_relative(system, omega_start)]])
    ig = FourierOmegaPoint.new_from_first_harmonic(Q1, omega=omega_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((1, 1), complex), omega=-1.0)

    t0 = perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = max_solutions,
        angular_frequency_range       = [omega_start, omega_end],
        solver_kwargs                 = solver_kwargs,
        step_length_adaptation_kwargs = step_kwargs,
        jacobian_update_frequency     = 1,
        verbose                       = verbose,
    )
    solve_time = perf_counter() - t0

    omega_hat  = np.array(ss.omega)
    omega_phys = omega_hat * system.omega_ref
    # Full multiharmonic spectrum of BOTH signals at each branch point, stored as
    # PHYSICAL two-sided Fourier coefficients  c_hat_k = c_k / N_time.  The solver's
    # coefficients are unnormalized (rfft scale ~ N_time), and N_time depends on the
    # harmonic order, so this normalization is what makes spectra at different H (e.g.
    # 21-harmonic variant vs 120-harmonic reference) directly comparable.  Parseval:
    # mean_t y^2 = |c_hat_0|^2 + 2 sum_{k>=1} |c_hat_k|^2.
    Nt = Fourier.number_of_time_samples
    Nh = Fourier.number_of_harmonics
    n_pts = len(omega_phys)
    peak      = {s: np.zeros(n_pts) for s in SIGNALS}
    harm_spec = {s: np.zeros((n_pts, Nh), complex) for s in SIGNALS}
    for i, (four, o_hat) in enumerate(zip(ss.fourier, omega_hat)):
        full = problem.compute_full_response(four, o_hat)
        cA = full.coefficients[:, system.rod_tip_idx, 0]
        cB = full.coefficients[:, system.tipB_idx, 0]
        harm_spec["tipA"][i] = cA / Nt
        harm_spec["xr"][i]   = (cA + cB) / Nt
        Fourier_Real.compute_time_series(full)
        uA = full.time_series[:, system.rod_tip_idx, 0]
        uB = full.time_series[:, system.tipB_idx, 0]
        peak["tipA"][i] = float(np.max(np.abs(uA)))
        peak["xr"][i]   = float(np.max(np.abs(uA + uB)))

    iters = np.asarray(ss.iterations, float)
    steps = np.asarray(ss.step_length, float)
    lo, hi = min(omega_start, omega_end), max(omega_start, omega_end)
    covered = (omega_hat.max() - omega_hat.min()) / (hi - lo) if len(omega_hat) else 0.0

    res = BranchResult(
        label=label, method=method_name, frf=frf_name, params=params or {},
        omega_phys=omega_phys, omega_hat=omega_hat,
        peak=peak, harmonics=harm_spec,
        solve_time=solve_time, n_points=len(omega_hat),
        total_iters=int(iters.sum()), mean_iters=float(iters.mean()) if len(iters) else 0.0,
        median_step=float(np.median(steps)) if len(steps) else 0.0,
        mean_step=float(steps.mean()) if len(steps) else 0.0,
        coverage=float(min(covered, 1.0)), solution_set=ss,
    )
    if verbose:
        print(f"  [{label}] {res.n_points} pts, t={solve_time:.1f}s, "
              f"iters={res.total_iters} (mean {res.mean_iters:.1f}), "
              f"mean step={res.mean_step:.2e}, coverage={res.coverage:.0%}")
    return res
