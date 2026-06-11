"""Branch runner + metrics for the rod example.

`run_branch` wires one (system, FRF provider, nonlinear method) triple into an
FBSProblem, runs the arc-length DLFT/AFT-HBM continuation across the frequency
window, post-processes the rod-tip peak amplitude, and bundles the branch with
the performance metrics into a :class:`BranchResult`.
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


# ============================ result container ==============================

@dataclass
class BranchResult:
    label:   str                    # human-readable variant name
    method:  str                    # "dlft" | "aft"
    frf:     str                    # "numerical" | "experimental"
    params:  dict                   # {k_rel, alpha, density, n_freq, ...}

    omega_phys: np.ndarray          # physical angular frequency [rad/s]
    peak:       np.ndarray          # ||u_B(t)||_inf along the branch [m]
    omega_hat:  np.ndarray          # nondimensional omega / omega_1
    uB_harmonics: np.ndarray = None # (n_points, Nh) complex spectrum of u_B

    # metrics
    solve_time:   float = 0.0       # wall-clock continuation time [s]
    n_points:     int   = 0         # converged continuation points
    total_iters:  int   = 0         # summed Newton iterations
    mean_iters:   float = 0.0
    median_step:  float = 0.0       # median arc-length step length
    mean_step:    float = 0.0
    coverage:     float = 0.0       # fraction of the omega window actually traced

    # trajectory relative error vs the stored reference (filled by plotting)
    traj_err:  np.ndarray = None    # per-point relative amplitude error
    traj_mean: float = float("nan")
    traj_max:  float = float("nan")
    traj_rms:  float = float("nan")

    solution_set: object = field(default=None, repr=False)


# ============================ linear reference =============================

def linear_relative(system, omega):
    """Linear (no-contact) relative interface response x_r = B u at frequency omega.

    Independent of k_obs (the obstacle carries no force until contact), so it is a
    good cold initial guess for every variant.  ``omega`` is nondimensional.
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
    peak = np.zeros_like(omega_phys)
    # Full multiharmonic spectrum of the rod tip u_B at each branch point, stored as
    # PHYSICAL two-sided Fourier coefficients  c_hat_k = c_k / N_time.  The solver's
    # coefficients are unnormalized (rfft scale ~ N_time), and N_time depends on the
    # harmonic order, so this normalization is what makes spectra at different H (e.g.
    # 21-harmonic variant vs 120-harmonic reference) directly comparable.  Parseval:
    # mean_t u_B^2 = |c_hat_0|^2 + 2 sum_{k>=1} |c_hat_k|^2.
    Nt = Fourier.number_of_time_samples
    uB_harmonics = np.zeros((len(omega_phys), Fourier.number_of_harmonics), complex)
    for i, (four, o_hat) in enumerate(zip(ss.fourier, omega_hat)):
        full = problem.compute_full_response(four, o_hat)
        uB_harmonics[i] = full.coefficients[:, system.rod_tip_idx, 0] / Nt
        Fourier_Real.compute_time_series(full)
        peak[i] = float(np.max(np.abs(full.time_series[:, system.rod_tip_idx, 0])))

    iters = np.asarray(ss.iterations, float)
    steps = np.asarray(ss.step_length, float)
    lo, hi = min(omega_start, omega_end), max(omega_start, omega_end)
    covered = (omega_hat.max() - omega_hat.min()) / (hi - lo) if len(omega_hat) else 0.0

    res = BranchResult(
        label=label, method=method_name, frf=frf_name, params=params or {},
        omega_phys=omega_phys, peak=peak, omega_hat=omega_hat,
        solve_time=solve_time, n_points=len(omega_hat),
        total_iters=int(iters.sum()), mean_iters=float(iters.mean()) if len(iters) else 0.0,
        median_step=float(np.median(steps)) if len(steps) else 0.0,
        mean_step=float(steps.mean()) if len(steps) else 0.0,
        coverage=float(min(covered, 1.0)), solution_set=ss,
        uB_harmonics=uB_harmonics,
    )
    if verbose:
        print(f"  [{label}] {res.n_points} pts, t={solve_time:.1f}s, "
              f"iters={res.total_iters} (mean {res.mean_iters:.1f}), "
              f"mean step={res.mean_step:.2e}, coverage={res.coverage:.0%}")
    return res
