"""
Two bar+beam elements with frictional contact -- thesis case study 6.4.

Traces the same forced-response branch TWICE, with two different solvers:

    second_order   HarmonicBalanceMethod(second_order_ode=...) on the 6 tip
                   degrees of freedom: M q'' + C q' + K q + f_nl = f_ext.
    first_order    HarmonicBalanceMethod(first_order_ode=...) on the 12-component
                   state z = [q ; qdot].

They share nothing but the force law in dynamical_system.py, so agreement
between the two branches is a real check. Each run writes one CSV into
results/, laid out exactly like the pyFBS two_bar_beam_friction example's, so a
file can be copied over there by hand and plotted next to its curves.

Model parameters live at the top of dynamical_system.py, solver settings at the
top of this file. Run with:

    python examples/two_bar_beam_friction/main.py
"""
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib.pyplot as plt

from pyhbm import (ExponentialAdaptation, Fourier, FourierOmegaPoint,
                   HarmonicBalanceMethod, TangentPredictorOne,
                   save_solution_csv)
# not re-exported by pyhbm/__init__.py -- imported from its defining module
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization

import dynamical_system as ds
from dynamical_system import (BarBeamFirstOrder, BarBeamSecondOrder, B_COUPLING,
                              C_MATRIX, DIMENSION, DOF_LABELS, IF_LABELS,
                              K_MATRIX, M_MATRIX, POLYNOMIAL_DEGREE)


# ===========================================================================
# Solver settings. Edit here.
# ===========================================================================

# harmonic 0 is MANDATORY: the clamping force P5 is static. The thesis reports
# that H = 15 is already very close to its H = 70 reference.
HARMONICS = list(range(0, 16))

OMEGA_START, OMEGA_END = 0.85, 1.09     # thesis continuation window
SWEEP = "up"                            # "up" | "down"

PARAMETERIZATION = ArcLengthParameterization
PREDICTOR = TangentPredictorOne
STEP_ADAPTATION = ExponentialAdaptation

SOLVER_KWARGS = {"maximum_iterations": 300, "absolute_tolerance": 1e-8}
# The arc length is measured over the WHOLE solution vector, whose entries are
# rFFT coefficients (N_t/2 times the physical amplitude), so |X| ~ 140 here.
# Step lengths therefore have to be O(1), not O(0.01): at 0.05 the branch
# advances omega by only 8e-5 per step and the window needs ~3000 points.
STEP_KWARGS = {"base": 3.0, "initial_step_length": 0.2,
               "maximum_step_length": 2.0, "minimum_step_length": 1e-6,
               "goal_number_of_iterations": 4}
JACOBIAN_UPDATE_FREQUENCY = 1
MAXIMUM_NUMBER_OF_SOLUTIONS = 5000

RESULTS = Path(__file__).resolve().parent / "results"

# every exported channel, in the order the harmonic block uses
LABELS = list(IF_LABELS) + list(DOF_LABELS)
OUT_LABEL = "q1"          # the summarized output channel: axial tip of element 1


# ===========================================================================
# Setup shared by both runs
# ===========================================================================

# Fourier state is class-level and FRFProblem snapshots it, so the harmonics
# must be registered BEFORE the problem is built.
HarmonicBalanceMethod.update_dependencies(HARMONICS, POLYNOMIAL_DEGREE)
N_TIME_SAMPLES = Fourier.number_of_time_samples

OMEGA_FIRST = OMEGA_START if SWEEP == "up" else OMEGA_END
DIRECTION = +1.0 if SWEEP == "up" else -1.0


def linear_response(omega):
    """Linear (open-contact) response amplitude a_1 to the axial excitation."""
    Z = -omega**2 * M_MATRIX + 1j * omega * C_MATRIX + K_MATRIX
    f = np.zeros((DIMENSION, 1), complex)
    f[ds.Q1, 0] = ds.P1
    return np.linalg.solve(Z, f)[:, 0]


def initial_guess(dimension, lift):
    """
    Cold start: the static contact state in harmonic 0 and the linear axial
    response in harmonic 1.

    Without the static seed the first Newton solve starts fully separated and
    has to find the preload from scratch. pyhbm's rFFT convention is
    c_0 = N_t a_0 and c_h = N_t/2 a_h for h > 0.

    :param lift: maps a 6-component displacement amplitude onto the solver's
        coordinates (identity for q, [q ; 1j w q] for the first-order state).
    """
    _, N = ds.static_contact_state()
    q_static = np.zeros(DIMENSION)
    q_static[ds.Q2] = N - ds.P5              # see static_contact_state
    q_static[ds.Q3] = 1.5 * q_static[ds.Q2]
    q_static[ds.Q5] = -q_static[ds.Q2]
    q_static[ds.Q6] = -q_static[ds.Q3]

    guess = FourierOmegaPoint.zero_amplitude(dimension=dimension, omega=OMEGA_FIRST)
    h0, h1 = HARMONICS.index(0), HARMONICS.index(1)
    guess.fourier.coefficients[h0, :, 0] = N_TIME_SAMPLES * lift(q_static, 0.0)
    guess.fourier.coefficients[h1, :, 0] = (N_TIME_SAMPLES / 2) * lift(
        linear_response(OMEGA_FIRST), OMEGA_FIRST)
    return guess


def header_lines(formulation, description, solve_time, n_points):
    """Plain-text CSV header: the model, the force laws, and every solver setting."""
    return [
        "two_bar_beam_friction -- Tiago Martins' MSc thesis, case study 6.4",
        f"formulation: {formulation}  ({description})",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        "",
        "structure: two identical cantilever bar+beam elements, uncoupled in the linear",
        "  system (block-diagonal M, K); tip DoFs [axial, transverse, slope] =",
        "  q1,q2,q3 (element 1) and q4,q5,q6 (element 2), thesis eq. (6.12)/(6.13)",
        f"  l={ds.L}, EA={ds.EA}, EI={ds.EI}, lambda={ds.LAM}, C=beta*K with beta={ds.BETA}",
        "interface: x_rel = [x_N ; x_T] = [q5 - q2 ; q1 - q4]; contact when x_N > eps",
        "normal (thesis eq. 6.15, regularized unilateral spring):",
        f"  N = ln(1 + exp(k*alpha2*(x_N - eps)))/alpha2, k={ds.K_C},"
        f" alpha2={ds.ALPHA2}, eps={ds.EPS}",
        "tangential (regularized dry friction):",
        f"  f_T = mu * N * tanh(alpha1 * d/dt x_T), mu={ds.MU}, alpha1={ds.ALPHA1}",
        "  f_nl = [f_T, -N, 0, -f_T, N, 0]^T",
        f"excitation: f_ext = [P1 cos(omega t), -P5, 0, 0, P5, 0]^T,"
        f" P1={ds.P1} (axial, harmonic), P5={ds.P5} (clamping, static)",
        "",
        f"harmonics: {HARMONICS}  (harmonic 0 carries the static clamping preload)",
        f"AFT sampling: polynomial_degree = {POLYNOMIAL_DEGREE} -> N_t = {N_TIME_SAMPLES}",
        f"omega range: [{OMEGA_START}, {OMEGA_END}], {SWEEP}ward sweep",
        f"parameterization: {PARAMETERIZATION.__name__}",
        f"predictor: {PREDICTOR.__name__}",
        f"step_length_adaptation: {STEP_ADAPTATION.__name__}",
        f"solver_kwargs: {SOLVER_KWARGS}"
        f"  (absolute_tolerance is rescaled by sqrt(2)/N_t inside solve_and_continue)",
        f"step_length_adaptation_kwargs: {STEP_KWARGS}",
        f"jacobian_update_frequency: {JACOBIAN_UPDATE_FREQUENCY}",
        f"maximum_number_of_solutions: {MAXIMUM_NUMBER_OF_SOLUTIONS}",
        "",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>:",
        "  x_rel_N, x_rel_T: the two interface relative coordinates",
        "  q1..q6: the six tip degrees of freedom",
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) )",
        "units: the thesis model is non-dimensional; omega_rad_s is the primary"
        " frequency axis (first undamped resonance at omega = 1), freq_hz ="
        " omega/(2 pi) is written for CSV compatibility only",
        "columns: freq_hz, omega_rad_s | iterations, step_length (corrector"
        f" diagnostics) | uout_h1_abs = |a_1({OUT_LABEL})|, uout_time_max ="
        f" max_t |{OUT_LABEL}(t)| | re/im of a_h, harmonic-major, then DoF order",
        f"uout (summarized output channel) = {OUT_LABEL}",
        f"dof order: {', '.join(LABELS)}",
    ]


def run(solver, guess, dimension):
    """Trace the branch and time it."""
    reference_direction = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((dimension, 1), complex), omega=DIRECTION)
    t0 = time()
    solution_set = solver.solve_and_continue(
        initial_guess=guess,
        initial_reference_direction=reference_direction,
        maximum_number_of_solutions=MAXIMUM_NUMBER_OF_SOLUTIONS,
        angular_frequency_range=[OMEGA_START, OMEGA_END],
        solver_kwargs=dict(SOLVER_KWARGS),
        step_length_adaptation_kwargs=dict(STEP_KWARGS),
        jacobian_update_frequency=JACOBIAN_UPDATE_FREQUENCY,
        verbose=True)
    return solution_set, time() - t0


# ===========================================================================
# The two runs
# ===========================================================================

def solve_second_order():
    """M q'' + C q' + K q + f_nl(q, q') = f_ext(tau) on the 6 tip DoFs."""
    system = BarBeamSecondOrder()
    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, second_order_ode=system,
        corrector_parameterization=PARAMETERIZATION,
        predictor=PREDICTOR, step_length_adaptation=STEP_ADAPTATION)

    guess = initial_guess(DIMENSION, lift=lambda q, w: q)
    solution_set, solve_time = run(solver, guess, DIMENSION)

    # solver coordinates are q itself; prepend the interface coordinates x_rel = B q
    recovery = np.vstack([B_COUPLING, np.eye(DIMENSION)])
    return solution_set, solve_time, recovery


def solve_first_order():
    """The same dynamics as a first-order system in z = [q ; qdot]."""
    system = BarBeamFirstOrder()
    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, first_order_ode=system,
        corrector_parameterization=PARAMETERIZATION,
        predictor=PREDICTOR, step_length_adaptation=STEP_ADAPTATION)

    guess = initial_guess(system.dimension,
                          lift=lambda q, w: np.concatenate([q, 1j * w * q]))
    solution_set, solve_time = run(solver, guess, system.dimension)

    # the state carries qdot too; export only [x_rel ; q], as the other run does
    recovery = np.zeros((len(LABELS), system.dimension))
    recovery[:2, :DIMENSION] = B_COUPLING
    recovery[2:, :DIMENSION] = np.eye(DIMENSION)
    return solution_set, solve_time, recovery


RUNS = {
    "second_order": (solve_second_order,
                     "second-order solver on the 6 tip DoFs"),
    "first_order":  (solve_first_order,
                     "first-order solver on the 12-component state"),
}


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"harmonics {HARMONICS[0]}..{HARMONICS[-1]}, polynomial_degree"
          f" {POLYNOMIAL_DEGREE} -> N_t = {N_TIME_SAMPLES}")
    print(f"continuation window: omega in [{OMEGA_START}, {OMEGA_END}]"
          f" ({SWEEP}ward sweep)\n")

    curves = {}
    for name, (solve, description) in RUNS.items():
        print(f"=== {name} ===")
        try:
            solution_set, solve_time, recovery = solve()
        except Exception as error:
            # One formulation failing must not discard the other's result, which
            # is already written: report and carry on.
            print(f"{name} FAILED: {type(error).__name__}: {error}\n")
            continue
        omega, _, uout_time_max = save_solution_csv(
            RESULTS / f"{name}.csv", solution_set, LABELS, OUT_LABEL,
            header_lines=header_lines(name, description, solve_time,
                                      len(solution_set)),
            recovery=recovery)
        peak = int(uout_time_max.argmax())
        print(f"{len(solution_set)} points in {solve_time:.1f} s;"
              f" peak max_t|{OUT_LABEL}| = {uout_time_max[peak]:.4f}"
              f" at omega = {omega[peak]:.4f}\n")
        curves[name] = (omega, uout_time_max)

    figure, axes = plt.subplots(figsize=(8, 5))
    for name, (omega, uout_time_max) in curves.items():
        axes.plot(omega, uout_time_max, label=name, linewidth=1.6)
    axes.set_xlabel(r"$\omega$  [rad/s]")
    axes.set_ylabel(rf"$\max_t |{OUT_LABEL}(t)|$")
    axes.set_title("Two bar+beam elements with frictional contact (thesis 6.4)")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    plt.show()
