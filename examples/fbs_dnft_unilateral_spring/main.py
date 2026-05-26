import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from pyhbm.frequency_domain import FBS_DLFT_numerical, FrequencyBasedSubstructuring_numerical,  Fourier_Real, FourierOmegaPoint
from pyhbm import HarmonicBalanceMethod
from dynamical_system import dlft_unilateral, aft_unilateral_spring
from time import time


# --- discretisation ---------------------------------------------------------
HARMONICS = list(range(0, 22))
EPSILON   = 1.0e6             # penalty: must be large (>~ 1e6 * stiffness). The
                              # converged solution is epsilon-independent (Vadcard 2022).
GAP       = 1.0               # wall offset g0 (contact when q_rel > g0)

system = dlft_unilateral(P=0.1)
system2 = aft_unilateral_spring(P=0.1)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)
ode    = FBS_DLFT_numerical(system, epsilon=EPSILON, g_zero=GAP)
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=ode)
ode2 = FrequencyBasedSubstructuring_numerical(system2)
solver2 = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=ode2)

initial_omega = 0.0
first_harmonic = np.array([[1.0 + 0j]])
static_amplitude = 0.1
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega= 1.0)
# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 300,
                 "absolute_tolerance": 0.1 * 1e-5,
                 }
#"relative_tolerance": 1e-6,
#"stagnation_tolerance": 1e-12

step_kwargs = {"base": 2.0,                    # was 2 — slower adaptation
               "initial_step_length": 0.005,   # was 0.01
               "maximum_step_length": 1.0,    # was 1.0 — critical change
               "minimum_step_length": 1e-6,
               "goal_number_of_iterations": 2} # was 3 — less aggressive

angular_frequency_range = [0.0, 2.0]

# --- FRF-Solver ---
t0 = time()
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=ode2)
solution_set = solver2.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time FRF:      {time() - t0:.3f} s")


full_resp = [ode.compute_full_response(f, w)
                     for f, w in zip(solution_set.fourier, solution_set.omega)]

def norm_dof(full_resp, dof):
    # L2 norm over harmonics of the Fourier coefficients (same amplitude measure
    # as the duffing_FBS_2DoF example).
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in full_resp]


dof_labels = ["DOF 0 (contact interface)", "DOF 1 (excited)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"DLFT-HBM FRC, 2-DOF unilateral spring (g0={GAP}, eps={EPSILON:.0e})")
for ax, dof, title in zip(axes, (0, 1), dof_labels):
    ax.plot(solution_set.omega, norm_dof(full_resp, dof),
            color="C0", linestyle="-", label="FBS DLFT")
    ax.set_xlabel("ω")
    ax.set_ylabel("||Q||")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
fig.tight_layout()

out = Path(__file__).parent / "frc_dlft.png"
fig.savefig(out, dpi=130)
print(f"FRC figure saved to {out}")
plt.show()