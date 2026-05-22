# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pyhbm.frequency_domain import FBS_DLFT_numerical
from pyhbm import HarmonicBalanceMethod, FourierOmegaPoint
from dynamical_system import dlft_unilateral
from pyhbm.frequency_domain import Fourier, Fourier_Real
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from time import time

harmonics = [1, 2, 3, 4, 5, 6, 7]
epsilon = 100

# --- System instances ---
system = dlft_unilateral(P=0.1)
HarmonicBalanceMethod.update_dependencies(harmonics, system.polynomial_degree)

continuation_kwargs = dict(
    maximum_number_of_solutions=2000,
    angular_frequency_range=[0.5, 0.8],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={"base": 2, "initial_step_length": 0.05,
                                   "maximum_step_length": 0.05, "minimum_step_length": 1e-12,
                                   "goal_number_of_iterations": 3},
)

# --- Branch 1: sweep upward from ω=0.5, zero initial guess (linear branch) ---
ode1 = FBS_DLFT_numerical(system, epsilon=epsilon, g_zero=1.0)
solver1 = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=ode1)
initial_guess1 = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=0.5)
initial_dir1   = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=1.0)
print("--- Branch 1: upward sweep from ω=0.5 ---")
solution1 = solver1.solve_and_continue(
    initial_guess=initial_guess1,
    initial_reference_direction=initial_dir1,
    **continuation_kwargs,
)

# --- Branch 2: sweep downward from ω=0.65, linear FRF as initial guess (contact branch) ---
ode2 = FBS_DLFT_numerical(system, epsilon=epsilon, g_zero=1.0)
solver2 = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=ode2)
omega_start = 0.8
x_init = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=omega_start)
Q_rel_linear = np.linalg.solve(ode2._get_Yr(x_init), ode2._get_Fext_admr(x_init))
Q_init_coeffs = Q_rel_linear.reshape(len(harmonics), system.dimension, 1)
initial_guess2 = FourierOmegaPoint(Fourier(Q_init_coeffs), omega_start)
initial_dir2   = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=-1.0)
print("--- Branch 2: downward sweep from ω=0.65 (linear FRF initial guess) ---")
solution2 = solver2.solve_and_continue(
    initial_guess=initial_guess2,
    initial_reference_direction=initial_dir2,
    **continuation_kwargs,
)

# --- Post-processing ---
def norm_dof(fourier_list, dof):
    return np.array([np.linalg.norm(f.coefficients[:, dof, 0]) for f in fourier_list])

full_resp1 = [ode1.compute_full_response(f, w) for f, w in zip(solution1.fourier, solution1.omega)]
full_resp2 = [ode2.compute_full_response(f, w) for f, w in zip(solution2.fourier, solution2.omega)]

omega1 = np.array(solution1.omega)
omega2 = np.array(solution2.omega)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('DLFT-HBM — 2-DOF oscillator with unilateral spring (g₀=1, ε=1)')

for ax, dof, label in zip(axes, [0, 1], ['DOF 1 (contact DOF)', 'DOF 2 (excited DOF)']):
    ax.plot(omega1, norm_dof(full_resp1, dof), 'C0', label='Branch 1 (upward, ε=1)')
    ax.plot(omega2, norm_dof(full_resp2, dof), 'C1--', label='Branch 2 (downward, ε=1)')
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(label)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
