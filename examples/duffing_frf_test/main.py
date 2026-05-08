# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

c, k, beta, P = 0.009, 1.0, 1.0, 1.0
harmonics = [1, 3, 5, 7, 9]

duffing = DuffingForced_SecondOrder(c=c, k=k, beta=beta, P=P)

# Fourier-Klasssenvariablen initialisieren bevor frf_ode und initial_guess erstellt werden
HarmonicBalanceMethod.update_dependencies(harmonics, duffing.polynomial_degree)

# --- Analytische FRF: Y(ω) = 1 / (k - ω^2 + j*c*ω), shape (N_freq, 1, 1) ---
omega_frf = np.linspace(0.00, 15.0, 5000)
Y_frf = (1.0 / (k - omega_frf**2 + 1j * c * omega_frf))[:, np.newaxis, np.newaxis]

# --- FrequencyDomainFRF aufbauen ---
frf_ode = FrequencyDomainFRF(
    nonlinear_ode=duffing,
    omega_frf=omega_frf,
    Y_frf=Y_frf,
    fd_step=1e-6,
)

initial_omega = 0.0
first_harmonic = np.array([[1.0 + 0j]])
static_amplitude = duffing.P / duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": P * 1e-6}
step_kwargs = {"base": 2, "initial_step_length": 0.1, "maximum_step_length": 5.0,
               "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}

# --- FRF-Solver ---
t0 = time()
frf_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=frf_ode)
solution_set_frf = frf_solver.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=3500,
    angular_frequency_range=[0.0, 15.0],
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time FRF:      {time() - t0:.3f} s")

# --- 2nd Order Referenz ---
t1 = time()
duffing_solver = HarmonicBalanceMethod(harmonics=harmonics, second_order_ode=duffing)
solution_set_2nd = duffing_solver.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=3500,
    angular_frequency_range=[0.0, 15.0],
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time 2nd Order: {time() - t1:.3f} s")

# --- Vergleich ---
def solution_norm(solution_set, dof=0):
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in solution_set.fourier]

fig, ax = plt.subplots()
ax.plot(solution_set_2nd.omega, solution_norm(solution_set_2nd), label='2nd Order', linewidth=2)
ax.plot(solution_set_frf.omega, solution_norm(solution_set_frf), label='FRF', linestyle='--')
ax.set_xlabel('ω')
ax.set_ylabel('||Q||')
ax.legend()
ax.set_title('Vergleich: 2nd Order vs. FRF-Formulierung')
plt.show()
