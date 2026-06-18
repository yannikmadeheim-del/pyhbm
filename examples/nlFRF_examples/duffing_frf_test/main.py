# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

c, k, beta, P, ome_dens = 0.009, 1.0, 1.0, 1.0, 1000
harmonics = [1, 3, 5, 7, 9]

duffing = DuffingForced_SecondOrder(c=c, k=k, beta=beta, P=P)
duffing_1st = DuffingForced(c=c, k=k, beta=beta, P=P)

# Fourier-Klasssenvariablen initialisieren bevor frf_ode und initial_guess erstellt werden
HarmonicBalanceMethod.update_dependencies(harmonics, duffing.polynomial_degree)

# --- Analytische FRF: Y(ω) = 1 / (k - ω^2 + j*c*ω), shape (N_freq, 1, 1) ---
omega_frf = np.linspace(0.00, 15.0*np.max(harmonics)*2, ome_dens*15*np.max(harmonics)*2)
Y_frf = (1.0 / (k - omega_frf**2 + 1j * c * omega_frf))[:, np.newaxis, np.newaxis]

# --- FrequencyDomainFRF aufbauen ---
frf_ode = FrequencyDomainFRF_experimental(
    nonlinear_ode=duffing,
    omega_frf=omega_frf,
    Y_frf=Y_frf,
    fd_step=1e-6,
)

initial_omega = 0.0
first_harmonic = np.array([[1.0 + 0j]])
static_amplitude = duffing.P / duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega= 2.0)
# initial_reference_direction = FourierOmegaPoint.zero_amplitude(dimension=duffing.dimension, omega=-1.0)

solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": P * 1e-6}
step_kwargs = {"base": 2, "initial_step_length": 0.1, "maximum_step_length": 3.0,
                     "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range=[0.0, 15.0]

# --- Numerical FRF aufbauen ---
frf_numerical_ode = FrequencyDomainFRF_numerical(nonlinear_ode=duffing)

# --- FRF-Solver ---
t0 = time()
frf_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=frf_ode)
solution_set_frf = frf_solver.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
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
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time 2nd Order: {time() - t1:.3f} s")

# --- 1st Order Referenz ---
t2 = time()
duffing_1st_solver = HarmonicBalanceMethod(harmonics=harmonics, first_order_ode=duffing_1st)
solution_set_1st = duffing_1st_solver.solve_and_continue(
    initial_guess=FourierOmegaPoint.zero_amplitude(dimension=duffing_1st.dimension, omega=initial_omega),
    initial_reference_direction=FourierOmegaPoint.zero_amplitude(dimension=duffing_1st.dimension, omega=1.0),
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time 1st Order: {time() - t2:.3f} s")

# --- Numerical FRF-Solver ---
t3 = time()
frf_numerical_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=frf_numerical_ode)
solution_set_frf_num = frf_numerical_solver.solve_and_continue(
    initial_guess=initial_guess,
    initial_reference_direction=initial_reference_direction,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time FRF numerical: {time() - t3:.3f} s")


# --- Vergleich ---
def solution_norm(solution_set, dof=0):
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in solution_set.fourier]

solvers = [
    (solution_set_2nd,      '2nd Order',      'C0', '-'),
    (solution_set_frf,      'FRF exp.',        'C1', '--'),
    (solution_set_1st,      '1st Order',       'C2', ':'),
    (solution_set_frf_num,  'FRF numerical',   'C3', '-.'),
]

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)

# --- Big comparison plot (top row, all 4 columns) ---
ax_all = fig.add_subplot(gs[0, :])
for sol, label, color, ls in solvers:
    ax_all.plot(sol.omega, solution_norm(sol), label=label, color=color, linestyle=ls)
ax_all.set_xlabel('ω')
ax_all.set_ylabel('||Q||')
ax_all.set_title('Vergleich aller Formulierungen')
ax_all.legend()

# --- 4 individual subplots (bottom row) ---
for i, (sol, label, color, ls) in enumerate(solvers):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(sol.omega, solution_norm(sol), color=color, linestyle=ls)
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(label)

plt.show()
