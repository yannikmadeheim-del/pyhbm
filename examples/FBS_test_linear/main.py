# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy as sp

d, k, P = 0.05, 200.0, 1.0
harmonics = [1, 3, 5, 7, 9]

duffing = DuffingForced_SecondOrder(d=d, k=k, P=P)

# Fourier-Klasssenvariablen initialisieren bevor frf_ode und initial_guess erstellt werden
HarmonicBalanceMethod.update_dependencies(harmonics, duffing.polynomial_degree)

# --- Analytische FRF: Y(ω) = 1 / (k - ω^2 + j*c*ω), shape (N_freq, 1, 1) ---
f_start = 0.01
f_end = 5
f_resolution = .01
freq = np.arange(f_start, f_end, f_resolution)

ome = 2 * np.pi * freq

M = np.diag([1,1,1,1])
C = np.array([
    [d, -d, 0, 0],
    [-d, d + d, -d, 0],
    [0, -d, d + d, -d],
    [0, 0,  -d,   d  ]
    ])
K = np.array([
    [k, -k, 0, 0],
    [-k, k + k, -k, 0],
    [0, -k, k + k, -k],
    [0, 0, -k, k]
])

Y_A = np.zeros([ome.shape[0], M.shape[0], M.shape[1]], dtype=complex)
for i, w in enumerate(ome):
    Y_A[i, :, :] = np.linalg.inv(-w ** 2 * M + 1j * w * C + K)
Y_B = Y_A
Y_A_B = np.zeros((Y_A.shape[0], Y_A.shape[1]+Y_B.shape[1], Y_A.shape[2]+Y_B.shape[2]), dtype=complex)
for i in range(Y_A.shape[0]):
    Y_A_B[i,:,:] = sp.linalg.block_diag(Y_A[i,:,:], Y_B[i,:,:])

# --- FrequencyDomainFRF aufbauen ---
# Y_A: FRF der einzelnen 4-DOF Substruktur (passt zu duffing.dimension=4)
# Y_A_B ist die assemblierte 8×8 FRF (für späteres NFBS)
frf_ode = FrequencyDomainFRF(
    nonlinear_ode=duffing,
    omega_frf=ome,
    Y_frf=Y_A,
    fd_step=1e-6,
)

initial_omega = 0.0
initial_guess = FourierOmegaPoint.zero_amplitude(dimension=duffing.dimension, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.zero_amplitude(dimension=duffing.dimension, omega=1.0)

solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": P * 1e-6}
step_kwargs_2nd = {"base": 2, "initial_step_length": 0.1, "maximum_step_length": 5.0,
                     "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}

step_kwargs_frf = {"base": 2, "initial_step_length": 0.1, "maximum_step_length": 5.0,
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
    step_length_adaptation_kwargs=step_kwargs_frf,
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
    step_length_adaptation_kwargs=step_kwargs_2nd,
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
