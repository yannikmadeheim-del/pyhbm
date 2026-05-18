# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

c1, c2, k1, k2, k3, beta, alpha, P = 0.09, 0.09, 1.0, 1.0, 0.0, 0.0, 0.9, 1.0
harmonics = [1, 3, 5, 7, 9]

# --- System instances ---
fbs_numerical    = System2DoF_FBS(c1=c1, c2=c2, k1=k1, k2=k2, beta=beta, P=P)
fbs_experimental = System2DoF_FBS_experimental(c1=c1, c2=c2, k1=k1, k2=k2, beta=beta, P=P)
reference        = System2DoF_1stOrder(c1=c1, c2=c2, k1=k1, k2=k2, beta=beta, P=P)

# Must be called before any FrequencyDomain* objects are created
HarmonicBalanceMethod.update_dependencies(harmonics, fbs_numerical.polynomial_degree)

# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": fbs_numerical.P * 1e-6}
step_kwargs_FBS   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 5.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
step_kwargs   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 5.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range = [0.0, 5.0]
initial_omega = 0.0

# FBS unknowns: u_rel (dimension = 1)
initial_guess_fbs = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=initial_omega)
initial_dir_fbs   = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=1.0)

# Reference 1st-order unknowns: [u1, v1, u2, v2] (dimension = 4)
initial_guess_ref = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=initial_omega)
initial_dir_ref   = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=1.0)

# --- FBS numerical ---
t0 = time()
fbs_num_ode    = FrequencyBasedSubstructuring_numerical(fbs_numerical)
fbs_num_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_num_ode)
solution_fbs_num = fbs_num_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs_FBS,
)
print(f"Time FBS numerical:    {time() - t0:.3f} s")

# --- FBS experimental ---
t1 = time()
fbs_exp_ode    = FrequencyBasedSubstructuring_experimental(fbs_experimental)
fbs_exp_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_exp_ode)
solution_fbs_exp = fbs_exp_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs_FBS,
)
print(f"Time FBS experimental: {time() - t1:.3f} s")

# --- Reference: 1st order ---
t2 = time()
ref_solver = HarmonicBalanceMethod(harmonics=harmonics, first_order_ode=reference)
solution_ref = ref_solver.solve_and_continue(
    initial_guess=initial_guess_ref,
    initial_reference_direction=initial_dir_ref,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time Reference 1st order: {time() - t2:.3f} s")

# --- Post-processing: recover full DOF response from FBS ---
# compute_full_response returns Fourier with coefficients shape (Nh, total_dimension, 1)
# DOF 0 = q1 (excited mass), DOF 1 = q2
full_resp_fbs_num = [fbs_num_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_num.fourier, solution_fbs_num.omega)]
full_resp_fbs_exp = [fbs_exp_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_exp.fourier, solution_fbs_exp.omega)]

# --- Helpers ---
def norm_dof(fourier_list, dof):
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in fourier_list]

# Reference 1st-order state: [u1, v1, u2, v2] → u1=DOF0, u2=DOF2
# FBS full response:         [q1, q2]          → q1=DOF0, q2=DOF1
solvers = [
    (solution_ref.omega,     solution_ref.fourier, None,              'Reference 1st order', 'C0', '-',  [0, 2]),
    (solution_fbs_num.omega, None,                 full_resp_fbs_num, 'FBS numerical',       'C1', '--', [0, 1]),
    (solution_fbs_exp.omega, None,                 full_resp_fbs_exp, 'FBS experimental',    'C2', ':',  [0, 1]),
]

dof_labels = ['DOF 1 (excited)', 'DOF 2']

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.35)
fig.suptitle('FBS vs Reference — Nonlinear 2-DOF Duffing')

# --- Top row: comparison per DOF (each spanning 3 columns) ---
for col, dof_label in enumerate(dof_labels):
    ax = fig.add_subplot(gs[0, col * 3: col * 3 + 3])
    for omegas, fourier_list, full_list, label, color, ls, dofs in solvers:
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[col]), label=label, color=color, linestyle=ls)
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(dof_label)
    ax.legend(fontsize=8)

# --- Rows 1–2: individual subplots per solver (3 cols), one row per DOF ---
for j, dof_label in enumerate(dof_labels):
    for i, (omegas, fourier_list, full_list, label, color, ls, dofs) in enumerate(solvers):
        ax = fig.add_subplot(gs[1 + j, i * 2: i * 2 + 2])
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[j]), color=color, linestyle=ls)
        ax.set_xlabel('ω')
        ax.set_ylabel('||Q||')
        ax.set_title(f'{label}\n{dof_label}', fontsize=8)


plt.show()
