# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

d, k, P = 0.009, 1.0, 1.0
harmonics = [1, 3, 5, 7, 9]

SAVE_PNG = False   # set True to write the figure to disk

# --- System instances ---
fbs_numerical   = linearFBS_System_numerical(d=d, k=k, P=P)
fbs_experimental = linearFBS_System_experimental(d=d, k=k, P=P)
reference       = linearFBS_System_Reference(d=d, k=k, P=P)

# Must be called before any FrequencyDomain* objects are created
HarmonicBalanceMethod.update_dependencies(harmonics, fbs_numerical.polynomial_degree)

# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": P * 1e-6}
step_kwargs   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 0.5,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range = [0.1, 2.5]

# Start at omega > 0: subsystem K is free-free (singular at omega=0)
initial_omega = 2.5

# Initial conditions for FBS (n_int = 3 DOFs)
initial_guess_fbs = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=initial_omega)
initial_dir_fbs   = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=-1.0)

# Initial conditions for reference (8 DOFs)
initial_guess_ref = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=initial_omega)
initial_dir_ref   = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=-1.0)

# --- FBS numerical ---
t0 = time()
fbs_num_ode    = FBSProblem(
    fbs_numerical,
    NumericalFRF(fbs_numerical.mass_matrix, fbs_numerical.damping_matrix, fbs_numerical.stiffness_matrix),
    AFT())
fbs_num_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_num_ode)
solution_fbs_num = fbs_num_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time FBS numerical:    {time() - t0:.3f} s")

# --- FBS experimental ---
t1 = time()
fbs_exp_ode    = FBSProblem(
    fbs_experimental,
    ExperimentalFRF(fbs_experimental.omega_frf, fbs_experimental.Y_frf, fd_step=1e-6),
    AFT())
fbs_exp_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_exp_ode)
solution_fbs_exp = fbs_exp_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time FBS experimental: {time() - t1:.3f} s")

# --- Reference: full 8-DOF 2nd order ---
t2 = time()
ref_solver   = HarmonicBalanceMethod(harmonics=harmonics, second_order_ode=reference)
solution_ref = ref_solver.solve_and_continue(
    initial_guess=initial_guess_ref,
    initial_reference_direction=initial_dir_ref,
    maximum_number_of_solutions=6000,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time Reference:        {time() - t2:.3f} s")

# --- Reference FRF numerical: avoids singular Jacobian at resonances ---
t3 = time()
ref_frf_ode    = FRFProblem(
    reference,
    NumericalFRF(reference.mass_matrix, reference.damping_matrix, reference.stiffness_matrix))
ref_frf_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=ref_frf_ode)
solution_ref_frf = ref_frf_solver.solve_and_continue(
    initial_guess=initial_guess_ref,
    initial_reference_direction=initial_dir_ref,
    maximum_number_of_solutions=6000,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs,
)
print(f"Time Reference FRF:    {time() - t3:.3f} s")

# --- Post-processing: reconstruct full DOF response from FBS solution ---
full_resp_fbs_num = [fbs_num_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_num.fourier, solution_fbs_num.omega)]
full_resp_fbs_exp = [fbs_exp_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_exp.fourier, solution_fbs_exp.omega)]

# --- Helpers ---
def norm_dof(fourier_list, dof=0):
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in fourier_list]

# --- Plot ---
solvers = [
    (solution_ref.omega,     solution_ref.fourier,     None,               'Reference 2nd order', 'C0', '-'),
    (solution_ref_frf.omega, solution_ref_frf.fourier, None,               'Reference FRF',       'C1', (0, (5, 1))),
    (solution_fbs_num.omega, None,                     full_resp_fbs_num,  'FBS numerical',       'C2', '--'),
    (solution_fbs_exp.omega, None,                     full_resp_fbs_exp,  'FBS experimental',    'C3', ':'),
]

dofs = [0, 4]
dof_labels = ['DOF 0 (Subsystem A, excited)', 'DOF 4 (Subsystem B)']

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)
fig.suptitle('FBS vs Reference — Linear 8-DOF System')

# --- Top row: comparison for DOF 0 and DOF 4 ---
for col, (dof, dof_label) in enumerate(zip(dofs, dof_labels)):
    ax = fig.add_subplot(gs[0, col*2 : col*2+2])
    for omegas, fourier_list, full_list, label, color, ls in solvers:
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dof), label=label, color=color, linestyle=ls)
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(dof_label)
    ax.legend(fontsize=8)

# --- Bottom two rows: individual subplots per solver ---
for i, (omegas, fourier_list, full_list, label, color, ls) in enumerate(solvers):
    for j, dof in enumerate(dofs):
        ax = fig.add_subplot(gs[1 + j, i])
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dof), color=color, linestyle=ls)
        ax.set_xlabel('ω')
        ax.set_ylabel('||Q||')
        ax.set_title(f'{label}\nDOF {dof}', fontsize=8)

if SAVE_PNG:
    out = Path(__file__).parent / "FBS_test_linear_frc.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")

plt.show()
