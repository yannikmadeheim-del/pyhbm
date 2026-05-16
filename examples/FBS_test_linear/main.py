# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from time import time

d, k, P = 0.05, 200.0, 1.0
harmonics = [1, 3, 5, 7, 9]

# --- System instances ---
fbs_numerical   = linearFBS_System_numerical(d=d, k=k, P=P)
fbs_experimental = linearFBS_System_experimental(d=d, k=k, P=P)
reference       = linearFBS_System_Reference(d=d, k=k, P=P)

# Must be called before any FrequencyDomain* objects are created
HarmonicBalanceMethod.update_dependencies(harmonics, fbs_numerical.polynomial_degree)

# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": P * 1e-6}
step_kwargs   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 0.01,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range = [0.5, 20.0]

# Start at omega > 0: subsystem K is free-free (singular at omega=0)
initial_omega = 0.5

# Initial conditions for FBS (n_int = 3 DOFs)
initial_guess_fbs = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=initial_omega)
initial_dir_fbs   = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=1.0)

# Initial conditions for reference (8 DOFs)
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
    step_length_adaptation_kwargs=step_kwargs,
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
ref_frf_ode    = FrequencyDomainFRF_numerical(reference)
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

# --- Plot: DOF 0 (excited DOF, subsystem A) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('FBS vs Reference — Linear 8-DOF System')

for ax, dof in zip(axes, [0, 4]):
    ax.plot(solution_ref.omega,     norm_dof(solution_ref.fourier, dof),     label='Reference 2nd order', linewidth=2)
    ax.plot(solution_ref_frf.omega, norm_dof(solution_ref_frf.fourier, dof), label='Reference FRF',       linewidth=2, linestyle=(0, (5, 1)))
    ax.plot(solution_fbs_num.omega, norm_dof(full_resp_fbs_num, dof),        label='FBS numerical',       linestyle='--')
    ax.plot(solution_fbs_exp.omega, norm_dof(full_resp_fbs_exp, dof),        label='FBS experimental',    linestyle=':')
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(f'DOF {dof} ({"Subsystem A" if dof < 4 else "Subsystem B"})')
    ax.legend()

plt.tight_layout()
plt.show()
