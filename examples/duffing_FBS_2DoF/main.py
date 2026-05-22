# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *
from pyhbm import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from time import time

c1, c2, k1, k2, k3, beta, alpha, P = 0.09, 0.09, 1.0, 1.0, 0.0, 1.0, 0.1, 1.0
harmonics = [1, 3, 5, 7, 9]

# --- System instances ---
fbs_numerical    = System2DoF_FBS(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)
fbs_experimental = System2DoF_FBS_experimental(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)
reference        = System2DoF_1stOrder(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)

# Must be called before any FrequencyDomain* objects are created
HarmonicBalanceMethod.update_dependencies(harmonics, fbs_numerical.polynomial_degree)

# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": fbs_numerical.P * 1e-6}
step_kwargs_FBS   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 1.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
step_kwargs   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 1.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range = [0.0, 5.0]
initial_omega = 5.0

# FBS unknowns: u_rel (dimension = 1)
initial_guess_fbs = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=initial_omega)
initial_dir_fbs   = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=-1.0)

# Reference 1st-order unknowns: [u1, v1, u2, v2] (dimension = 4)
initial_guess_ref = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=initial_omega)
initial_dir_ref   = FourierOmegaPoint.zero_amplitude(dimension=reference.dimension, omega=-1.0)

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
time_fbs_num = time() - t0
print(f"Time FBS numerical:    {time_fbs_num:.3f} s")

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
time_fbs_exp = time() - t1
print(f"Time FBS experimental: {time_fbs_exp:.3f} s")

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
time_ref = time() - t2
print(f"Time Reference 1st order: {time_ref:.3f} s")

# --- Post-processing: recover full DOF response from FBS ---
# compute_full_response returns Fourier with coefficients shape (Nh, total_dimension, 1)
# DOF 0 = q1 (excited mass), DOF 1 = q2
full_resp_fbs_num = [fbs_num_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_num.fourier, solution_fbs_num.omega)]
full_resp_fbs_exp = [fbs_exp_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_exp.fourier, solution_fbs_exp.omega)]

# --- Helpers ---
def norm_dof(fourier_list, dof):
    return np.array([np.linalg.norm(f.coefficients[:, dof, 0]) for f in fourier_list])

# --- Amplitude-parametrized relative error (Krack & Gross) ---
def amplitude_parametrized_error(omega_this, amp_this, omega_ref, amp_ref):
    """
    Sort both curves by amplitude, interpolate omega_this(amp) onto the reference
    amplitude grid, return (amplitudes, pointwise relative error in omega).
    """
    idx_this = np.argsort(amp_this)
    idx_ref  = np.argsort(amp_ref)
    amp_this_s   = amp_this[idx_this]
    omega_this_s = omega_this[idx_this]
    amp_ref_s    = amp_ref[idx_ref]
    omega_ref_s  = omega_ref[idx_ref]

    amp_min = max(amp_this_s.min(), amp_ref_s.min())
    amp_max = min(amp_this_s.max(), amp_ref_s.max())
    mask = (amp_ref_s >= amp_min) & (amp_ref_s <= amp_max)

    amp_eval       = amp_ref_s[mask]
    omega_ref_eval = omega_ref_s[mask]

    if len(amp_this_s) == 0 or len(amp_ref_s) == 0 or amp_min >= amp_max:
        return np.array([]), np.array([])

    f_interp = interp1d(amp_this_s, omega_this_s, kind='linear',
                        bounds_error=False, fill_value=np.nan)
    omega_this_eval = f_interp(amp_eval)

    valid = np.isfinite(omega_this_eval) & (omega_ref_eval > 1e-10)
    eps = np.abs(omega_this_eval[valid] - omega_ref_eval[valid]) / omega_ref_eval[valid]
    return amp_eval[valid], eps

# Reference 1st-order state: [u1, v1, u2, v2] → u1=DOF0, u2=DOF2
# FBS full response:         [q1, q2]          → q1=DOF0, q2=DOF1
omega_ref_arr = np.array(solution_ref.omega)
omega_num_arr = np.array(solution_fbs_num.omega)
omega_exp_arr = np.array(solution_fbs_exp.omega)

amp_ref_d1 = norm_dof(solution_ref.fourier, 0)
amp_ref_d2 = norm_dof(solution_ref.fourier, 2)
amp_num_d1 = norm_dof(full_resp_fbs_num, 0)
amp_num_d2 = norm_dof(full_resp_fbs_num, 1)
amp_exp_d1 = norm_dof(full_resp_fbs_exp, 0)
amp_exp_d2 = norm_dof(full_resp_fbs_exp, 1)

amp_err_num_d1, eps_num_d1 = amplitude_parametrized_error(omega_num_arr, amp_num_d1, omega_ref_arr, amp_ref_d1)
amp_err_num_d2, eps_num_d2 = amplitude_parametrized_error(omega_num_arr, amp_num_d2, omega_ref_arr, amp_ref_d2)
amp_err_exp_d1, eps_exp_d1 = amplitude_parametrized_error(omega_exp_arr, amp_exp_d1, omega_ref_arr, amp_ref_d1)
amp_err_exp_d2, eps_exp_d2 = amplitude_parametrized_error(omega_exp_arr, amp_exp_d2, omega_ref_arr, amp_ref_d2)

def _eps_max(e1, e2):
    vals = [e.max() for e in (e1, e2) if len(e) > 0]
    return f"{max(vals):.4e}" if vals else "N/A (empty solution set)"

print(f"ε_rel max FBS numerical:    {_eps_max(eps_num_d1, eps_num_d2)}")
print(f"ε_rel max FBS experimental: {_eps_max(eps_exp_d1, eps_exp_d2)}")

# --- Plot setup ---
solvers = [
    (solution_ref.omega,     solution_ref.fourier, None,              f'Reference 1st order ({time_ref:.2f} s)',      'C0', '-',  [0, 2]),
    (solution_fbs_num.omega, None,                 full_resp_fbs_num, f'FBS numerical ({time_fbs_num:.2f} s)',         'C1', '--', [0, 1]),
    (solution_fbs_exp.omega, None,                 full_resp_fbs_exp, f'FBS experimental ({time_fbs_exp:.2f} s)',      'C2', ':',  [0, 1]),
]

dof_labels = ['DOF 1 (excited)', 'DOF 2']

fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(4, 12, hspace=0.55, wspace=0.35)
fig.suptitle('FBS vs Reference — 2-DOF with cubic spring and cubic damping')

# --- Row 0: comparison per DOF (each spanning 3 columns) ---
for col, dof_label in enumerate(dof_labels):
    ax = fig.add_subplot(gs[0, col * 6: col * 6 + 5])
    for omegas, fourier_list, full_list, label, color, ls, dofs in solvers:
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[col]), label=label, color=color, linestyle=ls)
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(dof_label)
    ax.legend(fontsize=7)

# --- Rows 1–2: individual subplots per solver (3 cols), one row per DOF ---
for j, dof_label in enumerate(dof_labels):
    for i, (omegas, fourier_list, full_list, label, color, ls, dofs) in enumerate(solvers):
        ax = fig.add_subplot(gs[1 + j, i * 4: i * 4 + 3])
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[j]), color=color, linestyle=ls)
        ax.set_xlabel('ω')
        ax.set_ylabel('||Q||')
        ax.set_title(f'{label}\n{dof_label}', fontsize=8)

# --- Row 3: amplitude-parametrized relative error ---
# err_data = [
#     (amp_err_num_d1, eps_num_d1, amp_err_exp_d1, eps_exp_d1, 'DOF 1 (excited)'),
#     (amp_err_num_d2, eps_num_d2, amp_err_exp_d2, eps_exp_d2, 'DOF 2'),
# ]
# for col, (amp_n, eps_n, amp_e, eps_e, dof_label) in enumerate(err_data):
#     ax = fig.add_subplot(gs[3, col * 3: col * 3 + 3])
#     lbl_n = f'FBS numerical    $\\varepsilon_{{rel}}^{{max}}$ = {eps_n.max():.2e}' if len(eps_n) > 0 else 'FBS numerical (no solution)'
#     lbl_e = f'FBS experimental $\\varepsilon_{{rel}}^{{max}}$ = {eps_e.max():.2e}' if len(eps_e) > 0 else 'FBS experimental (no solution)'
#     if len(eps_n) > 0:
#         ax.semilogy(amp_n, eps_n, color='C1', linestyle='--', label=lbl_n)
#     if len(eps_e) > 0:
#         ax.semilogy(amp_e, eps_e, color='C2', linestyle=':', label=lbl_e)
#     ax.set_xlabel('||Q||')
#     ax.set_ylabel('$\\varepsilon_{rel}$')
#     ax.set_title(f'Relative error vs amplitude — {dof_label}', fontsize=9)
#     ax.legend(fontsize=7)

plt.show()
