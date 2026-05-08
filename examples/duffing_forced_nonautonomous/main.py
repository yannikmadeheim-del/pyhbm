# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dynamical_system import *

from pyhbm import *

from time import time

t0 = time()
duffing = DuffingForced_SecondOrder(c=0.009, k=1.0, beta=1.0, P=1.0)  # Create an instance of Duffing

duffing_solver = HarmonicBalanceMethod(
    second_order_ode = duffing,
    harmonics = [1,3,5,7,9],
)

# Define the initial guess after defining the harmonics of the HarmonicBalanceMethod
initial_omega = 0.0
first_harmonic = np.array([[1.0+0j]])
static_amplitude = duffing.P/duffing.k
initial_guess = FourierOmegaPoint.new_from_first_harmonic(first_harmonic * static_amplitude, omega=initial_omega)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(first_harmonic, omega=1)

solution_set = duffing_solver.solve_and_continue(
    initial_guess = initial_guess, 
    initial_reference_direction = initial_reference_direction, 
    maximum_number_of_solutions = 3500, 
    angular_frequency_range = [0.0, 15.0], 
    solver_kwargs = {
        "maximum_iterations": 200, 
        "absolute_tolerance": duffing.P * 1e-6
    }, 
    step_length_adaptation_kwargs = {
        "base": 2, 
        "initial_step_length": 0.1, 
        "maximum_step_length": 5.0,
        "minimum_step_length": 5e-6, 
        "goal_number_of_iterations": 3
    }
)
print(f"Time 2nd Order: {time() - t0:.3f} s")
from pyhbm import plot_FRF
from pyhbm.stability import BifurcationDetector, FloquetAnalyzer, SpecialPoint

# # Analyze stability
# print("Computing stability reports...")
# analyzer = FloquetAnalyzer(duffing)
# stability_reports = []
# for i, fourier in enumerate(solution_set.fourier):
#     if fourier.time_series is None:
#         fourier.compute_time_series()
#     report = analyzer.analyze(fourier.time_series, Fourier.adimensional_time_samples, solution_set.omega[i])
#     stability_reports.append(report)
# print(f"Computed {len(stability_reports)} stability reports")
#
# # Detect bifurcations
# print("Detecting bifurcations...")
# detector = BifurcationDetector()
# #bifurcations = detector.detect_all(solution_set, stability_reports)
# #print(f"Detected {len(bifurcations)} bifurcation points:")
# #for bif in bifurcations:
# #    print(bif.__str__(verbose=True), '\n')
#
# # Plot FRF with stability and bifurcations
# plot_FRF(
#     solution_set,
#     degrees_of_freedom=0,
#     time_domain_ode=duffing,
#     stability_reports=stability_reports, #bifurcations=bifurcations,
#     xscale='log',
#     yscale='log'
# )
#
#
# # %% Time-Domain Validation
# from pyhbm import TimeDomainValidator
#
# validator = TimeDomainValidator(duffing, integrator='RK45')
#
# index_to_validate = 1046
# fourier = solution_set.fourier[index_to_validate]
# omega = solution_set.omega[index_to_validate]
#
# time_series = fourier.time_series
#
# print(f"\nValidating solution at omega = {omega:.4f}")
#
# result = validator.validate(
#     time_series=time_series,
#     omega=omega,
#     multiplier_sampling_rate=40
# )
#
# print(f" Relative RMS error:\t{result.relative_rms_error:.6e}")
# print(f" Relative Max error:\t{result.relative_max_error:.6e}")
# print(f" Phase error:\t{result.phase_error:.6e}")
#
# validator.plot_comparison(result, degrees_of_freedom=0)#, show=False)
# %%"""
plot_FRF(solution_set, degrees_of_freedom=0)

import matplotlib.pyplot as plt
import numpy as np

t1 = time()
# --- Run 1st order for comparison ---
duffing_1st = DuffingForced(c=0.009, k=1.0, beta=1.0, P=1.0)

duffing_solver_1st = HarmonicBalanceMethod(
    first_order_ode=duffing_1st,
    harmonics=[1, 3, 5, 7, 9],
)

initial_omega_1st = 0.0
first_harmonic_1st = np.array([[1.0 + 0j], [1j * initial_omega_1st]])
static_amplitude_1st = duffing_1st.P / duffing_1st.k
initial_guess_1st = FourierOmegaPoint.new_from_first_harmonic(first_harmonic_1st * static_amplitude_1st,
                                                              omega=initial_omega_1st)
initial_reference_direction_1st = FourierOmegaPoint.new_from_first_harmonic(first_harmonic_1st, omega=1)

solution_set_1st = duffing_solver_1st.solve_and_continue(
    initial_guess=initial_guess_1st,
    initial_reference_direction=initial_reference_direction_1st,
    maximum_number_of_solutions=3500,
    angular_frequency_range=[0.0, 15.0],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": duffing_1st.P * 1e-6},
    step_length_adaptation_kwargs={"base": 2, "initial_step_length": 0.1, "maximum_step_length": 5.0,
                                   "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
)
print(f"Time 1st Order: {time() - t1:.3f} s")

# --- Compare on same plot ---
def solution_norm(solution_set, dof=0):
    return [np.linalg.norm(f.coefficients[:, dof, 0]) for f in solution_set.fourier]


fig, ax = plt.subplots()
ax.plot(solution_set.omega, solution_norm(solution_set), label='2nd order')
ax.plot(solution_set_1st.omega, solution_norm(solution_set_1st), label='1st order', linestyle='--')
ax.set_xlabel('ω')
ax.set_ylabel('||Q||')
ax.legend()
plt.show()