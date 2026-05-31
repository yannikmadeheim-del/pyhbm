"""Full HBM continuation 0.5 -> 1.5 with the ROBUST config, overlaid on NLvib.

Config under test (the two fixes that make the single-field DLFT track the
whole resonance instead of dying at contact onset):
  (1) accept on RELATIVE residual drop + STAGNATION, not the absolute floor
      (the absolute residual cannot reach 1e-6 because the non-smooth max()
       chatters at the machine level, but the *solution* is converged);
  (3) REDUCED harmonics (Nh=9 instead of 30) -> the impact corner is not
      over-resolved, so far fewer near-grazing samples flip the active set.

eps is a numerical penalty (rigid reference => eps-> infinity limit); eps=1
gives the most robust branch tracking. Penetration is O(1/eps); see
diag_epsindep.py for the eps-sweep toward the rigid NLvib curve.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyhbm import (FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
                   FBSProblem, NumericalFRF, DLFTContact,
                   HarmonicBalanceMethod)
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered


class SDOFVibroImpact(FBS_System):
    is_real_valued = True
    def __init__(self, m=1.0, c=0.05, k=1.0, F0=0.02, poly_deg=100):
        self.mass_matrix=np.array([[m]]); self.damping_matrix=np.array([[c]])
        self.stiffness_matrix=np.array([[k]]); self.B_coupling=np.array([[1.0]])
        self.total_dimension=1; self.dimension=1; self.polynomial_degree=poly_deg; self.F0=F0
    def external_term(self, tau):
        f=np.zeros((len(tau),1,1)); f[:,0,0]=self.F0*np.cos(tau); return f
    def interface_force(self,u,ud,t): return np.zeros((len(t),1,1))
    def jacobian_interface_force(self,u,ud,t): return np.zeros((len(t),1,1))
    def jacobian_interface_force_qdot(self,u,ud,t): return np.zeros((len(t),1,1))


PARAMS = dict(m=1.0, c=0.05, k=1.0, F0=0.02)
GAP = 0.1
NH = 9
EPSILON = 1.0
OMEGA_START, OMEGA_END = 0.5, 1.5
HARMONICS = list(range(0, NH))

system = SDOFVibroImpact(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)
provider = NumericalFRF(system.mass_matrix, system.damping_matrix, system.stiffness_matrix)
problem = FBSProblem(system, provider, DLFTContact(epsilon=EPSILON, g_zero=GAP))

Z = -OMEGA_START**2*system.mass_matrix + 1j*OMEGA_START*system.damping_matrix + system.stiffness_matrix
Q1 = np.linalg.solve(Z, np.array([PARAMS['F0']])).reshape(1, 1)
initial_guess = FourierOmegaPoint.new_from_first_harmonic(Q1, omega=OMEGA_START)
initial_ref_dir = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=1.0)

# (1) relative + stagnation acceptance, absolute floor kept loose as a backstop
solver_kwargs = {
    "maximum_iterations":   300,
    "absolute_tolerance":   1e-8,
    "relative_tolerance":   1e-4,
    "stagnation_tolerance": 1e-8,
}
step_kwargs = {
    "base":                      2.0,
    "initial_step_length":       0.005,
    "maximum_step_length":       0.05,
    "minimum_step_length":       1e-5,
    "goal_number_of_iterations": 4,
}

solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               predictor=TangentPredictorBordered)
print(f"Continuation: Nh={NH}, eps={EPSILON}, omega {OMEGA_START}->{OMEGA_END}, "
      f"rel+stag acceptance")
sol = solver.solve_and_continue(
    initial_guess               = initial_guess,
    initial_reference_direction = initial_ref_dir,
    maximum_number_of_solutions = 20000,
    angular_frequency_range     = [OMEGA_START, OMEGA_END],
    solver_kwargs               = solver_kwargs,
    step_length_adaptation_kwargs = step_kwargs,
    jacobian_update_frequency   = 1,
)

omegas = np.array(sol.omega)
print(f"  -> {len(omegas)} points, omega in [{omegas.min():.4f}, {omegas.max():.4f}]")
print(f"     [0.5,1.0): {int(((omegas>=0.5)&(omegas<1.0)).sum())}  "
      f"[1.0,1.5): {int(((omegas>=1.0)&(omegas<1.5)).sum())}  "
      f">=1.5: {int((omegas>=1.5).sum())}")

Nt = Fourier.number_of_time_samples
A1 = np.zeros_like(omegas); Apeak = np.zeros_like(omegas)
for i, (four, om) in enumerate(zip(sol.fourier, omegas)):
    full = problem.compute_full_response(four, om)
    A1[i] = 2.0/Nt*np.abs(full.coefficients[1, 0, 0])
    Fourier_Real.compute_time_series(full)
    Apeak[i] = float(np.max(np.abs(full.time_series[:, 0, 0])))

df = pd.read_csv(Path(__file__).parent/"nlvib_sdof_vibroimpact_shooting_frc.csv")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(df["omega"], df["A1"],     'o-', color="C1", ms=3, lw=1.0, label="NLvib -- 1st harmonic")
ax.plot(df["omega"], df["A_peak"], 'o-', color="C3", ms=3, lw=1.0, label="NLvib -- peak |q(t)|")
ax.plot(omegas, A1,    '-',  color="C0", lw=1.7, label="pyhbm DLFT -- 1st harmonic")
ax.plot(omegas, Apeak, '--', color="C2", lw=1.5, label="pyhbm DLFT -- peak |q(t)|")
ax.axhline(GAP, color="k", ls=":", lw=0.8, label=f"gap $g_0$={GAP}")
ax.set_xlabel(r"$\omega$"); ax.set_ylabel("amplitude")
ax.set_title(f"SDOF vibro-impact -- robust DLFT (Nh={NH}, eps={EPSILON}, rel+stag) vs NLvib")
ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.4)
plt.tight_layout()
out = Path(__file__).parent/"demo_continuation_robust.png"
fig.savefig(out, dpi=130)
print(f"Figure saved: {out}")
