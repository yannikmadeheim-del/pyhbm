"""Central configuration for the rod vibro-impact example.

Holds the physics/numerics that are shared by both the plain reproduction
(``flexible_main.py``) and the method study (``methodology_comparison.py``),
plus the comparison SWITCHBOARD that turns individual variants and parameter
sweeps on/off.
"""
import numpy as np

from systems import RodParams

# ============================ physics / discretization ======================

PARAMS = RodParams(F0=25e3, poly_deg=20)   # Vadcard Table 1 + forcing at node B

GAP        = 0.2e-3                          # g0: wall offset [m]
HARMONICS  = list(range(0, 21))             # 0..20  -> H = 20

# Frequency sweep in nondimensional omega_hat = omega / omega_1 (Vadcard Fig. 17).
OMEGA_START = 1.20
OMEGA_END   = 0.90

# DLFT penalty (stiffness units): large vs the interface dynamic stiffness ~k_rod.
# Converged solution is epsilon-INDEPENDENT (Vadcard 2022): epsilon only affects
# Newton convergence, not the answer.  Hence it is a FIXED setting here, not a
# comparison axis (an epsilon sweep would re-plot the same FRC N times).
EPSILON_REL = 10.0                          # epsilon = EPSILON_REL * k_rod

# AFT tanh regularization sharpness (alpha -> inf approaches the rigid, nonsmooth wall).
ALPHA = 1.0e4

# Obstacle stiffnesses to sweep, as multiples of k_rod = E*A/L  (Vadcard Fig. 17 a-d).
K_REL_VALUES = [0.4, 4.0, 20.0, 40.0]


# ============================ solver / continuation =========================

SOLVER_KWARGS = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
STEP_KWARGS = {
    "base":                      4.0,
    "initial_step_length":       0.002,
    "maximum_step_length":       0.005,     # narrow omega_hat window -> small steps
    "minimum_step_length":       1e-7,
    "goal_number_of_iterations": 3,
}
MAX_SOLUTIONS = 5000


# ============================ reference (ground truth) ======================
# make_reference.py runs DLFT on the EXACT (numerical) FRF with a very small step
# length and writes the branch to CSV.  methodology_comparison.py loads that CSV as
# the swappable "original" and measures each variant's trajectory error against it.

REFERENCE = ("dlft", "numerical")           # descriptor of how the CSV was produced
REFERENCE_LABEL = "DLFT/num (dense, CSV)"
REFERENCE_STEP_KWARGS = {
    "base":                      4.0,
    "initial_step_length":       1e-4,
    "maximum_step_length":       1e-4,      # very small -> dense, clean ground truth
    "minimum_step_length":       1e-8,
    "goal_number_of_iterations": 3,
}
REFERENCE_MAX_SOLUTIONS = 20000             # enough to fully trace the branch


# ============================ comparison switchboard ========================
# methodology_comparison.py reads this.  Flip a flag off to skip that variant;
# each sweep list controls how many branches are computed for that axis.

# Which (method, FRF source) combinations to run at all:
RUN = {
    "dlft_numerical":    True,
    "dlft_experimental": False,
    "aft_numerical":     True,
    "aft_experimental":  False,
}

# Parameter sweeps (one-axis-at-a-time around BASELINE).  A single-element list =
# no sweep on that axis (uses the baseline below); a multi-element list traces one
# branch per value.  Only axes that change the CONVERGED solution belong here --
# epsilon is deliberately absent (see EPSILON_REL above).
SWEEPS = {
    "k_rel":   [20],                       # obstacle stiffness k_obs/k_rod (all variants)
    "alpha":   [np.inf],              # AFT tanh sharpness            (AFT only)
    "density": [0.01, 0.1, 1.0],               # experimental FRF density [samples/Hz]
    #                                         (Delta f = 200 Hz, 50 Hz)  (experimental only)
    "noise":   [np.inf],        # measured-FRF SNR [dB]; inf = clean (exp. only)
    #                                         time-domain noise: lower dB => noisier
}

# Baseline values used for any axis NOT being swept in a given run.
BASELINE = {
    "k_rel":   20.0,
    "alpha":   5e05,
    "density": 1.0,                        # samples per Hz (Delta f = 50 Hz)
    "noise":   np.inf,                      # clean FRF by default (no measurement noise)
}

# Seed for the experimental-FRF measurement noise (reproducible branches).
NOISE_SEED = 1

# (Accuracy reference is configured in the "reference (ground truth)" section above:
#  REFERENCE / REFERENCE_LABEL / REFERENCE_STEP_KWARGS.  Trajectory error is measured
#  against the dense DLFT-numerical CSV produced by make_reference.py.)
