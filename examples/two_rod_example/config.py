"""Central configuration for the TWO-ROD vibro-impact example.

Holds the physics/numerics shared by the quicklook (``main.py``) and the method
study (``methodology_comparison.py``), plus the comparison SWITCHBOARD that turns
individual variants and parameter sweeps on/off.

Differences vs. the single-rod example
--------------------------------------
There is no obstacle spring: the obstacle is a second, identical clamped-free FE
rod.  Hence

    * DLFT has NO stiffness parameter at all -- it enforces the rigid unilateral
      tip-to-tip constraint x_r = u_A + u_B <= g0 exactly.
    * k_rel exists ONLY for AFT: it is the interface PENALTY stiffness
      k_c = k_rel * k_rod of the smooth one-sided contact spring.
    * alpha (AFT only) is the tanh regularization sharpness:
      finite  -> WITH regularization (smooth contact force),
      np.inf  -> WITHOUT regularization (hard max(0, .) ramp).
"""
from dataclasses import replace

import numpy as np

from systems import RodParams

# ============================ physics / discretization ======================

PARAMS = RodParams(F0=25e3, poly_deg=20)   # rod A: Vadcard Table 1 + forcing at tip A


def params_B_for(lb_rel: float) -> RodParams:
    """Rod B = rod A with length L_B = lb_rel * L_A (other material data equal).

    Shorter rod B (lb_rel < 1) -> STIFFER obstacle (k_B = E*A/L_B up) and higher
    rod-B modes; longer -> softer.  lb_rel = 1 reproduces the identical-rod case.
    """
    return replace(PARAMS, L=lb_rel * PARAMS.L)

GAP        = 0.2e-3                         # g0: initial tip-to-tip gap [m]
HARMONICS  = list(range(0, 21))             # 0..20  -> H = 20

# Frequency sweep in nondimensional omega_hat = omega / omega_1 (omega_1 = first
# axial mode of ONE rod; both rods are identical).
OMEGA_START = 1.20
OMEGA_END   = 0.90

# DLFT penalty (stiffness units): large vs the interface dynamic stiffness ~k_rod.
# Converged solution is epsilon-INDEPENDENT (Vadcard 2022): epsilon only affects
# Newton convergence, not the answer.  Hence it is a FIXED setting here, not a
# comparison axis.
EPSILON_REL = 1.0                          # epsilon = EPSILON_REL * k_rod


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
# make_reference.py runs DLFT (rigid tip-to-tip contact) on the EXACT (numerical)
# FRF with a very small step length and writes the branch to CSV.
# methodology_comparison.py loads that CSV as the swappable "original" and
# measures each variant's trajectory error against it.

REFERENCE_LABEL = "DLFT/num (CSV)"
# NOTE: deliberately NOT ultra-dense.  The rigid grazing corner makes the branch
# tangent jump discontinuously; with very small steps the sign-aligned predictor
# flips there and the continuation retraces the no-contact branch instead of
# entering contact.  Moderate adaptive steps leap over the corner (verified) and
# are plenty for a first-test reference.
REFERENCE_STEP_KWARGS = {
    "base":                      4.0,
    "initial_step_length":       1e-3,
    "maximum_step_length":       2e-3,
    "minimum_step_length":       1e-7,
    "goal_number_of_iterations": 3,
}
REFERENCE_MAX_SOLUTIONS = 20000             # enough to fully trace the branch


# ============================ comparison switchboard ========================
# methodology_comparison.py reads this.  Flip a flag off to skip that variant;
# each sweep list controls how many branches are computed for that axis.

# Which (method, FRF source) combinations to run at all:
RUN = {
    "dlft_numerical":    False,
    "dlft_experimental": False,
    "aft_numerical":     True,
    "aft_experimental":  False,
}

# Parameter sweeps (one-axis-at-a-time around BASELINE).  A single-element list =
# no sweep on that axis (uses the baseline below); a multi-element list traces one
# branch per value.
#   LB_rel -> ALL variants (rod-B length L_B / L_A; changes the SYSTEM, so each
#             value needs its own reference CSV -- make_reference.py handles that)
#   k_rel  -> AFT only (penalty stiffness k_c / k_rod; DLFT is penalty-free)
#   alpha  -> AFT only (np.inf = WITHOUT regularization, finite = WITH)
SWEEPS = {
    "LB_rel":  [1/4],                       # e.g. [0.5, 1.0, 2.0] to vary rod-B stiffness
    "k_rel":   [100],
    "alpha":   [2e07, 3e07],              # with regularization vs without
    "density": [1.0],                       # experimental FRF density [samples/Hz]
    "noise":   [np.inf],                    # measured-FRF SNR [dB]; inf = clean
}

# Baseline values used for any axis NOT being swept in a given run.
BASELINE = {
    "LB_rel":  1/4,
    "k_rel":   100,
    "alpha":   1e07,
    "density": 1.0,
    "noise":   np.inf,
}

# Seed for the experimental-FRF measurement noise (reproducible branches).
NOISE_SEED = 1
