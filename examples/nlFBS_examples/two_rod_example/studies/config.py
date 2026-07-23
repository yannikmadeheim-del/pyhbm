"""Central configuration for the TWO-ROD vibro-impact example.

Holds the physics/numerics and the comparison SWITCHBOARD for the method study
(``studies/methodology_comparison.py``) -- the variants and parameter sweeps that
compare AFT vs DLFT and numerical vs experimental FRF.

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

from dynamical_system import RodParams

# ============================ physics / discretization ======================

PARAMS = RodParams(F0=25e3, poly_deg=30)   # rod A: Vadcard Table 1 + forcing at tip A


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
OMEGA_START = 1.2
OMEGA_END   = 0.9

# DLFT penalty (stiffness units): large vs the interface dynamic stiffness ~k_rod.
# Converged solution is epsilon-INDEPENDENT (Vadcard 2022): epsilon only affects
# Newton convergence, not the answer.  Hence it is a FIXED setting here, not a
# comparison axis.
EPSILON_REL = 2                        # epsilon = EPSILON_REL * k_rod


# ============================ solver / continuation =========================

SOLVER_KWARGS = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
STEP_KWARGS = {
    "base":                      4.0,
    "initial_step_length":       0.002,
    "maximum_step_length":       0.005,     # narrow omega_hat window -> small steps
    "minimum_step_length":       1e-8,
    "goal_number_of_iterations": 3,
}
MAX_SOLUTIONS = 1000


# ============================ comparison switchboard ========================
# methodology_comparison.py reads this.  Flip a flag off to skip that variant;
# each sweep list controls how many branches are computed for that axis.

# Which (method, FRF source) combinations to run at all:
RUN = {
    "dlft_numerical":    True,
    "dlft_experimental": True,
    "aft_numerical":     True,
    "aft_experimental":  True,
}

# Parameter sweeps (one-axis-at-a-time around BASELINE).  A single-element list =
# no sweep on that axis (uses the baseline below); a multi-element list traces one
# branch per value.
#   LB_rel -> ALL variants (rod-B length L_B / L_A; changes the SYSTEM)
#   k_rel  -> AFT only (penalty stiffness k_c / k_rod; DLFT is penalty-free)
#   alpha  -> AFT only (np.inf = WITHOUT regularization, finite = WITH)
SWEEPS = {
    "LB_rel":  [1/20],                       # e.g. [0.5, 1.0, 2.0] to vary rod-B stiffness
    "k_rel":   [],
    "alpha":   [1e08, 1e07],              # with regularization vs without (np.inf)
    "density": [],                       # experimental FRF density [samples/Hz]
    "noise":   [40],                    # measured-FRF SNR [dB]; inf = clean
}

# Baseline values used for any axis NOT being swept in a given run.
BASELINE = {
    "LB_rel":  1/20,
    "k_rel":   100,
    "alpha":   1e08,
    "density": 0.01,
    "noise":   40,
}

# Seed for the experimental-FRF measurement noise (reproducible branches).
NOISE_SEED = 1


# ============================ uncoupled-FRF quicklook =======================
# True -> methodology_comparison.py ALSO plots one entry of the UNCOUPLED linear
# admittance |Y_ij(omega)| (block-diagonal system: no contact, rods uncoupled),
# sampled on the experimental grid (BASELINE density).  If SWEEPS["noise"]
# contains finite SNRs, the noisy FRF the solver would see is overlaid per level
# (same noise model and seed as the experimental provider).
#
# FRF_ENTRY uses 1-BASED DOF numbering:
#   1 .. n_elem            = rod A   (n_elem   = tip A)
#   n_elem+1 .. 2*n_elem   = rod B   (2*n_elem = tip B)
# e.g. (10, 10) = tip-A drive point (n_elem = 10);
#      (10, 11) = tip A -> rod B    (identically ZERO: rods are uncoupled).
PLOT_FRF  = True
FRF_ENTRY = (10, 10)
