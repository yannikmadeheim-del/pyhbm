"""Generate the dense DLFT-numerical reference NFRC(s) and store them as CSV.

This is the "ground truth" for the trajectory error in methodology_comparison.py:
a clamped-free rod vs. flexible obstacle solved with DLFT on the EXACT (numerical)
FRF and a very small continuation step (config.REFERENCE_STEP_KWARGS).  It is run
ONCE; the comparison then loads the CSV and never recomputes it.  Because it is a
plain CSV (columns omega, A_peak, A1 -- same as the NLvib references), it can be
swapped for any other solver's solution.

A reference is generated for every obstacle stiffness the comparison may need:
the baseline k_rel plus any k_rel in the sweep.

Run:  python make_reference.py
"""
import _setup  # noqa: F401

import numpy as np

from pyhbm import DLFTContact

import config as cfg
from systems import RodVibroImpactFlexible
from frf import make_numerical_provider
from run import run_branch
from reference import reference_path

OMEGA_RANGE = (cfg.OMEGA_START, cfg.OMEGA_END)
# Reference quality comes from the tiny STEP (REFERENCE_STEP_KWARGS), not an
# ultra-tight Newton tol: 1e-8 fails to converge at the fold and terminates the
# branch early.  1e-6 (as in the comparison runs) traverses the whole branch.
REF_SOLVER_KWARGS = {"maximum_iterations": 500, "absolute_tolerance": 1e-6}

REFERENCE_STEP_KWARGS = {
    "base":                      4.0,
    "initial_step_length":       1e-4,
    "maximum_step_length":       1e-4,      # very small -> dense, clean ground truth
    "minimum_step_length":       1e-8,
    "goal_number_of_iterations": 3,
}

HARMONICS = list(range(0,21))
REFERENCE_MAX_SOLUTIONS = 20000
REFERENCE_MAX_SOLUTIONS = 100000

def make_one(k_rel):
    system   = RodVibroImpactFlexible(cfg.PARAMS, k_rel=k_rel)
    provider = make_numerical_provider(system)
    method   = DLFTContact(epsilon=cfg.EPSILON_REL * system.k_rod, g_zero=cfg.GAP)

    print(f"\n=== reference  k_obs = {k_rel:g} k_rod  (k_obs = {system.k_obs:.3e} N/m) ===")
    res = run_branch(
        system, provider, method,
        harmonics=HARMONICS, omega_range=OMEGA_RANGE,
        solver_kwargs=REF_SOLVER_KWARGS, step_kwargs=REFERENCE_STEP_KWARGS,
        max_solutions=REFERENCE_MAX_SOLUTIONS,
        label=f"reference k_rel={k_rel:g}", method_name="dlft", frf_name="numerical",
        params={"k_rel": k_rel}, verbose=True,
    )

    # columns: omega, A_peak, then the u_B spectrum as PHYSICAL two-sided Fourier
    # coefficients c_hat_k = c_k / N_time (Re/Im per harmonic) -- H-independent, so a
    # high-H reference is directly comparable to lower-H variants (see reference.py).
    spec = res.uB_harmonics                                  # (n, Nh) complex c_hat
    re_im = np.empty((spec.shape[0], 2 * spec.shape[1]))
    re_im[:, 0::2] = spec.real
    re_im[:, 1::2] = spec.imag
    data = np.column_stack((res.omega_phys, res.peak, re_im))
    header = "omega,A_peak," + ",".join(
        f"uB_Re_{h},uB_Im_{h}" for h in range(spec.shape[1]))

    out = reference_path(k_rel)
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    print(f"  -> {len(res.omega_phys)} points, coverage {res.coverage:.0%}, "
          f"saved: {out.name}")


def main():
    k_rels = sorted(set(cfg.SWEEPS.get("k_rel", [])) | {cfg.BASELINE["k_rel"]})
    print(f"Generating dense DLFT-numerical reference(s) for k_rel = {k_rels}")
    for k_rel in k_rels:
        make_one(k_rel)


if __name__ == "__main__":
    main()
