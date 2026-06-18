"""Generate the dense DLFT-numerical reference NFRC and store it as CSV.

This is the "ground truth" for the trajectory error in methodology_comparison.py:
two identical clamped-free rods in rigid tip-to-tip DLFT contact, solved on the
EXACT (numerical) FRF with a very small continuation step
(config.REFERENCE_STEP_KWARGS).  It is run ONCE; the comparison then loads the
CSV and never recomputes it.  Because it is a plain CSV it can be swapped for any
other solver's solution.

DLFT has no stiffness parameter (the rigid constraint x_r <= g0 is exact), but a
reference is generated PER ROD-B LENGTH ratio LB_rel = L_B / L_A that the
comparison may need (baseline + sweep): a different rod B is a different system.

Run:  python make_reference.py
"""
import _setup  # noqa: F401

from pathlib import Path

import numpy as np

from pyhbm import DLFTContact

import config as cfg
from systems import TwoRodVibroImpact
from frf import make_numerical_provider
from run import run_branch, SIGNALS
from reference import reference_path

OMEGA_RANGE = (cfg.OMEGA_START, cfg.OMEGA_END)
# Reference quality comes from the tiny STEP (REFERENCE_STEP_KWARGS), not an
# ultra-tight Newton tol: 1e-8 fails to converge at the fold and terminates the
# branch early.  1e-6 (as in the comparison runs) traverses the whole branch.
REF_SOLVER_KWARGS = {"maximum_iterations": 500, "absolute_tolerance": 1e-6}

HARMONICS = list(range(0, 21))


def make_one(lb_rel):
    system   = TwoRodVibroImpact(cfg.PARAMS, cfg.params_B_for(lb_rel))
    provider = make_numerical_provider(system)
    method   = DLFTContact(epsilon=cfg.EPSILON_REL * system.k_rod, g_zero=cfg.GAP)

    print(f"\n=== two-rod reference  L_B = {lb_rel:g} L_A  "
          f"(k_B = {system.kB_rod:.3e} N/m, rigid DLFT, epsilon = "
          f"{cfg.EPSILON_REL:g} k_rod) ===")
    res = run_branch(
        system, provider, method,
        harmonics=HARMONICS, omega_range=OMEGA_RANGE,
        solver_kwargs=REF_SOLVER_KWARGS, step_kwargs=cfg.REFERENCE_STEP_KWARGS,
        max_solutions=cfg.REFERENCE_MAX_SOLUTIONS,
        label=f"reference LB={lb_rel:g}", method_name="dlft", frf_name="numerical",
        params={"LB_rel": lb_rel}, verbose=True,
    )

    # columns: omega, A_tipA, A_xr, then per signal the spectrum as PHYSICAL
    # two-sided Fourier coefficients c_hat_k = c_k / N_time (Re/Im per harmonic) --
    # H-independent, so a high-H reference is directly comparable to lower-H
    # variants (see reference.py).
    cols = [res.omega_phys] + [res.peak[s] for s in SIGNALS]
    header_parts = ["omega", "A_tipA", "A_xr"]
    for s in SIGNALS:
        spec = res.harmonics[s]                              # (n, Nh) complex c_hat
        re_im = np.empty((spec.shape[0], 2 * spec.shape[1]))
        re_im[:, 0::2] = spec.real
        re_im[:, 1::2] = spec.imag
        cols.append(re_im)
        header_parts += [f"{s}_Re_{h},{s}_Im_{h}" for h in range(spec.shape[1])]
    data = np.column_stack(cols)
    header = ",".join(header_parts)

    # Written NEXT TO this script, NOT into reference_csv/: a CSV only becomes
    # the comparison reference once YOU copy it there manually.
    out = Path(__file__).parent / reference_path(lb_rel).name
    np.savetxt(out, data, delimiter=",", header=header, comments="")
    print(f"  -> {len(res.omega_phys)} points, coverage {res.coverage:.0%}, "
          f"saved: {out.name}")
    print(f"     copy it manually into reference_csv/ to use it as reference")


def main():
    lb_rels = sorted(set(cfg.SWEEPS.get("LB_rel", [])) | {cfg.BASELINE["LB_rel"]})
    print(f"Generating two-rod DLFT-numerical reference(s) for LB_rel = {lb_rels}")
    for lb_rel in lb_rels:
        make_one(lb_rel)


if __name__ == "__main__":
    main()
