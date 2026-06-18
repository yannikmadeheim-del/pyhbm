"""Harmonic-truncation study: pyhbm HBM (H=20) vs NLvib shooting (truncation-free).

Three curves on the SAME two-rod system (rod B length LB_REL * L_A):

    NLvib shooting   -- time integration of the UNregularized penalty model
                        (k_c = K_REL * k_rod_A): no harmonic truncation; the
                        "exact" solution of the penalty model.
    pyhbm AFT no-reg -- the same penalty model solved by HBM with H harmonics:
                        difference to shooting = pure TRUNCATION error.
    pyhbm DLFT (CSV) -- rigid-contact HBM reference: difference to shooting =
                        penalty-model error + truncation.

The shooting CSV comes from NLvib/validation/nlvib_two_rod_shooting.m and must
be generated for the SAME (LB_REL, K_REL, n_elem) as configured here.

Run:  python truncation_study.py
"""
import _setup  # noqa: F401

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyhbm import AFT

import config as cfg
from systems import TwoRodPenaltyContact
from frf import make_numerical_provider
from run import run_branch, linear_relative, SIGNALS
from reference import ReferenceCurve, reference_path, load_reference_csv, relative_error
from plotting import SIGNAL_TEX, SIGNAL_DESC

# --- the system under study (must match the MATLAB script's parameters) ---
LB_REL = 1 / 20      # rod-B length L_B / L_A
K_REL  = 20.0        # penalty stiffness k_c / k_rod_A
ALPHA  = np.inf      # AFT regularization; np.inf = same unregularized ramp as NLvib

# The shooting CSV name encodes (LB_rel, k_rel, n_elem); it is produced by
# NLvib/validation/nlvib_two_rod_shooting.m and copied into reference_csv/
# MANUALLY.  If it is missing, the study still runs -- just without the
# shooting overlay/error.
from reference import REF_DIR

_DIR = Path(__file__).parent
SHOOT_CSV = REF_DIR / (f"nlvib_two_rod_shooting_LB_{LB_REL:g}_k_{K_REL:g}"
                       f"_nel_{cfg.PARAMS.n_elem}.csv")


def load_shooting_csv(path) -> ReferenceCurve:
    """Shooting CSV (omega, A_tipA, A_xr, spec_* ignored) as a ReferenceCurve."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return ReferenceCurve(
        omega=np.atleast_1d(data["omega"]),
        A_peak={"tipA": np.atleast_1d(data["A_tipA"]),
                "xr":   np.atleast_1d(data["A_xr"])},
        harmonics=None, source=str(path))


def main():
    shoot = None
    if SHOOT_CSV.exists():
        shoot = load_shooting_csv(SHOOT_CSV)
        print(f"shooting reference: {len(shoot.omega)} points "
              f"(omega {shoot.omega.min():.0f}..{shoot.omega.max():.0f} rad/s)")
    else:
        print(f"WARNING: shooting CSV not found: {SHOOT_CSV.name}")
        print("  Running WITHOUT the shooting overlay / truncation error.")
        print("  Generate it with NLvib/validation/nlvib_two_rod_shooting.m "
              "(same LB_rel, k_rel, n_el) and copy the CSV into this folder.")

    # --- pyhbm AFT branch on the identical penalty model (H = 20) ---
    alpha_lbl = "no-reg" if not np.isfinite(ALPHA) else f"a={ALPHA:g}"
    system = TwoRodPenaltyContact(cfg.PARAMS, cfg.params_B_for(LB_REL),
                                  k_rel=K_REL, g_zero=cfg.GAP, alpha=ALPHA)
    res = run_branch(
        system, make_numerical_provider(system), AFT(),
        harmonics=cfg.HARMONICS, omega_range=(cfg.OMEGA_START, cfg.OMEGA_END),
        solver_kwargs=cfg.SOLVER_KWARGS, step_kwargs=cfg.STEP_KWARGS,
        max_solutions=cfg.MAX_SOLUTIONS,
        label=f"AFT {alpha_lbl} k={K_REL:g} LB={LB_REL:g}", method_name="aft",
        frf_name="numerical", params={"LB_rel": LB_REL}, verbose=True,
    )

    # truncation error: AFT(H=20) vs shooting, amplitude-based (no spectrum in
    # CSV).  With finite ALPHA the difference also contains the regularization.
    if shoot is not None:
        for s in SIGNALS:
            e, mean, mx, rms, frac_out = relative_error(res, shoot, signal=s)
            print(f"  truncation error ({s}): RMS={rms:.2%}  max={mx:.2%}  "
                  f"(off-reference: {frac_out:.0%})")

    # --- rigid DLFT reference CSV (if present) ---
    dlft = None
    dlft_path = reference_path(LB_REL)
    if dlft_path.exists():
        dlft = load_reference_csv(dlft_path)
    else:
        print(f"(no rigid DLFT reference {dlft_path.name} -- overlay skipped)")

    # --- overlay figure, one panel per signal ---
    wh = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
    om_lin = wh * system.omega_ref
    peak_lin = np.array([abs(linear_relative(system, w)) for w in wh])

    S = 1.0e-4
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, sig in zip(axes, SIGNALS):
        tex = SIGNAL_TEX[sig]
        ax.plot(om_lin, peak_lin / S, ':k', lw=0.9, label="linear (no contact)")
        ax.axhline(cfg.GAP / S, color="red", ls="--", lw=1.0, label="$g_0$")
        if shoot is not None:
            ax.plot(shoot.omega, shoot.A_peak[sig] / S, '-', color="0.55", lw=3.0,
                    label="NLvib shooting (penalty, no truncation)")
        if dlft is not None:
            ax.plot(dlft.omega, dlft.A_peak[sig] / S, '-', color="#1F77B4",
                    lw=1.6, label="pyhbm DLFT rigid (H=20, CSV)")
        ax.plot(res.omega_phys, res.peak[sig] / S, '--', color="#E8820C", lw=1.6,
                label="pyhbm AFT no-reg (H=20)")
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(rf"$\|{tex}(t)\|_\infty$ [$\times 10^{{-4}}$ m]")
        ax.set_title(SIGNAL_DESC[sig], fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"Truncation study (L_B = {LB_REL:g} L_A, k_c = {K_REL:g} k_rod): "
                 "shooting vs HBM", fontsize=11)
    fig.tight_layout()

    out = _DIR / f"two_rod_truncation_study_LB_{LB_REL:g}.png"
    fig.savefig(out, dpi=140)
    print(f"Figure saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
