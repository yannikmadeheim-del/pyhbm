"""Clamped-free FE rod vs. FLEXIBLE obstacle -- the known-good reproduction.

Reproduces Vadcard, Batailly & Thouverez, JSV 531 (2022), Fig. 17: the nonlinear
frequency response of a clamped-free axial rod impacting a finite-stiffness wall
modelled as a grounded spring k_obs (2nd substructure), solved with rigid DLFT
contact on the relative interface DOF and FBS.  One panel per obstacle stiffness
k_obs = k_rel * k_rod, with the linear FRF, the contact threshold g0, and (when
present) the independent NLvib MATLAB reference overlaid.

This file is deliberately minimal and stable -- the method study lives in
``methodology_comparison.py``.

Run:  python flexible_main.py
"""
import _setup  # noqa: F401  (sys.path + utf-8 stdout)

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyhbm import DLFTContact

import config as cfg
from systems import RodVibroImpactFlexible
from frf import make_numerical_provider
from run import run_branch, linear_relative


def main():
    p = cfg.PARAMS
    ref = RodVibroImpactFlexible(p, k_rel=1.0)
    omega_1 = ref.omega_ref
    epsilon = cfg.EPSILON_REL * ref.k_rod

    print(f"first axial mode omega_1 = {omega_1:.1f} rad/s ({omega_1/2/np.pi:.1f} Hz)")
    print(f"k_rod = E*A/L = {ref.k_rod:.3e} N/m,  DLFT epsilon = {epsilon:.3e}")
    print(f"sweeping k_rel = k_obs/k_rod in {cfg.K_REL_VALUES}")

    # --- one DLFT continuation branch per obstacle stiffness ---
    results = {}
    for k_rel in cfg.K_REL_VALUES:
        system   = RodVibroImpactFlexible(p, k_rel=k_rel)
        provider = make_numerical_provider(system)
        method   = DLFTContact(epsilon=epsilon, g_zero=cfg.GAP)
        print(f"\nContinuation: k_obs = {k_rel:g} k_rod  (k_obs = {system.k_obs:.3e} N/m)")
        results[k_rel] = run_branch(
            system, provider, method,
            harmonics=cfg.HARMONICS, omega_range=(cfg.OMEGA_START, cfg.OMEGA_END),
            solver_kwargs=cfg.SOLVER_KWARGS, step_kwargs=cfg.STEP_KWARGS,
            max_solutions=cfg.MAX_SOLUTIONS, label=f"k_rel={k_rel:g}",
            method_name="dlft", frf_name="numerical", params={"k_rel": k_rel},
        )

    # --- linear (no-contact) reference FRF, same for every k_obs ---
    wh_lin   = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
    peak_lin = np.array([abs(linear_relative(ref, w)) for w in wh_lin])
    om_lin   = wh_lin * omega_1

    _plot_panels(results, om_lin, peak_lin, omega_1)


def _nlvib_csv(kv):
    return Path(__file__).parent / f"nlvib_rod_flexible_kobs_{kv:g}_krod.csv"


def _plot_panels(results, om_lin, peak_lin, omega_1):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    SCALE = 1.0e-4
    XLIM  = (5.8e4, 7.4e4)
    YLIM  = (0.0, 5.5)
    PANEL = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
    for ax, lbl, kv in zip(axes.ravel(), PANEL, cfg.K_REL_VALUES):
        res = results[kv]
        ax.plot(om_lin, peak_lin / SCALE, ':', color="k", lw=1.0,
                label="Linear FRF (no contact)")
        ax.axhline(cfg.GAP / SCALE, color="red", ls="--", lw=1.2,
                   label=r"Contact threshold $g_0$")
        ax.plot(res.omega_phys, res.peak / SCALE, '-', color="#E8820C", lw=1.8,
                label="pyhbm DLFT-FBS\n(2-substructure, flexible wall)")

        csv = _nlvib_csv(kv)
        if pd is not None and csv.exists():
            df = pd.read_csv(csv)
            ax.plot(df["omega"], df["A_peak"] / SCALE, '--', color="#2CA02C",
                    lw=1.4, zorder=5, label="NLvib HBM reference (MATLAB)")
        elif not csv.exists():
            print(f"  [overlay] CSV not found: {csv.name}")

        ax.set_title(rf"{lbl}  $k_\mathrm{{obs}} = {kv:g}\,k_\mathrm{{rod}}$", fontsize=11)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|x(t)\|_\infty$  [$\times 10^{-4}$ m]")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
    fig.suptitle(
        "NFRC: clamped-free FE rod vs. flexible obstacle (Vadcard JSV 531, 2022, Fig. 17)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).parent / "rod_vibroimpact_frc.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
