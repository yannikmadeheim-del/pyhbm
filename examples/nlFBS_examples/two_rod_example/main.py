"""Two identical clamped-free FE rods in tip-to-tip vibro-impact -- DLFT quicklook.

The left rod is harmonically driven at its tip; the right rod is an identical,
passive mirror clamped to the opposite wall.  Rigid unilateral DLFT contact
enforces x_r = u_A + u_B <= g0 on the relative tip approach.  Unlike the
single-rod example there is no obstacle-stiffness sweep -- the contact is exact --
so the quicklook is one branch, plotted in two windows (same layout each): the
driven tip u_A and the relative approach x_r.

This file is deliberately minimal and stable -- the method study lives in
``methodology_comparison.py``.
system configuration in config.py

Run:  python main.py
"""
import _setup  # noqa: F401  (sys.path + utf-8 stdout)

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyhbm import DLFTContact

import config as cfg
from systems import TwoRodVibroImpact
from frf import make_numerical_provider
from run import run_branch, linear_relative, SIGNALS, SIGNAL_TEX, SIGNAL_DESC


SAVE_PNG = False   # set True to write the figures to disk


def main():
    p  = cfg.PARAMS
    lb = cfg.BASELINE["LB_rel"]
    system  = TwoRodVibroImpact(p, cfg.params_B_for(lb))
    omega_1 = system.omega_ref
    epsilon = cfg.EPSILON_REL * system.k_rod

    print(f"first axial mode omega_1 = {omega_1:.1f} rad/s ({omega_1/2/np.pi:.1f} Hz)")
    print(f"rod A: k_rod = E*A/L = {system.k_rod:.3e} N/m,  DLFT epsilon = {epsilon:.3e}")
    print(f"rod B: L_B = {lb:g} L_A  ->  k_B = {system.kB_rod:.3e} N/m")
    print("rigid tip-to-tip DLFT contact (no penalty stiffness)")

    provider = make_numerical_provider(system)
    method   = DLFTContact(epsilon=epsilon, g_zero=cfg.GAP)
    res = run_branch(
        system, provider, method,
        harmonics=cfg.HARMONICS, omega_range=(cfg.OMEGA_START, cfg.OMEGA_END),
        solver_kwargs=cfg.SOLVER_KWARGS, step_kwargs=cfg.STEP_KWARGS,
        max_solutions=cfg.MAX_SOLUTIONS, label="two-rod DLFT",
        method_name="dlft", frf_name="numerical", params={},
    )

    # linear (no-contact) reference FRF: rod B is unforced without contact
    # (u_B = 0), so the same backdrop serves both windows.
    wh_lin   = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
    peak_lin = np.array([abs(linear_relative(system, w)) for w in wh_lin])
    om_lin   = wh_lin * omega_1

    SCALE = 1.0e-4
    XLIM  = (5.8e4, 7.4e4)
    YLIM  = (0.0, 5.5)

    for sig in SIGNALS:
        tex = SIGNAL_TEX[sig]
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.plot(om_lin, peak_lin / SCALE, ':', color="k", lw=1.0,
                label="Linear FRF (no contact)")
        ax.axhline(cfg.GAP / SCALE, color="red", ls="--", lw=1.2,
                   label=r"Contact threshold $g_0$")
        ax.plot(res.omega_phys, res.peak[sig] / SCALE, '-', color="#E8820C",
                lw=1.8, label="pyhbm DLFT-FBS\n(2 rods, rigid tip contact)")

        ax.set_title(f"NFRC: two-rod vibro-impact -- {SIGNAL_DESC[sig]}",
                     fontsize=11)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_ylabel(rf"$\|{tex}(t)\|_\infty$  [$\times 10^{{-4}}$ m]")
        ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
        fig.tight_layout()

        if SAVE_PNG:
            out = Path(__file__).parent / f"two_rod_vibroimpact_frc_{sig}.png"
            fig.savefig(out, dpi=150)
            print(f"Figure saved: {out}")

    plt.show()


if __name__ == "__main__":
    main()
