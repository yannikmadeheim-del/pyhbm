"""Quicklook of the stored two-rod reference NFRC CSV (no solving).

Plots both signals (driven tip u_A and relative approach x_r) against the linear
no-contact FRF and the gap line, plus the branch path order (point index) to
diagnose folds / coverage.

Run:  python plot_reference.py
"""
import _setup  # noqa: F401

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from systems import TwoRodVibroImpact
from run import linear_relative
from reference import reference_path, load_reference_csv

ref = load_reference_csv(reference_path())
print(f"{len(ref.omega)} points, omega in [{ref.omega.min():.0f}, "
      f"{ref.omega.max():.0f}] rad/s")

system = TwoRodVibroImpact(cfg.PARAMS)
wh = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
peak_lin = np.array([abs(linear_relative(system, w)) for w in wh])
om_lin = wh * system.omega_ref

SCALE = 1.0e-4
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
for ax, sig, tex in zip(axes, ("tipA", "xr"), (r"u_A", r"x_r")):
    ax.plot(om_lin, peak_lin / SCALE, ':', color="k", lw=1.0, label="linear FRF")
    ax.axhline(cfg.GAP / SCALE, color="red", ls="--", lw=1.0, label=r"$g_0$")
    sc = ax.scatter(ref.omega, ref.A_peak[sig] / SCALE, c=np.arange(len(ref.omega)),
                    s=2, cmap="viridis")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(rf"$\|{tex}(t)\|_\infty$ [$\times 10^{{-4}}$ m]")
    ax.set_title(f"reference: {sig}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
fig.colorbar(sc, ax=axes, label="branch point index", shrink=0.85)

out = Path(__file__).parent / "reference_quicklook.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Figure saved: {out}")
plt.show()
