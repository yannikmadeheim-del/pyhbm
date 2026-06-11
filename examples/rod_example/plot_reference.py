"""Quick-look at a stored reference: NFRC curve + a time-domain tip orbit.

Reads the dense reference CSV written by make_reference.py (it carries the full
physical u_B spectrum per branch point) and shows, side by side:
  * the NFRC (omega vs peak tip amplitude), with the inspected point marked, and
  * the reconstructed rod-tip orbit u_B(t) over a few periods at that point.

By default it picks the resonance peak (max amplitude).  Set TARGET_OMEGA to a
physical frequency [rad/s] to inspect a specific point instead.

Run:  python plot_reference.py
"""
import _setup  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from systems import RodVibroImpactFlexible
from reference import reference_path, load_reference_csv

# ---------------- choices ----------------
K_REL        = cfg.BASELINE["k_rel"]   # which reference to view
TARGET_OMEGA = None                    # None -> resonance peak; else physical rad/s
N_PERIODS    = 3
SCALE        = 1.0e-4                   # plot displacements in 1e-4 m
# -----------------------------------------

ref = load_reference_csv(reference_path(K_REL))
omega_ref = RodVibroImpactFlexible(cfg.PARAMS, k_rel=K_REL).omega_ref

# pick the branch point to inspect
if TARGET_OMEGA is None:
    i = int(np.argmax(ref.A_peak))
else:
    i = int(np.argmin(np.abs(ref.omega - TARGET_OMEGA)))
omega_i = ref.omega[i]

# reconstruct u_B(tau) from the stored physical two-sided coefficients c_hat_k:
#   u_B(tau) = c_hat_0 + 2 * sum_{k>=1} Re( c_hat_k e^{i k tau} )
chat = ref.uB_harmonics[i]                       # (H+1,) complex
tau  = np.linspace(0.0, 2 * np.pi * N_PERIODS, 2000)
uB   = np.full_like(tau, chat[0].real)
for k in range(1, len(chat)):
    uB += 2.0 * np.real(chat[k] * np.exp(1j * k * tau))

# ---------------- plot ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))

ax1.plot(ref.omega, ref.A_peak / SCALE, '-', color="#1F77B4", lw=1.4, label="reference NFRC")
ax1.plot(omega_i, ref.A_peak[i] / SCALE, 'o', color="#D62728", ms=7, label="inspected point")
ax1.axhline(cfg.GAP / SCALE, color="red", ls="--", lw=1.0, label=r"gap $g_0$")
ax1.set_xlabel(r"$\omega$ [rad/s]")
ax1.set_ylabel(r"$\|u_B\|_\infty$ [$\times10^{-4}$ m]")
ax1.set_title(f"Reference NFRC  (k_rel={K_REL:g}, H={len(chat) - 1}, {len(ref.omega)} pts)")
ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

ax2.plot(tau / (2 * np.pi), uB / SCALE, '-', color="#E8820C", lw=1.3, label=r"$u_B(t)$")
ax2.axhline(cfg.GAP / SCALE, color="red", ls="--", lw=1.0, label=r"gap $g_0$")
ax2.set_xlabel("t / T  (periods)")
ax2.set_ylabel(r"$u_B(t)$ [$\times10^{-4}$ m]")
ax2.set_title(rf"Tip orbit at $\omega$={omega_i:.0f} rad/s "
              rf"($\omega/\omega_1$={omega_i / omega_ref:.3f})")
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

fig.tight_layout()
out = reference_path(K_REL).with_name("reference_quicklook.png")
fig.savefig(out, dpi=130)
print(f"peak amplitude {ref.A_peak[i]:.3e} m at omega={omega_i:.1f} rad/s "
      f"(omega/omega_1={omega_i / omega_ref:.3f});  saved {out.name}")
plt.show()
