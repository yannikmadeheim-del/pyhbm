"""Plotting for the two-rod method comparison (AFT vs DLFT).

Per response signal (the comparison script calls each figure twice, once for the
driven tip u_A and once for the relative approach x_r = u_A + u_B):

    plot_frc_overview -- full-width overlay on top + one small-multiple per config
                         below; each panel shows that single configuration against
                         the linear (no-contact) backdrop.
    plot_metrics      -- grouped bars (solve time, total Newton iters, mean step
                         length, mean iters / point).

All figures are built unconditionally; passing ``out_path=None`` builds the figure
without writing a PNG (the comparison script gates this on its ``SAVE_PNG`` flag).
"""
import numpy as np
import matplotlib.pyplot as plt

from run import SIGNAL_TEX, SIGNAL_DESC


# ============================ FRC drawing helpers ==========================

def _draw_context(ax, om_lin, peak_lin, gap, scale):
    """Linear (no-contact) FRF + contact threshold backdrop."""
    if om_lin is not None:
        ax.plot(om_lin, peak_lin / scale, ':', color="k", lw=0.8,
                label="Linear FRF (no contact)")
    if gap is not None:
        ax.axhline(gap / scale, color="red", ls="--", lw=1.0,
                   label=r"Contact threshold $g_0$")


def _draw_overlay(ax, results, signal, scale):
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        style = "-" if r.method == "dlft" else "--"
        ax.plot(r.omega_phys, r.peak[signal] / scale, style, color=cmap(i % 10),
                lw=1.5, label=r.label)


def _draw_panel(ax, res, signal, om_lin, peak_lin, gap, scale):
    """One small-multiple: this config against the linear backdrop."""
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    col   = "#1F77B4" if res.method == "dlft" else "#E8820C"
    style = "-" if res.method == "dlft" else "--"
    ax.plot(res.omega_phys, res.peak[signal] / scale, style, color=col, lw=1.8,
            label="this config")
    ax.set_title(res.label, fontsize=8)


# ============================ overlay + per-config grid ====================

def plot_frc_overview(results, out_path, *, signal="tipA", om_lin=None,
                      peak_lin=None, gap=None, scale=1.0e-4,
                      xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5), ncols=3):
    """Combined figure for ONE signal: full-width overlay on top, one panel per
    config below.  ``out_path=None`` builds the figure without saving it.
    """
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    tex = SIGNAL_TEX[signal]

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(5.0 * ncols, 4.2 + 2.6 * nrows))
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[2.2] + [1.0] * nrows, hspace=0.55, wspace=0.30)

    # --- top: overlay across the full width ---
    ax_top = fig.add_subplot(gs[0, :])
    _draw_context(ax_top, om_lin, peak_lin, gap, scale)
    _draw_overlay(ax_top, results, signal, scale)
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_ylabel(rf"$\|{tex}(t)\|_\infty$  [$\times 10^{{-4}}$ m]")
    ax_top.set_title(f"Two-rod method comparison ({SIGNAL_DESC[signal]}): "
                     "AFT (dashed) vs DLFT (solid)", fontsize=11)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)

    # --- below: one small multiple per configuration ---
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _draw_panel(ax, res, signal, om_lin, peak_lin, gap, scale)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(rf"$\|{tex}\|_\infty$ [$\times10^{{-4}}$]", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\omega$ [rad/s]", fontsize=8)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"Figure saved: {out_path}")


# ============================ uncoupled linear FRF =========================

def plot_uncoupled_frf(system, entry_1based, out_path, *, harmonics,
                       omega_range, density_per_hz, noise_levels=(),
                       noise_seed=None, margin=1.05):
    """Plot one entry |Y_ij(omega)| of the UNCOUPLED linear admittance.

    The system is block-diagonal (open gap, no contact), so entries inside rod A
    or rod B are the substructure FRFs and any cross entry (rod A <-> rod B,
    e.g. tip A -> rod B) is identically zero.  ``entry_1based`` uses 1-based DOF
    numbering: 1..n_elem = rod A (n_elem = tip A), n_elem+1..2*n_elem = rod B.

    The grid mirrors :func:`frf.make_experimental_provider` (``density_per_hz``
    samples/Hz up to the top queried harmonic frequency), and for every FINITE
    SNR in ``noise_levels`` the same time-domain noise model (and seed) used by
    the experimental branches is overlaid -- so the plot shows exactly the FRF
    data the solver works with.  ``out_path=None`` builds the figure without
    saving it.
    """
    from frf import sample_admittance, add_measurement_noise

    i, j = entry_1based[0] - 1, entry_1based[1] - 1
    h_max = int(np.max(harmonics))
    omega_hi      = max(abs(omega_range[0]), abs(omega_range[1]))
    omega_max_hat = h_max * omega_hi * margin
    f_max  = omega_max_hat * system.omega_ref / (2.0 * np.pi)
    n_freq = max(2, int(round(density_per_hz * f_max)) + 1)
    omega_grid = np.linspace(0.0, omega_max_hat, n_freq)

    # Y from the nondimensional matrices at omega_hat equals the physical
    # admittance at omega = omega_hat * omega_ref (units m/N either way).
    Y = sample_admittance(system, omega_grid)
    om_phys = omega_grid * system.omega_ref

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    lo = min(abs(omega_range[0]), abs(omega_range[1])) * system.omega_ref
    ax.axvspan(lo, omega_hi * system.omega_ref, color="0.88", zorder=0,
               label="continuation window")

    for snr in noise_levels:
        if not np.isfinite(snr):
            continue
        Yn = add_measurement_noise(omega_grid, Y, snr,
                                   rng=np.random.default_rng(noise_seed))
        ax.plot(om_phys, np.abs(Yn[:, i, j]), lw=0.6, alpha=0.8,
                label=f"noisy (SNR = {snr:g} dB)")
    y_clean = np.abs(Y[:, i, j])
    ax.plot(om_phys, y_clean, 'k-', lw=1.3, label="clean")

    if np.any(y_clean > 0.0):
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.55, "identically zero:\nrods are uncoupled (no contact)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    n = getattr(system, "rod_tip_idx", 0) + 1            # n_elem (1-based tip A)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(rf"$|Y_{{{entry_1based[0]},{entry_1based[1]}}}(\omega)|$  [m/N]")
    ax.set_title(f"Uncoupled linear admittance, entry ({entry_1based[0]}, "
                 f"{entry_1based[1]})   [tip A = DOF {n}, tip B = DOF {2*n}]",
                 fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
        print(f"Figure saved: {out_path}")


# ============================ metrics bars =================================

def plot_metrics(results, out_path):
    labels = [r.label for r in results]
    x = np.arange(len(labels))

    panels = [
        ("Solve time [s]",          [r.solve_time  for r in results]),
        ("Total Newton iterations", [r.total_iters for r in results]),
        ("Mean step length",        [r.mean_step   for r in results]),
        ("Mean iters / point",      [r.mean_iters  for r in results]),
    ]
    colors = ["#1F77B4" if r.method == "dlft" else "#E8820C" for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for ax, (title, vals) in zip(axes.ravel(), panels):
        ax.bar(x, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        if "step" in title and np.nanmax(np.append(vals, 0.0)) > 0:
            ax.set_yscale("log")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#1F77B4", label="DLFT"),
                        Patch(color="#E8820C", label="AFT")],
               loc="upper right", fontsize=9)
    fig.suptitle("Two-rod method comparison metrics  (blue = DLFT, orange = AFT)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Figure saved: {out_path}")


def print_metrics_table(results):
    """Console summary table of all branches (performance metrics only)."""
    hdr = (f"{'variant':<34}{'pts':>5}{'time[s]':>9}{'iters':>7}"
           f"{'mean.iters':>11}{'mean.step':>11}{'cov':>6}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r.label[:34]:<34}{r.n_points:>5}{r.solve_time:>9.1f}"
              f"{r.total_iters:>7}{r.mean_iters:>11.2f}{r.mean_step:>11.2e}"
              f"{r.coverage:>6.0%}")
    print("=" * len(hdr))
