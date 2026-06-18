"""Plotting for the rod method comparison (AFT vs DLFT).

Outputs:
    plot_frc_overview -- full-width overlay on top + one small-multiple per config
                         below; each panel shows the config against the linear
                         (no-contact) backdrop.
    plot_metrics      -- grouped bar charts (solve time, total Newton iters, mean
                         step length, mean iters / point).

All figures are built unconditionally; passing ``out_path=None`` builds the figure
without writing a PNG (the comparison script gates this on its ``SAVE_PNG`` flag).
"""
import numpy as np
import matplotlib.pyplot as plt


# ============================ FRC drawing helpers ==========================

def _draw_context(ax, om_lin, peak_lin, gap, scale):
    """Linear (no-contact) FRF + contact threshold backdrop."""
    if om_lin is not None:
        ax.plot(om_lin, peak_lin / scale, ':', color="k", lw=0.8,
                label="Linear FRF (no contact)")
    if gap is not None:
        ax.axhline(gap / scale, color="red", ls="--", lw=1.0,
                   label=r"Contact threshold $g_0$")


def _draw_overlay(ax, results, scale):
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        style = "-" if r.method == "dlft" else "--"
        ax.plot(r.omega_phys, r.peak / scale, style, color=cmap(i % 10), lw=1.5,
                label=r.label)


def _draw_panel(ax, res, om_lin, peak_lin, gap, scale):
    """One small-multiple: this config against the linear backdrop."""
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    col   = "#1F77B4" if res.method == "dlft" else "#E8820C"
    style = "-" if res.method == "dlft" else "--"
    ax.plot(res.omega_phys, res.peak / scale, style, color=col, lw=1.8,
            label="this config")
    ax.set_title(res.label, fontsize=8)


# ============================ FRC overlay (standalone) =====================

def plot_frc_overlay(results, out_path, *, om_lin=None, peak_lin=None,
                     gap=None, scale=1.0e-4,
                     xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5)):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    _draw_overlay(ax, results, scale)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
    ax.set_ylabel(r"$\|u_B(t)\|_\infty$  [$\times 10^{-4}$ m]")
    ax.set_title("Method comparison: AFT (dashed) vs DLFT (solid), "
                 "numerical vs experimental FRF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=1)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Figure saved: {out_path}")


# ============================ overlay + per-config grid ====================

def plot_frc_overview(results, out_path, *, om_lin=None, peak_lin=None,
                      gap=None, scale=1.0e-4,
                      xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5), ncols=3):
    """Combined figure: full-width overlay on top, one panel per config below.
    ``out_path=None`` builds the figure without saving it.
    """
    n = len(results)
    nrows = (n + ncols - 1) // ncols

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(5.0 * ncols, 4.2 + 2.6 * nrows))
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[2.2] + [1.0] * nrows, hspace=0.55, wspace=0.30)

    # --- top: overlay across the full width ---
    ax_top = fig.add_subplot(gs[0, :])
    _draw_context(ax_top, om_lin, peak_lin, gap, scale)
    _draw_overlay(ax_top, results, scale)
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_ylabel(r"$\|u_B(t)\|_\infty$  [$\times 10^{-4}$ m]")
    ax_top.set_title("Method comparison overview: AFT (dashed) vs DLFT (solid), "
                     "numerical vs experimental FRF", fontsize=11)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)

    # --- below: one small multiple per configuration ---
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _draw_panel(ax, res, om_lin, peak_lin, gap, scale)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(r"$\|u_B\|_\infty$ [$\times10^{-4}$]", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\omega$ [rad/s]", fontsize=8)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
    fig.suptitle("Method comparison metrics  (blue = DLFT, orange = AFT)", fontsize=12)
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
