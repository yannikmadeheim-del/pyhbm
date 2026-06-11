"""Plotting + trajectory-error metrics for the method comparison.

Outputs:
    plot_frc_overview -- full-width overlay on top + one small-multiple per config
                         below; each panel shows the config against the stored
                         reference NFRC and its per-point relative error (twin axis).
    plot_metrics      -- grouped bar charts (solve time, total Newton iters, MEAN
                         step length, and the trajectory RMS / max relative error).

Trajectory error.  The NFRC folds (multivalued in omega near the overhang), so a
pointwise error(omega) is ill-posed.  For each variant point we take the nearest
point on the dense reference curve (normalized (omega, A) plane) and report the
pure pointwise relative amplitude error; see reference.relative_error.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reference import relative_error


# ============================ trajectory error =============================

def _ref_for(reference, res):
    """Resolve the reference for a result: a single curve or a {k_rel: curve} dict."""
    if isinstance(reference, dict):
        return reference.get(res.params.get("k_rel"))
    return reference


def compute_trajectory_error(results, reference):
    """Fill traj_err / traj_mean / traj_max / traj_rms on each result.

    ``reference`` is a ReferenceCurve, or a {k_rel: ReferenceCurve} mapping when
    different configs use different obstacle stiffnesses.  Warns when a config's
    branch extends beyond the reference's arc-length coverage (which would make the
    error there meaningless -- regenerate a fuller/denser reference).
    """
    for r in results:
        ref = _ref_for(reference, r)
        if ref is None:
            continue
        e, mean, mx, rms, frac_out = relative_error(r, ref)
        r.traj_err, r.traj_mean, r.traj_max, r.traj_rms = e, mean, mx, rms
        if frac_out > 0.01:
            print(f"  [traj-error] WARNING: {r.label}: {frac_out:.0%} of points lie "
                  f"beyond the reference's coverage -- regenerate a fuller reference.")


# ============================ FRC drawing helpers ==========================

def _draw_context(ax, om_lin, peak_lin, gap, scale):
    """Linear (no-contact) FRF + contact threshold backdrop."""
    if om_lin is not None:
        ax.plot(om_lin, peak_lin / scale, ':', color="k", lw=0.8,
                label="Linear FRF (no contact)")
    if gap is not None:
        ax.axhline(gap / scale, color="red", ls="--", lw=1.0,
                   label=r"Contact threshold $g_0$")


def _draw_reference(ax, reference, scale, lw=3.0):
    """Draw the stored reference NFRC(s) as a thick grey 'original' line."""
    refs = reference.values() if isinstance(reference, dict) else [reference]
    seen = set()
    for ref in refs:
        if ref is None or id(ref) in seen:
            continue
        seen.add(id(ref))
        ax.plot(ref.omega, ref.A_peak / scale, '-', color="0.6", lw=lw,
                zorder=1, label="reference (DLFT/num, CSV)")


def _draw_overlay(ax, results, scale, reference=None):
    if reference is not None:
        _draw_reference(ax, reference, scale, lw=3.5)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        style = "-" if r.method == "dlft" else "--"
        ax.plot(r.omega_phys, r.peak / scale, style, color=cmap(i % 10), lw=1.5,
                label=r.label)


def _draw_panel(ax, res, reference, om_lin, peak_lin, gap, scale):
    """One small-multiple: this config vs the stored reference + its rel. error."""
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    ref = _ref_for(reference, res)
    if ref is not None:
        ax.plot(ref.omega, ref.A_peak / scale, '-', color="0.6", lw=3.0,
                label="reference")
    col   = "#1F77B4" if res.method == "dlft" else "#E8820C"
    style = "-" if res.method == "dlft" else "--"
    ax.plot(res.omega_phys, res.peak / scale, style, color=col, lw=1.8,
            label="this config")

    # per-point relative error on a twin y-axis
    if res.traj_err is not None and len(res.traj_err):
        ax2 = ax.twinx()
        ax2.plot(res.omega_phys, 100.0 * res.traj_err, color="#D62728", lw=0.8,
                 alpha=0.7)
        ax2.set_ylim(bottom=0.0)
        ax2.set_ylabel("rel. err [%]", color="#D62728", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6, colors="#D62728")

    ax.set_title(res.label, fontsize=8)
    ax.text(0.025, 0.95, f"RMS={res.traj_rms:.1%}\nmax={res.traj_max:.1%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


# ============================ FRC overlay (standalone) =====================

def plot_frc_overlay(results, out_path, *, om_lin=None, peak_lin=None,
                     gap=None, reference=None, scale=1.0e-4,
                     xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5)):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    _draw_overlay(ax, results, scale, reference=reference)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
    ax.set_ylabel(r"$\|u_B(t)\|_\infty$  [$\times 10^{-4}$ m]")
    ax.set_title("Method comparison: AFT (dashed) vs DLFT (solid), "
                 "numerical vs experimental FRF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")


# ============================ overlay + per-config grid ====================

def plot_frc_overview(results, out_path, *, om_lin=None, peak_lin=None,
                      gap=None, reference=None, scale=1.0e-4,
                      xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5), ncols=3):
    """Combined figure: full-width overlay on top, one panel per config below.

    Each lower panel shows that single configuration against the stored reference
    ("original") NFRC plus its per-point relative error, so each variant can be
    validated on its own.
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
    _draw_overlay(ax_top, results, scale, reference=reference)
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_ylabel(r"$\|u_B(t)\|_\infty$  [$\times 10^{-4}$ m]")
    ax_top.set_title("Method comparison overview: AFT (dashed) vs DLFT (solid), "
                     "numerical vs experimental FRF", fontsize=11)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)

    # --- below: one small multiple per configuration ---
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _draw_panel(ax, res, reference, om_lin, peak_lin, gap, scale)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(r"$\|u_B\|_\infty$ [$\times10^{-4}$]", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\omega$ [rad/s]", fontsize=8)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


# ============================ metrics bars =================================

def plot_metrics(results, out_path):
    labels = [r.label for r in results]
    x = np.arange(len(labels))

    panels = [
        ("Solve time [s]",            [r.solve_time for r in results]),
        ("Total Newton iterations",   [r.total_iters for r in results]),
        ("Mean step length",          [r.mean_step  for r in results]),
        ("Mean iters / point",        [r.mean_iters for r in results]),
        ("Trajectory RMS rel. error", [r.traj_rms   for r in results]),
        ("Trajectory max rel. error", [r.traj_max   for r in results]),
    ]
    colors = ["#1F77B4" if r.method == "dlft" else "#E8820C" for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    for ax, (title, vals) in zip(axes.ravel(), panels):
        ax.bar(x, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        # log scale only when it is well-defined (at least one positive value)
        if ("error" in title or "step" in title) and np.nanmax(np.append(vals, 0.0)) > 0:
            ax.set_yscale("log")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#1F77B4", label="DLFT"),
                        Patch(color="#E8820C", label="AFT")],
               loc="upper right", fontsize=9)
    fig.suptitle("Method comparison metrics  (blue = DLFT, orange = AFT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")


def print_metrics_table(results, reference_label=None):
    """Console summary table of all branches."""
    hdr = (f"{'variant':<34}{'pts':>5}{'time[s]':>9}{'iters':>7}"
           f"{'mean.step':>11}{'rms.err':>10}{'max.err':>10}{'cov':>6}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r.label[:34]:<34}{r.n_points:>5}{r.solve_time:>9.1f}"
              f"{r.total_iters:>7}{r.mean_step:>11.2e}{r.traj_rms:>10.2e}"
              f"{r.traj_max:>10.2e}{r.coverage:>6.0%}")
    print("=" * len(hdr))
    if reference_label:
        print(f"trajectory error reference = {reference_label}")
