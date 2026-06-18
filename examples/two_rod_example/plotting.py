"""Plotting + trajectory-error metrics for the two-rod method comparison.

Same layouts as the single-rod example, evaluated PER SIGNAL -- the comparison
script calls each figure twice, once for the driven tip u_A and once for the
relative approach x_r = u_A + u_B (two separate windows/files):

    plot_frc_overview -- full-width overlay on top + one small-multiple per config
                         below; each panel shows the config against the stored
                         reference NFRC and its per-point relative error (twin axis).
    plot_metrics      -- grouped bars (solve time, total Newton iters, MEAN step
                         length, and the trajectory RMS / max relative error).

Trajectory error.  The NFRC folds (multivalued in omega near the overhang), so a
pointwise error(omega) is ill-posed.  For each variant point we take the nearest
point on the dense reference curve (normalized (omega, A) plane) and report the
pure pointwise relative amplitude error; see reference.relative_error.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reference import relative_error, SIGNALS

SIGNAL_TEX = {"tipA": r"u_A", "xr": r"x_r"}
SIGNAL_DESC = {"tipA": "driven tip $u_A$",
               "xr":   "relative approach $x_r = u_A + u_B$"}


# ============================ trajectory error =============================

def _ref_for(reference, res):
    """Resolve the reference for a result: a single curve or a {LB_rel: curve} dict."""
    if isinstance(reference, dict):
        return reference.get(res.params.get("LB_rel"))
    return reference


def compute_trajectory_error(results, reference):
    """Fill traj_err / traj_mean / traj_max / traj_rms (dicts per signal) on each
    result.

    ``reference`` is a ReferenceCurve, or a {LB_rel: ReferenceCurve} mapping when
    different configs use different rod-B lengths.  Warns when a config's branch
    extends beyond the reference's arc-length coverage (which would make the
    error there meaningless -- regenerate a fuller/denser reference).
    """
    for r in results:
        ref = _ref_for(reference, r)
        if ref is None:
            continue
        for s in SIGNALS:
            e, mean, mx, rms, frac_out = relative_error(r, ref, signal=s)
            r.traj_err[s], r.traj_mean[s] = e, mean
            r.traj_max[s], r.traj_rms[s]  = mx, rms
            if frac_out > 0.01:
                print(f"  [traj-error] WARNING: {r.label} ({s}): {frac_out:.0%} of "
                      f"points lie beyond the reference's coverage -- regenerate a "
                      f"fuller reference.")


# ============================ FRC drawing helpers ==========================

def _draw_context(ax, om_lin, peak_lin, gap, scale):
    """Linear (no-contact) FRF + contact threshold backdrop."""
    if om_lin is not None:
        ax.plot(om_lin, peak_lin / scale, ':', color="k", lw=0.8,
                label="Linear FRF (no contact)")
    if gap is not None:
        ax.axhline(gap / scale, color="red", ls="--", lw=1.0,
                   label=r"Contact threshold $g_0$")


def _draw_reference(ax, reference, signal, scale, lw=3.0):
    """Draw the stored reference NFRC(s) as a thick grey 'original' line."""
    refs = reference.values() if isinstance(reference, dict) else [reference]
    seen = set()
    for ref in refs:
        if ref is None or id(ref) in seen:
            continue
        seen.add(id(ref))
        ax.plot(ref.omega, ref.A_peak[signal] / scale, '-', color="0.6",
                lw=lw, zorder=1, label="reference (DLFT/num, CSV)")


def _draw_overlay(ax, results, signal, scale, reference=None):
    if reference is not None:
        _draw_reference(ax, reference, signal, scale, lw=3.5)
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        style = "-" if r.method == "dlft" else "--"
        ax.plot(r.omega_phys, r.peak[signal] / scale, style, color=cmap(i % 10),
                lw=1.5, label=r.label)


def _draw_panel(ax, res, reference, signal, om_lin, peak_lin, gap, scale):
    """One small-multiple: this config vs the stored reference + its rel. error."""
    _draw_context(ax, om_lin, peak_lin, gap, scale)
    ref = _ref_for(reference, res)
    if ref is not None:
        ax.plot(ref.omega, ref.A_peak[signal] / scale, '-',
                color="0.6", lw=3.0, label="reference")
    col   = "#1F77B4" if res.method == "dlft" else "#E8820C"
    style = "-" if res.method == "dlft" else "--"
    ax.plot(res.omega_phys, res.peak[signal] / scale, style, color=col, lw=1.8,
            label="this config")

    # per-point relative error on a twin y-axis
    err = res.traj_err.get(signal)
    if err is not None and len(err):
        ax2 = ax.twinx()
        ax2.plot(res.omega_phys, 100.0 * err, color="#D62728", lw=0.8,
                 alpha=0.7)
        ax2.set_ylim(bottom=0.0)
        ax2.set_ylabel("rel. err [%]", color="#D62728", fontsize=7)
        ax2.tick_params(axis="y", labelsize=6, colors="#D62728")

    ax.set_title(res.label, fontsize=8)
    rms = res.traj_rms.get(signal, float("nan"))
    mx  = res.traj_max.get(signal, float("nan"))
    txt = "no reference" if np.isnan(rms) else f"RMS={rms:.1%}\nmax={mx:.1%}"
    ax.text(0.025, 0.95, txt,
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


# ============================ overlay + per-config grid ====================

def plot_frc_overview(results, out_path, *, signal="tipA", om_lin=None,
                      peak_lin=None, gap=None, reference=None, scale=1.0e-4,
                      xlim=(5.8e4, 7.4e4), ylim=(0.0, 5.5), ncols=3):
    """Combined figure for ONE signal: full-width overlay on top, one panel per
    config below.

    Each lower panel shows that single configuration against the stored reference
    ("original") NFRC plus its per-point relative error, so each variant can be
    validated on its own.
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
    _draw_overlay(ax_top, results, signal, scale, reference=reference)
    ax_top.set_xlim(*xlim); ax_top.set_ylim(*ylim)
    ax_top.set_ylabel(rf"$\|{tex}(t)\|_\infty$  [$\times 10^{{-4}}$ m]")
    ax_top.set_title(f"Two-rod method comparison ({SIGNAL_DESC[signal]}): "
                     "AFT (dashed) vs DLFT (solid)", fontsize=11)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)

    # --- below: one small multiple per configuration ---
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        _draw_panel(ax, res, reference, signal, om_lin, peak_lin, gap, scale)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel(rf"$\|{tex}\|_\infty$ [$\times10^{{-4}}$]", fontsize=8)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\omega$ [rad/s]", fontsize=8)
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

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
    data the solver works with.
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
    fig.savefig(out_path, dpi=140)
    print(f"Figure saved: {out_path}")


# ============================ metrics bars =================================

def plot_metrics(results, out_path, *, signal="tipA"):
    labels = [r.label for r in results]
    x = np.arange(len(labels))
    tex = SIGNAL_TEX[signal]

    panels = [
        ("Solve time [s]",            [r.solve_time for r in results]),
        ("Total Newton iterations",   [r.total_iters for r in results]),
        ("Mean step length",          [r.mean_step  for r in results]),
        ("Mean iters / point",        [r.mean_iters for r in results]),
        (rf"Trajectory RMS rel. error (${tex}$)",
         [r.traj_rms.get(signal, float("nan")) for r in results]),
        (rf"Trajectory max rel. error (${tex}$)",
         [r.traj_max.get(signal, float("nan")) for r in results]),
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
    fig.suptitle("Two-rod method comparison metrics  (blue = DLFT, orange = AFT)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")


def print_metrics_table(results, reference_label=None):
    """Console summary table of all branches (errors per signal)."""
    hdr = (f"{'variant':<34}{'pts':>5}{'time[s]':>9}{'iters':>7}"
           f"{'mean.step':>11}{'rms.uA':>10}{'rms.xr':>10}{'max.uA':>10}"
           f"{'max.xr':>10}{'cov':>6}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    nan = float("nan")
    for r in results:
        print(f"{r.label[:34]:<34}{r.n_points:>5}{r.solve_time:>9.1f}"
              f"{r.total_iters:>7}{r.mean_step:>11.2e}"
              f"{r.traj_rms.get('tipA', nan):>10.2e}"
              f"{r.traj_rms.get('xr',  nan):>10.2e}"
              f"{r.traj_max.get('tipA', nan):>10.2e}"
              f"{r.traj_max.get('xr',  nan):>10.2e}{r.coverage:>6.0%}")
    print("=" * len(hdr))
    if reference_label:
        print(f"trajectory error reference = {reference_label}")
