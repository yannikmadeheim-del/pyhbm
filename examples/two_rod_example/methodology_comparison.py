"""AFT vs DLFT method study on the TWO-ROD vibro-impact problem.

Two identical clamped-free FE rods face each other across a gap g0; the left rod
is driven at its tip.  For each enabled (method, FRF source) combination in
``config.RUN`` this traces one continuation branch per swept parameter value
(one-axis-at-a-time around the baseline), collects performance + accuracy
metrics, and produces -- per response signal (driven tip u_A and relative
approach x_r = u_A + u_B), in TWO separate windows with the same layout as the
single-rod example --

    two_rod_method_comparison_frc_tipA.png  -- u_A branches overlaid + per-config panels
    two_rod_method_comparison_frc_xr.png    -- x_r branches overlaid + per-config panels
    two_rod_method_comparison_metrics.png   -- grouped bars (time / iters / step / error)
    + a console summary table

Variations are toggled entirely from ``config.py``:
    RUN[...]      = False   -> skip a whole (method, frf) family
    SWEEPS[axis]  = [v]     -> single value (no sweep on that axis)
    SWEEPS[axis]  = [a, b]  -> one branch per value

Swept axes and where they apply (the DLFT penalty epsilon is fixed, not swept;
DLFT itself is parameter-free -- rigid tip-to-tip contact):
    k_rel   -> AFT only          (penalty stiffness k_c / k_rod)
    alpha   -> AFT only          (tanh regularization sharpness;
                                  np.inf = WITHOUT regularization)
    density -> experimental only (measured-FRF density, samples per Hz)
    noise   -> experimental only (measured-FRF SNR in dB; inf = clean)

Run:  python methodology_comparison.py
"""
import _setup  # noqa: F401

from pathlib import Path

import numpy as np

from pyhbm import DLFTContact, AFT

import config as cfg
from systems import TwoRodVibroImpact, TwoRodPenaltyContact
from frf import make_numerical_provider, make_experimental_provider
from run import run_branch, linear_relative, SIGNALS
import reference as refmod
import plotting

_SHORT = {"LB_rel": "LB", "k_rel": "k", "alpha": "a", "density": "d", "noise": "snr"}
OMEGA_RANGE = (cfg.OMEGA_START, cfg.OMEGA_END)


def _applicable_axes(method, frf):
    # LB_rel (rod-B length, i.e. obstacle-rod stiffness) changes the SYSTEM and
    # applies to every variant.  The rigid tip-to-tip DLFT contact itself is
    # parameter-free (epsilon is not a comparison axis: converged solution is
    # epsilon-independent); k_rel and alpha exist only for the AFT penalty model.
    axes = ["LB_rel"]
    if method == "aft":
        axes.append("k_rel")
        axes.append("alpha")
    if frf == "experimental":
        axes.append("density")
        axes.append("noise")
    return axes


def _fmt(ax, v):
    if ax == "noise":
        return "clean" if not np.isfinite(v) else f"{v:g}dB"
    if ax == "alpha":
        return "no-reg" if not np.isfinite(v) else f"{v:g}"
    return f"{v:g}"


def _label(method, frf, params, axes):
    parts = [f"{method}/{frf}"] + [f"{_SHORT[a]}={_fmt(a, params[a])}" for a in axes]
    return " ".join(parts)


def _build(method, frf, params):
    """Return (system, provider, method_object) for one variant."""
    p   = cfg.PARAMS
    p_B = cfg.params_B_for(params["LB_rel"])
    if method == "dlft":
        system = TwoRodVibroImpact(p, p_B)
        nl = DLFTContact(epsilon=cfg.EPSILON_REL * system.k_rod, g_zero=cfg.GAP)
    else:  # aft
        system = TwoRodPenaltyContact(p, p_B, k_rel=params["k_rel"],
                                      g_zero=cfg.GAP, alpha=params["alpha"])
        nl = AFT()

    if frf == "numerical":
        provider = make_numerical_provider(system)
    else:
        provider, n_freq = make_experimental_provider(
            system, density_per_hz=params["density"],
            harmonics=cfg.HARMONICS, omega_range=OMEGA_RANGE,
            snr_db=params["noise"], noise_seed=cfg.NOISE_SEED)
        params["n_freq"] = n_freq          # resolved sample count, for reporting
    return system, provider, nl


def _collect_specs():
    """Build the deduplicated list of (method, frf, params, label) to run."""
    specs = {}   # label -> spec
    for key, enabled in cfg.RUN.items():
        if not enabled:
            continue
        method, frf = key.split("_")
        axes = _applicable_axes(method, frf)
        # one-axis-at-a-time around the baseline
        for ax in axes:
            for val in cfg.SWEEPS.get(ax, [cfg.BASELINE[ax]]):
                params = {a: cfg.BASELINE[a] for a in axes}
                params[ax] = val
                lbl = _label(method, frf, params, axes)
                specs.setdefault(lbl, (method, frf, params, axes))
    return specs


def _load_references(specs):
    """Load the reference NFRC CSV for every LB_rel present in the specs.

    Missing CSVs are NOT fatal: a warning is printed and the affected variants
    run and plot without a reference (no grey "original" curve, no trajectory
    error -- the metrics show NaN there).  Returns {LB_rel: ReferenceCurve}.
    """
    lb_rels = sorted({params["LB_rel"] for (_, _, params, _) in specs.values()})
    curves, missing = {}, []
    for lb_rel in lb_rels:
        path = refmod.reference_path(lb_rel)
        if path.exists():
            curves[lb_rel] = refmod.load_reference_csv(path)
        else:
            missing.append(path)
    if missing:
        print("WARNING: missing reference CSV(s) -- the affected variants run "
              "WITHOUT trajectory error:")
        for p in missing:
            print("   ", p.name)
        print("Generate them with:  python make_reference.py\n")
    return curves


def main():
    specs = _collect_specs()
    ref_curves = _load_references(specs)

    print(f"Running {len(specs)} variant(s); trajectory error vs "
          f"{cfg.REFERENCE_LABEL} (where a reference exists)\n")

    results = []
    for lbl, (method, frf, params, axes) in specs.items():
        print(f"--- {lbl}")
        system, provider, nl = _build(method, frf, params)
        res = run_branch(
            system, provider, nl,
            harmonics=cfg.HARMONICS, omega_range=OMEGA_RANGE,
            solver_kwargs=cfg.SOLVER_KWARGS, step_kwargs=cfg.STEP_KWARGS,
            max_solutions=cfg.MAX_SOLUTIONS,
            label=lbl, method_name=method, frf_name=frf, params=params,
            verbose=True,
        )
        results.append(res)

    plotting.compute_trajectory_error(results, ref_curves)
    plotting.print_metrics_table(results, reference_label=cfg.REFERENCE_LABEL)

    # linear FRF backdrop: without contact rod B is unforced (u_B = 0), so the
    # same curve backs both the u_A and the x_r window -- and it is independent
    # of rod B entirely.
    ref_sys = TwoRodVibroImpact(cfg.PARAMS)
    wh = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
    peak_lin = np.array([abs(linear_relative(ref_sys, w)) for w in wh])
    om_lin = wh * ref_sys.omega_ref

    here = Path(__file__).parent
    for sig in SIGNALS:
        plotting.plot_frc_overview(
            results, here / f"two_rod_method_comparison_frc_{sig}.png",
            signal=sig, om_lin=om_lin, peak_lin=peak_lin, gap=cfg.GAP,
            reference=ref_curves)
    plotting.plot_metrics(results, here / "two_rod_method_comparison_metrics.png",
                          signal="tipA")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
