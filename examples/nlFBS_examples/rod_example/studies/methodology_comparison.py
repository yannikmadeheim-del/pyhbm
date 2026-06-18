"""AFT vs DLFT method study on the clamped-free rod vibro-impact problem.

For each enabled (method, FRF source) combination in ``config.RUN`` this traces
one continuation branch per swept parameter value (one-axis-at-a-time around the
baseline), collects performance + accuracy metrics, and produces:

    rod_method_comparison_frc.png      -- all branches overlaid (visual accuracy)
    rod_method_comparison_metrics.png  -- grouped bars (time / iters / step / error)
    + a console summary table

Variations are toggled entirely from ``config.py``:
    RUN[...]      = False   -> skip a whole (method, frf) family
    SWEEPS[axis]  = [v]     -> single value (no sweep on that axis)
    SWEEPS[axis]  = [a, b]  -> one branch per value

Swept axes and where they apply (only quantities that change the CONVERGED
solution -- the DLFT penalty epsilon is fixed, not swept):
    k_rel   -> all variants      (obstacle stiffness k_obs / k_rod)
    alpha   -> AFT only          (tanh regularization sharpness)
    density -> experimental only (measured-FRF density, samples per Hz)
    noise   -> experimental only (measured-FRF SNR in dB; inf = clean.  Noise is
                                  injected in the TIME domain, then transformed
                                  back, to mimic real measurement noise.)

Run:  python methodology_comparison.py
"""
import _setup  # noqa: F401

from pathlib import Path

import numpy as np

from pyhbm import DLFTContact, AFT

import config as cfg
from systems import RodVibroImpactFlexible, RodPenaltyContact
from frf import make_numerical_provider, make_experimental_provider
from run import run_branch, linear_relative
import plotting

SAVE_PNG = False   # set True to write the comparison figures to disk

_SHORT = {"k_rel": "k", "alpha": "a", "density": "d", "noise": "snr"}
OMEGA_RANGE = (cfg.OMEGA_START, cfg.OMEGA_END)


def _applicable_axes(method, frf):
    # epsilon is NOT here: DLFT's converged solution is epsilon-independent, so it
    # is a fixed setting (cfg.EPSILON_REL), not a sweepable comparison axis.
    axes = ["k_rel"]
    if method == "aft":
        axes.append("alpha")
    if frf == "experimental":
        axes.append("density")
        axes.append("noise")
    return axes


def _fmt(ax, v):
    if ax == "noise":
        return "clean" if not np.isfinite(v) else f"{v:g}dB"
    return f"{v:g}"


def _label(method, frf, params, axes):
    parts = [f"{method}/{frf}"] + [f"{_SHORT[a]}={_fmt(a, params[a])}" for a in axes]
    return " ".join(parts)


def _build(method, frf, params):
    """Return (system, provider, method_object) for one variant."""
    p = cfg.PARAMS
    if method == "dlft":
        system = RodVibroImpactFlexible(p, k_rel=params["k_rel"])
        nl = DLFTContact(epsilon=cfg.EPSILON_REL * system.k_rod, g_zero=cfg.GAP)
    else:  # aft
        system = RodPenaltyContact(p, k_rel=params["k_rel"],
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


def main():
    specs = _collect_specs()

    print(f"Running {len(specs)} variant(s)\n")

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

    plotting.print_metrics_table(results)

    # linear FRF backdrop (any system gives the same no-contact relative response)
    ref_sys = RodVibroImpactFlexible(cfg.PARAMS, k_rel=cfg.BASELINE["k_rel"])
    wh = np.linspace(cfg.OMEGA_START, cfg.OMEGA_END, 600)
    peak_lin = np.array([abs(linear_relative(ref_sys, w)) for w in wh])
    om_lin = wh * ref_sys.omega_ref

    here = Path(__file__).parent
    plotting.plot_frc_overview(
        results,
        here / "rod_method_comparison_frc.png" if SAVE_PNG else None,
        om_lin=om_lin, peak_lin=peak_lin, gap=cfg.GAP)
    plotting.plot_metrics(
        results,
        here / "rod_method_comparison_metrics.png" if SAVE_PNG else None)

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
