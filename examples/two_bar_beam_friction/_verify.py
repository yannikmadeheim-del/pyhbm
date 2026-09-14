"""
Throwaway verification for the two_bar_beam_friction example -- DELETE ANYTIME.

Nothing imports this file; dynamical_system.py and main.py do not depend on it
and stay assert-free. It checks, in order:

  1. AFT sampling      N_t next to the pyFBS example's sample_number
  2. Force laws        finite-difference check of both analytical Jacobians
  3. Branches          the two solver formulations against each other
  4. pyFBS             any pyFBS CSV dropped into results/, against ours

Run after main.py:

    python examples/two_bar_beam_friction/_verify.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

import dynamical_system as ds
from dynamical_system import (jacobian_nonlinear_force,
                              jacobian_nonlinear_force_qdot, nonlinear_force)

RESULTS = Path(__file__).resolve().parent / "results"
OURS = ("second_order", "first_order")

failures = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------- 1. sampling
section("1. AFT sampling")

h_max = 15
n_t = (ds.POLYNOMIAL_DEGREE + 1) * h_max + 1
print(f"  polynomial_degree={ds.POLYNOMIAL_DEGREE}, h_max={h_max} -> N_t={n_t}"
      f"  (pyFBS sample_number: 512)")


# ------------------------------------------------------------- 2. force laws
section("2. Force laws: analytical Jacobians vs finite differences")

rng = np.random.default_rng(0)
for scale, regime in ((2e-3, "partial slip (tanh unsaturated)"),
                      (5e-1, "gross slip (tanh saturated)")):
    q = rng.normal(scale=0.01, size=(8, 6, 1))
    q[:, ds.Q2, 0] -= 0.005
    q[:, ds.Q5, 0] += 0.005                      # push the tips into contact
    qdot = rng.normal(scale=scale, size=(8, 6, 1))

    step = 1e-8
    for name, analytic, perturb in (
            ("d f_nl / d q   ", jacobian_nonlinear_force(q, qdot), "q"),
            ("d f_nl / d qdot", jacobian_nonlinear_force_qdot(q, qdot), "qdot")):
        numeric = np.zeros_like(analytic)
        for j in range(6):
            plus, minus = q.copy(), q.copy()
            if perturb == "qdot":
                plus, minus = qdot.copy(), qdot.copy()
            plus[:, j, 0] += step
            minus[:, j, 0] -= step
            if perturb == "q":
                column = nonlinear_force(plus, qdot) - nonlinear_force(minus, qdot)
            else:
                column = nonlinear_force(q, plus) - nonlinear_force(q, minus)
            numeric[:, :, j] = (column / (2 * step))[:, :, 0]
        scale_ref = max(np.abs(numeric).max(), 1e-12)
        error = np.abs(analytic - numeric).max() / scale_ref
        check(f"{name} -- {regime}", error < 1e-5, f"rel. error {error:.2e}")


# --------------------------------------------------------------- 3. branches
section("3. Solver formulations against each other")


def read(path):
    """One result CSV -> {column name: values} (no pandas needed).

    The comment header carries commas of its own (solver_kwargs, harmonics),
    which is why the '#' lines are dropped by hand rather than left to
    genfromtxt's comment handling.
    """
    with open(path) as handle:
        lines = [line for line in handle if not line.startswith("#")]
    names = lines[0].strip().split(",")
    return dict(zip(names, np.loadtxt(lines[1:], delimiter=",").T))


def peak(curve):
    """(omega, max_t|q1|) at the branch maximum."""
    index = int(np.nanargmax(curve["uout_time_max"]))
    return curve["omega_rad_s"][index], curve["uout_time_max"][index]


def compare(curves, reference):
    """Interpolate every curve onto the reference omega grid and report."""
    w_ref = curves[reference]["omega_rad_s"]
    order = np.argsort(w_ref)
    for name, curve in curves.items():
        w_peak, u_peak = peak(curve)
        line = (f"  {name:22s} {len(curve['omega_rad_s']):5d} points,"
                f" peak max_t|q1| = {u_peak:.6f} at omega = {w_peak:.6f}")
        if name != reference:
            w, u = curve["omega_rad_s"], curve["uout_time_max"]
            inside = (w_ref >= w.min()) & (w_ref <= w.max())
            resampled = np.interp(w_ref[order][inside[order]],
                                  *(lambda s: (w[s], u[s]))(np.argsort(w)))
            target = curves[reference]["uout_time_max"][order][inside[order]]
            deviation = np.abs(resampled - target).max() / np.abs(target).max()
            line += f",  max rel. deviation vs {reference}: {deviation:.2e}"
        print(line)


available = {name: read(RESULTS / f"{name}.csv") for name in OURS
             if (RESULTS / f"{name}.csv").exists()}
missing = [name for name in OURS if name not in available]
if missing:
    print(f"  note: {', '.join(missing)} not in results/ -- run main.py first")
if available:
    compare(available, reference=next(iter(available)))
    if len(available) == 2:
        a, b = (available[name]["uout_time_max"] for name in OURS)
        check("both formulations reach the same peak",
              abs(a.max() - b.max()) / a.max() < 1e-3,
              f"{a.max():.6f} vs {b.max():.6f}")
else:
    check("results/ contains at least one branch", False)


# ------------------------------------------------------------------ 4. pyFBS
section("4. pyFBS CSVs dropped into results/ (optional)")

foreign = sorted(p for p in RESULTS.glob("*.csv")
                 if p.stem not in OURS)
if not foreign:
    print("  none found -- copy a pyFBS result CSV into results/ to compare here")
elif available:
    everything = dict(available)
    everything.update({p.stem: read(p) for p in foreign})
    compare(everything, reference=next(iter(available)))


# ----------------------------------------------------------------- verdict
print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
