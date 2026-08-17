"""
Throwaway verification for the two_bar_beam_friction example -- DELETE ANYTIME.

Nothing imports this file; dynamical_system.py and main.py do not depend on it
and stay assert-free. It checks, in order:

  1. AFT sampling      N_t matches the pyFBS example's sample_number
  2. Force laws        finite-difference check of both analytical Jacobians
  3. Statics           the two relations Ke itself enforces, and what the
                       thesis reports against them
  4. Branches          the two solver formulations against each other
  5. pyFBS             any pyFBS CSV dropped into results/, against ours

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
check("N_t == 256 (pyFBS sample_number)", n_t == 256,
      f"polynomial_degree={ds.POLYNOMIAL_DEGREE}, h_max={h_max} -> N_t={n_t}")


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


# ---------------------------------------------------------------- 3. statics
section("3. Statics -- what Ke enforces, and what the thesis reports")

Kb = ds.K_MATRIX[1:3, 1:3]                       # bending block of one element
x_N, N = ds.static_contact_state()
q2 = N - ds.P5
q3 = 1.5 * q2

print(f"  bending block of Ke: {Kb.tolist()}")
print(f"  our static state: N = {N:.7f}, x_N = {x_N:.7f},"
      f" q2 = {q2:.7e}, q3 = {q3:.7e}, q5 = {-q2:.7e}")

# neither f_ext nor f_nl has a moment component, so the slope row must vanish
check("slope row of Ke closes: -6EI/l^2 q2 + 4EI/l q3 == 0",
      abs((Kb @ [q2, q3])[1]) < 1e-14, f"residual moment {(Kb @ [q2, q3])[1]:.2e}")
# and the transverse row must reproduce the equilibrium q2 = N - P5
check("transverse row closes: 4 q2 - 2 q3 - N == -P5",
      abs((Kb @ [q2, q3])[0] - N + ds.P5) < 1e-12)
check("x_N agrees with pyFBS's AFT penetration 0.0101848",
      abs(x_N - 0.0101848) < 1e-6, f"x_N = {x_N:.7f}")

print("\n  thesis-reported static values, tested against the SAME Ke:")
t_q2, t_q3, t_q5, t_q6 = (-6.020127e-3, -6.653825e-3, 4.119034e-3, -2.851639e-3)
print(f"    q3/q2 = {t_q3 / t_q2:.4f} and q6/q5 = {t_q6 / t_q5:.4f}   (Ke forces 1.5)")
print(f"    q5 = {t_q5:.6e} vs -q2 = {-t_q2:.6e}   (Ke forces q5 = -q2)")
print(f"    residual moment left by Ke: element 1 {(Kb @ [t_q2, t_q3])[1]:+.4e},"
      f" element 2 {(Kb @ [t_q5, t_q6])[1]:+.4e}   (must be 0)")
print("    -> the four individual values are NOT reproducible from the thesis'"
      " own Ke.")

# ... but their DIFFERENCE is sound: it reproduces the reported contact state
q_perp = ds.EPS + t_q2 - t_q5
N_thesis, slope_thesis = ds.normal_force(np.array([-q_perp + ds.EPS]))
print("\n  their difference q2 - q5, however, is self-consistent:")
check("q_perp from their q2,q5 == their penetration -1.3916133e-4",
      abs(q_perp + 1.3916133e-4) < 1e-9, f"q_perp = {q_perp:.7e}")
check("eq. (6.15) on it == their N = 0.3825733",
      abs(N_thesis[0] - 0.3825733) < 1e-6, f"N = {N_thesis[0]:.7f}")
check("d f_nl_2/d q2 == their 267.367",
      abs(slope_thesis[0] - 267.367) < 1e-2, f"{slope_thesis[0]:.3f}")


# --------------------------------------------------------------- 4. branches
section("4. Solver formulations against each other")


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


# ------------------------------------------------------------------ 5. pyFBS
section("5. pyFBS CSVs dropped into results/ (optional)")

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
