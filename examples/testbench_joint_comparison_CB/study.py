"""
Parameter permutations of the composable joint and of the reduction, one CSV per run.

Edit the top block: BASE holds the defaults (a CONFIG exactly like main.py's),
JOINT_SETS lists the joint compositions to compare, and GLOBAL_SWEEP varies
non-joint config keys -- including the reduction axes ``condensation``,
``n_modes``, ``modal_damping``, ``interface_method`` and ``rbe3_weights``, which
is what this study adds over its pyFBS counterpart. Inside a joint set ANY
scalar field may be written as a LIST of values; the study then runs every
combination of those lists, crossed with every combination of GLOBAL_SWEEP.

Values are varied INSIDE the joint spec rather than in one global sweep table
because a parameter then only multiplies the runs whose joint actually has it: a
sweep over ``alpha`` in the cubic set produces no duplicate runs in the linear or
friction sets. ``dofs`` and ``spin_dof`` are excluded from sweeping by name --
they are legitimately tuples/strings; to compare different DoF assignments,
write a second joint set.

The Ansys import runs once for the whole study. Runs are then grouped by the
settings that determine the (expensive) Craig-Bampton reduction, so the
reduction and the coupled assembly happen once per group and only the solve is
repeated. Results go to results/<STUDY_NAME>/, together with a manifest.csv;
existing CSVs are skipped so an interrupted study resumes, and a failing run is
logged and does not abort the rest.

Run it with no arguments, or::

    python study.py --only 000,003 --results results
"""

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import main as run

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
STUDY_NAME = "cubic_vs_amplitude_condensation"

BASE = dict(
    joints = [],                   # filled per run from JOINT_SETS
    F0 = 200.0, modal_damping = 0.005,
    solver = "pyhbm-cb",
    condensation = "RBE_average",
    n_modes = 60,                  # fixed-interface modes per substructure; 20
                                   # tops the basis out at ~7 kHz, below the
                                   # 5th harmonic of the 2 kHz sweep end
    interface_method = "descriptor", rbe3_weights = None,
    harmonics = [1, 3, 5], polynomial_degree = 3,
    f_lo = 1.0, f_hi = 2000.0, sweep = "down",
    parameterization = "ArcLengthParameterization",
    predictor = "TangentPredictorBordered",
    step_adaptation = "ExponentialAdaptation",
    # absolute_tolerance is filled in per run by resolve_tolerance, scaled with
    # that run's F0; write one here to pin it instead
    solver_kwargs = {"maximum_iterations": 300},
    # step lengths live in the arc-length metric (omega_scale = 1 -> rad/s).
    # This is the cubic-spring reference's tuning: fine enough to round the
    # hardening folds instead of stepping over them. Raising the maximum is not
    # a safe way to shorten a run -- at a fold the corrector converges in one or
    # two iterations, so the adaptation never shrinks the step on its own.
    step_kwargs = {"base": 2, "initial_step_length": 0.1,
                   "maximum_step_length": 1.0,
                   "minimum_step_length": 1e-6,
                   "goal_number_of_iterations": 2},
    omega_scale = 1.0,
    # 1..2000 Hz at the step length above needs >12500 points; a smaller cap
    # would end a run mid-sweep, and the CSV would look complete
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
)

ALL6 = ("ux", "uy", "uz", "rx", "ry", "rz")
JOINT_SETS = [
    # identical linear backbone in every run; alpha = 0 is the linear reference,
    # so all curves share one element list and stay comparable in the legend
    [dict(type="linear", k=1.0e6, c=0.5, dofs=ALL6),
     dict(type="cubic",  alpha=[0.0, 1.0e8, 1.0e9], dofs=ALL6)],
]

# Amplitude crossed with the interface condensation: 3 alphas x 3 F0 x 2
# condensations = 18 runs, but only ``condensation`` is a reduction key, so the
# nine runs sharing one condensation also share its Craig-Bampton reduction.
# The directional condensations take their boundary from the workbook VPT rows
# and therefore IGNORE ``interface_method`` and ``rbe3_weights``: sweeping
# either of those alongside them only produces duplicate runs.
GLOBAL_SWEEP = dict(F0=[50.0, 200.0, 800.0],
                    condensation=["RBE_average", "RBE_rigid"])
# ---------------------------------------------------------------------------

SWEEP_EXEMPT = ("dofs", "spin_dof")   # tuples/strings, not sweepable value lists
# everything the Craig-Bampton build depends on: runs sharing these values reuse
# one reduction and one coupled assembly
REDUCTION_KEYS = ("condensation", "n_modes", "modal_damping",
                  "interface_method", "rbe3_weights")
MANIFEST_COLUMNS = ("run_id", "joint_types", "varied", "status", "n_points",
                    "runtime_s", "csv", "error")
# corrector tolerance per newton of excitation: the residual is a force, so one
# fixed absolute value would mean a different convergence quality at every
# amplitude -- and F0 is a sweepable key like any other.
TOLERANCE_PER_F0 = 1e-6


def joint_prefixes(specs):
    """``linear``, ``linear2``, ``cubic``, ... -- one name per spec, so a repeated
    joint type stays distinguishable in the manifest (same scheme as the
    automatic plot labels)."""
    seen, out = {}, []
    for spec in specs:
        kind = spec.get("type", "?")
        n = seen.get(kind, 0)
        seen[kind] = n + 1
        out.append(kind if n == 0 else f"{kind}{n + 1}")
    return out


def joint_slug(specs):
    return "+".join(dict.fromkeys(spec["type"] for spec in specs))


def resolve_tolerance(cfg):
    """``solver_kwargs`` of one run, with its F0-scaled corrector tolerance.

    An ``absolute_tolerance`` written into BASE wins, so a study can still pin
    one value across an F0 sweep.
    """
    kwargs = dict(cfg["solver_kwargs"])
    kwargs.setdefault("absolute_tolerance", TOLERANCE_PER_F0 * cfg["F0"])
    return kwargs


def build_runs(base, joint_sets, global_sweep):
    """Expand the study into ``(run_id, cfg, varied)`` triples, in run order."""
    global_keys = list(global_sweep)
    runs = []
    for specs in joint_sets:
        prefixes = joint_prefixes(specs)
        axes = [(i, key, list(value)) for i, spec in enumerate(specs)
                for key, value in spec.items()
                if key not in SWEEP_EXEMPT and isinstance(value, list)]
        for joint_combo in itertools.product(*[values for _, _, values in axes]):
            resolved = [dict(spec) for spec in specs]
            varied_joint = {}
            for (i, key, _), value in zip(axes, joint_combo):
                resolved[i][key] = value
                varied_joint[f"{prefixes[i]}.{key}"] = value
            for global_combo in itertools.product(*global_sweep.values()):
                cfg = dict(base)
                cfg["joints"] = [dict(spec) for spec in resolved]
                cfg.update(zip(global_keys, global_combo))
                cfg["solver_kwargs"] = resolve_tolerance(cfg)
                varied = dict(varied_joint, **dict(zip(global_keys, global_combo)))
                runs.append((f"{len(runs):03d}_{joint_slug(resolved)}", cfg, varied))
    return runs


def group_key(cfg):
    """The reduction settings, hashable -- ``rbe3_weights`` may be a list."""
    return tuple(json.dumps(cfg[key]) for key in REDUCTION_KEYS)


def manifest_varied(cfg, varied):
    """Manifest description of a run: the values this study sweeps, plus the
    reduction settings it holds fixed, so every row identifies its run without
    the study source."""
    fixed = [f"{key}={cfg[key]}" for key in REDUCTION_KEYS if key not in varied]
    return ", ".join([f"{k}={v}" for k, v in varied.items()] + fixed)


def append_manifest(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated run ids (default: all)")
    parser.add_argument("--results", default="results",
                        help="results folder name in this example folder")
    args = parser.parse_args()

    out_dir = HERE / args.results / STUDY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    runs = build_runs(BASE, JOINT_SETS, GLOBAL_SWEEP)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs
                if r[0] in keep or r[0].split("_", 1)[0] in keep]
    print(f"study '{STUDY_NAME}': {len(runs)} run(s) -> {out_dir}")

    groups = {}
    for entry in runs:
        groups.setdefault(group_key(entry[1]), []).append(entry)

    context = run.load_substructures()

    done = 0
    for group in groups.values():
        pending = [e for e in group if not (out_dir / f"{e[0]}.csv").exists()]
        for run_id, _, _ in group:
            if (out_dir / f"{run_id}.csv").exists():
                done += 1
                print(f"[{done}/{len(runs)}] {run_id}: skipped (result exists)")
        if not pending:
            continue

        cfg0 = pending[0][1]
        print(f"\nreducing for "
              + ", ".join(f"{key}={cfg0[key]}" for key in REDUCTION_KEYS)
              + f" ({len(pending)} run(s)) ...")
        t0 = time.perf_counter()
        reduction = run.build_reduced(cfg0, context)
        print(f"reduction done in {time.perf_counter() - t0:.0f} s")

        for run_id, cfg, varied in pending:
            done += 1
            shown = ", ".join(f"{k}={v}" for k, v in varied.items())
            print(f"\n[{done}/{len(runs)}] {run_id}: {shown}")
            csv_path = out_dir / f"{run_id}.csv"
            t0 = time.perf_counter()
            try:
                system, ss, solve_time = run.solve_config(cfg, reduction)
                run.save_solution(csv_path, cfg, context, reduction, ss,
                                  solve_time)
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, joint_slug(cfg["joints"]),
                                           manifest_varied(cfg, varied), "ok",
                                           len(ss.omega), f"{runtime:.1f}",
                                           csv_path.name, ""])
                print(f"[{run_id}] ok: {len(ss.omega)} points in {runtime:.0f} s")
            except Exception as exc:
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, joint_slug(cfg["joints"]),
                                           manifest_varied(cfg, varied),
                                           "failed", 0, f"{runtime:.1f}", "",
                                           f"{type(exc).__name__}: {exc}"])
                print(f"[{run_id}] FAILED after {runtime:.0f} s: {exc}")
                traceback.print_exc()

    print(f"\nstudy '{STUDY_NAME}' finished -- manifest: {manifest}")


if __name__ == "__main__":
    main()
