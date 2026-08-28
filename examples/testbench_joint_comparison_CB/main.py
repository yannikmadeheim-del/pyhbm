"""
One configurable Craig-Bampton solve of the lab testbench with a composable joint.

Edit CONFIG at the top -- it holds the joint (one entry per element, each with
the interface DoFs it acts on), the excitation, the interface condensation and
reduction size, the continuation window and the solver parts -- then run this
file. It imports the FE data, reduces both substructures, traces the forced-
response branch by AFT + arc-length HBM continuation, exports the branch in
PHYSICAL coordinates to results/<name>.csv and plots it.

The CSV is the shared export of the testbench_*_CB examples plus one extra
header line, ``# config_json: {...}``, recording the exact CONFIG that produced
it. That makes a run reproducible and lets the pyFBS testbench_joint_comparison
plot_comparison.py label it automatically: the same joint description drives
both pipelines, so an FBS and a Craig-Bampton branch of one joint overlay in a
single figure.

``load_substructures``, ``build_reduced``, ``solve_config`` and ``save_solution``
are importable, so study.py runs parameter permutations without duplicating any
of this.
"""

import json
import sys
import time
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

# imported first: loading the sibling example's infrastructure also resolves the
# repo's src/ onto sys.path, which is what makes the pyhbm imports below work
# from an uninstalled checkout.
from dynamical_system import (CB_DIR, CoupledJointCB, ReducedSubstructure,
                              assemble_coupled, channel_header_lines,
                              channel_snap_info, check_vpt_rows,
                              condensation_header_text,
                              get_boundary_nodes, load_or_export, make_joints,
                              nearest_node, output_channel_label,
                              physical_recovery, read_channels, read_descriptor,
                              read_vpt_rows, save_physical_solution)
from measurements import DEFAULT as DEFAULT_WORKBOOK, resolve as resolve_workbook

from pyhbm import (BiExponentialAdaptation, ExponentialAdaptation,
                   FourierOmegaPoint, HarmonicBalanceMethod,
                   OrthogonalParameterization, TangentPredictorOne,
                   TangentPredictorRobust, TangentPredictorTwo)
# not re-exported by pyhbm/__init__.py -- imported from their defining modules
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered

HERE = Path(__file__).resolve().parent

# FE data and npz caches are the sibling cubic-spring example's copy of the
# pyFBS lab_testbench data (see dynamical_system) -- 292 MB, shared.
FEM_DIR = CB_DIR / "lab_testbench" / "FEM"
# The workbooks and their descriptors are LOCAL to this example: the interface
# rows are collocated, so RBE_rigid, RBE_average and the pyFBS VPT all see the
# SAME interface DoF set. The two sheets list that set in a different row order,
# so Tf equals Tu.T only up to that permutation, which both pipelines apply
# consistently. Each workbook has its own descriptor, written by pyFBS's
# export_substructure_descriptor.py. The cubic-spring and dry-friction examples
# keep the as-measured sheets and are unaffected.


def descriptor_path(workbook):
    return HERE / f"substructure_descriptor_{workbook}.json"

# ---------------------------------------------------------------------------
# CONFIG -- system, then reduction, then solver. Plain JSON types only: the
# whole dict is written into the result CSV header and read back by the pyFBS
# comparison plotter. Keys shared with the pyFBS testbench_joint_comparison
# example keep the SAME names, so a mixed legend compares them directly.
# ---------------------------------------------------------------------------
CONFIG = dict(
    # --- system: identical schema to the pyFBS example ---------------------
    # one entry per joint element; "dofs" says which interface DoFs it acts on,
    # named out of ("ux", "uy", "uz", "rx", "ry", "rz"). Elements are summed, so
    # several may share a DoF (e.g. a linear and a cubic spring on uz).
    joints = [dict(type="linear", k=1.0e6, c=0.5, dofs=("ux", "uy", "uz")),
              dict(type="linear", k=1.0e3, c=0.1, dofs=("rx", "ry", "rz")),
              dict(type="cubic",  alpha=1.0e8,    dofs=("uz",))],
    F0 = 200.0,                    # harmonic excitation amplitude [N]
    modal_damping = 0.005,         # modal damping ratio of both substructures;
                                   # the pyFBS name for what the other CB
                                   # examples call ZETA -- sharing the name is
                                   # what makes the parameter comparable across
                                   # the two pipelines
    solver = "pyhbm-cb",           # names the pipeline in mixed comparisons
    workbook = "collocated_impact_locations",   # table in measurements/

    # --- reduction (pyhbm-only) -------------------------------------------
    condensation = "RBE_average",  # whole-node boundary: "rbe2" (rigid MPC,
                                   # interface condensed to the 6-DoF VP) |
                                   # "rbe3" (interpolation MPC, flexible).
                                   # directional (VPT) boundary: "RBE_rigid" |
                                   # "RBE_average" -- only the DoFs the workbook
                                   # names, as pyFBS's VPT sees them
    n_modes = 20,                  # fixed-interface modes per substructure
    interface_method = "descriptor",   # see get_boundary_nodes for alternatives
    rbe3_weights = None,           # None -> uniform; see rbe3_vp_operator

    # --- solver -----------------------------------------------------------
    harmonics = [1, 3, 5, 7],
    polynomial_degree = 3,         # AFT sampling: N_t = (deg+1)*max(h)+1; exact
                                   # for linear/cubic, raise it for friction
    f_lo = 200.0, f_hi = 500.0, sweep = "down",       # "down" | "up"
    parameterization = "ArcLengthParameterization",   # | OrthogonalParameterization
    predictor = "TangentPredictorBordered",           # | TangentPredictorOne
                                                      # | TangentPredictorTwo
                                                      # | TangentPredictorRobust
    step_adaptation = "ExponentialAdaptation",        # | BiExponentialAdaptation
    solver_kwargs = {"maximum_iterations": 300},   # tolerance filled in below
    step_kwargs = {"base": 1.5, "initial_step_length": 0.1 * 2 * np.pi,
                   "maximum_step_length": 1.0 * 2 * np.pi,
                   "minimum_step_length": 1e-4,
                   "goal_number_of_iterations": 3},
    omega_scale = 1.0,             # rescales the step length and the
                                   # continuation metric (no pyFBS equivalent)
    maximum_number_of_solutions = 10000, jacobian_update_frequency = 1,
)

# The corrector residual is a force, so a fixed absolute tolerance would mean a
# different convergence quality at every excitation level. It is set here rather
# than inside CONFIG because F0 is a keyword there, not yet a name. study.py
# does the same per run, see its TOLERANCE_PER_F0.
CONFIG["solver_kwargs"]["absolute_tolerance"] = 1e-6 * CONFIG["F0"]

# --- output ---------------------------------------------------------------
# Kept for schema symmetry with the pyFBS example. The shared
# save_physical_solution always writes the complete physical recovery and is
# used unchanged rather than forked, so this flag does not currently shrink the
# export.
SAVE_FULL_RESPONSE = True
RESULT_NAME = "linear+cubic_RBE_average_impactloc.csv"   # None -> auto-generated
                            # from the config; the auto name carries the joint and
                            # the condensation but NOT the workbook

# solver parts addressable by name from CONFIG (and hence from a CSV header)
SOLVER_PARTS = {cls.__name__: cls for cls in (
    ArcLengthParameterization, OrthogonalParameterization,
    TangentPredictorBordered, TangentPredictorOne, TangentPredictorTwo,
    TangentPredictorRobust, ExponentialAdaptation, BiExponentialAdaptation)}


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------

def load_substructures(workbook=DEFAULT_WORKBOOK):
    """
    Everything that depends on neither the joint nor the reduction settings: the
    Ansys import (npz-cached) of A and B, the pyFBS descriptor that defines the
    virtual point, the excitation and the output channel, the workbook channel
    tables and the drive node. A whole study needs this exactly once.

    :returns: context dict consumed by :func:`build_reduced` and
        :func:`save_solution`.
    """
    print(f"FEM data: {FEM_DIR}")
    xlsx_path = resolve_workbook(workbook)
    descriptor = read_descriptor(descriptor_path(workbook))
    substructures = {name: load_or_export(name, FEM_DIR, CB_DIR)
                     for name in ("A", "B")}

    inp_pos = np.array(descriptor["excitation"]["position"])
    inp_dir = np.array(descriptor["excitation"]["direction"])
    # retain the excitation node as a CB attachment DoF (static completeness of
    # the applied load). The measurement node stays interior -- it carries no
    # load and is recovered exactly from [Psi | Phi] after the solve.
    drive_node_B = int(np.argmin(
        np.linalg.norm(substructures["B"]["nodes"] - inp_pos, axis=1)))

    channels = {name: read_channels(xlsx_path, name) for name in ("A", "B")}
    chan_info = {name: channel_snap_info(substructures[name], channels[name])
                 for name in ("A", "B")}
    out_label = output_channel_label(channels["A"],
                                     np.array(descriptor["output"]["position"]),
                                     np.array(descriptor["output"]["direction"]))
    print(f"channels: A {len(channels['A'])}, B {len(channels['B'])} | "
          f"plotted output channel: {out_label}")

    # the directional condensations need the workbook rows themselves, plus the
    # single I/O DoF each side retains (drive point on B, measured channel on A)
    out_pos = np.array(descriptor["output"]["position"])
    out_dir = np.array(descriptor["output"]["direction"])
    out_node_A = int(np.argmin(
        np.linalg.norm(substructures["A"]["nodes"] - out_pos, axis=1)))
    vpt_rows = {name: read_vpt_rows(xlsx_path, name, substructures[name]["nodes"])
                for name in ("A", "B")}
    for name in ("A", "B"):
        check_vpt_rows(vpt_rows[name], descriptor, name,
                       substructures[name]["nnum"])

    return dict(descriptor=descriptor, substructures=substructures,
                xlsx_path=xlsx_path,
                vp_xyz=np.array(descriptor["vp"]["position"]),
                inp_pos=inp_pos, inp_dir=inp_dir, drive_node_B=drive_node_B,
                channels=channels, chan_info=chan_info, out_label=out_label,
                vpt_rows=vpt_rows,
                io_rows={"A": [("out", out_node_A, out_dir)],
                         "B": [("in", drive_node_B, inp_dir)]})


def build_reduced(cfg, ctx):
    """
    Craig-Bampton reduction of both substructures and the uncoupled assembly --
    the expensive, JOINT-INDEPENDENT step, so study.py builds it once per group
    of runs sharing the reduction settings.

    The interface node sets are resolved here rather than in
    :func:`load_substructures` because ``interface_method`` is a config axis
    like the other reduction keys; the lookup itself is cheap.

    :returns: (M, C, K, Bc, Bc_load, f_r, sub_A, sub_B) -- the block-diagonal
        reduced matrices, the 6-DoF gap operator x = Bc q = VP_A - VP_B, the map
        spreading the joint wrench back onto q (Bc.T except for RBE_average), the
        reduced load vector of the unit drive force and both substructures.
    """
    idx_A, idx_B = get_boundary_nodes(
        cfg["interface_method"], ctx["substructures"]["A"],
        ctx["substructures"]["B"], xlsx_path=ctx["xlsx_path"],
        descriptor=ctx["descriptor"])

    if cfg["condensation"] in ("RBE_rigid", "RBE_average"):
        extra = {name: dict(vpt_rows=ctx["vpt_rows"][name],
                            io_rows=ctx["io_rows"][name]) for name in ("A", "B")}
    else:
        extra = {"A": {}, "B": dict(attachment_idx=[ctx["drive_node_B"]])}

    sub_A = ReducedSubstructure.build(
        "A", ctx["substructures"]["A"], idx_A, ctx["vp_xyz"], cfg["n_modes"],
        cfg["modal_damping"], condensation=cfg["condensation"],
        weights=cfg["rbe3_weights"], **extra["A"])
    sub_B = ReducedSubstructure.build(
        "B", ctx["substructures"]["B"], idx_B, ctx["vp_xyz"], cfg["n_modes"],
        cfg["modal_damping"], condensation=cfg["condensation"],
        weights=cfg["rbe3_weights"], **extra["B"])

    M, C, K, Bc, Bc_load = assemble_coupled(sub_A, sub_B)
    f_r = np.concatenate([np.zeros(sub_A.M_r.shape[0]),
                          sub_B.recovery_row(ctx["inp_pos"], ctx["inp_dir"])])
    return M, C, K, Bc, Bc_load, f_r, sub_A, sub_B


def solve_config(cfg, reduced):
    """
    Trace the forced-response branch of ``cfg`` on a pre-built reduction.

    The wiring order is load-bearing: ``HarmonicBalanceMethod.__init__`` itself
    calls ``update_dependencies(harmonics, ode.polynomial_degree)``, which
    re-sizes the class-level ``Fourier`` tables. The system must therefore carry
    its ``polynomial_degree`` before the solver is constructed, and every run
    with different harmonics or a different degree needs a fresh solver.

    :returns: (system, solution_set, solve_time_s)
    """
    M, C, K, Bc, Bc_load, f_r, _, _ = reduced
    system = CoupledJointCB(M, C, K, Bc, f_r, cfg["F0"],
                            make_joints(cfg["joints"]),
                            cfg["polynomial_degree"], Bc_load=Bc_load)
    solver = HarmonicBalanceMethod(
        harmonics=list(cfg["harmonics"]), second_order_ode=system,
        corrector_parameterization=SOLVER_PARTS[cfg["parameterization"]],
        predictor=SOLVER_PARTS[cfg["predictor"]],
        step_length_adaptation=SOLVER_PARTS[cfg["step_adaptation"]])

    w_lo, w_hi = 2.0 * np.pi * cfg["f_lo"], 2.0 * np.pi * cfg["f_hi"]
    # cold zero start at the end of the window the sweep starts from; the
    # reference direction orients the first tangent. The dimension is the
    # REDUCED coupled one, not the 6 interface DoFs of the FBS formulation.
    if cfg["sweep"] == "down":
        w_start, direction = w_hi, -1.0
    elif cfg["sweep"] == "up":
        w_start, direction = w_lo, +1.0
    else:
        raise ValueError(f"sweep must be 'down' or 'up', got {cfg['sweep']!r}")
    ig = FourierOmegaPoint.zero_amplitude(dimension=system.dimension,
                                          omega=w_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((system.dimension, 1), dtype=complex), omega=direction)

    t0 = time.perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = cfg["maximum_number_of_solutions"],
        angular_frequency_range       = [w_lo, w_hi],
        solver_kwargs                 = dict(cfg["solver_kwargs"]),
        step_length_adaptation_kwargs = dict(cfg["step_kwargs"]),
        omega_scale                   = cfg["omega_scale"],
        jacobian_update_frequency     = cfg["jacobian_update_frequency"],
        verbose                       = True,
    )
    return system, ss, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Physical-solution CSV export
# ---------------------------------------------------------------------------

def export_header(cfg, ctx, sub_A, sub_B, solve_time, n_points, iface):
    """
    Comment header that makes the CSV self-contained: run metadata, the joint
    element list, the exact time-reconstruction formula and units, the meaning
    of every column family, and the id/position/direction of every exported
    physical DoF. The last entry is the machine-readable ``config_json`` line
    the pyFBS reader turns into legend labels.

    :param iface: {"A": (ansys_ids, xyz (nb,3)), "B": ...} interface node info
    """
    vp_txt, iface_txt = condensation_header_text(cfg["condensation"])
    in_id, in_xyz = nearest_node(ctx["substructures"]["B"], ctx["inp_pos"])
    lines = [
        f"testbench_joint_comparison_CB physical forced-response -- "
        f"{cfg['condensation'].upper()} interface + Craig-Bampton "
        f"({cfg['n_modes']} fixed-interface modes/substructure, "
        f"zeta = {cfg['modal_damping']})",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(cfg['harmonics'])}",
        f"AFT sampling: polynomial_degree = {cfg['polynomial_degree']}"
        f" -> N_t = {(cfg['polynomial_degree'] + 1) * max(cfg['harmonics']) + 1}",
        f"reduced dimension: A {sub_A.M_r.shape[0]} + B {sub_B.M_r.shape[0]} DoFs",
        f"joint: {len(cfg['joints'])} element(s) on the 6-DoF VP gap "
        f"x = A_vp - B_vp, forces summed",
    ]
    for spec in cfg["joints"]:
        spec = dict(spec)
        kind = spec.pop("type")
        args = ", ".join(f"{k}={v}" for k, v in spec.items())
        lines.append(f"  {kind}: {args}")
    lines += [
        f"excitation: f(t) = F0 cos(omega t) at 'uin', F0 = {cfg['F0']} N",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the PHYSICAL solution -- the reduced (Craig-Bampton) solver"
        " coordinates are already mapped back:",
        "  uin and the A_/B_ sensor channels: displacement recovered through"
        " the CB basis [Psi | Phi] (inverse Craig-Bampton) at the snapped FE"
        " nodes listed below",
        f"  A_vp_* / B_vp_*: 6-DoF virtual-point motion, {vp_txt}; the joint"
        " elements listed above act on the gap x = A_vp - B_vp and on its"
        " velocity",
        f"  A_n<id>_u* / B_n<id>_u*: interface node displacements, {iface_txt}",
        "  (modal coordinates eta are not exported; responses at other internal"
        " nodes need a rerun)",
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) );  channel"
        " acceleration: acc_h = -(h omega)^2 a_h",
        "units: displacement amplitudes m, vp rotations rad, freq_hz Hz,"
        " omega_rad_s rad/s",
        f"columns: freq_hz, omega_rad_s | iterations, step_length (corrector"
        f" diagnostics) | uout_h1_abs_m = |a_1({ctx['out_label']})|,"
        f" uout_time_max_m = max_t |{ctx['out_label']}(t)| (the two plotted"
        f" curves) | re/im of a_h, harmonic-major, then DoF order as listed",
        f"uout (plotted output channel) = {ctx['out_label']}",
        f"uin:  substructure B node B_n{in_id} at ({in_xyz[0]:.6f},"
        f" {in_xyz[1]:.6f}, {in_xyz[2]:.6f}) m, direction"
        f" {np.round(ctx['inp_dir'], 7).tolist()}",
        f"virtual point at {np.round(ctx['vp_xyz'], 7).tolist()} m",
    ]
    for name, chan_info in ctx["chan_info"].items():
        lines += channel_header_lines(name, chan_info)
    for name, (ids, xyz) in iface.items():
        lines.append(f"interface {name}: {len(ids)} nodes")
        lines += [f"  {name}_n{int(i)}: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m"
                  for i, p in zip(ids, xyz)]
    lines.append(f"config_json: {json.dumps(cfg, sort_keys=True)}")
    return lines


def save_solution(path, cfg, ctx, reduced, solution_set, solve_time):
    """
    Write the converged branch in PHYSICAL coordinates through the shared
    exporter, with the joint-aware header of :func:`export_header`.

    :returns: (freq_hz, uout_h1_abs, uout_time_max) for the plot.
    """
    _, _, _, _, _, f_r, sub_A, sub_B = reduced
    # the exported interface node set, in interface_recovery()'s row order --
    # NOT boundary_idx[:n_interface], which counts DoFs in the directional modes
    iface_idx = {sub.name: sub.interface_nodes() for sub in (sub_A, sub_B)}
    ids = {name: ctx["substructures"][name]["nnum"][idx]
           for name, idx in iface_idx.items()}

    labels, T = physical_recovery(sub_A, sub_B, f_r, ids["A"], ids["B"],
                                  ctx["channels"]["A"], ctx["channels"]["B"])
    header = export_header(
        cfg, ctx, sub_A, sub_B, solve_time, len(solution_set),
        iface={name: (ids[name], ctx["substructures"][name]["nodes"][idx])
               for name, idx in iface_idx.items()})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_physical_solution(solution_set, solve_time, path, labels, T,
                                  header, ctx["out_label"])


def result_path(cfg, name=None):
    """results/<name>.csv; the auto name is the joint types plus the interface
    condensation (e.g. linear+cubic_rbe3.csv)."""
    if name is None:
        types = list(dict.fromkeys(spec["type"] for spec in cfg["joints"]))
        name = f"{'+'.join(types)}_{cfg['condensation']}.csv"
    return HERE / "results" / name


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(f"Continuation window: {CONFIG['f_lo']:.1f}..{CONFIG['f_hi']:.1f} Hz"
          f" ({CONFIG['sweep']}ward sweep)")
    context = load_substructures(CONFIG["workbook"])

    print(f"\n=== condensation: {CONFIG['condensation'].upper()} ===")
    reduction = build_reduced(CONFIG, context)
    system, solution_set, solve_time = solve_config(CONFIG, reduction)

    out_csv = result_path(CONFIG, RESULT_NAME)
    freq, abs_h1, max_time = save_solution(out_csv, CONFIG, context, reduction,
                                           solution_set, solve_time)

    out_label = context["out_label"]
    fig, (ax_max, ax_h1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax_max.semilogy(freq, max_time, "-", color="#1f77b4", lw=1.2,
                    label=f"max|{out_label}(t)|")
    ax_max.set_ylabel(f"max|{out_label}(t)|  [m]")
    ax_max.set_title(f"{out_csv.stem}: {CONFIG['condensation'].upper()} + "
                     f"Craig-Bampton forced response")
    ax_max.grid(True, which="both", alpha=0.3)
    ax_max.legend()

    ax_h1.semilogy(freq, abs_h1, "-", color="#1f77b4", lw=1.2,
                   label="|1st harmonic amplitude|")
    ax_h1.set_xlim(CONFIG["f_lo"], CONFIG["f_hi"])
    ax_h1.set_xlabel("Frequency [Hz]")
    ax_h1.set_ylabel(f"|1st harmonic {out_label}|  [m]")
    ax_h1.grid(True, which="both", alpha=0.3)
    ax_h1.legend()
    fig.tight_layout()
    plt.show()
