"""
Minimal reference for the pyFBS testbench_cubicSpring example:
RBE2 / RBE3 interface + Craig-Bampton reduction + pyhbm second-order HBM.

For each condensation method in CONDENSATION_METHODS the coupled reduced system
is assembled and swept with the HBM continuation. The solution branch is
exported to one CSV per method (reference_<method>_cb_hbm.csv) in PHYSICAL
coordinates: per continuation point omega, the corrector diagnostics and, for
every harmonic, the complex amplitudes of every pyFBS response channel and the
drive-point displacement (inverse Craig-Bampton through [Psi | Phi]), the
6-DoF virtual-point motion of each substructure and the physical interface
node displacements (inverse RBE2 u_Gamma = T_b q_m, resp. the retained RBE3
interface DoFs) -- see physical_recovery / save_physical_solution. The comment
header and the column names make the file self-contained: both plotted curves
and the physical solution at every frequency point can be recovered from the
CSV alone.
For a quick look the output DoF response is also plotted (not saved).
"""

import sys
import time
from pathlib import Path

from pyhbm import OrthogonalParameterization
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered

try:                          # pyhbm's progress line prints unicode (Δω);
    sys.stdout.reconfigure(   # Windows consoles default to cp1252
        encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

from dynamical_system import (CoupledCubicCB, ReducedSubstructure,
                              assemble_coupled, channel_header_lines,
                              channel_snap_info, check_vpt_rows,
                              condensation_header_text,
                              get_boundary_nodes, load_or_export, nearest_node,
                              output_channel_label, physical_recovery,
                              read_channels, read_descriptor, read_vpt_rows,
                              save_physical_solution)

# ---------------------------------------------------------------------------
# Paths -- lab_testbench is a local copy of the pyFBS example data (FEM, STL,
# Measurements).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FEM_DIR = HERE / "lab_testbench" / "FEM"
XLSX_PATH = HERE / "lab_testbench" / "Measurements" / "coupling_example.xlsx"

# ---------------------------------------------------------------------------
# Joint / interface / I/O definition -- exported from the pyFBS example, so
# pyFBS's VPT is the single source of truth.
# ---------------------------------------------------------------------------
DESCRIPTOR = read_descriptor(HERE / "substructure_descriptor.json")

VP_XYZ = np.array(DESCRIPTOR["vp"]["position"])  # (0.038895, 0.348107, 0.007)

K_DIAG     = np.array(DESCRIPTOR["joint"]["k_diag"])      # [N/m x3, Nm/rad x3]
C_DIAG     = np.array(DESCRIPTOR["joint"]["c_diag"])      # linear viscous damper [N s/m x3, Nm s/rad x3]
ALPHA_DIAG = np.array(DESCRIPTOR["joint"]["alpha_diag"])  # cubic stiffness
BETA_DIAG  = np.array(DESCRIPTOR["joint"]["beta_diag"])   # cubic damping

# excitation: F0*cos(w t) at impact H28 on B (first B reference impact)
F0      = DESCRIPTOR["excitation"]["F0"]                  # [N]
INP_POS = np.array(DESCRIPTOR["excitation"]["position"])
INP_DIR = np.array(DESCRIPTOR["excitation"]["direction"])

# output: displacement at channel S1 X on A (first A reference channel)
OUT_POS = np.array(DESCRIPTOR["output"]["position"])
OUT_DIR = np.array(DESCRIPTOR["output"]["direction"])

# ---------------------------------------------------------------------------
# Solver / reduction parameters
# ---------------------------------------------------------------------------
HARMONICS = [1, 3, 5, 7]                         # cubic forcing -> odd harmonics
F_LO, F_HI = 1, 500                        # continuation window [Hz]
ZETA = 0.005                                     # modal damping per substructure
N_MODES = 20                                     # fixed-interface modes per substructure

# interface condensation -- run any subset. The first two retain all 3 DoFs of
# every interface node; the last two retain only the DoFs the workbook names,
# which is the DoF set pyFBS's VPT actually works with:
#   "rbe2"        -- rigid MPC, interface condensed to the 6-DoF VP (stiffens the joint)
#   "rbe3"        -- interpolation MPC, interface flexible, VP by weighted average
#   "RBE_rigid"   -- rigid MPC on the VPT DoFs only
#   "RBE_average" -- VP observed/loaded through pyFBS's Tu / Tf
CONDENSATION_METHODS = ("RBE_rigid", "RBE_average")
RBE3_WEIGHTS = None                              # None -> uniform; see rbe3_vp_operator
VPT_WU = VPT_WF = None                           # RBE_average weighting; None -> identity

# condensations whose boundary is the directional (VPT) DoF set
DIRECTIONAL_METHODS = ("RBE_rigid", "RBE_average")

# interface (RBE2 slave) definition -- see get_boundary_nodes:
#   "descriptor": the exact nodes pyFBS's VPT uses, from the exported JSON (default)
INTERFACE_METHOD = "descriptor"
IFACE_A_TXT = FEM_DIR / "IFACE_A.txt"
IFACE_B_TXT = FEM_DIR / "IFACE_B.txt"
MATING_TOL = 1e-6                                # coincidence tolerance [m]


def run_hbm(system):
    """
    Multiharmonic balance + arc-length continuation of the coupled system,
    swept downward from F_HI to F_LO like the pyFBS example (cold zero start
    at the top of the window, reference direction pointing to lower omega).
    """
    from pyhbm import FourierOmegaPoint, HarmonicBalanceMethod

    solver = HarmonicBalanceMethod(harmonics=HARMONICS, second_order_ode=system, corrector_parameterization=ArcLengthParameterization, predictor=TangentPredictorBordered)
    w_lo, w_hi = 2.0 * np.pi * F_LO, 2.0 * np.pi * F_HI
    ig = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=300*2*np.pi)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((system.dimension, 1), dtype=complex), omega=-1.0)

    return solver.solve_and_continue(
        initial_guess=ig,
        initial_reference_direction=rd,
        maximum_number_of_solutions= 10000,
        angular_frequency_range=[w_lo, w_hi],
        solver_kwargs={"maximum_iterations": 300,
                       "absolute_tolerance": F0*1e-6},
        omega_scale= 1,
        step_length_adaptation_kwargs={"base": 2,
                                       "initial_step_length": 0.1,
                                       "maximum_step_length": 1.0,
                                       "minimum_step_length": 1e-4,
                                       "goal_number_of_iterations": 2},
        jacobian_update_frequency=1,
        verbose=True,
    )


def export_header(method, solve_time, n_points, iface, channels, out_label,
                  in_node):
    """
    Comment header that makes the CSV self-contained: run metadata, the exact
    time-reconstruction formula and units, the meaning of every column family,
    and the id/position of every exported physical DoF.

    :param iface: {"A": (ansys_ids, xyz (nb,3)), "B": ...} interface node info
    :param channels: {"A": channel_snap_info(...), "B": ...}
    :param out_label: channel column feeding the plotted uout_* summary columns
    :param in_node: (ansys_id, xyz) of the snapped drive node
    """
    vp_txt, iface_txt = condensation_header_text(method)
    in_id, in_xyz = in_node
    lines = [
        f"testbench_cubicSpring_CB physical forced-response reference -- "
        f"{method.upper()} interface + Craig-Bampton "
        f"({N_MODES} fixed-interface modes/substructure, zeta = {ZETA})",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(HARMONICS)}",
        f"excitation: f(t) = F0 cos(omega t) at 'uin', F0 = {F0} N",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the PHYSICAL solution -- the reduced (Craig-Bampton) solver"
        " coordinates are already mapped back:",
        "  uin and the A_/B_ sensor channels: displacement recovered through"
        " the CB basis [Psi | Phi] (inverse Craig-Bampton) at the snapped FE"
        " nodes listed below",
        f"  A_vp_* / B_vp_*: 6-DoF virtual-point motion, {vp_txt}",
        f"  A_n<id>_u* / B_n<id>_u*: interface node displacements, {iface_txt}",
        "  (modal coordinates eta are not exported; responses at other internal"
        " nodes need a rerun)",
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) );  channel"
        " acceleration: acc_h = -(h omega)^2 a_h",
        "joint: x = A_vp - B_vp (6 relative VP DoFs); f_joint = K_DIAG x"
        " + C_DIAG xdot + ALPHA_DIAG x^3 + BETA_DIAG xdot^3 (per-DoF diagonals)",
        f"joint stiffness k_diag [N/m x3, Nm/rad x3]: {K_DIAG.tolist()}",
        f"joint viscous damping c_diag [N s/m x3, Nm s/rad x3]: {C_DIAG.tolist()}",
        f"joint cubic alpha_diag: {ALPHA_DIAG.tolist()}",
        f"joint cubic beta_diag: {BETA_DIAG.tolist()}",
        "units: displacement amplitudes m, vp rotations rad, freq_hz Hz,"
        " omega_rad_s rad/s",
        f"columns: freq_hz, omega_rad_s | iterations, step_length (corrector"
        f" diagnostics) | uout_h1_abs_m = |a_1({out_label})|, uout_time_max_m ="
        f" max_t |{out_label}(t)| (the two plotted curves) | re/im of a_h,"
        f" harmonic-major, then DoF order as listed",
        f"uout (plotted output channel) = {out_label}",
        f"uin:  substructure B node B_n{in_id} at ({in_xyz[0]:.6f},"
        f" {in_xyz[1]:.6f}, {in_xyz[2]:.6f}) m, direction"
        f" {np.round(INP_DIR, 7).tolist()}",
        f"virtual point at {np.round(VP_XYZ, 7).tolist()} m",
    ]
    for name, chan_info in channels.items():
        lines += channel_header_lines(name, chan_info)
    for name, (ids, xyz) in iface.items():
        lines.append(f"interface {name}: {len(ids)} nodes")
        lines += [f"  {name}_n{int(i)}: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m"
                  for i, p in zip(ids, xyz)]
    return lines


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(f"FEM data: {FEM_DIR}")
    substructures = {name: load_or_export(name, FEM_DIR, HERE)
                     for name in ("A", "B")}
    idx_A, idx_B = get_boundary_nodes(
        INTERFACE_METHOD, substructures["A"], substructures["B"],
        xlsx_path=XLSX_PATH, tol=MATING_TOL,
        file_A=IFACE_A_TXT, file_B=IFACE_B_TXT, descriptor=DESCRIPTOR)

    # retain the excitation node as a CB attachment DoF (static completeness of
    # the applied load). The measurement node stays interior -- it carries no
    # load and is recovered exactly from [Psi | Phi] after the solve.
    drive_node_B = int(np.argmin(
        np.linalg.norm(substructures["B"]["nodes"] - INP_POS, axis=1)))

    channels = {name: read_channels(XLSX_PATH, name) for name in ("A", "B")}
    chan_info = {name: channel_snap_info(substructures[name], channels[name])
                 for name in ("A", "B")}
    out_label = output_channel_label(channels["A"], OUT_POS, OUT_DIR)
    print(f"channels: A {len(channels['A'])}, B {len(channels['B'])} | "
          f"plotted output channel: {out_label}")

    # the directional condensations need the workbook rows themselves, not just
    # the node ids, plus the one I/O DoF each side retains (drive point on B,
    # measured channel on A) -- see build_directional_boundary.
    vpt_rows = {name: read_vpt_rows(XLSX_PATH, name, substructures[name]["nodes"])
                for name in ("A", "B")}
    for name in ("A", "B"):
        check_vpt_rows(vpt_rows[name], DESCRIPTOR, name,
                       substructures[name]["nnum"])
    out_node_A = int(np.argmin(
        np.linalg.norm(substructures["A"]["nodes"] - OUT_POS, axis=1)))
    io_rows = {"A": [("out", out_node_A, OUT_DIR)],
               "B": [("in", drive_node_B, INP_DIR)]}

    colors = {"rbe2": "#d62728", "rbe3": "#1f77b4",
              "RBE_rigid": "#ff7f0e", "RBE_average": "#2ca02c"}
    fig, (ax_max, ax_h1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for method in CONDENSATION_METHODS:
        print(f"\n=== condensation: {method} ===")
        if method in DIRECTIONAL_METHODS:
            extra = {name: dict(vpt_rows=vpt_rows[name], io_rows=io_rows[name],
                                wu=VPT_WU, wf=VPT_WF) for name in ("A", "B")}
        else:
            extra = {"A": {}, "B": dict(attachment_idx=[drive_node_B])}
        sub_A = ReducedSubstructure.build("A", substructures["A"], idx_A, VP_XYZ,
                                          N_MODES, ZETA, condensation=method,
                                          weights=RBE3_WEIGHTS, **extra["A"])
        sub_B = ReducedSubstructure.build("B", substructures["B"], idx_B, VP_XYZ,
                                          N_MODES, ZETA, condensation=method,
                                          weights=RBE3_WEIGHTS, **extra["B"])
        M, C, K, Bc, Bc_load = assemble_coupled(sub_A, sub_B)
        f_r = np.concatenate([np.zeros(sub_A.M_r.shape[0]),
                              sub_B.recovery_row(INP_POS, INP_DIR)])
        system = CoupledCubicCB(M, C, K, Bc, K_DIAG, C_DIAG,
                                ALPHA_DIAG, BETA_DIAG, f_r, F0, Bc_load=Bc_load)

        t0 = time.perf_counter()
        solution_set = run_hbm(system)
        solve_time = time.perf_counter() - t0

        ids_A_ansys = substructures["A"]["nnum"][idx_A]
        ids_B_ansys = substructures["B"]["nnum"][idx_B]
        labels, T = physical_recovery(sub_A, sub_B, f_r, ids_A_ansys,
                                      ids_B_ansys, channels["A"], channels["B"])
        header = export_header(
            method, solve_time, len(solution_set),
            iface={"A": (ids_A_ansys, substructures["A"]["nodes"][idx_A]),
                   "B": (ids_B_ansys, substructures["B"]["nodes"][idx_B])},
            channels=chan_info, out_label=out_label,
            in_node=nearest_node(substructures["B"], INP_POS))
        freq, abs_h1, max_time = save_physical_solution(
            solution_set, solve_time, HERE / f"reference_{method}_cb_hbm.csv",
            labels, T, header, out_label)

        ax_max.semilogy(freq, max_time, "-", color=colors.get(method), lw=1.2,
                        label=f"{method.upper()} max|{out_label}(t)|")
        ax_h1.semilogy(freq, abs_h1, "-", color=colors.get(method), lw=1.2,
                       label=f"{method.upper()} |1st harmonic amplitude|")

    ax_max.set_ylabel(f"max|{out_label}(t)|  [m]")
    ax_max.set_title("Cubic-spring testbench: RBE2 vs RBE3 forced response")
    ax_max.grid(True, which="both", alpha=0.3)
    ax_max.legend()

    ax_h1.set_xlim(F_LO, F_HI)
    ax_h1.set_xlabel("Frequency [Hz]")
    ax_h1.set_ylabel(f"|1st harmonic {out_label}|  [m]")
    ax_h1.grid(True, which="both", alpha=0.3)
    ax_h1.legend()
    fig.tight_layout()
    plt.show()
