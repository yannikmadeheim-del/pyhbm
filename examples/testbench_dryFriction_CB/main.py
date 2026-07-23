"""
Craig-Bampton counterpart of the pyFBS testbench_dry_friction example:
RBE2 / RBE3 interface + Craig-Bampton reduction + pyhbm second-order HBM, with
the SAME dry-friction joint, FE model, excitation (impact H28, F0) and output
channel (A_S1X) as the pyFBS FBS solve -- so both branches solve one problem by
two routes and can be overlaid.

For each condensation method in CONDENSATION_METHODS the coupled reduced system
is assembled and swept with the HBM continuation. The branch is exported to one
CSV per method (reference_<method>_cb_dryfriction_hbm.csv) in PHYSICAL
coordinates, in exactly the convention of the pyFBS export: per continuation
point omega, the corrector diagnostics and, for every harmonic, the complex
amplitudes of every pyFBS response channel and the drive-point displacement
(inverse Craig-Bampton through [Psi | Phi]), the 6-DoF virtual-point motion of
each substructure and the physical interface node displacements. The CSVs drop
straight into the pyFBS example's plot_diagnostics_comparison.py next to
reference_modal_fbs_hbm.csv (it reads the gap as A_vp_* - B_vp_* and the output
channel as A_S1X). The forced response is also plotted here (not saved).

Runtime: with HARMONICS 1..19 and POLY_DEG = 52 the AFT sampling is N_t = 1008
like pyFBS, which makes the per-Newton-step Jacobian array (N_t, d, d) -- about
150 MB for RBE2 (d = 135) and 340 MB for RBE3 (d = 204), plus its FFT. A full
sweep is an overnight run; POLY_DEG, N_MODES and CONDENSATION_METHODS are the
knobs, and lowering them does not change the exported format.
"""

import sys
import time
from pathlib import Path

from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered

try:                          # pyhbm's progress line prints unicode (Δω);
    sys.stdout.reconfigure(   # Windows consoles default to cp1252
        encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

from dynamical_system import (CB_DIR, CoupledDryFrictionCB, ReducedSubstructure,
                              assemble_coupled, channel_header_lines,
                              channel_snap_info, get_boundary_nodes,
                              load_or_export, nearest_node,
                              output_channel_label, physical_recovery,
                              read_channels, read_descriptor,
                              save_physical_solution)

# ---------------------------------------------------------------------------
# Paths -- FE data, workbook, npz cache and descriptor are the sibling cubic-
# spring example's copy of the pyFBS lab_testbench data (see dynamical_system).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FEM_DIR = CB_DIR / "lab_testbench" / "FEM"
XLSX_PATH = CB_DIR / "lab_testbench" / "Measurements" / "coupling_example.xlsx"

# ---------------------------------------------------------------------------
# Interface / I/O definition -- exported from the pyFBS example, so pyFBS's VPT
# is the single source of truth. Its "joint" entry (cubic spring) is unused
# here; the dry-friction parameters below replace it.
# ---------------------------------------------------------------------------
DESCRIPTOR = read_descriptor(CB_DIR / "substructure_descriptor.json")

VP_XYZ = np.array(DESCRIPTOR["vp"]["position"])  # (0.038895, 0.348107, 0.007)

# excitation: F0*cos(w t) at impact H28 on B (first B reference impact = the
# DoF pyFBS excites); F0 from the pyFBS dry-friction main, not the descriptor.
INP_POS = np.array(DESCRIPTOR["excitation"]["position"])
INP_DIR = np.array(DESCRIPTOR["excitation"]["direction"])

# output: displacement at channel S1 X on A (first A reference channel)
OUT_POS = np.array(DESCRIPTOR["output"]["position"])
OUT_DIR = np.array(DESCRIPTOR["output"]["direction"])

# ---------------------------------------------------------------------------
# Joint parameters -- copied from the pyFBS testbench_dry_friction main.
#   MU_TRANS : friction coefficient [-]
#   N_CLAMP  : bolt clamping force [N]; both clamp faces carry MU_TRANS*N_CLAMP,
#              so the x-y friction force saturates at 2*MU_TRANS*N_CLAMP
#   K_TRANS  : penalty stiffness [N/m] tying the normal (z) interface gap
#   G        : effective contact radius [m] of the torsional friction; the spin
#              moment saturates at 2*MU_TRANS*N_CLAMP*G
#   K_ROT    : rotational penalty stiffness [Nm/rad] tying the tilt gap rx, ry
#   ALPHA    : tanh regularization sharpness [s/m]; near sticking the joint acts
#              like a viscous damper c_eff = 2*MU_TRANS*N_CLAMP*ALPHA
#   F0       : harmonic excitation amplitude [N]
# ---------------------------------------------------------------------------
MU_TRANS = 0.3
N_CLAMP = 200.0
K_TRANS = 1.0e8
G = 0.00
K_ROT = 0
ALPHA = 1.0e3
F0 = 1000.0

# ---------------------------------------------------------------------------
# Solver / reduction parameters
# ---------------------------------------------------------------------------
HARMONICS = list(range(1, 20, 2))   # friction force is odd in v -> odd harmonics
POLY_DEG = 10                       # AFT sampling: N_t = (POLY_DEG+1)*19+1 = 1008
F_LO, F_HI = 40.0, 500          # continuation window [Hz]
ZETA = 0.005                        # modal damping per substructure
N_MODES = 60                        # fixed-interface modes per substructure

# interface condensation -- run any subset (both by default):
#   "rbe2" -- rigid MPC, interface condensed to the 6-DoF VP (stiffens the joint)
#   "rbe3" -- interpolation MPC, interface kept flexible, VP by weighted average
CONDENSATION_METHODS = ("rbe2", "rbe3")
RBE3_WEIGHTS = None                 # None -> uniform; see rbe3_vp_operator

# interface (RBE2 slave) definition: "descriptor" = the exact nodes pyFBS's VPT
# uses, from the exported JSON (see get_boundary_nodes for the alternatives)
INTERFACE_METHOD = "descriptor"


def run_hbm(system):
    """
    Multiharmonic balance + arc-length continuation of the coupled system,
    swept downward from F_HI to F_LO like the pyFBS example (cold zero start at
    the top of the window, reference direction pointing to lower omega). The
    continuation settings are the pyFBS dry-friction ones -- in particular the
    Jacobian is rebuilt every iteration, because the friction Jacobian changes
    with the sliding state.
    """
    from pyhbm import FourierOmegaPoint, HarmonicBalanceMethod

    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, second_order_ode=system,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)
    w_lo, w_hi = 2.0 * np.pi * F_LO, 2.0 * np.pi * F_HI
    ig = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=w_hi)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((system.dimension, 1), dtype=complex), omega=-1.0)

    return solver.solve_and_continue(
        initial_guess=ig,
        initial_reference_direction=rd,
        maximum_number_of_solutions=5000,
        angular_frequency_range=[w_lo, w_hi],
        solver_kwargs={"maximum_iterations": 500,
                       "absolute_tolerance": 1e-6},
        omega_scale=1,
        step_length_adaptation_kwargs={"base": 3.0,
                                       "initial_step_length": 0.01 * 2 * np.pi,
                                       "maximum_step_length": 1.0 * 2 * np.pi,
                                       "minimum_step_length": 1e-6,
                                       "goal_number_of_iterations": 8},
        jacobian_update_frequency=1,
        verbose=True,
    )


def export_header(method, solve_time, n_points, iface, channels, out_label,
                  in_node):
    """
    Comment header that makes the CSV self-contained: run metadata, the exact
    time-reconstruction formula and units, the meaning of every column family,
    and the id/position of every exported physical DoF. The joint lines are
    worded like the pyFBS export so both references document the same law.

    :param iface: {"A": (ansys_ids, xyz (nb,3)), "B": ...} interface node info
    :param channels: {"A": channel_snap_info(...), "B": ...}
    :param out_label: channel column feeding the plotted uout_* summary columns
    :param in_node: (ansys_id, xyz) of the snapped drive node
    """
    if method == "rbe2":
        vp_txt = "the RBE2 master DoFs (= the condensed CB boundary block)"
        iface_txt = ("u_Gamma = T_b q_m -- rigid expansion of the VP master "
                     "(inverse RBE2)")
    else:
        vp_txt = "the RBE3 weighted interface average G^T q_Gamma"
        iface_txt = ("the retained (flexible) RBE3 interface DoFs "
                     "= CB boundary block, no expansion needed")
    in_id, in_xyz = in_node
    lines = [
        f"testbench_dryFriction_CB physical forced-response reference -- "
        f"{method.upper()} interface + Craig-Bampton "
        f"({N_MODES} fixed-interface modes/substructure, zeta = {ZETA})",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(HARMONICS)}",
        f"joint friction: mu_trans = {MU_TRANS}, clamping force N = {N_CLAMP} N,"
        f" slip force 2*mu_trans*N = {2.0 * MU_TRANS * N_CLAMP} N,"
        f" alpha = {ALPHA} s/m",
        f"joint torsional friction (rz): geometry factor G = {G} m,"
        f" slip moment 2*mu_trans*N*G = {2.0 * MU_TRANS * N_CLAMP * G} Nm",
        f"joint penalty stiffness: k_trans (z) = {K_TRANS} N/m,"
        f" k_rot (rx, ry) = {K_ROT} Nm/rad",
        f"excitation: f(t) = F0 cos(omega t) at 'uin', F0 = {F0} N",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the PHYSICAL solution -- the reduced (Craig-Bampton) solver"
        " coordinates are already mapped back:",
        "  uin and the A_/B_ sensor channels: displacement recovered through"
        " the CB basis [Psi | Phi] (inverse Craig-Bampton) at the snapped FE"
        " nodes listed below",
        f"  A_vp_* / B_vp_*: 6-DoF virtual-point motion, {vp_txt}; the joint"
        " gap is x = A_vp - B_vp and the joint force is"
        " f_T = 2*mu_trans*N*tanh(alpha*||v_T||)*v_T/||v_T|| on the [ux, uy]"
        " gap velocity, f_z = k_trans*x on uz, M_rxy = k_rot*x on rx/ry and"
        " M_rz = 2*mu_trans*N*G*tanh(alpha*G*xdot) on rz",
        f"  A_n<id>_u* / B_n<id>_u*: interface node displacements, {iface_txt}",
        "  (modal coordinates eta are not exported; responses at other internal"
        " nodes need a rerun)",
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) );  channel"
        " acceleration: acc_h = -(h omega)^2 a_h",
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
    substructures = {name: load_or_export(name, FEM_DIR, CB_DIR)
                     for name in ("A", "B")}
    idx_A, idx_B = get_boundary_nodes(
        INTERFACE_METHOD, substructures["A"], substructures["B"],
        xlsx_path=XLSX_PATH, descriptor=DESCRIPTOR)

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

    colors = {"rbe2": "#d62728", "rbe3": "#1f77b4"}
    fig, (ax_max, ax_h1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for method in CONDENSATION_METHODS:
        print(f"\n=== condensation: {method.upper()} ===")
        sub_A = ReducedSubstructure.build("A", substructures["A"], idx_A, VP_XYZ,
                                          N_MODES, ZETA, condensation=method,
                                          weights=RBE3_WEIGHTS)
        sub_B = ReducedSubstructure.build("B", substructures["B"], idx_B, VP_XYZ,
                                          N_MODES, ZETA, condensation=method,
                                          weights=RBE3_WEIGHTS,
                                          attachment_idx=[drive_node_B])
        M, C, K, Bc = assemble_coupled(sub_A, sub_B)
        f_r = np.concatenate([np.zeros(sub_A.M_r.shape[0]),
                              sub_B.recovery_row(INP_POS, INP_DIR)])
        system = CoupledDryFrictionCB(M, C, K, Bc, f_r, F0, MU_TRANS, N_CLAMP,
                                      ALPHA, K_TRANS, K_ROT, G, POLY_DEG)

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
            solution_set, solve_time,
            HERE / f"reference_{method}_cb_dryfriction_hbm.csv",
            labels, T, header, out_label)

        ax_max.semilogy(freq, max_time, "-", color=colors.get(method), lw=1.2,
                        label=f"{method.upper()} max|{out_label}(t)|")
        ax_h1.semilogy(freq, abs_h1, "-", color=colors.get(method), lw=1.2,
                       label=f"{method.upper()} |1st harmonic amplitude|")

    ax_max.set_ylabel(f"max|{out_label}(t)|  [m]")
    ax_max.set_title(f"Dry-friction testbench: RBE2 vs RBE3 forced response "
                     f"(mu = {MU_TRANS:g}, N = {N_CLAMP:g} N, F0 = {F0:g} N)")
    ax_max.grid(True, which="both", alpha=0.3)
    ax_max.legend()

    ax_h1.set_xlim(F_LO, F_HI)
    ax_h1.set_xlabel("Frequency [Hz]")
    ax_h1.set_ylabel(f"|1st harmonic {out_label}|  [m]")
    ax_h1.grid(True, which="both", alpha=0.3)
    ax_h1.legend()
    fig.tight_layout()
    plt.show()
