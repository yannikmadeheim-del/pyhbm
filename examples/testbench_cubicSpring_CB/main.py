"""
Independent reference for the pyFBS testbench_cubicSpring example:
RBE2 interface + Craig-Bampton reduction + pyhbm second-order HBM.

Stages (run this file; later stages activate as the work packages land):
  1. WP1: import full M/K of A and B from Ansys, free-free eigencheck   [active]
  2. WP2-4: boundary nodes -> RBE2 -> Craig-Bampton (Yannik)            [pending]
  3. WP5: assembly + linear FRF check vs pyFBS linear backbone (Claude) [pending]
  4. WP6: HBM sweep 1000 -> 20 Hz -> reference_rbe2_cb_hbm.csv (Claude) [pending]

The parameters below mirror pyFBS_clone/.../examples/testbench_cubicSpring/main.py
exactly -- any change there must be reflected here, otherwise the reference and
the pyFBS curve solve different problems.
"""

import sys
from pathlib import Path

try:                          # pyhbm's progress line prints unicode (Δω);
    sys.stdout.reconfigure(   # Windows consoles default to cp1252
        encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

from dynamical_system import (CoupledCubicCB, ReducedSubstructure,
                              assemble_coupled, get_boundary_nodes,
                              load_or_export, natural_frequencies,
                              read_vp_definition, report_interface)

# ---------------------------------------------------------------------------
# Paths -- lab_testbench is a local copy of the pyFBS example data (FEM, STL,
# Measurements); PYFBS_EXAMPLE_DIR is only used to hand the reference CSV over.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[3]                    # Fast_numerical_solution_...
PYFBS_EXAMPLE_DIR = (THESIS_ROOT / "pyFBS_clone" / "pyFBS" / "pyfbs"
                     / "nonlinearFBS" / "examples" / "testbench_cubicSpring")
FEM_DIR = HERE / "lab_testbench" / "FEM"
XLSX_PATH = HERE / "lab_testbench" / "Measurements" / "coupling_example.xlsx"
CSV_OUT = HERE / "reference_rbe2_cb_hbm.csv"

# ---------------------------------------------------------------------------
# Joint definition (identical to the pyFBS dual-FBS example)
# ---------------------------------------------------------------------------
# Virtual point = RBE2 master. Defined in vp_definition.csv (same row
# structure as the pyFBS VP_Channels sheet, but local to this example):
# edit the positions there to move the joint. It must stay consistent with
# the pyFBS VPT definition -- and the exported IFACE_*.txt node sets do NOT
# move with it, reselect those in Mechanical when the joint really moves.
VP_CSV = HERE / "vp_definition.csv"
VP_XYZ = read_vp_definition(VP_CSV)              # (0.038895, 0.348107, 0.007)

k_trans, k_rot         = 1.0e3, 1.0e3            # linear stiffness [N/m], [Nm/rad]
alpha_trans, alpha_rot = 1.0e4, 1.0e4            # cubic stiffness [N/m^3], [Nm/rad^3]
beta_trans, beta_rot   = 1.0e0, 1.0e0            # cubic damping [N s^3/m^3], [Nm s^3/rad^3]
K_DIAG     = np.array([k_trans] * 3 + [k_rot] * 3)
ALPHA_DIAG = np.array([alpha_trans] * 3 + [alpha_rot] * 3)
BETA_DIAG  = np.array([beta_trans] * 3 + [beta_rot] * 3)

# excitation: F0*cos(w t) at impact H28 on B (first B reference impact)
F0 = 50.0                                        # [N]
INP_POS = np.array([0.112567, 0.258976, -0.009414])
INP_DIR = np.array([-0.0871559, 0.9961947, 0.0])

# output: displacement at channel S1 X on A (first A reference channel)
OUT_POS = np.array([-0.076519, 0.142987, 0.022])
OUT_DIR = np.array([0.7050572, 0.7091504, 0.0])

# ---------------------------------------------------------------------------
# Solver / reduction parameters
# ---------------------------------------------------------------------------
HARMONICS = [1, 3, 5, 7]                         # cubic forcing -> odd harmonics
F_LO, F_HI = 20.0, 1000.0                        # continuation window [Hz]
ZETA = 0.003                                     # modal damping per substructure
N_MODES = 60                                     # fixed-interface modes per substructure

# interface (RBE2 slave) definition -- see get_boundary_nodes:
#   "file":   bore-wall node lists exported from Ansys Mechanical (default)
#   "mating": the 7 coincident assembly nodes, "vpt": VPT sensor/impact nodes
INTERFACE_METHOD = "file"
IFACE_A_TXT = FEM_DIR / "IFACE_A.txt"
IFACE_B_TXT = FEM_DIR / "IFACE_B.txt"
MATING_TOL = 1e-6                                # coincidence tolerance [m]

# linear verification: pyFBS LM-FBS backbone (alpha=0), dumped once from the
# pyFBS example data at 1 Hz resolution -- see linear_frf_check
BACKBONE_CSV = HERE / "linear_backbone.csv"
LINEAR_PNG = HERE / "linear_check.png"


RUN_HBM = True                                   # Stage 5 (takes ~minutes)
NFRC_PNG = HERE / "nfrc.png"


def run_hbm(system):
    """
    Multiharmonic balance + arc-length continuation of the coupled system,
    swept downward from F_HI to F_LO like the pyFBS example (cold zero start
    at the top of the window, reference direction pointing to lower omega).
    """
    from pyhbm import FourierOmegaPoint, HarmonicBalanceMethod

    solver = HarmonicBalanceMethod(harmonics=HARMONICS, second_order_ode=system)
    w_lo, w_hi = 2.0 * np.pi * F_LO, 2.0 * np.pi * F_HI
    ig = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=w_hi)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((system.dimension, 1), dtype=complex), omega=-1.0)

    return solver.solve_and_continue(
        initial_guess=ig,
        initial_reference_direction=rd,
        maximum_number_of_solutions=20000,
        angular_frequency_range=[w_lo, w_hi],
        solver_kwargs={"maximum_iterations": 300,
                       "absolute_tolerance": system.F0 * 1e-6},
        step_length_adaptation_kwargs={"base": 4.0,
                                       "initial_step_length": 0.1 * 2 * np.pi,
                                       "maximum_step_length": 5 * 2 * np.pi,
                                       "minimum_step_length": 1e-7,
                                       "goal_number_of_iterations": 3},
        jacobian_update_frequency=1,
    )


def export_csv(solution_set, t_out):
    """
    Reference CSV along the continuation branch (NOT sorted by frequency --
    the NFRC is multivalued around the bent resonances). amp_m is the peak
    physical displacement over one period at the output DoF, the exact
    quantity the pyFBS example plots; amp_h1_m is the first-harmonic
    amplitude alone.
    """
    import shutil

    from pyhbm import Fourier, Fourier_Real

    h1 = list(Fourier.harmonics).index(1)
    n = len(solution_set.omega)
    freq = np.asarray(solution_set.omega) / (2.0 * np.pi)
    amp = np.empty(n)
    amp_h1 = np.empty(n)
    for i, fourier in enumerate(solution_set.fourier):
        Fourier_Real.compute_time_series(fourier)
        u_out = fourier.time_series[:, :, 0] @ t_out          # (Nt,)
        amp[i] = np.abs(u_out).max()
        c1 = fourier.coefficients[h1, :, 0] @ t_out
        amp_h1[i] = 2.0 * abs(c1) / Fourier.number_of_time_samples

    np.savetxt(CSV_OUT, np.c_[freq, np.asarray(solution_set.omega), amp, amp_h1],
               delimiter=",", comments="",
               header="freq_hz,omega_rad_s,amp_m,amp_h1_m")
    print(f"reference written: {CSV_OUT} ({n} branch points)")

    pyfbs_copy = PYFBS_EXAMPLE_DIR / CSV_OUT.name
    if PYFBS_EXAMPLE_DIR.exists():
        shutil.copy2(CSV_OUT, pyfbs_copy)
        print(f"copied to pyFBS example: {pyfbs_copy}")
    return freq, amp


def linear_frf_check(M, C, K, t_out, f_r):
    """
    Receptance |u_out / F_in| of the coupled LINEAR system (cubic terms off,
    linear spring in K) by direct solves, overlaid with the pyFBS linear
    backbone. This isolates reduction/mapping/damping errors before any HBM;
    the remaining gap is the modelling difference RBE2 vs VPT.
    """
    import matplotlib.pyplot as plt

    freqs = np.arange(F_LO, F_HI + 0.5, 1.0)
    mag = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        w = 2.0 * np.pi * f
        Z = -w**2 * M + 1j * w * C + K
        mag[i] = abs(t_out @ np.linalg.solve(Z, f_r))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(freqs, mag, "-", color="#d62728", lw=1.2,
                label="RBE2 + CB reduced model (direct solve)")
    if BACKBONE_CSV.exists():
        bb = np.loadtxt(BACKBONE_CSV, delimiter=",", comments="#")
        sel = (bb[:, 0] >= F_LO) & (bb[:, 0] <= F_HI)
        ax.semilogy(bb[sel, 0], bb[sel, 1], "--", color="#555555", lw=1.0,
                    label="pyFBS linear backbone (VPT + LM-FBS)")
        ratio = np.interp(bb[sel, 0], freqs, mag) / bb[sel, 1]
        print(f"linear check vs pyFBS backbone: |Y| ratio median "
              f"{np.median(ratio):.3f} (min {ratio.min():.3f}, "
              f"max {ratio.max():.3f})")
    else:
        print(f"note: {BACKBONE_CSV.name} not found -- plotting reduced model only")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|u_out / F_in|  [m/N]")
    ax.set_title("Linear verification: RBE2+CB reduction vs pyFBS backbone")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LINEAR_PNG, dpi=110)
    print(f"linear check figure: {LINEAR_PNG}")
    return freqs, mag


if __name__ == "__main__":
    # --- Stage 1 (WP1): full matrices + free-free eigencheck ---------------
    print(f"FEM data: {FEM_DIR}")
    substructures = {}
    for name in ("A", "B"):
        data = load_or_export(name, FEM_DIR, HERE)
        substructures[name] = data
        f = natural_frequencies(data["K"], data["M"], n=12)
        print(f"[{name}] n_nodes = {data['nodes'].shape[0]}, "
              f"n_dofs = {data['K'].shape[0]}")
        print(f"[{name}] lowest 12 natural frequencies [Hz] "
              f"(A: fixed-base, no rigid-body modes; B: free-free, 6 x ~0):")
        print("    " + np.array2string(f, precision=3, suppress_small=True))

    # --- Stage 2 (WP2): boundary node sets ----------------------------------
    missing = [p.name for p in (IFACE_A_TXT, IFACE_B_TXT) if not p.exists()]
    if INTERFACE_METHOD == "file" and missing:
        print(f"Stage 2 stopped: waiting for {', '.join(missing)} in {FEM_DIR}")
        raise SystemExit(0)
    idx_A, idx_B = get_boundary_nodes(
        INTERFACE_METHOD, substructures["A"], substructures["B"],
        xlsx_path=XLSX_PATH, tol=MATING_TOL,
        file_A=IFACE_A_TXT, file_B=IFACE_B_TXT)
    report_interface("A", substructures["A"]["nodes"], idx_A, VP_XYZ)
    report_interface("B", substructures["B"]["nodes"], idx_B, VP_XYZ)

    # --- Stage 3 (WP3+4): RBE2 + Craig-Bampton reduction --------------------
    sub_A = ReducedSubstructure.build("A", substructures["A"], idx_A, VP_XYZ, N_MODES, ZETA)
    sub_B = ReducedSubstructure.build("B", substructures["B"], idx_B, VP_XYZ, N_MODES, ZETA)

    # --- Stage 4 (WP5): assembly + linear FRF check --------------------------
    M, C, K, Bc = assemble_coupled(sub_A, sub_B, K_DIAG)
    t_out = np.concatenate([sub_A.recovery_row(OUT_POS, OUT_DIR),
                            np.zeros(sub_B.M_r.shape[0])])
    f_r = np.concatenate([np.zeros(sub_A.M_r.shape[0]),
                          sub_B.recovery_row(INP_POS, INP_DIR)])
    lin_freqs, lin_mag = linear_frf_check(M, C, K, t_out, f_r)

    # --- Stage 5 (WP6): HBM sweep -> reference CSV ---------------------------
    if RUN_HBM:
        import matplotlib.pyplot as plt

        system = CoupledCubicCB(M, C, K, Bc, ALPHA_DIAG, BETA_DIAG, f_r, F0)
        solution_set = run_hbm(system)
        freq_nl, amp_nl = export_csv(solution_set, t_out)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(lin_freqs, lin_mag * F0, "-", color="#999999", lw=1.2,
                    label=f"linear (alpha=0) x F0={F0:g} N")
        ax.semilogy(freq_nl, amp_nl, "-", color="#d62728", lw=1.2,
                    label="RBE2+CB+HBM nonlinear forced response")
        ax.set_xlim(F_LO, F_HI)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude |u_out|  [m]")
        ax.set_title("Cubic-spring testbench: independent RBE2+CB+HBM reference")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(NFRC_PNG, dpi=110)
        print(f"NFRC figure: {NFRC_PNG}")
        plt.show()
