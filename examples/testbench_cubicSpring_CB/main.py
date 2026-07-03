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

from pathlib import Path

import numpy as np

from dynamical_system import load_or_export, natural_frequencies

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
VP_XYZ = np.array([0.038895, 0.348107, 0.007])   # virtual point = RBE2 master [m]

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
INTERFACE_METHOD = "mating"                      # "mating" | "vpt"  (see WP2)
MATING_TOL = 1e-6                                # coincidence tolerance [m]


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

    # --- Stage 2 (WP2-4, Yannik): reduction ---------------------------------
    # idx_A, idx_B = get_boundary_nodes(INTERFACE_METHOD, substructures["A"],
    #                                   substructures["B"], XLSX_PATH, MATING_TOL)
    # sub_A = ReducedSubstructure.build("A", substructures["A"], idx_A, VP_XYZ, N_MODES, ZETA)
    # sub_B = ReducedSubstructure.build("B", substructures["B"], idx_B, VP_XYZ, N_MODES, ZETA)

    # --- Stage 3 (WP5, Claude): assembly + linear FRF check -----------------
    # --- Stage 4 (WP6, Claude): HBM sweep -> CSV -----------------------------
