"""
RBE3 weighting from the interface node geometry.

Computes, per substructure, the L2 distance r_j = ||x_j - x_VP|| of every
interface node to the virtual point and turns it into a diagonal RBE3 weighting

    w_j = (r_j / r_ref)**p,     r_ref = mean(r_j)      (p configurable)

which enters G^T = (D^T W D)^-1 D^T W (see rbe3_vp_operator). A global scale of
W cancels in G^T, so only the *spread* of the weights matters -- that is exactly
the knob on the conditioning of D^T W D, the 6x6 matrix that is inverted to
build the VP map and therefore sits inside the coupling block of the Jacobian.

Run it to see the distance statistics and cond(D^T W D) over a range of
exponents, then write the chosen weights to rbe3_weights.json:

    python rbe3_weights.py                 # report only
    python rbe3_weights.py --exponent 1 --write

main.py can then load the file instead of RBE3_WEIGHTS = None:

    import json
    W = json.load(open(HERE / "rbe3_weights.json"))["weights"]
    ... ReducedSubstructure.build("A", ..., weights=np.array(W["A"]))

Note on what weighting can and cannot fix: D has an identity translation block
and a rotation block of size ~r (a few mm here), so its columns differ by ~1e2
in magnitude and D^T W D is ill-conditioned for *unit* reasons, which no nodal
weighting removes -- W is a row scaling, the column scaling is geometric. The
report therefore also prints the conditioning after non-dimensionalising the
rotation columns by r_ref, which is the part a weighting can actually improve.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dynamical_system import (get_boundary_nodes, load_or_export,
                              rbe2_transformation, read_descriptor)

HERE = Path(__file__).resolve().parent
FEM_DIR = HERE / "lab_testbench" / "FEM"
XLSX_PATH = HERE / "lab_testbench" / "Measurements" / "coupling_example.xlsx"
OUT_JSON = HERE / "rbe3_weights.json"

INTERFACE_METHOD = "descriptor"                  # keep in sync with main.py
MATING_TOL = 1e-6
EXPONENTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)   # scanned in the report


def interface_distances(nodes, boundary_idx, vp_xyz):
    """
    L2 distance of every interface node to the VP.

    :param nodes: (n, 3) node coordinates [m]
    :param boundary_idx: (nb,) 0-based interface node indices
    :param vp_xyz: (3,) virtual point position [m]
    :return: (nb,) distances [m], in the order of boundary_idx
    """
    return np.linalg.norm(nodes[boundary_idx] - np.asarray(vp_xyz), axis=1)


def distance_weights(r, exponent=1.0):
    """
    Per-node RBE3 weights w_j = (r_j / mean(r))**exponent, normalised to mean 1
    so the numbers are readable (the normalisation itself cancels in G^T).

    exponent > 0 favours the far nodes (they carry the rotational lever arm),
    exponent < 0 favours the near ones, exponent = 0 reproduces uniform RBE3.

    :param r: (nb,) node-to-VP distances [m]
    :param exponent: p in w = r**p
    :return: (nb,) weights, mean 1
    """
    w = (r / r.mean()) ** exponent
    return w / w.mean()


def conditioning(D, w, r_ref):
    """
    Conditioning of the 6x6 matrix inverted in the RBE3 map, both as it is built
    and after scaling the three rotation columns by r_ref (which makes all six
    columns dimensionless and comparable -- the honest measure of how well the
    *weighting* spreads the interface over the VP DoFs).

    :param D: (3nb, 6) rigid-body matrix D_Gamma
    :param w: (nb,) per-node weights
    :param r_ref: characteristic radius [m] used to non-dimensionalise
    :return: (cond_raw, cond_scaled, norm_GT)
    """
    w_dof = np.repeat(w, 3)                       # per node -> per DoF (x, y, z)
    WD = w_dof[:, None] * D
    A = D.T @ WD
    scale = np.array([1.0, 1.0, 1.0, r_ref, r_ref, r_ref])
    A_s = A / scale[:, None] / scale[None, :]     # symmetric column+row scaling
    GT = np.linalg.solve(A, WD.T)                 # (6, 3nb), same as rbe3_vp_operator
    return np.linalg.cond(A), np.linalg.cond(A_s), np.linalg.norm(GT, 2)


def report(name, nodes, boundary_idx, vp_xyz):
    """Print the distance statistics and the conditioning scan for one substructure."""
    r = interface_distances(nodes, boundary_idx, vp_xyz)
    D = rbe2_transformation(nodes, boundary_idx, vp_xyz)
    r_ref = r.mean()

    print(f"\n=== substructure {name}: {len(r)} interface nodes ===")
    print(f"  r to VP [mm]: min {r.min()*1e3:8.3f}  mean {r_ref*1e3:8.3f}  "
          f"max {r.max()*1e3:8.3f}  max/min {r.max()/r.min():6.2f}")
    print("  p      w_min    w_max   cond(D^T W D)   cond(scaled)   ||G^T||_2")
    for p in EXPONENTS:
        w = distance_weights(r, p)
        c_raw, c_scaled, n_GT = conditioning(D, w, r_ref)
        print(f"  {p:+4.1f}  {w.min():7.3f}  {w.max():7.3f}   "
              f"{c_raw:12.4e}   {c_scaled:10.4e}   {n_GT:9.4e}")
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--exponent", type=float, default=1.0,
                        help="p in w = r**p for the written weights (default 1)")
    parser.add_argument("--write", action="store_true",
                        help=f"write the weights to {OUT_JSON.name}")
    args = parser.parse_args()

    descriptor = read_descriptor(HERE / "substructure_descriptor.json")
    vp_xyz = np.array(descriptor["vp"]["position"])
    substructures = {name: load_or_export(name, FEM_DIR, HERE)
                     for name in ("A", "B")}
    idx = dict(zip(("A", "B"), get_boundary_nodes(
        INTERFACE_METHOD, substructures["A"], substructures["B"],
        xlsx_path=XLSX_PATH, tol=MATING_TOL, descriptor=descriptor)))

    distances = {name: report(name, substructures[name]["nodes"], idx[name], vp_xyz)
                 for name in ("A", "B")}

    if args.write:
        payload = {
            "exponent": args.exponent,
            "vp_position": vp_xyz.tolist(),
            "interface_node_ids": {
                name: substructures[name]["nnum"][idx[name]].tolist()
                for name in ("A", "B")},
            "distance_m": {name: r.tolist() for name, r in distances.items()},
            "weights": {name: distance_weights(r, args.exponent).tolist()
                        for name, r in distances.items()},
        }
        with open(OUT_JSON, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwritten: {OUT_JSON}  (p = {args.exponent}, one weight per node, "
              "same order as interface_node_ids)")
