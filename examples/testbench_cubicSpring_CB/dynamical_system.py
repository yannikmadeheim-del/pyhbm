"""
RBE2 + Craig-Bampton verification model for the pyFBS testbench_cubicSpring example.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh, splu


def load_ansys_substructure(rst_path, full_path):
    """
    Read one substructure's FE data from the Ansys binary files.

    :param rst_path:  .rst result file (node coordinates)
    :param full_path: .full file (assembled K and M)
    :return: dict with nodes (n,3) [m], dof_ref (3n,2) [node id, dof], K, M (csr)
    """
    from ansys.mapdl import reader as pymapdl_reader

    rst = pymapdl_reader.read_binary(str(rst_path))
    nodes = np.asarray(rst.mesh.nodes, dtype=float)      # (n_nodes, 3) [m]
    nnum = np.asarray(rst.mesh.nnum)                     # Ansys node ids

    full = pymapdl_reader.read_binary(str(full_path))
    dof_ref, k_triu, m_triu = full.load_km(sort=True)    # upper-triangular sparse
    dof_ref = np.asarray(dof_ref)
    K = (k_triu + sparse.triu(k_triu, 1).T).tocsr()
    M = (m_triu + sparse.triu(m_triu, 1).T).tocsr()

    n_nodes = nodes.shape[0]
    assert dof_ref.shape[0] == 3 * n_nodes, (
        f"expected 3 translational DoFs per node (SOLID, no BCs): "
        f"{dof_ref.shape[0]} DoFs vs {n_nodes} nodes")
    assert np.array_equal(dof_ref[::3, 0], nnum), \
        ".full node ordering does not match the .rst mesh -- DoF mapping invalid"
    assert np.array_equal(dof_ref[:3, 1], [0, 1, 2]), \
        f"per-node DoF order is not x,y,z: {dof_ref[:3, 1]}"

    # pyfbs convention: node ids consecutive starting at 1
    if dof_ref[0, 0] != 1:
        dof_ref[:, 0] = dof_ref[:, 0] - (dof_ref[0, 0] - 1)
    assert np.array_equal(dof_ref[::3, 0], np.arange(1, n_nodes + 1)), \
        "node ids are not consecutive -- the 3*i+d DoF indexing would be wrong"

    asym = abs(K - K.T).max()
    assert asym <= 1e-9 * abs(K).max(), f"K not symmetric after triu fix: {asym:g}"
    span = np.ptp(nodes, axis=0)
    assert 0.01 < span.max() < 1.0, f"node coordinates not in metres? span = {span}"

    # Drop orphan nodes (mesh nodes not attached to any element): their K and M
    # rows are completely empty, which makes every factorization exactly
    # singular and would poison the boundary/internal partition later (an
    # orphan in the internal set makes K_ii singular in the Craig-Bampton
    # step). pyfbs keeps them and perturbs the diagonal instead; for the
    # reduction pipeline removing them is the clean choice.
    empty_dof = (np.diff(K.indptr) == 0) & (np.diff(M.indptr) == 0)
    if empty_dof.any():
        per_node = empty_dof.reshape(-1, 3)
        assert np.array_equal(per_node.all(axis=1), per_node.any(axis=1)), \
            "empty DoF rows do not come in whole-node triples"
        keep_nodes = ~per_node.all(axis=1)
        keep_dofs = np.where(np.repeat(keep_nodes, 3))[0]
        K = K[keep_dofs][:, keep_dofs].tocsr()
        M = M[keep_dofs][:, keep_dofs].tocsr()
        nodes = nodes[keep_nodes]
        nnum = nnum[keep_nodes]
        n_nodes = nodes.shape[0]
        dof_ref = np.column_stack((np.repeat(np.arange(1, n_nodes + 1), 3),
                                   np.tile([0, 1, 2], n_nodes)))
        print(f"  dropped {np.count_nonzero(~keep_nodes)} orphan nodes "
              f"({3 * np.count_nonzero(~keep_nodes)} empty DoFs), "
              f"{n_nodes} nodes remain")

    # nnum: original Ansys node numbers of the kept nodes (traceability only;
    # all downstream indexing is 0-based into ``nodes``)
    return dict(nodes=nodes, dof_ref=dof_ref, K=K, M=M, nnum=nnum)


def load_or_export(name, fem_dir, cache_dir):
    """
    NPZ cache around :func:`load_ansys_substructure`.

    Reads <cache_dir>/<name>_full.npz if present, otherwise imports
    <fem_dir>/<name>.rst + <name>.full and writes the cache (sparse matrices
    stored as their CSR components, because np.savez cannot hold scipy sparse).
    """
    cache = Path(cache_dir) / f"{name}_full.npz"
    if cache.exists():
        d = np.load(cache)

        def csr(prefix):
            return sparse.csr_matrix(
                (d[f"{prefix}_data"], d[f"{prefix}_indices"], d[f"{prefix}_indptr"]),
                shape=tuple(d[f"{prefix}_shape"]))

        return dict(nodes=d["nodes"], dof_ref=d["dof_ref"], nnum=d["nnum"],
                    K=csr("K"), M=csr("M"))

    data = load_ansys_substructure(Path(fem_dir) / f"{name}.rst",
                                   Path(fem_dir) / f"{name}.full")
    K, M = data["K"], data["M"]
    np.savez_compressed(
        cache, nodes=data["nodes"], dof_ref=data["dof_ref"], nnum=data["nnum"],
        K_data=K.data, K_indices=K.indices, K_indptr=K.indptr, K_shape=np.array(K.shape),
        M_data=M.data, M_indices=M.indices, M_indptr=M.indptr, M_shape=np.array(M.shape))
    return data


def natural_frequencies(K, M, n=12):
    """
    Lowest ``n`` natural frequencies [Hz] of a substructure, sorted.
    """
    lam = eigsh(K.tocsc(), k=n, M=M.tocsc(), sigma=-1.0e3, which="LM",
                return_eigenvectors=False)
    return np.sort(np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi))


def read_vp_definition(csv_path, grouping=None):
    """
    All rows must share one position, and the directions must be the global
    axes -- a rotated VP frame would change the meaning of the 6 master DoFs
    (and of the spring's k/alpha/beta diagonals) and is rejected.

    :param csv_path: vp_definition.csv next to main.py
    :param grouping: VP grouping id; None -> all rows (single-VP file)
    :return: (3,) VP position [m]
    """
    import csv

    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if grouping is None or int(r["Grouping"]) == grouping]
    assert rows, f"{csv_path}: no VP rows (grouping={grouping})"

    pos = np.array([[float(r[f"Position_{i}"]) for i in (1, 2, 3)] for r in rows])
    dirs = np.array([[float(r[f"Direction_{i}"]) for i in (1, 2, 3)] for r in rows])
    assert np.ptp(pos, axis=0).max() < 1e-12, \
        f"{csv_path}: rows differ in position -- multiple VPs? pass grouping="
    assert np.allclose(dirs, np.vstack([np.eye(3)] * 2)[:len(dirs)]), \
        f"{csv_path}: directions are not the global axes -- rotated VP frames unsupported"
    return pos[0]


def find_nodes_by_ansys_id(nodes, nnum, ids):
    """
    :param nodes: (n, 3) node coordinates of ONE substructure [m]
    :param nnum: (n,) original Ansys node ids of these nodes
    :param ids: iterable of Ansys node ids
    :return: sorted int array of 0-based node indices, duplicates removed
    """
    ids = [int(i) for i in ids]
    assert ids, "empty node id list"
    pos = {int(n): i for i, n in enumerate(nnum)}
    missing = [i for i in ids if i not in pos]
    assert not missing, (
        f"{len(missing)} node ids unknown to the imported model "
        f"(e.g. {missing[:5]}) -- renumbered mesh or eliminated nodes?")
    return np.unique([pos[i] for i in ids])


def find_file_nodes(nodes, nnum, path):
    """
    Interface definition ("file"): a node list exported from an Ansys Mechanical
    named selection / node component (bore wall of the VP hole).

    Expected format: a header line ("Knotennummer" / "Node Number") followed by
    one Ansys node id per line; extra columns are ignored. The ids use the
    ORIGINAL Ansys numbering, which survives the chain cdb -> blocked cdb ->
    Mechanical unchanged (verified bit-identical).

    :param nodes: (n, 3) node coordinates of ONE substructure [m]
    :param nnum: (n,) original Ansys node ids of these nodes
    :param path: exported .txt file
    :return: sorted int array of 0-based node indices, duplicates removed
    """
    ids = []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if tok and tok[0].isdigit():
                ids.append(int(tok[0]))
    assert ids, f"no node ids found in {path}"
    return find_nodes_by_ansys_id(nodes, nnum, ids)


def read_descriptor(json_path):
    """
    pyFBS-exported substructure descriptor (see the testbench_cubicSpring
    exporter export_substructure_descriptor.py): VP frame, joint (k/alpha/beta),
    excitation/output DoFs, and per substructure the interface node ids VPT uses
    (ORIGINAL Ansys ids). Lets pyhbm reproduce the pyFBS interface exactly while
    still re-reading M/K independently from the same .full/.rst.

    :param json_path: substructure_descriptor.json next to main.py
    :return: parsed dict
    """
    with open(json_path) as fh:
        return json.load(fh)


def find_mating_nodes(nodes_A, nodes_B, tol):
    """
    Coincident-node detection between the two substructure meshes ("mating"):
    the discrete points where A and B share nodes in the assembly (here: 7
    pairs along the joint strip). Kept as the automatic baseline method.

    :param nodes_A: (nA, 3) node coordinates of substructure A [m]
    :param nodes_B: (nB, 3) node coordinates of substructure B [m]
    :param tol: coincidence tolerance [m]; anything in 1e-8..1e-4 gives the
        same 7 pairs (next-nearest distance is 2e-4 m)
    :return: (idx_A, idx_B) int arrays, one entry per mating node pair
    """
    from scipy.spatial import cKDTree

    dist, j = cKDTree(nodes_B).query(nodes_A)
    mask = dist <= tol
    idx_A = np.nonzero(mask)[0]
    idx_B = j[mask]
    assert len(idx_A) > 0, "no coincident nodes -- meshes are non-conforming"
    assert len(np.unique(idx_B)) == len(idx_B), "tol too loose: pairing not 1:1"
    return idx_A, idx_B


def find_vpt_nodes(nodes, xlsx_path, substructure, grouping=10):
    """
    Alternative interface definition ("vpt"): the FE nodes the virtual-point
    transformation actually sees -- all Grouping==``grouping`` rows of the
    sheets Channels_<substructure> and Impacts_<substructure>, positions
    snapped to the nearest FE node, duplicates removed.

    :param nodes: (n, 3) node coordinates of ONE substructure [m]
    :param xlsx_path: coupling_example.xlsx of the pyFBS example
    :param substructure: "A" or "B" (selects the sheet names)
    :return: sorted int array of 0-based node indices
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    pos = np.vstack([
        df.loc[df["Grouping"] == grouping,
               ["Position_1", "Position_2", "Position_3"]].to_numpy(float)
        for df in (pd.read_excel(xlsx_path, sheet_name=f"{kind}_{substructure}")
                   for kind in ("Channels", "Impacts"))])
    assert len(pos), f"no Grouping=={grouping} rows for substructure {substructure}"
    dist, idx = cKDTree(nodes).query(pos)
    return np.unique(idx)


def get_boundary_nodes(method, data_A, data_B, xlsx_path=None, tol=1e-6,
                       file_A=None, file_B=None, descriptor=None):
    """
    The single switch point for the interface definition.

    method == "descriptor": interface node ids from the pyFBS-exported descriptor
                            (the exact nodes VPT uses; default)
    method == "file":       node lists exported from Ansys Mechanical (file_A/B)
    method == "mating":     coincident-node detection on both meshes
    method == "vpt":        VPT sensor/impact nodes per substructure

    :return: (idx_A, idx_B) -- 0-based boundary node indices per substructure
    """
    if method == "descriptor":
        subs = descriptor["substructures"]
        return (find_nodes_by_ansys_id(data_A["nodes"], data_A["nnum"],
                                       subs["A"]["interface_node_ids"]),
                find_nodes_by_ansys_id(data_B["nodes"], data_B["nnum"],
                                       subs["B"]["interface_node_ids"]))
    if method == "file":
        return (find_file_nodes(data_A["nodes"], data_A["nnum"], file_A),
                find_file_nodes(data_B["nodes"], data_B["nnum"], file_B))
    if method == "mating":
        return find_mating_nodes(data_A["nodes"], data_B["nodes"], tol)
    if method == "vpt":
        return (find_vpt_nodes(data_A["nodes"], xlsx_path, "A"),
                find_vpt_nodes(data_B["nodes"], xlsx_path, "B"))
    raise ValueError(f"unknown interface method {method!r}")


def report_interface(name, nodes, idx, vp_xyz):
    """
    Plausibility report for a boundary node set: count, in-plane distance to
    the virtual point (bore wall -> a ring of a few mm) and z extent.
    """
    sel = nodes[idx]
    r_xy = np.linalg.norm(sel[:, :2] - np.asarray(vp_xyz)[:2], axis=1) * 1e3
    z = sel[:, 2] * 1e3
    print(f"[{name}] {len(idx)} boundary nodes | r_xy to VP "
          f"{r_xy.min():.2f}..{r_xy.max():.2f} mm | z {z.min():.2f}..{z.max():.2f} mm")



def skew(r):
    """
    Skew-symmetric cross-product matrix:  skew(v) @ w == np.cross(v, w).

        [[  0, -vz,  vy],
         [ vz,   0, -vx],
         [-vy,  vx,   0]]
    """
    return np.array([[0, -r[2], r[1]],
                    [r[2], 0, -r[0]],
                    [-r[1], r[0], 0]])


def rbe2_transformation(nodes, boundary_idx, master_xyz):
    """
    RBE2 kinematics: every boundary (slave) node moves rigidly with the 6-DoF
    master q_m = [ux, uy, uz, rx, ry, rz] at position r_m (small rotations):

        u_j = u_m + theta x (r_j - r_m)   =>   u_j = [ I3 | -skew(r_j - r_m) ] q_m

    Stack the (3 x 6) blocks in the order of ``boundary_idx``:

        T_b (3*nb, 6),  rows 3*p..3*p+2  <->  boundary node boundary_idx[p]

    Acceptance (WP3 rigid-body test, done in review): unit rigid motions of the
    master reproduce exact rigid displacement fields at the slaves.

    :param nodes: (n, 3) node coordinates of the substructure [m]
    :param boundary_idx: (nb,) 0-based boundary node indices
    :param master_xyz: (3,) master/VP position [m]
    :return: T_b (3*nb, 6) dense
    """
    d = nodes[boundary_idx] - np.asarray(master_xyz)  # (nb, 3) Hebelarme
    T_b = np.zeros((3 * len(boundary_idx), 6))
    for p, dp in enumerate(d):
        T_b[3 * p:3 * p + 3, :3] = np.eye(3)
        T_b[3 * p:3 * p + 3, 3:] = -skew(dp)  # u_j = u_m + θ×d  ⇒  −skew(d)
    return T_b


def partition_dofs(n_nodes, boundary_idx):
    """DoF-Permutation für die Boundary-first-Sortierung (Folie, Schritt 2).

    :return: (perm, internal_idx)   # K_sorted = K[perm][:, perm]
    """
    internal_idx = np.setdiff1d(np.arange(n_nodes), boundary_idx)  # aufsteigend
    node_order = np.concatenate([boundary_idx, internal_idx])
    perm = (3 * node_order[:, None] + np.arange(3)).ravel()
    return perm, internal_idx


def apply_rbe2(K, M, T_b, perm, n_b, n_att=0):
    """Sortieren + RBE2-Kondensation der Boundary-DoFs auf den 6-DoF-Master.

    K_s = K[perm][:, perm]          # jetzt: [b-Block | i-Block] zusammenhängend
    nb3 = 3 * n_b
    K_bb = K_s[:nb3, :nb3]; K_bi = K_s[:nb3, nb3:]; K_ii = K_s[nb3:, nb3:]
    ->  K_bb6 = T_b.T @ (K_bb @ T_b)      (6, 6)    dicht
        K_bi6 = (K_bi.T @ T_b).T          (6, n_i)  dicht -- über die sparse Seite rechnen!
        K_ii  bleibt sparse (csc für splu in WP4)
    gleiches für M.  :return: dict(K_bb, K_bi, K_ii, M_bb, M_bi, M_ii)
    """
    n_if3 = 3 * n_b
    nb3 = n_if3 + 3 * n_att
    if n_att:                                       # interface -> 6 master (T_b),
        T_red = np.zeros((nb3, 6 + 3 * n_att))      # attachment nodes -> own 3 DoFs
        T_red[:n_if3, :6] = T_b
        T_red[n_if3:, 6:] = np.eye(3 * n_att)
    else:
        T_red = T_b

    def transform_blocks(A):
        # symmetric A => A_ib = A_bi.T, so only three blocks are returned
        # (a stored ib copy could silently drift from bi); A_ii stays sparse
        # for the splu/eigsh factorizations in the Craig-Bampton step.
        A_s = A[perm][:, perm]
        A_bb = T_red.T @ (A_s[:nb3, :nb3] @ T_red)  # (6+3n_att, 6+3n_att) dense
        A_bi = (A_s[:nb3, nb3:].T @ T_red).T        # (6+3n_att, n_i) dense, sparse-side
        A_ii = A_s[nb3:, nb3:].tocsc()
        return A_bb, A_bi, A_ii

    K_bb, K_bi, K_ii = transform_blocks(K)
    M_bb, M_bi, M_ii = transform_blocks(M)
    return dict(K_bb=K_bb, K_bi=K_bi, K_ii=K_ii,
                M_bb=M_bb, M_bi=M_bi, M_ii=M_ii)


def rbe3_vp_operator(nodes, boundary_idx, master_xyz, weights=None):
    """
    RBE3 interpolation map: the virtual point (VP) motion is the W-weighted
    least-squares average of the interface node displacements,

        q_m = G^T q_Gamma,   G = W D (D^T W D)^-1,   G^T = (D^T W D)^-1 D^T W

    with D = D_Gamma the rigid-body matrix of :func:`rbe2_transformation` and W a
    diagonal weighting over the 3*nb interface DoFs. Unlike RBE2 this does NOT
    reduce or stiffen the interface -- G^T only observes the VP from the (still
    free) interface, and by reciprocity a VP wrench distributes back to the nodes
    as f_Gamma = G w. Note G^T D == I6 (left inverse), so a *symmetric* 6-DoF
    reduction with this map would collapse to RBE2; RBE3 therefore keeps the
    interface DoFs as the Craig-Bampton boundary set and uses G^T only to couple.

    :param nodes: (n, 3) node coordinates [m]
    :param boundary_idx: (nb,) 0-based interface node indices
    :param master_xyz: (3,) VP position [m]
    :param weights: None -> uniform; else per-node (nb,) or per-DoF (3nb,) weights.
        A uniform scale cancels in G, so W only matters for irregular meshes.
    :return: G^T (6, 3*nb) dense
    """
    D = rbe2_transformation(nodes, boundary_idx, master_xyz)      # (3nb, 6) D_Gamma
    if weights is None:
        w = np.ones(D.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if w.size == len(boundary_idx):          # one weight per node -> per DoF
            w = np.repeat(w, 3)
    WD = w[:, None] * D                           # W D
    return np.linalg.solve(D.T @ WD, WD.T)        # (D^T W D)^-1 D^T W  = G^T


# ===========================================================================
# Directional (pyFBS-VPT) interface -- the DoF set behind the RBE_rigid and
# RBE_average condensations.
#
# pyFBS's virtual point transformation never sees a whole node: every workbook
# row contributes ONE scalar DoF, the response (or force) along that row's own
# direction, at the FE node its position snaps to. A triaxial sensor supplies
# three rows and so happens to span its node completely; a single impact
# supplies one and leaves the other two directions of that node untouched.
# These helpers reproduce that DoF set exactly, in contrast to the full
# 3-DoF-per-node idealisation the "rbe2"/"rbe3" condensations use.
# ===========================================================================

VPT_GROUPING = 10                # workbook Grouping id of the virtual-point interface


def read_vpt_rows(xlsx_path, substructure, nodes, grouping=VPT_GROUPING):
    """
    The interface rows pyFBS's VPT actually uses, snapped to the FE mesh.

    Reads the ``grouping`` rows of Channels_<substructure> / Impacts_<substructure>
    and snaps each position to the nearest FE node -- the same rule as pyfbs
    Model.find_nearest_locations, verified to select the identical nodes.

    :param nodes: (n, 3) node coordinates of ONE substructure [m]
    :return: dict(channels=[...], impacts=[...]), each a list of
        (name, node index, direction (3,)) in WORKBOOK ROW ORDER. The order is
        what makes Ru/Rf comparable to vpt.ru/vpt.rf row by row.
    """
    import pandas as pd

    out = {}
    for kind, sheet in (("channels", "Channels"), ("impacts", "Impacts")):
        df = pd.read_excel(xlsx_path, sheet_name=f"{sheet}_{substructure}")
        rows = []
        for _, r in df[df["Grouping"] == grouping].iterrows():
            pos = r[["Position_1", "Position_2", "Position_3"]].to_numpy(float)
            dvec = r[["Direction_1", "Direction_2", "Direction_3"]].to_numpy(float)
            j = int(np.argmin(np.linalg.norm(nodes - pos, axis=1)))
            rows.append((str(r["Name"]), j, dvec))
        assert rows, f"no Grouping=={grouping} rows in {sheet}_{substructure}"
        out[kind] = rows
    return out


def check_vpt_rows(vpt_rows, descriptor, substructure, nnum):
    """
    Guard against the two-workbook-copies hazard.

    The descriptor is written by pyFBS from ITS copy of coupling_example.xlsx;
    ``vpt_rows`` come from the copy next to this example. Editing one and not the
    other -- or forgetting to re-run export_substructure_descriptor.py -- would
    leave the two pipelines silently describing different interfaces, so compare
    them row by row. Skipped for descriptors written before these fields existed.
    """
    sub = descriptor["substructures"][substructure]
    for kind in ("channels", "impacts"):
        ref = sub.get(f"interface_{kind}")
        if ref is None:
            continue
        got = vpt_rows[kind]
        hint = ("descriptor and the local coupling_example.xlsx disagree -- re-run "
                "export_substructure_descriptor.py, or the two workbook copies "
                "have drifted apart")
        assert len(ref) == len(got), \
            f"[{substructure}] {kind}: {len(ref)} rows in descriptor vs {len(got)} local -- {hint}"
        for k, (exp, (name, j, dvec)) in enumerate(zip(ref, got)):
            assert int(exp["node_id"]) == int(nnum[j]), \
                (f"[{substructure}] {kind} row {k} ({name}): descriptor node "
                 f"{exp['node_id']} vs local {int(nnum[j])} -- {hint}")
            assert np.allclose(exp["direction"], dvec, atol=1e-9), \
                f"[{substructure}] {kind} row {k} ({name}): direction differs -- {hint}"


def direction_basis(directions, tol=1e-8):
    """
    Split one node's 3D space into the directions the workbook names and the rest.

    The SVD is not cosmetic: once Channels_<X> and Impacts_<X> are made identical
    (so that Tu == Tf.T), every interface node carries DUPLICATED directions --
    an impact row and its mirrored channel row point the same way -- so the raw
    stack is rank deficient and could not be completed to a basis directly. The
    rank detection below handles that as a matter of course.

    :param directions: (m, 3) unit directions named at this node -- three for a
        triax, one for a single impact, with duplicates allowed.
    :return: (C, N) with C (k, 3) an ORTHONORMAL basis of their span (k = rank)
        and N (3-k, 3) its orthonormal complement, so vstack([C, N]) is a
        rotation. In that frame the k constrained coordinates can be retained as
        Craig-Bampton boundary DoFs while the remaining 3-k stay interior.
    """
    _, s, Vt = np.linalg.svd(np.atleast_2d(directions))
    k = int(np.count_nonzero(s > tol * max(1.0, s[0])))
    return Vt[:k], Vt[k:]


@dataclass
class DirectionalBoundary:
    """
    The (constrained | free) split of every retained node, and the layout of the
    resulting boundary vector ``a``. ``interface_mask`` marks which boundary DoFs
    belong to the joint interface -- the excitation/output DoFs are retained for
    static completeness but must NOT enter the virtual-point map.
    """
    node_idx: np.ndarray               # (nb,) retained node indices, ascending
    C: dict                            # node index -> (k_j, 3) constrained basis
    N: dict                            # node index -> (3-k_j, 3) complement
    slot: dict                         # node index -> slice into ``a``
    n_boundary: int                    # sum of k_j over retained nodes
    interface_mask: np.ndarray         # (n_boundary,) bool
    b_idx: np.ndarray                  # (n_boundary,) boundary DoFs, ROTATED global index
    i_idx: np.ndarray                  # (n_dof - n_boundary,) interior DoFs, ditto

    def basis(self, node):
        return self.C[int(node)]

    def free_slot(self, node):
        """Positions of node ``node``'s free (complement) coordinates within the
        INTERIOR vector -- what :func:`ReducedSubstructure.recovery_row` needs to
        pick the Psi/Phi rows of a partially constrained node."""
        j, k = int(node), self.C[int(node)].shape[0]
        return np.searchsorted(self.i_idx, 3 * j + np.arange(k, 3))


def build_directional_boundary(interface_rows, io_rows, n_nodes):
    """
    Assemble the directional boundary from the workbook rows.

    :param interface_rows: (name, node, direction) of every joint-interface row
        -- the UNION of the channel and impact rows, so the boundary set does not
        change when the two sheets are later made identical.
    :param io_rows: same tuples for the excitation / output DoFs; retained so
        their static response is exact, but excluded from the VP map.
    :param n_nodes: total node count of the substructure (sets the interior set)
    :return: :class:`DirectionalBoundary`
    """
    per_node, iface_nodes = {}, set()
    for rows, is_iface in ((interface_rows, True), (io_rows, False)):
        for _, j, dvec in rows:
            per_node.setdefault(int(j), []).append(dvec)
            if is_iface:
                iface_nodes.add(int(j))

    node_idx = np.array(sorted(per_node), dtype=int)
    C, N, slot, mask, b_idx, off = {}, {}, {}, [], [], 0
    for j in node_idx:
        C[int(j)], N[int(j)] = direction_basis(np.vstack(per_node[int(j)]))
        k = C[int(j)].shape[0]
        slot[int(j)] = slice(off, off + k)
        mask.append(np.full(k, int(j) in iface_nodes))
        b_idx.append(3 * int(j) + np.arange(k))          # rotated-frame DoF index
        off += k
    b_idx = np.concatenate(b_idx)
    return DirectionalBoundary(
        node_idx=node_idx, C=C, N=N, slot=slot, n_boundary=off,
        interface_mask=np.concatenate(mask), b_idx=b_idx,
        i_idx=np.setdiff1d(np.arange(3 * n_nodes), b_idx))


def rotate_and_partition(K, M, boundary):
    """
    Rotate every retained node into its (constrained | free) frame and return the
    boundary-first blocks in the same dict layout as :func:`apply_rbe2` /
    :func:`partition_blocks`, so :func:`craig_bampton` is shared by all four
    condensations.

    The rotation u_j = Q_j^T [a_j; b_j] with Q_j = [C_j; N_j] is orthogonal, so
    it is exact and does not touch conditioning; only the constrained
    coordinates a_j become boundary DoFs, the free b_j join the interior.
    """
    n_dof = K.shape[0]
    T = sparse.eye(n_dof, format="lil")
    for j in boundary.node_idx:
        Q = np.vstack([boundary.C[int(j)], boundary.N[int(j)]])   # (3, 3) rotation
        T[3 * j:3 * j + 3, 3 * j:3 * j + 3] = Q.T
    T = T.tocsr()

    perm = np.concatenate([boundary.b_idx, boundary.i_idx])
    nb = boundary.n_boundary

    def blocks(A):
        A_z = (T.T @ A @ T).tocsr()[perm][:, perm]
        return A_z[:nb, :nb].toarray(), A_z[:nb, nb:].toarray(), A_z[nb:, nb:].tocsc()

    K_bb, K_bi, K_ii = blocks(K)
    M_bb, M_bi, M_ii = blocks(M)
    return dict(K_bb=K_bb, K_bi=K_bi, K_ii=K_ii,
                M_bb=M_bb, M_bi=M_bi, M_ii=M_ii)


def vpt_selection(rows, boundary):
    """
    (n_rows, n_boundary) map from the boundary coordinates ``a`` to the scalar
    channel / impact amplitudes: the amplitude of row (node j, direction d) is
    d.u_j = (C_j d).a_j, because d lies in the span of C_j by construction.
    Every other entry is exactly zero -- that is what makes this reduction
    faithful to the VPT rather than an approximation of it, so it is asserted.
    """
    S = np.zeros((len(rows), boundary.n_boundary))
    for r, (name, j, dvec) in enumerate(rows):
        C = boundary.basis(j)
        residual = np.linalg.norm(dvec - C.T @ (C @ dvec))
        assert residual < 1e-10, (
            f"row {name!r}: direction is not in the retained span of node {j} "
            f"(residual {residual:g}) -- the boundary is missing this direction")
        S[r, boundary.slot[int(j)]] = C @ dvec
    return S


def vpt_idm(nodes, rows, master_xyz):
    """
    pyFBS's Ru / Rf: one row per workbook row, ``direction @ [I3 | -skew(r)]``
    with r the lever arm from the virtual point to the snapped node. Built on
    :func:`rbe2_transformation`, which is elementwise identical to vpt.py's
    r_matrix_u block (and its transpose is the r_matrix_f block).
    """
    return np.vstack([dvec @ rbe2_transformation(nodes, [j], master_xyz)
                      for _, j, dvec in rows])


def vpt_transformations(Ru, Rf, wu=None, wf=None):
    """
    Tu and Tf exactly as pyfbs/interface/vpt.py builds them (define_idm_u /
    define_idm_f), pinv included so the degenerate cases behave identically.

    ``wu`` and ``wf`` stay INDEPENDENT here, as they are in pyFBS. The single
    ``weights`` argument of :func:`rbe3_vp_operator` cannot be reused: it plays
    the role of wu and of wf^-1 at once, which forces Tf == Tu.T and is exactly
    the reciprocity the VPT does not have.

    :return: (Tu (6, n_chn), Tf (n_imp, 6))
    """
    Wu = np.eye(Ru.shape[0]) if wu is None else np.asarray(wu, dtype=float)
    Wf = np.eye(Rf.shape[0]) if wf is None else np.asarray(wf, dtype=float)
    Tu = np.linalg.pinv(Ru.T @ Wu @ Ru) @ Ru.T @ Wu
    Tf = np.linalg.pinv(Wf) @ Rf @ np.linalg.pinv(Rf.T @ np.linalg.pinv(Wf) @ Rf)
    return Tu, Tf


def condense_boundary(blocks, T_red):
    """
    Apply a boundary reduction a_b = T_red q_b to a partitioned block dict,
    leaving the interior untouched. The RBE_rigid counterpart of the in-place
    condensation :func:`apply_rbe2` performs, kept separate so the partition and
    the constraint stay independent steps.
    """
    out = {}
    for sym in ("K", "M"):
        out[f"{sym}_bb"] = T_red.T @ blocks[f"{sym}_bb"] @ T_red
        out[f"{sym}_bi"] = T_red.T @ blocks[f"{sym}_bi"]
        out[f"{sym}_ii"] = blocks[f"{sym}_ii"]
    return out


def rigid_boundary_map(rows, boundary, R):
    """
    Reduction a_b = T_red [q_m; q_io] for RBE_rigid: the interface coordinates
    follow the 6-DoF master rigidly, the excitation/output coordinates stay free.

    The constraint is stated on the workbook rows, "the amplitude along row r
    equals R[r] q_m", i.e. ``S_iface a_iface = R q_m``. Because the boundary
    coordinates are the ROTATED node coordinates (see :func:`direction_basis`),
    not the raw amplitudes, that has to be inverted rather than read off:

        a_iface = pinv(S_iface) R q_m

    ``pinv`` -- not ``solve`` -- because after the two sheets are unified every
    direction appears twice, making ``S_iface`` overdetermined but consistent.
    The consistency is asserted; a nonzero residual would mean the boundary
    cannot represent the constraint and the reduction would be silently wrong.

    :param rows: the union rows the interface boundary was built from
    :param R: (n_rows, 6) their IDM, from :func:`vpt_idm`
    :return: T_red (n_boundary, 6 + n_io)
    """
    iface = boundary.interface_mask
    S_iface = vpt_selection(rows, boundary)[:, iface]
    G = np.linalg.pinv(S_iface) @ R                     # (n_iface_boundary, 6)

    residual = np.abs(S_iface @ G - R).max()
    assert residual < 1e-9 * max(1.0, np.abs(R).max()), (
        f"rigid constraint is not representable in the retained boundary "
        f"(residual {residual:g}) -- a workbook direction is missing from it")

    n_io = int(np.count_nonzero(~iface))
    T_red = np.zeros((boundary.n_boundary, 6 + n_io))
    T_red[iface, :6] = G
    T_red[~iface, 6:] = np.eye(n_io)                    # I/O DoFs stay independent
    return T_red


def partition_blocks(K, M, perm, n_b):
    """Boundary-first partition WITHOUT condensation -- the RBE3 counterpart of
    :func:`apply_rbe2`. The interface DoFs stay the boundary set (RBE3 does not
    reduce them), so the boundary block is the full (3nb, 3nb) sub-matrix.
    Returns the same dict layout as :func:`apply_rbe2` so :func:`craig_bampton`
    is shared by both methods.
    """
    nb3 = 3 * n_b

    def blocks(A):
        A_s = A[perm][:, perm]
        A_bb = A_s[:nb3, :nb3].toarray()          # (3nb, 3nb) dense
        A_bi = A_s[:nb3, nb3:].toarray()          # (3nb, n_i) dense
        A_ii = A_s[nb3:, nb3:].tocsc()            # sparse for splu/eigsh
        return A_bb, A_bi, A_ii

    K_bb, K_bi, K_ii = blocks(K)
    M_bb, M_bi, M_ii = blocks(M)
    return dict(K_bb=K_bb, K_bi=K_bi, K_ii=K_ii,
                M_bb=M_bb, M_bi=M_bi, M_ii=M_ii)



def craig_bampton(blocks, n_modes):
    """
    Craig-Bampton reduction. The boundary set is either the 6 RBE2 master DoFs
    (:func:`apply_rbe2`) or the 3*n_b flexible interface DoFs (RBE3,
    :func:`partition_blocks`); its size ``nb`` is inferred from ``K_bb``, so the
    same code serves both methods.

    Static constraint modes (unit master motion, interior follows statically):

        Psi (ni, 6) = -splu(K_ii).solve(K_ib)        with K_ib = K_bi.T (dense)

    Fixed-interface vibration modes (boundary clamped; K_ii is nonsingular now):

        lam, Phi = eigsh(K_ii, k=n_modes, M=M_ii, sigma=0)
        mass-normalize:  Phi /= sqrt(diag(Phi.T @ M_ii @ Phi))

    Reduction basis  u = R q_r,  q_r = [q_m (6); eta (n_modes)]:

        R = [[I6, 0], [Psi, Phi]]

    Reduced matrices, assembled block-wise (cheaper and clearer than R.T@()@R):

        K_r = [[K_bb + K_bi @ Psi,        0        ],
               [        0,           diag(lam)     ]]
        M_bb_r = M_bb + M_bi@Psi + Psi.T@M_bi.T + Psi.T@(M_ii@Psi)
        M_bm_r = M_bi@Phi + Psi.T@(M_ii@Phi)
        M_r = [[M_bb_r,   M_bm_r        ],
               [M_bm_r.T, eye(n_modes)  ]]

    Acceptance (review): M_r modal block == I and K_r modal block == diag(lam)
    to ~1e-8; boundary-modal coupling of K_r exactly 0 by construction; free-
    master eigenfrequencies of (K_r, M_r) match the full RBE2-transformed
    substructure to < 0.5 % for the first ~15 elastic modes.

    :param blocks: output of :func:`apply_rbe2`
    :param n_modes: number of fixed-interface modes to keep
    :return: (M_r, K_r, Psi, Phi, f_fixed_hz) with f_fixed_hz = sqrt(lam)/2pi
    """

    K_bb, K_bi, K_ii = blocks["K_bb"], blocks["K_bi"], blocks["K_ii"]
    M_bb, M_bi, M_ii = blocks["M_bb"], blocks["M_bi"], blocks["M_ii"]
    nb = K_bb.shape[0]                                   # 6 (RBE2) or 3*n_b (RBE3)

    # static constraint modes: unit master motion, interior follows statically
    Psi = -splu(K_ii).solve(K_bi.T)                     # (n_i, nb)

    # fixed-interface modes of the clamped interior (K_ii nonsingular), sorted
    # ascending and mass-normalized so that Phi.T M_ii Phi == I exactly
    lam, Phi = eigsh(K_ii, k=n_modes, M=M_ii, sigma=0)
    order = np.argsort(lam)
    lam, Phi = lam[order], Phi[:, order]
    Phi = Phi / np.sqrt(np.diag(Phi.T @ (M_ii @ Phi)))

    # reduced matrices: R^T () R with R = [[I,0],[Psi,Phi]], multiplied out
    # (see docstring; the zero coupling and diag(lam) are exact by construction)
    K_r = np.block([[K_bb + K_bi @ Psi, np.zeros((nb, n_modes))],
                    [np.zeros((n_modes, nb)), np.diag(lam)]])

    M_bb_r = M_bb + M_bi @ Psi + Psi.T @ M_bi.T + Psi.T @ (M_ii @ Psi)
    M_bm_r = M_bi @ Phi + Psi.T @ (M_ii @ Phi)
    M_r = np.block([[M_bb_r, M_bm_r],
                    [M_bm_r.T, np.eye(n_modes)]])

    f_fixed_hz = np.sqrt(lam) / (2.0 * np.pi)
    return M_r, K_r, Psi, Phi, f_fixed_hz



def modal_damping_matrix(M_r, K_r, zeta, f_rbm_tol=1.0):
    """
    Viscous damping equivalent to ``zeta`` modal damping on every elastic mode
    of the reduced substructure (rigid-body modes stay undamped):

        C_r = M_r @ V @ diag(2 zeta w) @ V.T @ M_r,   w_k < 2*pi*f_rbm_tol -> 0

    with (w^2, V) = eigh(K_r, M_r), V mass-normalized. Matches the 0.3 % modal
    damping of the pyFBS FRF synthesis per substructure.
    """
    from scipy.linalg import eigh

    lam, V = eigh(K_r, M_r)                     # V is M_r-orthonormal
    w = np.sqrt(np.clip(lam, 0.0, None))
    w[w < 2.0 * np.pi * f_rbm_tol] = 0.0        # rigid-body modes stay undamped
    return M_r @ V @ np.diag(2.0 * zeta * w) @ V.T @ M_r


@dataclass
class ReducedSubstructure:
    """One Craig-Bampton reduced substructure, in one of four condensations.

    Whole-node boundary (all 3 translations of every interface node retained):
      "rbe2"        rigid interface condensed to the 6-DoF VP; q_r = [q_m; eta]
      "rbe3"        interface kept flexible; q_r = [q_Gamma (3nb); eta]

    Directional boundary (only the DoFs the workbook names, plus excitation and
    output -- see :func:`build_directional_boundary`):
      "RBE_rigid"   those DoFs condensed rigidly to the VP; q_r = [q_m; q_io; eta]
      "RBE_average" they stay free; q_r = [a (n_boundary); eta]

    ``vp_operator`` observes the 6 VP DoFs from q_r and ``vp_load`` spreads a VP
    wrench back onto it. They are transposes of one another in every mode EXCEPT
    "RBE_average", where Tu and Tf are built from different workbook rows -- which
    is precisely the asymmetry pyFBS's VPT has and the reciprocal RBE3 map has not.
    """
    name: str
    M_r: np.ndarray            # (nr, nr), nr = nb + n_modes
    C_r: np.ndarray            # (nr, nr)
    K_r: np.ndarray            # (nr, nr)
    T_b: np.ndarray            # (3nb, 6) rigid-body map D_Gamma of the boundary nodes
    Psi: np.ndarray            # (ni, nb) static constraint modes
    Phi: np.ndarray            # (ni, n_modes) fixed-interface modes
    nodes: np.ndarray          # (n, 3) all node coordinates
    boundary_idx: np.ndarray   # (nb,) 0-based RETAINED node indices: joint
                               # interface first, then any attachment nodes
    internal_idx: np.ndarray   # (n - nb,) remaining node indices
    condensation: str          # "rbe2" | "rbe3" | "RBE_rigid" | "RBE_average"
    vp_operator: np.ndarray    # (6, nr) coupling map: q_m = vp_operator @ q_r
    n_interface: int           # # of joint-interface nodes (first entries of
                               # boundary_idx); the remainder are attachment DoFs
    vp_load: np.ndarray        # (nr, 6) load map: f_r = vp_load @ w_VP
    directional: object = None  # DirectionalBoundary for the RBE_* modes, else None
    T_red: np.ndarray = None   # (n_boundary, 6+n_io) RBE_rigid constraint, else None

    @classmethod
    def build(cls, name, data, boundary_idx, master_xyz, n_modes, zeta,
              condensation="rbe2", weights=None, attachment_idx=None,
              vpt_rows=None, io_rows=None, wu=None, wf=None):
        """
        Full reduction of one substructure: interface transformation -> boundary-
        first partition -> Craig-Bampton -> modal damping.

        :param name: "A" or "B" (report label)
        :param data: dict from :func:`load_or_export`
        :param boundary_idx: interface node indices (from get_boundary_nodes);
            used by the whole-node modes only
        :param master_xyz: (3,) master / VP position
        :param n_modes: fixed-interface modes to keep
        :param zeta: modal damping ratio per elastic mode
        :param condensation: see the class docstring -- "rbe2", "rbe3",
            "RBE_rigid" or "RBE_average"
        :param weights: RBE3 weighting W (see :func:`rbe3_vp_operator`); ignored
            by every other mode.
        :param attachment_idx: extra nodes retained in the CB boundary set (e.g. a
            load point) for static completeness, but excluded from the joint
            interface (zero columns in the VP map). Whole-node modes only; the
            directional modes retain ``io_rows`` instead. None -> interface only.
        :param vpt_rows: dict from :func:`read_vpt_rows` -- REQUIRED by the
            directional modes, ignored by the whole-node ones.
        :param io_rows: (name, node, direction) of the excitation / output DoFs to
            retain, directional modes only.
        :param wu: response weighting of the VPT (RBE_average only), pyFBS ``wu``.
        :param wf: force weighting of the VPT (RBE_average only), pyFBS ``wf``.
            Deliberately independent of ``wu`` -- see :func:`vpt_transformations`.
        """
        nodes = data["nodes"]
        interface_idx = np.asarray(boundary_idx)         # joint interface (Gamma)
        n_iface = len(interface_idx)

        if condensation in ("RBE_rigid", "RBE_average"):
            return cls._build_directional(
                name, data, master_xyz, n_modes, zeta, condensation,
                vpt_rows, io_rows or [], wu, wf)

        # attachment DoFs: extra nodes retained in the CB boundary set (e.g. a
        # load point) so their static response is captured exactly, but which
        # are NOT on the joint interface and so get zero columns in the VP map.
        if attachment_idx is None or len(attachment_idx) == 0:
            attachment_idx = np.array([], dtype=int)
        else:
            attachment_idx = np.setdiff1d(np.asarray(attachment_idx), interface_idx)
        retained_idx = np.concatenate([interface_idx, attachment_idx]).astype(int)
        n_ret = len(retained_idx)

        T_b = rbe2_transformation(nodes, interface_idx, master_xyz)  # D_Gamma (joint)
        perm, internal_idx = partition_dofs(len(nodes), retained_idx)

        if condensation == "rbe2":
            blocks = apply_rbe2(data["K"], data["M"], T_b, perm, n_iface,
                                n_att=len(attachment_idx))
            vp_boundary = np.zeros((6, 6 + 3 * len(attachment_idx)))
            vp_boundary[:, :6] = np.eye(6)              # VP = the 6 master DoFs
        elif condensation == "rbe3":
            blocks = partition_blocks(data["K"], data["M"], perm, n_ret)
            GT = rbe3_vp_operator(nodes, interface_idx, master_xyz,
                                  weights)              # G^T: (6, 3*n_iface)
            vp_boundary = np.zeros((6, 3 * n_ret))      # attachment cols stay zero
            vp_boundary[:, :3 * n_iface] = GT
        else:
            raise ValueError(
                f"unknown condensation {condensation!r} (expected 'rbe2', "
                f"'rbe3', 'RBE_rigid' or 'RBE_average')")

        M_r, K_r, Psi, Phi, f_fixed_hz = craig_bampton(blocks, n_modes)
        C_r = modal_damping_matrix(M_r, K_r, zeta)

        # coupling map q_m = vp_operator @ q_r, q_r = [boundary; eta]; the modal
        # part never enters the VP, so its columns are zero.
        vp_operator = np.hstack([vp_boundary, np.zeros((6, n_modes))])

        n_att = n_ret - n_iface
        print(f"[{name}] {condensation.upper()} reduced {3 * len(nodes)} -> "
              f"{M_r.shape[0]} DoFs | fixed-interface modes "
              f"{f_fixed_hz[0]:.1f}..{f_fixed_hz[-1]:.1f} Hz"
              + (f" | +{n_att} attachment node(s) retained" if n_att else ""))
        # rbe2 and rbe3 are reciprocal by construction (see the class docstring),
        # so the load map is simply the transpose of the observation map.
        return cls(name=name, M_r=M_r, C_r=C_r, K_r=K_r, T_b=T_b, Psi=Psi,
                   Phi=Phi, nodes=nodes, boundary_idx=retained_idx,
                   internal_idx=internal_idx, condensation=condensation,
                   vp_operator=vp_operator, n_interface=n_iface,
                   vp_load=vp_operator.T)

    @classmethod
    def _build_directional(cls, name, data, master_xyz, n_modes, zeta,
                           condensation, vpt_rows, io_rows, wu, wf):
        """
        The RBE_rigid / RBE_average reduction: keep only the DoFs the workbook
        names (plus excitation and output), then either tie the interface ones
        rigidly to the virtual point or observe/load it through pyFBS's Tu / Tf.

        Both modes share the boundary, so they differ ONLY in the virtual-point
        map -- which is what makes a comparison between them a clean measurement
        of the rigid-vs-averaged interface assumption.
        """
        assert vpt_rows is not None, (
            f"condensation {condensation!r} needs vpt_rows -- call read_vpt_rows("
            f"xlsx, '{name}', nodes) and pass the result")
        nodes = data["nodes"]

        # the UNION of channel and impact rows: the boundary must carry both, and
        # staying with the union keeps it invariant when the sheets are unified.
        union = vpt_rows["channels"] + vpt_rows["impacts"]
        boundary = build_directional_boundary(union, io_rows, len(nodes))
        blocks = rotate_and_partition(data["K"], data["M"], boundary)

        Ru = vpt_idm(nodes, vpt_rows["channels"], master_xyz)
        Rf = vpt_idm(nodes, vpt_rows["impacts"], master_xyz)
        n_io = int(np.count_nonzero(~boundary.interface_mask))
        T_red = None

        if condensation == "RBE_rigid":
            T_red = rigid_boundary_map(union, boundary,
                                       vpt_idm(nodes, union, master_xyz))
            blocks = condense_boundary(blocks, T_red)
            vp_boundary = np.zeros((6, 6 + n_io))
            vp_boundary[:, :6] = np.eye(6)               # VP = the 6 master DoFs
            vp_boundary_load = vp_boundary.T             # rigid MPC is reciprocal
        else:                                            # RBE_average
            Tu, Tf = vpt_transformations(Ru, Rf, wu, wf)
            vp_boundary = Tu @ vpt_selection(vpt_rows["channels"], boundary)
            vp_boundary_load = vpt_selection(vpt_rows["impacts"], boundary).T @ Tf

        M_r, K_r, Psi, Phi, f_fixed_hz = craig_bampton(blocks, n_modes)
        C_r = modal_damping_matrix(M_r, K_r, zeta)

        vp_operator = np.hstack([vp_boundary, np.zeros((6, n_modes))])
        vp_load = np.vstack([vp_boundary_load, np.zeros((n_modes, 6))])

        n_if_dof = int(np.count_nonzero(boundary.interface_mask))
        print(f"[{name}] {condensation} reduced {3 * len(nodes)} -> "
              f"{M_r.shape[0]} DoFs | fixed-interface modes "
              f"{f_fixed_hz[0]:.1f}..{f_fixed_hz[-1]:.1f} Hz | "
              f"{n_if_dof} directional interface DoFs over "
              f"{len(boundary.node_idx) - n_io} nodes"
              + (f" + {n_io} I/O DoF(s)" if n_io else ""))
        return cls(name=name, M_r=M_r, C_r=C_r, K_r=K_r,
                   T_b=rbe2_transformation(nodes, boundary.node_idx, master_xyz),
                   Psi=Psi, Phi=Phi, nodes=nodes,
                   boundary_idx=boundary.node_idx, internal_idx=boundary.i_idx,
                   condensation=condensation, vp_operator=vp_operator,
                   n_interface=n_if_dof, vp_load=vp_load,
                   directional=boundary, T_red=T_red)

    def recovery_row(self, position, direction):
        """
        Row t (nr,) of the physical<->reduced map for the scalar DoF
        "displacement at ``position`` in ``direction``": u = t @ q_r, and by
        the transpose of the same map a point force F*direction there enters
        the reduced equations as f_r = t * F.

        The position snaps to the nearest FE node (like pyfbs
        update_locations_df). Internal node -> direction projected onto its
        [Psi | Phi] rows; interface node -> its T_b rows for RBE2 (moves rigidly
        with the master) or its own retained DoFs for RBE3. The directional modes
        take :meth:`_recovery_row_directional`, where a node can be retained in
        SOME directions only.
        """
        pos = np.asarray(position, dtype=float)
        dvec = np.asarray(direction, dtype=float)
        dist = np.linalg.norm(self.nodes - pos, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 5e-3:
            print(f"[{self.name}] recovery_row: snapped {dist[j] * 1e3:.2f} mm "
                  f"to node {j} -- check the position")

        if self.directional is not None:
            return self._recovery_row_directional(j, dvec)

        row = np.zeros(self.M_r.shape[0])
        nb = self.Psi.shape[1]                         # boundary block width
        hit = np.nonzero(self.boundary_idx == j)[0]
        if hit.size:                                   # interface (boundary) node
            p = int(hit[0])
            if self.condensation == "rbe2":
                if p < self.n_interface:               # rigid slave of the 6-DoF master
                    row[:6] = dvec @ self.T_b[3 * p:3 * p + 3, :]
                else:                                  # retained attachment DoF
                    a = p - self.n_interface
                    row[6 + 3 * a:6 + 3 * a + 3] = dvec
            else:                                      # RBE3: node DoFs retained
                row[3 * p:3 * p + 3] = dvec
        else:                                          # internal node
            q = int(np.searchsorted(self.internal_idx, j))
            assert self.internal_idx[q] == j
            row[:nb] = dvec @ self.Psi[3 * q:3 * q + 3, :]
            row[nb:] = dvec @ self.Phi[3 * q:3 * q + 3, :]
        return row

    def _recovery_row_directional(self, j, dvec):
        """
        :meth:`recovery_row` for the RBE_rigid / RBE_average boundaries, where a
        node may be retained in only some of its directions.

        In the rotated frame of a retained node, u_j = C_j^T a_j + N_j^T b_j with
        a_j the retained (boundary) coordinates and b_j the free ones, which sit
        in the INTERIOR partition. So a direction picks up both parts:

            d.u_j = (C_j d).a_j + (N_j d).b_j

        For RBE_rigid the boundary was condensed further, a = T_red q_b, so the
        first term goes through T_red. A node that is not retained at all has all
        three DoFs in the interior and behaves like the whole-node modes.
        """
        row = np.zeros(self.M_r.shape[0])
        nb = self.Psi.shape[1]                         # boundary block width
        b = self.directional

        def interior(rows, weights):
            """accumulate weights . [Psi | Phi][rows] into the row"""
            row[:nb] += weights @ self.Psi[rows, :]
            row[nb:] += weights @ self.Phi[rows, :]

        if int(j) in b.slot:                           # partially/fully retained
            a_part = b.C[int(j)] @ dvec                # (k_j,) boundary weights
            if self.T_red is not None:                 # RBE_rigid: a = T_red q_b
                row[:nb] += a_part @ self.T_red[b.slot[int(j)], :]
            else:                                      # RBE_average: a IS q_b
                row[b.slot[int(j)]] += a_part
            free = b.N[int(j)]
            if free.size:                              # the unretained directions
                interior(b.free_slot(j), free @ dvec)
        else:                                          # fully interior node
            rows = np.searchsorted(b.i_idx, 3 * int(j) + np.arange(3))
            assert np.array_equal(b.i_idx[rows], 3 * int(j) + np.arange(3)), \
                f"node {j} is neither retained nor fully interior"
            interior(rows, dvec)
        return row

    def interface_recovery(self):
        """
        Map U (3*nb_nodes, nr) from reduced coordinates to the PHYSICAL
        interface node displacements, u_Gamma = U @ q_r, ordered like
        ``boundary_idx`` (x, y, z per node). RBE2: the rigid expansion T_b of
        the 6 master DoFs (the inverse of the RBE2 condensation); RBE3:
        identity on the retained interface DoFs. The directional modes rebuild
        each node from its retained and free parts, so unlike RBE2/RBE3 their
        modal columns are NOT zero -- a partially retained node still moves with
        the fixed-interface modes along its unretained directions.
        """
        if self.directional is not None:
            b = self.directional
            iface_nodes = [int(j) for k, j in enumerate(b.node_idx)
                           if b.interface_mask[b.slot[int(j)]].any()]
            U = np.zeros((3 * len(iface_nodes), self.M_r.shape[0]))
            for p, j in enumerate(iface_nodes):
                U[3 * p:3 * p + 3, :] = np.vstack(
                    [self._recovery_row_directional(j, e) for e in np.eye(3)])
            return U

        if self.condensation == "rbe2":
            n_att3 = self.Psi.shape[1] - 6             # attachment cols in boundary block
            U_b = np.hstack([self.T_b, np.zeros((self.T_b.shape[0], n_att3))])
        else:                                          # RBE3: joint-interface DoFs only
            n_if3 = 3 * self.n_interface
            U_b = np.zeros((n_if3, self.Psi.shape[1]))
            U_b[:, :n_if3] = np.eye(n_if3)
        return np.hstack([U_b, np.zeros((U_b.shape[0], self.Phi.shape[1]))])



def assemble_coupled(sub_A, sub_B):
    """
    Assemble the two reduced substructures WITHOUT any joint terms.

    Coordinates q = [q_mA (6), eta_A | q_mB (6), eta_B]; M, C, K stay purely
    block diagonal (linearly uncoupled). The ENTIRE joint force -- linear
    spring k and damper c as well as the cubic terms -- is applied by
    :class:`CoupledCubicCB` through the pyhbm nonlinear term on

        x_r = Bc q  (6 relative master DoFs, +1 on A, -1 on B -- same signs
        as the pyFBS example's signed-Boolean matrix)

    The linear terms are degree-1 polynomials, so the AFT evaluation at the
    polynomial_degree=3 sampling stays exact. The damper still matters: the
    joint (rigid-body-vs-spring) modes carry no substructure modal damping,
    and c_diag keeps them damped through jacobian_nonlinear_term_qdot --
    without it their hardened resonances stall the continuation.

    :param sub_A/sub_B: :class:`ReducedSubstructure`
    :return: (M, C, K, Bc, Bc_load) dense, d = nrA + nrB
    """
    from scipy.linalg import block_diag

    nrA = sub_A.M_r.shape[0]
    d = nrA + sub_B.M_r.shape[0]

    # x_r = q_mA - q_mB through each substructure's VP operator (q_m = P q_r).
    # RBE2: P = [I6 | 0] reproduces the signed-Boolean master coupling exactly;
    # RBE3: P = [G^T | 0] gathers the VP from the flexible interface DoFs;
    # RBE_average: P = [Tu S_u | 0] reads the VP off the sensor channels alone.
    Bc = np.zeros((6, d))
    Bc[:, :nrA] = sub_A.vp_operator          # +A VP
    Bc[:, nrA:] = -sub_B.vp_operator         # -B VP

    # ...and the reverse map, spreading the joint wrench back onto q. It is NOT
    # simply Bc.T for RBE_average: there the VP is observed through the channel
    # rows but loaded through the impact rows, which are different DoFs. The
    # other three condensations are reciprocal, so Bc_load == Bc.T for them.
    Bc_load = np.zeros((d, 6))
    Bc_load[:nrA] = sub_A.vp_load
    Bc_load[nrA:] = -sub_B.vp_load

    M = block_diag(sub_A.M_r, sub_B.M_r)
    C = block_diag(sub_A.C_r, sub_B.C_r)
    K = block_diag(sub_A.K_r, sub_B.K_r)
    return M, C, K, Bc, Bc_load


# ===========================================================================
# Physical recovery + CSV export
#
# Shared by the testbench_*_CB examples: the cubic-spring and the dry-friction
# main differ only in their joint law, i.e. in export_header -- everything
# below is joint-independent.
# ===========================================================================

VP_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")

# How each condensation defines the two exported physical DoF families, for the
# CSV comment header. Shared by the testbench_*_CB examples so the four
# descriptions cannot drift apart between them.
CONDENSATION_HEADER_TEXT = {
    "rbe2": (
        "the RBE2 master DoFs (= the condensed CB boundary block)",
        "u_Gamma = T_b q_m -- rigid expansion of the VP master (inverse RBE2)"),
    "rbe3": (
        "the RBE3 weighted interface average G^T q_Gamma",
        "the retained (flexible) RBE3 interface DoFs = CB boundary block, no "
        "expansion needed"),
    "RBE_rigid": (
        "the RBE_rigid master DoFs -- a rigid MPC on the workbook's "
        "channel/impact directions only",
        "u_Gamma rebuilt from the retained directions (rigidly tied to the "
        "master) plus the free ones via [Psi | Phi]"),
    "RBE_average": (
        "pyFBS's VPT response map Tu applied to the sensor channels (loads go "
        "back through Tf on the impact rows -- not its transpose)",
        "u_Gamma rebuilt from the retained directional DoFs plus the free ones "
        "via [Psi | Phi]"),
}


def condensation_header_text(condensation):
    """(vp_txt, iface_txt) describing how ``condensation`` defines the exported
    ``*_vp_*`` and ``*_n<id>_u*`` columns."""
    try:
        return CONDENSATION_HEADER_TEXT[condensation]
    except KeyError:
        raise ValueError(f"unknown condensation {condensation!r}") from None


def nearest_node(data, position):
    """(ansys_id, xyz) of the FE node closest to ``position`` -- the same snap
    rule as ReducedSubstructure.recovery_row, used to document where each
    exported channel / the drive point lands on the FE mesh."""
    j = int(np.argmin(np.linalg.norm(data["nodes"] - np.asarray(position), axis=1)))
    return int(data["nnum"][j]), data["nodes"][j]


def read_channels(xlsx_path, substructure):
    """
    Every directional response channel of one substructure from the pyFBS
    workbook (sheet Channels_<substructure>, all groupings -- reference and
    interface sensors alike): one (label, name, grouping, position, direction)
    tuple per row, label = the sheet's Name without the blank ("S1 X" -> "S1X").
    The raw name and grouping are kept only for the CSV header, which mirrors
    the pyFBS export format.
    """
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name=f"Channels_{substructure}")
    return [(str(row["Name"]).replace(" ", ""), str(row["Name"]), row.get("Grouping"),
             row[["Position_1", "Position_2", "Position_3"]].to_numpy(float),
             row[["Direction_1", "Direction_2", "Direction_3"]].to_numpy(float))
            for _, row in df.iterrows()]


def channel_snap_info(data, channels):
    """:func:`read_channels` tuples extended by (snapped ansys node id, snap
    distance [mm]) -- documents in the CSV header where each channel lands on
    the FE mesh (sensor housings sit a few mm above the surface)."""
    info = []
    for label, raw, grouping, pos, dvec in channels:
        nid, xyz = nearest_node(data, pos)
        info.append((label, raw, grouping, pos, dvec, nid,
                     1e3 * np.linalg.norm(xyz - pos)))
    return info


def channel_header_lines(name, chan_info):
    """
    Per-channel CSV header block, written in the pyFBS export format

        <label>: '<workbook name>' grouping <g> at (x, y, z) m, direction [...]

    so that pyFBS's plot_diagnostics_comparison.py can place the exported DoFs
    in space, and extended by the FE node each channel snapped to.
    """
    lines = [f"channels {name}: {len(chan_info)} (sheet Channels_{name},"
             f" xlsx position -> snapped FE node)"]
    lines += [f"  {name}_{lab}: {raw!r} grouping {grp} at"
              f" ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m, direction"
              f" [{d[0]:.7f}, {d[1]:.7f}, {d[2]:.7f}]"
              f" -> node {name}_n{nid} (snap {snap:.2f} mm)"
              for lab, raw, grp, p, d, nid, snap in chan_info]
    return lines


def output_channel_label(channels, position, direction, prefix="A"):
    """Label of the descriptor's output channel among ``channels`` (exact
    position/direction match) -- the source of the plotted uout_* curves."""
    for label, _, _, pos, dvec in channels:
        if (np.allclose(pos, position, atol=1e-8)
                and np.allclose(dvec, direction, atol=1e-6)):
            return f"{prefix}_{label}"
    raise ValueError("descriptor output channel not found in the channel list "
                     "-- cannot define the plotted uout column")


def physical_recovery(sub_A, sub_B, t_in, ids_A, ids_B, chan_A, chan_B):
    """
    Reduced -> physical map of the exported solution set: returns (labels, T)
    with u_phys = T @ q, q = [q_rA | q_rB] the coupled reduced coordinates and
    one clearly named label per row of T. Exported DoFs, in column order:

      uin                 drive-point displacement (B), recovered through the
                          Craig-Bampton basis [Psi | Phi] (inverse CB)
      {A,B}_<Sn><XYZ>     every pyFBS response channel (read_channels), same
                          inverse-CB recovery at the snapped FE node
      {A,B}_vp_{ux..rz}   6-DoF virtual-point motion: the RBE2 master DoFs,
                          resp. the RBE3 weighted average G^T q_Gamma
      {A,B}_n<id>_u{xyz}  physical interface node displacements: inverse RBE2
                          u_Gamma = T_b q_m, resp. the retained RBE3 interface
                          DoFs; <id> is the original Ansys node id
    """
    nrA = sub_A.M_r.shape[0]
    dim = nrA + sub_B.M_r.shape[0]

    def embedded(rows, offset):            # substructure rows -> coupled q
        full = np.zeros((rows.shape[0], dim))
        full[:, offset:offset + rows.shape[1]] = rows
        return full

    labels, blocks = ["uin"], [t_in[None, :]]
    for sub, chans, offset in ((sub_A, chan_A, 0), (sub_B, chan_B, nrA)):
        labels += [f"{sub.name}_{label}" for label, _, _, _, _ in chans]
        blocks.append(embedded(np.array([sub.recovery_row(pos, dvec)
                                         for _, _, _, pos, dvec in chans]), offset))
    for sub, offset in ((sub_A, 0), (sub_B, nrA)):
        labels += [f"{sub.name}_vp_{dof}" for dof in VP_DOFS]
        blocks.append(embedded(sub.vp_operator, offset))
    for sub, ids, offset in ((sub_A, ids_A, 0), (sub_B, ids_B, nrA)):
        labels += [f"{sub.name}_n{int(nid)}_u{ax}" for nid in ids for ax in "xyz"]
        blocks.append(embedded(sub.interface_recovery(), offset))
    return labels, np.vstack(blocks)


def save_physical_solution(solution_set, solve_time, csv_out, labels, T,
                           header_lines, out_label):
    """
    Export the branch in PHYSICAL coordinates, one row per continuation point:
    freq/omega, corrector diagnostics, the two plotted curves of the output
    channel ``out_label`` and, per harmonic h and exported DoF (see
    physical_recovery), the complex amplitude a_h as a re/im pair, normalized
    such that u(t) = Re(sum_h a_h e^{1j h w t}) -- i.e. |a_h| is the physical
    amplitude of harmonic h, NOT the raw rFFT-scaled solver coefficient. Read
    back with pandas.read_csv(csv_out, comment="#"); the comment header
    documents every column. Returns (freq_hz, uout_h1_abs, uout_time_max).
    """
    from numpy.fft import irfft

    from pyhbm import Fourier

    harmonics = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples

    # (n, Nh, dim) raw rFFT-convention coefficients -> physical (n, Nh, n_phys);
    # 2/n_t (1/n_t for a DC harmonic) rescales to physical amplitudes a_h
    raw = np.array([f.coefficients[:, :, 0] for f in solution_set.fourier])
    n = raw.shape[0]
    u_raw = raw @ T.T
    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])
    amp = u_raw * scale[None, :, None]

    # plotted curves: |a_1| of the output channel and the one-period peak of
    # its time signal, same irfft/sampling as pyhbm's Fourier_Real
    i_out, i_h1 = labels.index(out_label), harmonics.index(1)
    padded = np.zeros((n, max(harmonics) + 1), dtype=complex)
    padded[:, harmonics] = u_raw[:, :, i_out]
    uout_time_max = np.abs(irfft(padded, n=n_t, axis=1)).max(axis=1)
    uout_h1_abs = np.abs(amp[:, i_h1, i_out])

    omega = np.asarray(solution_set.omega, dtype=float)
    freq = omega / (2.0 * np.pi)
    reim = np.stack([amp.real, amp.imag], axis=-1)     # re/im adjacent per DoF
    data = np.column_stack([
        freq, omega,
        np.asarray(solution_set.iterations, dtype=float),
        np.asarray(solution_set.step_length, dtype=float),
        uout_h1_abs, uout_time_max,
        reim.reshape(n, -1),                           # harmonic-major, then DoF
    ])
    cols = (["freq_hz", "omega_rad_s", "iterations", "step_length",
             "uout_h1_abs_m", "uout_time_max_m"]
            + [f"{p}_h{h}_{lab}" for h in harmonics for lab in labels
               for p in ("re", "im")])
    assert data.shape[1] == len(cols)

    with open(csv_out, "w", newline="") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        fh.write(f"# uout_time_max_m: max |{out_label}(t)| over the {n_t} AFT"
                 f" time samples of one period\n")
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, data, delimiter=",", fmt="%.10e")
    print(f"physical solution written: {csv_out}  ({n} points, {len(labels)}"
          f" DoFs x {len(harmonics)} harmonics, {solve_time:.1f} s solve)")
    return freq, uout_h1_abs, uout_time_max


# ===========================================================================
# WP6 -- coupled pyhbm system
# ===========================================================================

try:                                     # pyhbm is not pip-installed: resolve
    from pyhbm import SecondOrderODE     # the repo's src/ relative to this file
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from pyhbm import SecondOrderODE


class CoupledCubicCB(SecondOrderODE):
    """
    The coupled reduced testbench as a pyhbm second-order system:

        M q'' + C q' + K q
            + Bc_load (k * x + c * xdot + alpha * x^3 + beta * xdot^3)
            = f_r F0 cos(tau)

    with x = Bc q the 6 relative master DoFs (VP_A - VP_B). M, C, K are the
    linearly UNCOUPLED block-diagonal matrices (see assemble_coupled); the
    complete joint -- linear spring k, linear damper c and the cubic terms --
    enters through the nonlinear term, the same bushing law as the pyFBS
    example's TestbenchCubicSpring. qdot passed by pyhbm is the PHYSICAL
    velocity, so the xdot terms need no extra omega scaling.

    ``Bc_load`` defaults to Bc.T, which is exact for every reciprocal
    condensation (rbe2, rbe3, RBE_rigid). RBE_average observes the virtual point
    through the sensor channels and loads it through the impact rows -- different
    DoFs -- so there it is a genuinely different matrix and the resulting
    Jacobian Bc_load @ J @ Bc is NOT symmetric.
    """
    is_real_valued = True

    def __init__(self, M, C, K, Bc, k_diag, c_diag, alpha_diag, beta_diag, f_r, F0,
                 Bc_load=None):
        self.mass_matrix = M
        self.damping_matrix = C
        self.stiffness_matrix = K
        self.dimension = M.shape[0]
        self.polynomial_degree = 3            # sets the AFT sampling (exact)
        self.Bc = Bc
        self.Bc_load = Bc.T if Bc_load is None else Bc_load
        self.k_diag = np.asarray(k_diag, dtype=float)
        self.c_diag = np.asarray(c_diag, dtype=float)
        self.alpha_diag = np.asarray(alpha_diag, dtype=float)
        self.beta_diag = np.asarray(beta_diag, dtype=float)
        self.f_r = np.asarray(f_r, dtype=float)
        self.F0 = float(F0)

    def external_term(self, adimensional_time):
        tau = np.asarray(adimensional_time)
        return (self.F0 * np.cos(tau))[:, None, None] * self.f_r[None, :, None]

    def nonlinear_term(self, q, q_dot, adimensional_time):
        x = np.einsum("ij,tjk->tik", self.Bc, q)          # (Nt, 6, 1)
        xd = np.einsum("ij,tjk->tik", self.Bc, q_dot)
        f_int = (self.k_diag[None, :, None] * x
                 + self.c_diag[None, :, None] * xd
                 + self.alpha_diag[None, :, None] * x ** 3
                 + self.beta_diag[None, :, None] * xd ** 3)
        return np.einsum("ij,tjk->tik", self.Bc_load, f_int)   # Bc_load f_int

    def jacobian_nonlinear_term(self, q, q_dot, adimensional_time):
        x = np.einsum("ij,tjk->tik", self.Bc, q)[:, :, 0]           # (Nt, 6)
        diag = self.k_diag[None, :] + 3.0 * self.alpha_diag[None, :] * x ** 2
        # Bc_load @ diag(.) @ Bc -- 'ij,tj,jk' contracts the diagonal in place
        return np.einsum("ij,tj,jk->tik", self.Bc_load, diag, self.Bc)  # (Nt, d, d)

    def jacobian_nonlinear_term_qdot(self, q, q_dot, adimensional_time):
        xd = np.einsum("ij,tjk->tik", self.Bc, q_dot)[:, :, 0]
        diag = self.c_diag[None, :] + 3.0 * self.beta_diag[None, :] * xd ** 2
        return np.einsum("ij,tj,jk->tik", self.Bc_load, diag, self.Bc)
