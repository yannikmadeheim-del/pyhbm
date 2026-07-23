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
    """One Craig-Bampton reduced substructure. Reduced coordinates q_r are
    [q_m (6); eta] for RBE2 (rigid interface condensed to the VP) or
    [q_Gamma (3nb); eta] for RBE3 (interface kept flexible); ``vp_operator`` maps
    q_r to the 6 VP DoFs used for coupling."""
    name: str
    M_r: np.ndarray            # (nr, nr), nr = nb + n_modes (nb = 6 RBE2 / 3nb RBE3)
    C_r: np.ndarray            # (nr, nr)
    K_r: np.ndarray            # (nr, nr)
    T_b: np.ndarray            # (3nb, 6) rigid-body map D_Gamma of the boundary nodes
    Psi: np.ndarray            # (ni, nb) static constraint modes
    Phi: np.ndarray            # (ni, n_modes) fixed-interface modes
    nodes: np.ndarray          # (n, 3) all node coordinates
    boundary_idx: np.ndarray   # (nb,) 0-based RETAINED node indices: joint
                               # interface first, then any attachment nodes
    internal_idx: np.ndarray   # (n - nb,) remaining node indices
    condensation: str          # "rbe2" or "rbe3"
    vp_operator: np.ndarray    # (6, nr) coupling map: q_m = vp_operator @ q_r
    n_interface: int           # # of joint-interface nodes (first entries of
                               # boundary_idx); the remainder are attachment DoFs

    @classmethod
    def build(cls, name, data, boundary_idx, master_xyz, n_modes, zeta,
              condensation="rbe2", weights=None, attachment_idx=None):
        """
        Full reduction of one substructure: interface transformation -> boundary-
        first partition -> Craig-Bampton -> modal damping.

        :param name: "A" or "B" (report label)
        :param data: dict from :func:`load_or_export`
        :param boundary_idx: interface node indices (from get_boundary_nodes)
        :param master_xyz: (3,) master / VP position
        :param n_modes: fixed-interface modes to keep
        :param zeta: modal damping ratio per elastic mode
        :param condensation: "rbe2" -> rigid interface condensed to the 6-DoF VP
            (reduced boundary); "rbe3" -> interface kept flexible as the boundary
            set, VP defined by the weighted map for coupling only.
        :param weights: RBE3 weighting W (see :func:`rbe3_vp_operator`); ignored
            for RBE2.
        :param attachment_idx: extra nodes retained in the CB boundary set (e.g. a
            load point) for static completeness, but excluded from the joint
            interface (zero columns in the VP map). None -> interface only.
        """
        nodes = data["nodes"]
        interface_idx = np.asarray(boundary_idx)         # joint interface (Gamma)
        n_iface = len(interface_idx)

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
            raise ValueError(f"unknown condensation {condensation!r} "
                             f"(expected 'rbe2' or 'rbe3')")

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
        return cls(name=name, M_r=M_r, C_r=C_r, K_r=K_r, T_b=T_b, Psi=Psi,
                   Phi=Phi, nodes=nodes, boundary_idx=retained_idx,
                   internal_idx=internal_idx, condensation=condensation,
                   vp_operator=vp_operator, n_interface=n_iface)

    def recovery_row(self, position, direction):
        """
        Row t (nr,) of the physical<->reduced map for the scalar DoF
        "displacement at ``position`` in ``direction``": u = t @ q_r, and by
        the transpose of the same map a point force F*direction there enters
        the reduced equations as f_r = t * F.

        The position snaps to the nearest FE node (like pyfbs
        update_locations_df). Internal node -> direction projected onto its
        [Psi | Phi] rows; interface node -> its T_b rows for RBE2 (moves rigidly
        with the master) or its own retained DoFs for RBE3.
        """
        pos = np.asarray(position, dtype=float)
        dvec = np.asarray(direction, dtype=float)
        dist = np.linalg.norm(self.nodes - pos, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 5e-3:
            print(f"[{self.name}] recovery_row: snapped {dist[j] * 1e3:.2f} mm "
                  f"to node {j} -- check the position")

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

    def interface_recovery(self):
        """
        Map U (3*nb_nodes, nr) from reduced coordinates to the PHYSICAL
        interface node displacements, u_Gamma = U @ q_r, ordered like
        ``boundary_idx`` (x, y, z per node). RBE2: the rigid expansion T_b of
        the 6 master DoFs (the inverse of the RBE2 condensation); RBE3:
        identity on the retained interface DoFs. The modal columns are zero --
        fixed-interface modes do not move the boundary by construction.
        """
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
    :return: (M, C, K, Bc) dense, d = nrA + nrB
    """
    from scipy.linalg import block_diag

    nrA = sub_A.M_r.shape[0]
    d = nrA + sub_B.M_r.shape[0]

    # x_r = q_mA - q_mB through each substructure's VP operator (q_m = P q_r).
    # RBE2: P = [I6 | 0] reproduces the signed-Boolean master coupling exactly;
    # RBE3: P = [G^T | 0] gathers the VP from the flexible interface DoFs.
    Bc = np.zeros((6, d))
    Bc[:, :nrA] = sub_A.vp_operator          # +A VP
    Bc[:, nrA:] = -sub_B.vp_operator         # -B VP

    M = block_diag(sub_A.M_r, sub_B.M_r)
    C = block_diag(sub_A.C_r, sub_B.C_r)
    K = block_diag(sub_A.K_r, sub_B.K_r)
    return M, C, K, Bc


# ===========================================================================
# Physical recovery + CSV export
#
# Shared by the testbench_*_CB examples: the cubic-spring and the dry-friction
# main differ only in their joint law, i.e. in export_header -- everything
# below is joint-independent.
# ===========================================================================

VP_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")


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
            + Bc^T (k * x + c * xdot + alpha * x^3 + beta * xdot^3)
            = f_r F0 cos(tau)

    with x = Bc q the 6 relative master DoFs (VP_A - VP_B). M, C, K are the
    linearly UNCOUPLED block-diagonal matrices (see assemble_coupled); the
    complete joint -- linear spring k, linear damper c and the cubic terms --
    enters through the nonlinear term, the same bushing law as the pyFBS
    example's TestbenchCubicSpring. qdot passed by pyhbm is the PHYSICAL
    velocity, so the xdot terms need no extra omega scaling.
    """
    is_real_valued = True

    def __init__(self, M, C, K, Bc, k_diag, c_diag, alpha_diag, beta_diag, f_r, F0):
        self.mass_matrix = M
        self.damping_matrix = C
        self.stiffness_matrix = K
        self.dimension = M.shape[0]
        self.polynomial_degree = 3            # sets the AFT sampling (exact)
        self.Bc = Bc
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
        return np.einsum("ji,tjk->tik", self.Bc, f_int)   # Bc^T f_int

    def jacobian_nonlinear_term(self, q, q_dot, adimensional_time):
        x = np.einsum("ij,tjk->tik", self.Bc, q)[:, :, 0]           # (Nt, 6)
        diag = self.k_diag[None, :] + 3.0 * self.alpha_diag[None, :] * x ** 2
        return np.einsum("ji,tj,jk->tik", self.Bc, diag, self.Bc)   # (Nt, d, d)

    def jacobian_nonlinear_term_qdot(self, q, q_dot, adimensional_time):
        xd = np.einsum("ij,tjk->tik", self.Bc, q_dot)[:, :, 0]
        diag = self.c_diag[None, :] + 3.0 * self.beta_diag[None, :] * xd ** 2
        return np.einsum("ji,tj,jk->tik", self.Bc, diag, self.Bc)
