"""
RBE2 + Craig-Bampton verification model for the pyFBS testbench_cubicSpring example.

Pipeline (procedure of "Substructuring in Commercial Tools", slide 10):
  1. Export full M and K of substructures A and B from the Ansys .full/.rst files.
  2. Partition the DoFs into boundary (interface) and internal sets.
  3. RBE2: rigidify each interface node set to a 6-DoF master at the virtual point.
  4. Craig-Bampton: static constraint modes + fixed-interface modes -> reduced model.
  5. Couple the reduced substructures through the same cubic joint as the pyFBS
     dual-FBS example and solve with pyhbm's second-order HBM (see main.py).

Work split:
  WP1 (Ansys export)            -- implemented (Claude)
  WP2 (interface node sets)     -- implemented (Claude); node lists exported
                                   from Ansys Mechanical named selections
  WP3 (RBE2)                    -- implemented (Yannik, reviewed)
  WP4 (Craig-Bampton)           -- implemented (Yannik + Claude)
  WP5 (recovery/assembly/checks)-- Claude
  WP6 (pyhbm system + HBM)      -- Claude

Every stub's docstring contains the exact math, shapes and the acceptance check
it must pass. All positions are in metres, global Ansys coordinate system.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh, splu


# ===========================================================================
# WP1 -- Ansys import: .full/.rst -> nodes, dof_ref, K, M   (implemented)
# ===========================================================================

def load_ansys_substructure(rst_path, full_path):
    """
    Read one substructure's FE data from the Ansys binary files.

    Mirrors pyfbs.mck.Model.from_ansys so both pipelines see the exact same
    matrices: load_km(sort=True) returns the UPPER-TRIANGULAR K and M, which
    are symmetrized here; the node ids are renumbered to consecutive 1..n.

    Row 3*i + d of K/M belongs to node i (0-based row of ``nodes``) and
    direction d (0=x, 1=y, 2=z) -- the asserts protect exactly this alignment.

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

    Works for constrained and free-free structures alike: with a free-free K
    (singular, 6 rigid-body modes) shift-invert must not factorize K itself,
    so a small negative sigma keeps (K - sigma*M) nonsingular while the
    eigenvalues nearest sigma are still the rigid-body + lowest elastic modes.

    Note the imported testbench: A is FIXED-BASE (its 88 support nodes come
    back as empty rows from the .full and are dropped at import -> no
    rigid-body modes, first mode 221.5 Hz), B is free-free (6 x ~0 Hz).
    """
    lam = eigsh(K.tocsc(), k=n, M=M.tocsc(), sigma=-1.0e3, which="LM",
                return_eigenvectors=False)
    return np.sort(np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi))


# ===========================================================================
# WP2 -- interface node sets (boundary partition)
# ===========================================================================

def read_vp_definition(csv_path, grouping=None):
    """
    Virtual-point / RBE2-master definition of THIS example (self-contained,
    no pyFBS dependency): vp_definition.csv mirrors the row structure of the
    pyFBS VP_Channels sheet -- one row per VP DoF (ux, uy, uz, rx, ry, rz)
    with Grouping, Position_1..3 [m] and Direction_1..3. Move the joint by
    editing the positions there.

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


def find_file_nodes(nodes, nnum, path):
    """
    Primary interface definition ("file"): a node list exported from an Ansys
    Mechanical named selection / node component (bore wall of the VP hole).

    Expected format: a header line ("Knotennummer" / "Node Number") followed
    by one Ansys node id per line; extra columns are ignored. The ids use the
    ORIGINAL Ansys numbering, which survives the chain cdb -> blocked cdb ->
    Mechanical unchanged (verified bit-identical), and are mapped onto 0-based
    rows of ``nodes`` via ``nnum``. Unknown ids abort hard -- they would mean
    Mechanical renumbered the mesh or an eliminated support node was selected.

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

    pos = {int(n): i for i, n in enumerate(nnum)}
    missing = [i for i in ids if i not in pos]
    assert not missing, (
        f"{path}: {len(missing)} node ids unknown to the imported model "
        f"(e.g. {missing[:5]}) -- renumbered mesh or eliminated nodes?")
    return np.unique([pos[i] for i in ids])


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
                       file_A=None, file_B=None):
    """
    The single switch point for the interface definition.

    method == "file":   node lists exported from Ansys Mechanical (file_A/B)
    method == "mating": coincident-node detection on both meshes
    method == "vpt":    VPT sensor/impact nodes per substructure

    :return: (idx_A, idx_B) -- 0-based boundary node indices per substructure
    """
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


# ===========================================================================
# WP3 -- RBE2: rigidify the boundary nodes to a 6-DoF master
# ===========================================================================

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


def apply_rbe2(K, M, T_b, perm, n_b):
    """Sortieren + RBE2-Kondensation der Boundary-DoFs auf den 6-DoF-Master.

    K_s = K[perm][:, perm]          # jetzt: [b-Block | i-Block] zusammenhängend
    nb3 = 3 * n_b
    K_bb = K_s[:nb3, :nb3]; K_bi = K_s[:nb3, nb3:]; K_ii = K_s[nb3:, nb3:]
    ->  K_bb6 = T_b.T @ (K_bb @ T_b)      (6, 6)    dicht
        K_bi6 = (K_bi.T @ T_b).T          (6, n_i)  dicht -- über die sparse Seite rechnen!
        K_ii  bleibt sparse (csc für splu in WP4)
    gleiches für M.  :return: dict(K_bb, K_bi, K_ii, M_bb, M_bi, M_ii)
    """
    nb3 = 3 * n_b

    def transform_blocks(A):
        # symmetric A => A_ib = A_bi.T, so only three blocks are returned
        # (a stored ib copy could silently drift from bi); A_ii stays sparse
        # for the splu/eigsh factorizations in the Craig-Bampton step.
        A_s = A[perm][:, perm]
        A_bb = T_b.T @ (A_s[:nb3, :nb3] @ T_b)     # (6, 6) dense
        A_bi = (A_s[:nb3, nb3:].T @ T_b).T         # (6, n_i) dense, sparse-side product
        A_ii = A_s[nb3:, nb3:].tocsc()
        return A_bb, A_bi, A_ii

    K_bb, K_bi, K_ii = transform_blocks(K)
    M_bb, M_bi, M_ii = transform_blocks(M)
    return dict(K_bb=K_bb, K_bi=K_bi, K_ii=K_ii,
                M_bb=M_bb, M_bi=M_bi, M_ii=M_ii)


# ===========================================================================
# WP4 -- Craig-Bampton reduction
# ===========================================================================

def craig_bampton(blocks, n_modes):
    """
    Craig-Bampton with the 6 RBE2 master DoFs as boundary set.

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

    # static constraint modes: unit master motion, interior follows statically
    Psi = -splu(K_ii).solve(K_bi.T)                     # (n_i, 6)

    # fixed-interface modes of the clamped interior (K_ii nonsingular), sorted
    # ascending and mass-normalized so that Phi.T M_ii Phi == I exactly
    lam, Phi = eigsh(K_ii, k=n_modes, M=M_ii, sigma=0)
    order = np.argsort(lam)
    lam, Phi = lam[order], Phi[:, order]
    Phi = Phi / np.sqrt(np.diag(Phi.T @ (M_ii @ Phi)))

    # reduced matrices: R^T () R with R = [[I,0],[Psi,Phi]], multiplied out
    # (see docstring; the zero coupling and diag(lam) are exact by construction)
    K_r = np.block([[K_bb + K_bi @ Psi, np.zeros((6, n_modes))],
                    [np.zeros((n_modes, 6)), np.diag(lam)]])

    M_bb_r = M_bb + M_bi @ Psi + Psi.T @ M_bi.T + Psi.T @ (M_ii @ Psi)
    M_bm_r = M_bi @ Phi + Psi.T @ (M_ii @ Phi)
    M_r = np.block([[M_bb_r, M_bm_r],
                    [M_bm_r.T, np.eye(n_modes)]])

    f_fixed_hz = np.sqrt(lam) / (2.0 * np.pi)
    return M_r, K_r, Psi, Phi, f_fixed_hz


# ===========================================================================
# WP5 -- damping + recovery                        (Claude, after WP2-4 review)
# ===========================================================================

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
    """One RBE2 + Craig-Bampton reduced substructure, q_r = [q_m (6); eta]."""
    name: str
    M_r: np.ndarray            # (nr, nr), nr = 6 + n_modes
    C_r: np.ndarray            # (nr, nr)
    K_r: np.ndarray            # (nr, nr)
    T_b: np.ndarray            # (3nb, 6) RBE2 map of the boundary nodes
    Psi: np.ndarray            # (ni, 6)  static constraint modes
    Phi: np.ndarray            # (ni, n_modes) fixed-interface modes
    nodes: np.ndarray          # (n, 3) all node coordinates
    boundary_idx: np.ndarray   # (nb,) 0-based boundary node indices
    internal_idx: np.ndarray   # (n - nb,) remaining node indices

    @classmethod
    def build(cls, name, data, boundary_idx, master_xyz, n_modes, zeta):
        """
        Full reduction of one substructure: RBE2 transformation -> boundary-
        first partition -> Craig-Bampton -> modal damping.

        :param name: "A" or "B" (report label)
        :param data: dict from :func:`load_or_export`
        :param boundary_idx: RBE2 slave node indices (from get_boundary_nodes)
        :param master_xyz: (3,) RBE2 master position = VP
        :param n_modes: fixed-interface modes to keep
        :param zeta: modal damping ratio per elastic mode
        """
        nodes = data["nodes"]
        boundary_idx = np.asarray(boundary_idx)

        T_b = rbe2_transformation(nodes, boundary_idx, master_xyz)
        perm, internal_idx = partition_dofs(len(nodes), boundary_idx)
        blocks = apply_rbe2(data["K"], data["M"], T_b, perm, len(boundary_idx))
        M_r, K_r, Psi, Phi, f_fixed_hz = craig_bampton(blocks, n_modes)
        C_r = modal_damping_matrix(M_r, K_r, zeta)

        print(f"[{name}] reduced {3 * len(nodes)} -> {M_r.shape[0]} DoFs | "
              f"fixed-interface modes {f_fixed_hz[0]:.1f}..{f_fixed_hz[-1]:.1f} Hz")
        return cls(name=name, M_r=M_r, C_r=C_r, K_r=K_r, T_b=T_b, Psi=Psi,
                   Phi=Phi, nodes=nodes, boundary_idx=boundary_idx,
                   internal_idx=internal_idx)

    def recovery_row(self, position, direction):
        """
        Row t (nr,) of the physical<->reduced map for the scalar DoF
        "displacement at ``position`` in ``direction``": u = t @ q_r, and by
        the transpose of the same map a point force F*direction there enters
        the reduced equations as f_r = t * F.

        The position snaps to the nearest FE node (like pyfbs
        update_locations_df). Internal node -> direction projected onto its
        [Psi | Phi] rows; boundary node -> onto its T_b rows (eta part zero,
        the node moves rigidly with the master).
        """
        pos = np.asarray(position, dtype=float)
        dvec = np.asarray(direction, dtype=float)
        dist = np.linalg.norm(self.nodes - pos, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 5e-3:
            print(f"[{self.name}] recovery_row: snapped {dist[j] * 1e3:.2f} mm "
                  f"to node {j} -- check the position")

        row = np.zeros(self.M_r.shape[0])
        hit = np.nonzero(self.boundary_idx == j)[0]
        if hit.size:                                   # boundary (RBE2 slave) node
            p = int(hit[0])
            row[:6] = dvec @ self.T_b[3 * p:3 * p + 3, :]
        else:                                          # internal node
            q = int(np.searchsorted(self.internal_idx, j))
            assert self.internal_idx[q] == j
            row[:6] = dvec @ self.Psi[3 * q:3 * q + 3, :]
            row[6:] = dvec @ self.Phi[3 * q:3 * q + 3, :]
        return row


# ===========================================================================
# WP5 -- coupled linear system (assembly + spring)
# ===========================================================================

def assemble_coupled(sub_A, sub_B, k_diag):
    """
    Couple the two reduced substructures through the linear part of the joint.

    Coordinates q = [q_mA (6), eta_A | q_mB (6), eta_B]; M, C, K are block
    diagonal, and the LINEAR spring stiffness goes into K:

        x_r = Bc q  (6 relative master DoFs, +1 on A, -1 on B -- same signs
        as the pyFBS example's signed-Boolean matrix)
        K += Bc.T @ diag(k_diag) @ Bc

    The cubic terms (alpha x^3 + beta xdot^3) stay nonlinear and are applied
    by the pyhbm system class in WP6.

    :param sub_A/sub_B: :class:`ReducedSubstructure`
    :param k_diag: (6,) linear joint stiffness [N/m, N/m, N/m, Nm/rad x3]
    :return: (M, C, K, Bc) dense, d = nrA + nrB
    """
    from scipy.linalg import block_diag

    nrA = sub_A.M_r.shape[0]
    d = nrA + sub_B.M_r.shape[0]

    Bc = np.zeros((6, d))
    Bc[:, :6] = np.eye(6)                    # A master
    Bc[:, nrA:nrA + 6] = -np.eye(6)          # B master

    M = block_diag(sub_A.M_r, sub_B.M_r)
    C = block_diag(sub_A.C_r, sub_B.C_r)
    K = block_diag(sub_A.K_r, sub_B.K_r) + Bc.T @ np.diag(k_diag) @ Bc
    return M, C, K, Bc


# ===========================================================================
# WP6 -- coupled pyhbm system                                        (Claude)
# ===========================================================================
# class CoupledCubicCB(pyhbm.SecondOrderODE) and build_coupled_system() will be
# added here: f_nl = Bc.T (alpha x^3 + beta xdot^3) on x_r = Bc q.
