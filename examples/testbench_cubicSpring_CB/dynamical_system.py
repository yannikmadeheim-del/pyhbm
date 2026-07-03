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
  WP2 (interface node sets)     -- stub, Yannik
  WP3 (RBE2)                    -- stub, Yannik
  WP4 (Craig-Bampton)           -- stub, Yannik
  WP5 (damping/assembly/checks) -- Claude, after WP2-4 review
  WP6 (pyhbm system + HBM)      -- Claude, after WP2-4 review

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
# WP2 -- interface node sets (boundary partition)          (stubs -- Yannik)
# ===========================================================================

def find_mating_nodes(nodes_A, nodes_B, tol):
    """
    Coincident-node detection between the two substructure meshes: the bolted
    footprint where A and B touch (primary interface definition, "mating").

    Approach: tree = scipy.spatial.cKDTree(nodes_B); dist, j = tree.query(nodes_A);
    a node pair "mates" when dist <= tol. Return the 0-based node indices
    (idx_A, idx_B) of the paired nodes, each without duplicates.

    Acceptance: len(idx_A) == len(idx_B) > 0; print the count and the bounding
    box of the selected nodes -- it must sit around the VP (0.0389, 0.3481, 0.007).
    If the count is 0 the meshes are non-conforming: stop and we pick a fallback
    together at review.

    :param nodes_A: (nA, 3) node coordinates of substructure A [m]
    :param nodes_B: (nB, 3) node coordinates of substructure B [m]
    :param tol: coincidence tolerance [m], e.g. 1e-6
    :return: (idx_A, idx_B) int arrays, one entry per mating node pair
    """
    raise NotImplementedError("WP2 -- Yannik")


def find_vpt_nodes(nodes, xlsx_path, substructure, grouping=10):
    """
    Alternative interface definition ("vpt"): the FE nodes the virtual-point
    transformation actually sees -- all Grouping==``grouping`` rows of the
    sheets Channels_<substructure> and Impacts_<substructure>, positions
    snapped to the nearest FE node (cKDTree.query), duplicates removed.

    :param nodes: (n, 3) node coordinates of ONE substructure [m]
    :param xlsx_path: coupling_example.xlsx (pandas.read_excel, one sheet each)
    :param substructure: "A" or "B" (selects the sheet names)
    :return: int array of 0-based node indices
    """
    raise NotImplementedError("WP2 -- Yannik")


def get_boundary_nodes(method, data_A, data_B, xlsx_path=None, tol=1e-6):
    """
    The single switch point for the interface definition.

    method == "mating": find_mating_nodes(...) on both meshes.
    method == "vpt":    find_vpt_nodes(...) per substructure.

    :return: (idx_A, idx_B) -- 0-based boundary node indices per substructure
    """
    raise NotImplementedError("WP2 -- Yannik")


# ===========================================================================
# WP3 -- RBE2: rigidify the boundary nodes to a 6-DoF master (stubs -- Yannik)
# ===========================================================================

def skew(v):
    """
    Skew-symmetric cross-product matrix:  skew(v) @ w == np.cross(v, w).

        [[  0, -vz,  vy],
         [ vz,   0, -vx],
         [-vy,  vx,   0]]
    """
    raise NotImplementedError("WP3 -- Yannik")


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
    raise NotImplementedError("WP3 -- Yannik")


def partition_dofs(n_nodes, boundary_idx):
    """
    DoF index sets for the boundary-first partition (slide procedure step 2).

    Node i owns DoF rows [3*i, 3*i+1, 3*i+2] (see load_ansys_substructure).

        b_dofs = (3*boundary_idx[:, None] + [0, 1, 2]).ravel()   # order of T_b!
        i_dofs = all remaining DoFs, ascending

    :return: (b_dofs, i_dofs) int arrays, disjoint, together all 3*n_nodes DoFs
    """
    raise NotImplementedError("WP3 -- Yannik")


def apply_rbe2(K, M, T_b, b_dofs, i_dofs):
    """
    Condense the boundary-node DoFs onto the 6 master DoFs.

    With u_b = T_b q_m the transformed blocks are (same for M):

        K_bb (6, 6)   = T_b.T @ K[b, b] @ T_b        (dense)
        K_bi (6, ni)  = T_b.T @ K[b, i]              (dense -- only 6 rows)
        K_ii (ni, ni) = K[i, i]                      (keep sparse, csc)

    Slicing pattern for sparse csr: K[np.ix_(b_dofs, b_dofs)] densifies -- do
    NOT do that for K_ii; use K[b_dofs][:, i_dofs] style slicing and convert
    only the small results to dense.

    Acceptance (review): K_bb symmetric; rigid-body test K_tilde @ z_rig ~ 0;
    z.T @ M_tilde @ z for a unit rigid translation equals the substructure mass.

    :return: dict(K_bb, K_bi, K_ii, M_bb, M_bi, M_ii)
    """
    raise NotImplementedError("WP3 -- Yannik")


# ===========================================================================
# WP4 -- Craig-Bampton reduction                            (stubs -- Yannik)
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
    raise NotImplementedError("WP4 -- Yannik")


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
    raise NotImplementedError("WP5 -- Claude")


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
        WP4 wrap-up (Yannik): partition -> rbe2 -> craig_bampton -> damping.
        ``data`` is the dict from load_or_export.
        """
        raise NotImplementedError("WP4 -- Yannik")

    def recovery_row(self, position, direction):
        """
        WP5 (Claude). Row t (nr,) of the physical<->reduced map for the DoF
        "displacement at `position` in `direction`": u = t @ q_r, and the
        generalized force of a point force F*direction there is f_r = t * F.
        Position snaps to the nearest FE node; internal node -> direction @
        [Psi Phi] rows, boundary node -> direction @ T_b rows (eta part zero).
        """
        raise NotImplementedError("WP5 -- Claude")


# ===========================================================================
# WP6 -- coupled pyhbm system                     (Claude, after WP2-4 review)
# ===========================================================================
# class CoupledCubicCB(pyhbm.SecondOrderODE) and build_coupled_system() will be
# added here: block-diagonal assembly of A and B, signed-Boolean Bc on the two
# 6-DoF masters, linear spring k into K, f_nl = Bc.T (alpha x^3 + beta xdot^3).
