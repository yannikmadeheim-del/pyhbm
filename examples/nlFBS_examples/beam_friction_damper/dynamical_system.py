"""System definition for the dry-friction-damper beam example.

Clamped-free 2-D frame FE beam with a flexible Coulomb-friction damper pressed
against it at 0.3L, assembled in FBS form (beam + damper coupled only through the
nonlinear contact).  See ``main.py`` for the full model description (Nacivet,
Pierre, Thouverez, Jezequel, J. Sound Vib. 265(1), 2003, Ex. 1 analogue).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from numpy import zeros, sin
from scipy.linalg import eigh

from pyhbm import FBS_System


# ============================ FE beam assembly ==============================

def frame_beam_matrices(n_elem, L, E, rho, A, I):
    """Clamped-free 2-D frame beam: 3 DOF/node [axial u_z, transverse w_x, rot].

    Returns (M, K) with the clamped node 0 removed. Global DOF of node i:
        3*i + 0 = axial (Z),  3*i + 1 = transverse (X),  3*i + 2 = rotation.
    """
    le = L / n_elem
    ndof = 3 * (n_elem + 1)
    M = zeros((ndof, ndof)); K = zeros((ndof, ndof))
    EA, EI, rA = E * A, E * I, rho * A

    Ka = (EA / le) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    Ma = (rA * le / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Kb = (EI / le**3) * np.array([
        [12.0,    6*le,   -12.0,   6*le],
        [6*le,  4*le**2,  -6*le,  2*le**2],
        [-12.0,  -6*le,    12.0,  -6*le],
        [6*le,  2*le**2,  -6*le,  4*le**2]])
    Mb = (rA * le / 420.0) * np.array([
        [156.0,    22*le,    54.0,   -13*le],
        [22*le,  4*le**2,   13*le,  -3*le**2],
        [54.0,    13*le,   156.0,   -22*le],
        [-13*le, -3*le**2, -22*le,  4*le**2]])

    for e in range(n_elem):
        ax = [3*e, 3*(e+1)]
        bd = [3*e+1, 3*e+2, 3*(e+1)+1, 3*(e+1)+2]
        for a in range(2):
            for b in range(2):
                K[ax[a], ax[b]] += Ka[a, b]; M[ax[a], ax[b]] += Ma[a, b]
        for a in range(4):
            for b in range(4):
                K[bd[a], bd[b]] += Kb[a, b]; M[bd[a], bd[b]] += Mb[a, b]

    keep = slice(3, ndof)              # clamp node 0
    return M[keep, keep], K[keep, keep]


# ============================ system definition =============================

class BeamFrictionDamper(FBS_System):
    is_real_valued = True

    def __init__(self, n_elem=10, L=0.5, E=210e9, rho=7800.0,
                 bx=0.01, by=0.1, k_T=2.4e7, k_N=2.4e3,
                 preload=1500.0, F0=1.0, xi=0.01, poly_deg=16):
        A = bx * by
        I = by * bx**3 / 12.0            # bending about Y for transverse-X deflection

        M_b, K_b = frame_beam_matrices(n_elem, L, E, rho, A, I)
        nb = M_b.shape[0]                # = 3*n_elem

        w2, Phi = eigh(K_b, M_b)
        omega_modes = np.sqrt(np.clip(w2, 0.0, None))
        C_b = M_b @ Phi @ np.diag(2.0 * xi * omega_modes) @ Phi.T @ M_b

        damper_node = int(round(0.3 * n_elem)); tip_node = n_elem
        dn = 3 * (damper_node - 1); tn = 3 * (tip_node - 1)
        self.beam_axial_damper = dn + 0     # u_z at 0.3L -> normal beam side
        self.beam_trans_damper = dn + 1     # w_x at 0.3L -> tangential beam side
        self.beam_trans_tip    = tn + 1     # w_x at tip  -> F_X here

        d_tot = nb + 2
        self.Dx = nb + 0                    # damper X (tangential, grounded by k_T)
        self.Dz = nb + 1                    # damper Z (normal,     grounded by k_N)
        self.k_T, self.k_N = k_T, k_N

        M = zeros((d_tot, d_tot)); M[:nb, :nb] = M_b
        C = zeros((d_tot, d_tot)); C[:nb, :nb] = C_b
        K = zeros((d_tot, d_tot)); K[:nb, :nb] = K_b
        K[self.Dx, self.Dx] = k_T; K[self.Dz, self.Dz] = k_N

        # reference frequency = first mode of the stuck+in-contact linear system
        K_stuck = K_b.copy()
        K_stuck[self.beam_trans_damper, self.beam_trans_damper] += k_T
        K_stuck[self.beam_axial_damper, self.beam_axial_damper] += k_N
        w2s, _ = eigh(K_stuck, M_b)
        self.omega_stuck = np.sqrt(np.clip(w2s, 0.0, None))
        omega_ref = self.omega_stuck[0]
        self.omega_ref = omega_ref

        self.mass_matrix      = omega_ref**2 * M
        self.damping_matrix   = omega_ref     * C
        self.stiffness_matrix = K

        B = zeros((2, d_tot))
        B[0, self.beam_axial_damper] = +1.0; B[0, self.Dz] = -1.0   # normal     (Z)
        B[1, self.beam_trans_damper] = +1.0; B[1, self.Dx] = -1.0   # tangential (X)
        self.B_coupling = B

        self.dimension        = 2
        self.total_dimension  = d_tot
        self.polynomial_degree = poly_deg
        self.F0 = F0
        self.preload = preload
        self.omega_modes = omega_modes

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.beam_trans_tip, 0] = self.F0 * sin(tau)     # F_X = F0 sin(w t)
        f[:, self.Dz, 0]            = -self.preload           # static normal pre-load
        return f

    def interface_force(self, u, ud, tau):
        return zeros((len(tau), self.dimension, 1))
    def jacobian_interface_force(self, u, ud, tau):
        return zeros((len(tau), self.dimension, self.dimension))
    def jacobian_interface_force_qdot(self, u, ud, tau):
        return zeros((len(tau), self.dimension, self.dimension))
