"""Dynamical-system definitions for the clamped-free rod vibro-impact example.

Three substructure models of the SAME physics (Vadcard, Batailly & Thouverez,
JSV 531 (2022), Fig. 17) -- a clamped-free axial FE rod whose free end B hits an
obstacle a gap g0 away:

    RodVibroImpactFlexible  -- DLFT, 2 substructures (rod + grounded spring k_obs).
                               The unilateral contact acts on x_r = u_B - u_w.
    RodVibroImpactRigid     -- DLFT, exact rigid wall (no obstacle DOF); the
                               contact enforces u_B <= g0 directly.
    RodPenaltyContact       -- AFT, single rod substructure.  The obstacle is a
                               tanh-REGULARIZED one-sided penalty spring k_obs on
                               the rod tip.  This is the smooth-force model AFT
                               needs, and reduces to the rigid wall as k_obs->inf.

All three share the same rod assembly and the same omega_1 frequency
nondimensionalization, so they can be compared point-for-point.

Material / geometry (Vadcard Table 1) live in :class:`RodParams`.
"""
from dataclasses import dataclass

import numpy as np
from numpy import zeros, cos
from scipy.linalg import eigh

from pyhbm import FBS_System


# ============================ rod assembly ==================================

@dataclass
class RodParams:
    """Clamped-free axial bar parameters (Vadcard Table 1 defaults)."""
    n_elem: int = 20          # number of two-node bar elements
    L:      float = 0.13      # rod length [m]
    E:      float = 210e9     # Young's modulus [Pa]
    rho:    float = 7800.0    # density [kg/m^3]
    A:      float = 15.6e-4   # cross section [m^2]   (15.6 cm^2)
    xi:     float = 7.5e-3    # modal damping ratio
    F0:     float = 25e3      # harmonic forcing at the free end B [N]
    poly_deg: int = 30        # AFT/aliasing polynomial degree

    @property
    def k_rod(self) -> float:
        """Static axial stiffness E*A/L [N/m] -- the obstacle-stiffness reference."""
        return self.E * self.A / self.L


def assemble_rod(p: RodParams):
    """Assemble the clamped-free rod and return its FE operators.

    :returns: dict with
        M_rod, K_rod, C_rod   -- (n, n) rod matrices (clamped node dropped),
        omega_modes           -- (n,) undamped natural frequencies [rad/s],
        omega_ref             -- omega_modes[0], the nondimensionalization base,
        n                     -- free DOF count (= n_elem).
    """
    n = p.n_elem
    l = p.L / n
    Me = (p.rho * p.A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Ke = (p.E * p.A / l)         * np.array([[1.0, -1.0], [-1.0, 1.0]])

    M_rod = zeros((n + 1, n + 1))
    K_rod = zeros((n + 1, n + 1))
    for e in range(n):
        M_rod[e:e + 2, e:e + 2] += Me
        K_rod[e:e + 2, e:e + 2] += Ke
    M_rod = M_rod[1:, 1:]            # drop clamped node 0
    K_rod = K_rod[1:, 1:]

    # modal damping  C = M Phi diag(2 xi w_i) Phi^T M   (Phi mass-normalized)
    w2, Phi = eigh(K_rod, M_rod)
    omega_modes = np.sqrt(np.clip(w2, 0.0, None))
    C_rod = M_rod @ Phi @ np.diag(2.0 * p.xi * omega_modes) @ Phi.T @ M_rod

    return dict(M_rod=M_rod, K_rod=K_rod, C_rod=C_rod,
                omega_modes=omega_modes, omega_ref=omega_modes[0], n=n)


def _zero_interface(dimension):
    """interface_force stub (Nt, d, 1) for DLFT systems where the method supplies F."""
    def _f(self, u_rel, udot_rel, tau):
        return zeros((len(tau), dimension, 1))
    return _f


# ============================ DLFT: flexible wall ===========================

class RodVibroImpactFlexible(FBS_System):
    """Rod + grounded obstacle spring k_obs, coupled by FBS; rigid DLFT on x_r.

    M = blkdiag(M_rod, 0),  C = blkdiag(C_rod, 0),  K = blkdiag(K_rod, k_obs)
    B = [ ... +1 (rod tip) ... -1 (obstacle node) ]   ->  x_r = u_B - u_w

    k_obs = k_rel * k_rod;  the rigid wall is the limit k_rel -> infinity.
    The obstacle compliance enters the interface admittance automatically, so the
    DLFT contact code is unchanged.
    """
    is_real_valued = True

    def __init__(self, params: RodParams = None, k_rel: float = 1.0):
        p = params or RodParams()
        if k_rel <= 0.0:
            raise ValueError("k_rel must be > 0 (k_obs = k_rel * k_rod).")
        r = assemble_rod(p)
        n = r["n"]

        k_obs = k_rel * p.k_rod
        self.k_rod, self.k_obs, self.k_rel = p.k_rod, k_obs, k_rel
        self.k_ref = p.k_rod                                  # back-compat alias

        d_tot = n + 1                                        # rod + obstacle node
        M = zeros((d_tot, d_tot)); M[:n, :n] = r["M_rod"]
        C = zeros((d_tot, d_tot)); C[:n, :n] = r["C_rod"]
        K = zeros((d_tot, d_tot)); K[:n, :n] = r["K_rod"]
        K[n, n] = k_obs                                      # obstacle spring to ground

        omega_ref = r["omega_ref"]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * M
        self.damping_matrix   = omega_ref      * C
        self.stiffness_matrix = K

        self.rod_tip_idx  = n - 1
        self.obstacle_idx = n
        B = zeros((1, d_tot))
        B[0, self.rod_tip_idx]  = +1.0
        B[0, self.obstacle_idx] = -1.0
        self.B_coupling = B

        self.dimension         = 1
        self.total_dimension   = d_tot
        self.polynomial_degree = p.poly_deg
        self.F0 = p.F0
        self.omega_modes = r["omega_modes"]

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)
        return f

    interface_force               = _zero_interface(1)
    jacobian_interface_force      = lambda self, u, ud, tau: zeros((len(tau), 1, 1))
    jacobian_interface_force_qdot = lambda self, u, ud, tau: zeros((len(tau), 1, 1))


# ============================ DLFT: rigid wall ==============================

class RodVibroImpactRigid(FBS_System):
    """Clamped-free rod with an EXACT rigid wall (DLFT, no obstacle DOF).

    Interface DOF is u_B (rod free end); DLFTContact enforces u_B <= g0.  The
    converged solution is the true unilateral solution (no penalty approximation),
    but it is numerically stiff cold -- warm-start from a near-rigid flexible run.
    """
    is_real_valued = True

    def __init__(self, params: RodParams = None):
        p = params or RodParams()
        r = assemble_rod(p)
        n = r["n"]

        omega_ref = r["omega_ref"]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * r["M_rod"]
        self.damping_matrix   = omega_ref      * r["C_rod"]
        self.stiffness_matrix = r["K_rod"]

        self.rod_tip_idx = n - 1
        B = zeros((1, n)); B[0, -1] = 1.0
        self.B_coupling = B

        self.dimension         = 1
        self.total_dimension   = n
        self.polynomial_degree = p.poly_deg
        self.F0 = p.F0
        self.omega_modes = r["omega_modes"]
        self.k_rod = self.k_ref = p.k_rod

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)
        return f

    interface_force               = _zero_interface(1)
    jacobian_interface_force      = lambda self, u, ud, tau: zeros((len(tau), 1, 1))
    jacobian_interface_force_qdot = lambda self, u, ud, tau: zeros((len(tau), 1, 1))


# ============================ AFT: regularized penalty wall =================

def _smoothed_ramp_1(d, alpha, k):
    """tanh-regularized ramp:  f(d) = k * 0.5*d*(1+tanh(alpha*d)) -> k*max(0,d)."""
    if not np.isfinite(alpha):
        r  = np.where(d > 0.0, k * d, 0.0)
        dr = np.where(d > 0.0, k,     0.0)
        return r, dr
    t  = np.tanh(alpha * d)
    H  = 0.5 * (1.0 + t)
    r  = k * (d * H)
    dr = k * (H + 0.5 * alpha * d * (1.0 - t * t))
    return r, dr

def _smoothed_ramp_2(d, alpha, k):
    """softplus-regularized ramp:  f(d) = k * ln(1+exp(alpha*d))/alpha -> k*max(0,d)."""
    if not np.isfinite(alpha):
        r  = np.where(d > 0.0, k * d, 0.0)
        dr = np.where(d > 0.0, k,     0.0)
        return r, dr
    r = k * np.logaddexp(0.0, alpha * d) / alpha  # = k*ln(1+exp(alpha*d))/alpha, overflow-safe
    dr = k * (1.0 / (1.0 + np.exp(-alpha * d)))  # = k*sigmoid(alpha*d)
    return r, dr


class RodPenaltyContact(FBS_System):
    """Clamped-free rod with a tanh-REGULARIZED one-sided penalty spring (AFT).

    Single substructure (the rod); the obstacle is modelled as a smooth contact
    force on the rod tip u_B:

        f_c(u_B) = k_obs * r(u_B - g0),   r = tanh-smoothed ramp ~ max(0, .)

    Positive f_c (compression) resists penetration, matching the DLFT sign
    convention, so the FBS residual r(x) = Q_rel + Y_r f_c~ - F_adm is directly
    comparable.  k_obs = k_rel * k_rod is the same obstacle stiffness as the DLFT
    flexible model; alpha is the regularization sharpness (alpha->inf => rigid,
    nonsmooth).  Velocity-independent, so the qdot Jacobian is zero.
    """
    is_real_valued = True

    def __init__(self, params: RodParams = None, k_rel: float = 1.0,
                 g_zero: float = 0.0, alpha: float = 1.0e4):
        p = params or RodParams()
        if k_rel <= 0.0:
            raise ValueError("k_rel must be > 0 (k_obs = k_rel * k_rod).")
        r = assemble_rod(p)
        n = r["n"]

        self.k_rod = self.k_ref = p.k_rod
        self.k_obs = k_rel * p.k_rod
        self.k_rel = k_rel
        self.g_zero = g_zero
        self.alpha  = alpha

        omega_ref = r["omega_ref"]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * r["M_rod"]
        self.damping_matrix   = omega_ref      * r["C_rod"]
        self.stiffness_matrix = r["K_rod"]

        self.rod_tip_idx = n - 1
        B = zeros((1, n)); B[0, -1] = 1.0
        self.B_coupling = B

        self.dimension         = 1
        self.total_dimension   = n
        self.polynomial_degree = p.poly_deg
        self.F0 = p.F0
        self.omega_modes = r["omega_modes"]

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)
        return f

    # --- AFT contact force on the relative interface DOF x_r = u_B ---
    def interface_force(self, u_rel, udot_rel, tau):
        d = u_rel[:, :, 0] - self.g_zero               # (Nt, 1)
        r, _ = _smoothed_ramp_1(d, self.alpha, self.k_obs)
        return r[:, :, None]            # (Nt, 1, 1)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        d = u_rel[:, :, 0] - self.g_zero
        _, dr = _smoothed_ramp_1(d, self.alpha, self.k_obs)
        return dr[:, :, None]           # (Nt, 1, 1)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), 1, 1))
