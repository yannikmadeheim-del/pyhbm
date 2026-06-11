"""Two bar+beam elements connected by a frictional contact element.

Reproduction of the thesis case study 6.4 ("Two bar+beam elements with
frictional contact", figure 6.13): two identical cantilever bar+beam elements,
each clamped at one node and carrying three tip DOFs

    element 1 tip :  q1 (axial / longitudinal),  q2 (transverse),  q3 (slope)
    element 2 tip :  q4 (axial / longitudinal),  q5 (transverse),  q6 (slope)

The two elements are uncoupled in the LINEAR system (block-diagonal M, K). They
are coupled only through a single nonlinear contact element acting between the
tips, with two nonlinearities:

    * NORMAL (vertical) -- unilateral contact / contact-separation between q2 and
      q5 across a small vertical gap eps.  Penetration  p = q5 - q2 - eps  > 0
      means contact (compression).
    * TANGENTIAL (horizontal) -- dry (Coulomb) friction along the axial
      direction, driven by the relative tangential velocity  d/dt(q1 - q4).
      The friction bound is  mu * N  with N the normal compression force.

Two interface relative coordinates are used (ordered [normal ; tangential] to
match :class:`DLFTFriction`, index 0 = normal, index 1 = tangential)::

    x_r^N = q5 - q2          (normal,     contact when x_r^N > eps)
    x_r^T = q1 - q4          (tangential, sliding)

The structural contact force  B^T lambda  with lambda = [N, f_T] reproduces the
thesis nonlinear force  f_nl = [f_T, -N, 0, -f_T, N, 0]^T  exactly.

Two models of the SAME physics are provided:

    BarBeamFrictionDLFT      -- DLFT (Dynamic Lagrangian Frequency-Time).
        RIGID unilateral contact + EXACT Coulomb friction, NO regularization.
        The contact force is slaved to x_r by the prediction-correction at every
        residue evaluation (DLFTFriction).  This is the "ground-truth" model.

    BarBeamFrictionAFT       -- AFT (Alternating Frequency-Time), the thesis
        MHBM model with the PROPOSED REGULARIZATION:
          normal     N    = softplus(k*alpha2*(x_r^N - eps)) / alpha2   (eq. 6.15)
          tangential f_T  = mu * N * tanh(alpha1 * d/dt x_r^T)          (eq. 6.11)
        It approaches BarBeamFrictionDLFT as k, alpha1, alpha2 -> infinity.

Numerical values (thesis):  l = 1, EA = 1/3, EI = 1/3, lambda = 1;
contact  alpha2 = 2, k = 500, eps = 0.01, alpha1 = 150, mu = 0.1;
forcing  P1 = 0.1 (axial, harmonic), P5 = 0.4 (clamping, static);
damping  C = 0.05 * K  (first undamped resonance at omega = 1).
"""
from dataclasses import dataclass

import numpy as np
from numpy import zeros, cos, tanh, exp, log1p

from pyhbm import FBS_System


# DOF ordering of the assembled 6-DOF system (figure 6.13).
Q1, Q2, Q3, Q4, Q5, Q6 = 0, 1, 2, 3, 4, 5


# ============================ FE assembly ==================================

@dataclass
class BarBeamParams:
    """Bar+beam element + contact parameters (thesis case study 6.4 defaults)."""
    l:      float = 1.0       # element length
    EA:     float = 1.0 / 3   # axial rigidity  E*A
    EI:     float = 1.0 / 3   # bending rigidity E*I
    lam:    float = 1.0       # linear density  lambda

    # contact / friction
    k:      float = 500.0     # normal contact (surface) stiffness
    alpha2: float = 2.0       # normal softplus regularization constant
    eps:    float = 0.01      # vertical gap between the tips
    alpha1: float = 150.0     # tangential tanh regularization (stick slope)
    mu:     float = 0.1       # Coulomb friction coefficient

    # forcing / damping
    P1:     float = 0.1       # amplitude of axial harmonic excitation on q1
    P5:     float = 0.4       # static clamping force (pushes the tips together)
    beta:   float = 0.05      # stiffness-proportional damping  C = beta * K

    poly_deg: int = 20        # AFT/aliasing polynomial degree


def element_matrices(p: BarBeamParams):
    """Elementary 3x3 stiffness and mass matrices (thesis eq. 6.12).

    DOF order per element: [axial, transverse, slope].
    """
    l, EA, EI, lam = p.l, p.EA, p.EI, p.lam
    Ke = np.array([
        [EA / l,        0.0,             0.0          ],
        [0.0,      12.0 * EI / l**3, -6.0 * EI / l**2 ],
        [0.0,      -6.0 * EI / l**2,  4.0 * EI / l    ],
    ])
    Me = (lam * l / 210.0) * np.array([
        [70.0,   0.0,        0.0      ],
        [0.0,   78.0,      -11.0 * l  ],
        [0.0,  -11.0 * l,    2.0 * l**2],
    ])
    return Ke, Me


def assemble(p: BarBeamParams):
    """Assemble the global 6x6 block-diagonal M, K, C (thesis eq. 6.13).

    The two elements do not share DOFs, so the linear system is uncoupled:
        K = blkdiag(Ke, Ke),  M = blkdiag(Me, Me),  C = beta * K.
    """
    Ke, Me = element_matrices(p)
    M = zeros((6, 6)); K = zeros((6, 6))
    M[:3, :3] = Me; M[3:, 3:] = Me
    K[:3, :3] = Ke; K[3:, 3:] = Ke
    C = p.beta * K
    return M, C, K


# ============================ common base ==================================

class _BarBeamBase(FBS_System):
    """Shared assembly, forcing, coupling and reference frequency.

    The thesis works in coordinates where the first undamped (axial) resonance is
    at omega = 1, so the nondimensionalization base is omega_ref = 1 and the
    matrices are used directly.
    """
    is_real_valued = True

    def _build(self, p: BarBeamParams):
        self.p = p
        M, C, K = assemble(p)

        self.omega_ref        = 1.0                 # first undamped resonance
        self.mass_matrix      = M
        self.damping_matrix   = C
        self.stiffness_matrix = K

        # interface coupling  x_r = B u,  ordered [normal ; tangential]
        B = zeros((2, 6))
        B[0, Q5] = +1.0; B[0, Q2] = -1.0            # normal      x_r^N = q5 - q2
        B[1, Q1] = +1.0; B[1, Q4] = -1.0            # tangential  x_r^T = q1 - q4
        self.B_coupling = B

        self.dimension         = 2                  # interface DOFs (Newton unknowns)
        self.total_dimension   = 6
        self.polynomial_degree = p.poly_deg

        # undamped natural frequencies (for reference / labelling)
        from scipy.linalg import eigh
        w2, _ = eigh(K, M)
        self.omega_modes = np.sqrt(np.clip(w2, 0.0, None))

    def external_term(self, tau):
        """f_ext = [P1 cos(tau), -P5, 0, 0, P5, 0]^T  (thesis)."""
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, Q1, 0] = self.p.P1 * cos(tau)          # axial harmonic excitation
        f[:, Q2, 0] = -self.p.P5                     # clamping: push tip 1 down
        f[:, Q5, 0] = +self.p.P5                     # clamping: push tip 2 up
        return f


# ============================ DLFT (no regularization) =====================

def _zero_interface(dimension):
    """interface_force stub for DLFT systems (the method supplies the force)."""
    def _f(self, u_rel, udot_rel, tau):
        return zeros((len(tau), dimension, 1))
    return _f


class BarBeamFrictionDLFT(_BarBeamBase):
    """DLFT model: rigid unilateral contact + exact Coulomb friction.

    No regularization -- the contact/friction force is computed by
    :class:`DLFTFriction` (prediction-correction) on the relative interface
    coordinate x_r.  Pair this system with ``DLFTFriction(g_zero=eps, mu=mu,
    n_tangential=1)`` in an :class:`FBSProblem`.
    """
    def __init__(self, params: BarBeamParams = None):
        self._build(params or BarBeamParams())

    interface_force               = _zero_interface(2)
    jacobian_interface_force      = lambda self, u, ud, tau: zeros((len(tau), 2, 2))
    jacobian_interface_force_qdot = lambda self, u, ud, tau: zeros((len(tau), 2, 2))


# ============================ AFT (proposed regularization) ================

def _softplus_normal(xN, p: BarBeamParams):
    """Regularized normal compression force and its slope (thesis eq. 6.15).

        N    = softplus(c) / alpha2,   c = k*alpha2*(xN - eps)  (>=0)
        dN/dxN = k * sigmoid(c)

    softplus(c) = log(1 + exp(c)) is evaluated stably for large c.
    """
    c = p.k * p.alpha2 * (xN - p.eps)
    # stable softplus: log1p(exp(-|c|)) + max(c, 0)
    softplus = log1p(exp(-np.abs(c))) + np.maximum(c, 0.0)
    N        = softplus / p.alpha2
    sigmoid  = 1.0 / (1.0 + exp(-c))
    dN_dxN   = p.k * sigmoid
    return N, dN_dxN


class BarBeamFrictionAFT(_BarBeamBase):
    """AFT model: the thesis MHBM with the proposed smooth regularization.

    Interface (relative) contact force, ordered [normal ; tangential]::

        lambda_N = N                 = softplus(k*alpha2*(x_r^N - eps)) / alpha2
        lambda_T = mu * N * tanh(alpha1 * d/dt x_r^T)

    so that  B^T lambda = [f_T, -N, 0, -f_T, N, 0]^T  matches the thesis f_nl.
    The normal force depends on displacement only; the tangential force depends
    on the normal displacement (through N) and on the tangential velocity.
    """
    def __init__(self, params: BarBeamParams = None):
        self._build(params or BarBeamParams())

    def interface_force(self, u_rel, udot_rel, tau):
        p = self.p
        xN     = u_rel[:, 0, 0]                      # x_r^N = q5 - q2
        vT     = udot_rel[:, 1, 0]                   # d/dt x_r^T = q1dot - q4dot
        N, _   = _softplus_normal(xN, p)
        f = zeros((len(tau), 2, 1))
        f[:, 0, 0] = N                               # lambda_N
        f[:, 1, 0] = p.mu * N * tanh(p.alpha1 * vT)  # lambda_T
        return f

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        """d lambda / d x_r  (displacement)."""
        p = self.p
        xN       = u_rel[:, 0, 0]
        vT       = udot_rel[:, 1, 0]
        N, dN    = _softplus_normal(xN, p)
        J = zeros((len(tau), 2, 2))
        J[:, 0, 0] = dN                              # dN/dxN
        J[:, 1, 0] = p.mu * tanh(p.alpha1 * vT) * dN # df_T/dxN  (through N)
        return J

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        """d lambda / d (d/dt x_r)  (velocity)."""
        p = self.p
        xN     = u_rel[:, 0, 0]
        vT     = udot_rel[:, 1, 0]
        N, _   = _softplus_normal(xN, p)
        th     = tanh(p.alpha1 * vT)
        J = zeros((len(tau), 2, 2))
        J[:, 1, 1] = p.mu * N * p.alpha1 * (1.0 - th * th)   # df_T/dvT
        return J
