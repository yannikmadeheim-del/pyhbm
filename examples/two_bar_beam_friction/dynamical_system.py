"""
Two bar+beam elements with frictional contact -- Tiago Martins' MSc thesis,
case study 6.4 (figure 6.13), rebuilt from the thesis text.

Two identical cantilever bar+beam elements, each clamped at one node and
carrying three tip degrees of freedom

    element 1 tip :  q1 (longitudinal), q2 (transverse), q3 (slope)
    element 2 tip :  q4 (longitudinal), q5 (transverse), q6 (slope)

The elements share no degrees of freedom, so they are UNCOUPLED in the linear
system (M and K are block-diagonal). They are coupled only through a single
contact element between the tips, which carries two nonlinearities: unilateral
contact with contact-separation in the normal (vertical) direction, and
stick-slip dry friction in the tangential (horizontal) direction.

Equations, quoted from the thesis (section 6.4):

  (6.12)  Ke = (E/l) [[A, 0, 0], [0, 12 I/l^2, -6 I/l], [0, -6 I/l, 4 I]]
          Me = (lam l/210) [[70, 0, 0], [0, 78, -11 l], [0, -11 l, 2 l^2]]
  (6.13)  K = blkdiag(Ke, Ke),  M = blkdiag(Me, Me)
  (6.15)  normal distance  q_perp = eps + q2 - q5,  and
          N = ln(exp(-k alpha2 q_perp) + 1)/alpha2 >= 0
          so  f_nl_2 = -N = -ln(exp(k alpha2 [q5 - q2 - eps]) + 1)/alpha2
          friction   f_nl_1 = -mu f_nl_2 tanh(alpha1 [q1dot - q4dot])
          action-reaction    f_nl = [f_nl_1, f_nl_2, 0, -f_nl_1, -f_nl_2, 0]^T
          excitation         f_ext = [P1 cos(w t), -P5, 0, 0, P5, 0]^T

Damping is Rayleigh, C = beta K, chosen so that it is roughly 5% of the
restitution force near the first resonance, which the thesis places at w = 1.

Two systems of the SAME physics are provided, for the two solvers:

    BarBeamSecondOrder -- M q'' + C q' + K q + f_nl(q, q') = f_ext(tau),
        the form the 2nd-order FRF solver (FRFProblem + NumericalFRF) consumes.

    BarBeamFirstOrder  -- the same dynamics as a first-order system in the state
        z = [q ; q'], for the original first-order solver
        HarmonicBalanceMethod(first_order_ode=...).

Both delegate to the single force law below, so they cannot drift apart.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numpy import zeros, cos, tanh, exp, log1p
from scipy.special import expit

from pyhbm.dynamical_system import FirstOrderODE, SecondOrderODE


# ===========================================================================
# Parameters -- the thesis values. Edit here.
# ===========================================================================

L      = 1.0        # element length
EA     = 1.0 / 3    # axial rigidity   E*A
EI     = 1.0 / 3    # bending rigidity E*I
LAM    = 1.0        # linear density

K_C    = 500.0      # contact surface stiffness k
ALPHA2 = 2.0        # normal regularization constant
EPS    = 0.01       # vertical gap between the tips
ALPHA1 = 150.0      # friction regularization (slope at the stick condition)
MU     = 0.1        # kinetic friction coefficient

P1     = 0.1        # amplitude of the axial harmonic excitation on q1
P5     = 0.4        # static clamping force
BETA   = 0.05       # Rayleigh damping, C = BETA * K

POLYNOMIAL_DEGREE = 16   # AFT sampling: N_t = (deg+1)*max(harmonic) + 1


# degree-of-freedom order of the assembled 6-DOF system (figure 6.13)
Q1, Q2, Q3, Q4, Q5, Q6 = 0, 1, 2, 3, 4, 5
DOF_LABELS = ("q1", "q2", "q3", "q4", "q5", "q6")

# the two interface relative coordinates, ordered [normal ; tangential]
IF_LABELS = ("x_rel_N", "x_rel_T")
DIMENSION = 6


# ===========================================================================
# Linear system -- thesis (6.12) and (6.13)
# ===========================================================================

def element_matrices():
    """
    Elementary stiffness and mass matrices, thesis (6.12), for one cantilever
    bar+beam element. The clamped node's three degrees of freedom are already
    removed, so this is the tip block. Order: [axial, transverse, slope].

    E and A (resp. E and I) only ever appear as the products EA and EI, so the
    (E/l) prefactor of (6.12) is distributed into the entries here.
    """
    Ke = np.array([
        [EA / L,        0.0,             0.0         ],
        [0.0,      12.0 * EI / L**3, -6.0 * EI / L**2],
        [0.0,      -6.0 * EI / L**2,  4.0 * EI / L   ],
    ])
    Me = (LAM * L / 210.0) * np.array([
        [70.0,   0.0,        0.0       ],
        [0.0,   78.0,      -11.0 * L   ],
        [0.0,  -11.0 * L,    2.0 * L**2],
    ])
    return Ke, Me


def assemble():
    """Global M, C, K, thesis (6.13). Block-diagonal: the elements are uncoupled."""
    Ke, Me = element_matrices()
    M = zeros((6, 6)); K = zeros((6, 6))
    M[:3, :3] = Me; M[3:, 3:] = Me
    K[:3, :3] = Ke; K[3:, 3:] = Ke
    return M, BETA * K, K


def coupling_matrix():
    """
    Signed-Boolean B with x_rel = B q, ordered [normal ; tangential]:

        x_N = q5 - q2   (normal;      the thesis measures q_perp = eps - x_N)
        x_T = q1 - q4   (tangential;  its rate drives the friction)
    """
    B = zeros((2, 6))
    B[0, Q5] = +1.0; B[0, Q2] = -1.0
    B[1, Q1] = +1.0; B[1, Q4] = -1.0
    return B


M_MATRIX, C_MATRIX, K_MATRIX = assemble()
B_COUPLING = coupling_matrix()


# ===========================================================================
# Nonlinear force -- thesis (6.15) and the friction law
# ===========================================================================

def normal_force(x_N):
    """
    Regularized normal compression force N >= 0 and its slope, thesis (6.15)
    written in x_N = q5 - q2 = eps - q_perp:

        N      = softplus(c)/alpha2,  c = k alpha2 (x_N - eps)
        dN/dxN = k sigmoid(c)

    Both are evaluated in the overflow-safe forms: softplus as
    log1p(exp(-|c|)) + max(c, 0), and the sigmoid through scipy's expit, since
    a plain 1/(1+exp(-c)) overflows for c << 0.
    """
    c = K_C * ALPHA2 * (x_N - EPS)
    N = (log1p(exp(-np.abs(c))) + np.maximum(c, 0.0)) / ALPHA2
    return N, K_C * expit(c)


def _interface(q, qdot):
    """(x_N, d/dt x_T) from the full time series q, qdot of shape (N_t, 6, 1)."""
    return (q[:, Q5, 0] - q[:, Q2, 0]), (qdot[:, Q1, 0] - qdot[:, Q4, 0])


def nonlinear_force(q, qdot):
    """
    The thesis nonlinear force f_nl = [f1, f2, 0, -f1, -f2, 0]^T, with
    f2 = -N and f1 = -mu f2 tanh(alpha1 [q1dot - q4dot]) = mu N tanh(...).

    Written as B^T lambda with lambda = [N, f_T] on the interface, which
    expands to exactly that vector.

    :param q, qdot: (N_t, 6, 1) time series
    :returns: (N_t, 6, 1)
    """
    x_N, v_T = _interface(q, qdot)
    N, _ = normal_force(x_N)
    lam = zeros((q.shape[0], 2, 1))
    lam[:, 0, 0] = N
    lam[:, 1, 0] = MU * N * tanh(ALPHA1 * v_T)
    return B_COUPLING.T @ lam


def jacobian_nonlinear_force(q, qdot):
    """d f_nl / d q = B^T (d lambda / d x_rel) B.   -> (N_t, 6, 6)"""
    x_N, v_T = _interface(q, qdot)
    _, dN = normal_force(x_N)
    dlam = zeros((q.shape[0], 2, 2))
    dlam[:, 0, 0] = dN                                  # dN/dx_N
    dlam[:, 1, 0] = MU * tanh(ALPHA1 * v_T) * dN        # df_T/dx_N, through N
    return B_COUPLING.T @ dlam @ B_COUPLING


def jacobian_nonlinear_force_qdot(q, qdot):
    """d f_nl / d qdot = B^T (d lambda / d xdot_rel) B.   -> (N_t, 6, 6)"""
    x_N, v_T = _interface(q, qdot)
    N, _ = normal_force(x_N)
    th = tanh(ALPHA1 * v_T)
    dlam = zeros((q.shape[0], 2, 2))
    dlam[:, 1, 1] = MU * N * ALPHA1 * (1.0 - th * th)   # df_T/dv_T
    return B_COUPLING.T @ dlam @ B_COUPLING


def external_force(adimensional_time):
    """f_ext = [P1 cos(tau), -P5, 0, 0, P5, 0]^T.   -> (N_t, 6, 1)

    The clamping force is static, so this has a non-zero mean: harmonic 0 MUST
    be in the harmonic set or the preload is silently dropped.
    """
    f = zeros((len(adimensional_time), 6, 1))
    f[:, Q1, 0] = P1 * cos(adimensional_time)
    f[:, Q2, 0] = -P5
    f[:, Q5, 0] = +P5
    return f


def static_contact_state():
    """
    The static (harmonic-0) contact state under the clamping force alone, used
    to seed the continuation.

    Bending and axial dynamics are uncoupled and the clamping force is
    constant, so the normal contact is essentially static. Neither f_ext nor
    f_nl has a moment component, so the slope row of Ke gives q3 = 1.5 q2, and
    the transverse row then reduces to q2 = N - P5. The two elements are
    identical under exactly opposite loads, hence q5 = -q2 and

        x_N = q5 - q2 = 2 (P5 - N),

    which is 2 * (tip compliance l^3/3EI) * (P5 - N) with the thesis values.
    Closing this with (6.15) leaves one scalar equation, solved by bisection:
    N_law(x_N(N)) - N is strictly decreasing in N.

    :returns: (x_N, N)
    """
    def gap_of(N):
        return 2.0 * (L**3 / (3.0 * EI)) * (P5 - N)

    lo, hi = 0.0, 2.0 * P5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        N_law, _ = normal_force(np.array([gap_of(mid)]))
        lo, hi = (mid, hi) if N_law[0] > mid else (lo, mid)
    N = 0.5 * (lo + hi)
    return gap_of(N), N


# ===========================================================================
# Second-order system -- for the 2nd-order FRF solver
# ===========================================================================

class BarBeamSecondOrder(SecondOrderODE):
    """
    M q'' + C q' + K q + f_nl(q, q') = f_ext(tau), the 6 tip degrees of freedom.

    Pair with FRFProblem(system, NumericalFRF(M, C, K)), which solves it through
    the admittance Y = Z^-1 of the linear part.
    """
    is_real_valued = True

    def __init__(self):
        self.mass_matrix = M_MATRIX
        self.damping_matrix = C_MATRIX
        self.stiffness_matrix = K_MATRIX
        self.dimension = DIMENSION
        self.polynomial_degree = POLYNOMIAL_DEGREE

    def external_term(self, adimensional_time):
        return external_force(adimensional_time)

    def nonlinear_term(self, q, qdot, adimensional_time):
        return nonlinear_force(q, qdot)

    def jacobian_nonlinear_term(self, q, qdot, adimensional_time):
        return jacobian_nonlinear_force(q, qdot)

    def jacobian_nonlinear_term_qdot(self, q, qdot, adimensional_time):
        return jacobian_nonlinear_force_qdot(q, qdot)


# ===========================================================================
# First-order system -- for the original first-order solver
# ===========================================================================

class BarBeamFirstOrder(FirstOrderODE):
    """
    The same dynamics in the state z = [q ; q'] (12 components):

        zdot = A z + f_nl_z(z, tau) + f_ext_z(tau),
        A = [[0, I], [-M^-1 K, -M^-1 C]],
        f_nl_z = [0 ; -M^-1 f_nl],   f_ext_z = [0 ; +M^-1 f_ext].

    SIGN NOTE: the first-order residual of pyhbm is A Q - w Q' + f_nl + f_ext,
    i.e. BOTH the nonlinear and the external term are ADDED -- opposite to the
    second-order form, where f_nl is added and f_ext subtracted. That is the
    same convention as the original duffing_forced_nonautonomous example.
    """
    is_real_valued = True

    def __init__(self):
        M_inv = np.linalg.inv(M_MATRIX)
        self.M_inv = M_inv

        A = zeros((2 * DIMENSION, 2 * DIMENSION))
        A[:DIMENSION, DIMENSION:] = np.eye(DIMENSION)
        A[DIMENSION:, :DIMENSION] = -M_inv @ K_MATRIX
        A[DIMENSION:, DIMENSION:] = -M_inv @ C_MATRIX
        self.linear_coefficient = A

        self.dimension = 2 * DIMENSION
        self.polynomial_degree = POLYNOMIAL_DEGREE

    @staticmethod
    def _split(state):
        """State time series (N_t, 12, 1) -> (q, qdot), each (N_t, 6, 1)."""
        return state[:, :DIMENSION, :], state[:, DIMENSION:, :]

    def external_term(self, adimensional_time):
        f = zeros((len(adimensional_time), 2 * DIMENSION, 1))
        f[:, DIMENSION:, :] = self.M_inv @ external_force(adimensional_time)
        return f

    def linear_term(self, state):
        return self.linear_coefficient @ state

    def nonlinear_term(self, state, adimensional_time):
        q, qdot = self._split(state)
        f = zeros((state.shape[0], 2 * DIMENSION, 1))
        f[:, DIMENSION:, :] = -self.M_inv @ nonlinear_force(q, qdot)
        return f

    def jacobian_nonlinear_term(self, state, adimensional_time):
        """
        d f_nl_z / d z. The velocity IS part of the state here, so the friction
        law's velocity dependence enters this single Jacobian -- the first-order
        solver has no separate qdot path.
        """
        q, qdot = self._split(state)
        J = zeros((state.shape[0], 2 * DIMENSION, 2 * DIMENSION))
        J[:, DIMENSION:, :DIMENSION] = -self.M_inv @ jacobian_nonlinear_force(q, qdot)
        J[:, DIMENSION:, DIMENSION:] = -self.M_inv @ jacobian_nonlinear_force_qdot(q, qdot)
        return J
