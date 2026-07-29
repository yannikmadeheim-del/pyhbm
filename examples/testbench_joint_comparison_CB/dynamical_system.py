"""
Composable joint models on the RBE2/RBE3 + Craig-Bampton testbench model.

The Ansys import, the interface condensation, the Craig-Bampton reduction and
the physical-CSV export are shared with the sibling example
``testbench_cubicSpring_CB``, which also holds the FE data, the npz caches, the
workbook and the descriptor; only the joint is new. The joint element library --
linear / cubic / dry-friction contributions, each assigned to a named subset of
the six virtual-point interface DoFs and summed into one interface force -- is
the one of the pyFBS ``testbench_joint_comparison`` example, ported unchanged.
The same CONFIG therefore describes the same physical joint in both pipelines,
and the Craig-Bampton branch of this example overlays with the FBS branch of
that one.
"""

import importlib.util
from pathlib import Path

import numpy as np

# The shared infrastructure lives in the sibling example, in a module with the
# same basename as this file -- so it is loaded by explicit path (a plain
# "from dynamical_system import ..." would re-import this module instead).
CB_DIR = Path(__file__).resolve().parent.parent / "testbench_cubicSpring_CB"
_spec = importlib.util.spec_from_file_location("cb_infra",
                                               CB_DIR / "dynamical_system.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

# re-exported so that main.py and study.py import everything from one module
SecondOrderODE = cb.SecondOrderODE
ReducedSubstructure = cb.ReducedSubstructure
assemble_coupled = cb.assemble_coupled
get_boundary_nodes = cb.get_boundary_nodes
load_or_export = cb.load_or_export
read_descriptor = cb.read_descriptor
nearest_node = cb.nearest_node
read_channels = cb.read_channels
channel_snap_info = cb.channel_snap_info
channel_header_lines = cb.channel_header_lines
output_channel_label = cb.output_channel_label
physical_recovery = cb.physical_recovery
save_physical_solution = cb.save_physical_solution

N_IF = 6                                                # virtual-point interface DoFs
VP_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")          # their names, in row order
ALL_DOFS = VP_DOFS                                      # default assignment of an element


# ---------------------------------------------------------------------------
# Joint elements
#
# Every element acts on a named subset of the six interface DoFs and returns a
# CONTRIBUTION on all six rows; CoupledJointCB sums them. Rows outside the
# element's DoFs stay zero, so elements never overwrite each other and any
# number of them may share a DoF.
#
# All coefficients are SCALARS. Different values on different DoFs are expressed
# by a second entry of the same type rather than by per-DoF vectors or
# *_trans/*_rot shorthands: study.py can then read "a list value means sweep
# this parameter" without having to guess whether a list is a sweep or a
# per-DoF vector.
#
# The elements are byte-for-byte the pyFBS ones because both frameworks hand the
# law the same two arguments: the interface gap and its PHYSICAL relative
# velocity (pyhbm has already multiplied the adimensional derivative by omega
# when it calls nonlinear_term). No omega scaling is needed on either side.
# ---------------------------------------------------------------------------

def resolve_dofs(dofs):
    """Interface-DoF indices for a sequence of names out of :data:`VP_DOFS`."""
    idx = []
    for name in dofs:
        if name not in VP_DOFS:
            raise ValueError(f"unknown interface DoF {name!r}; valid names are "
                             f"{', '.join(VP_DOFS)}")
        idx.append(VP_DOFS.index(name))
    return np.array(idx, dtype=int)


class JointElement:
    """
    Base class: one joint contribution on the DoFs named in ``dofs``.

    Subclasses implement, all on the physical interface gap ``u_rel`` (Nt, 6, 1)
    and its physical relative velocity ``udot_rel`` (Nt, 6, 1):

        force(u_rel, udot_rel)    -> (Nt, 6, 1)
        jac_u(u_rel, udot_rel)    -> (Nt, 6, 6)   df/du_rel
        jac_udot(u_rel, udot_rel) -> (Nt, 6, 6)   df/dudot_rel
    """

    def __init__(self, dofs):
        self.dofs = tuple(dofs)
        self.idx = resolve_dofs(self.dofs)

    def constrained_dofs(self):
        """DoF indices this element actually ties (stiffness or friction).
        Used only for the ill-conditioning warning in :class:`CoupledJointCB`."""
        return ()

    def _zeros_f(self, u_rel):
        return np.zeros((u_rel.shape[0], N_IF, 1))

    def _zeros_J(self, u_rel):
        return np.zeros((u_rel.shape[0], N_IF, N_IF))


class LinearJoint(JointElement):
    """
    Linear spring + viscous damper, decoupled per DoF:  f = k*u + c*v.

    :param k: stiffness [N/m] resp. [Nm/rad].
    :param c: viscous damping [N s/m] resp. [Nm s/rad]. The joint
        (rigid-body-vs-spring) modes carry no substructure modal damping, so
        only this damper limits their resonances.
    """

    def __init__(self, k, c=0.0, dofs=ALL_DOFS):
        super().__init__(dofs)
        self.k = float(k)
        self.c = float(c)

    def constrained_dofs(self):
        return tuple(self.idx) if self.k != 0.0 else ()

    def force(self, u_rel, udot_rel):
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (self.k * u_rel[:, self.idx, 0]
                             + self.c * udot_rel[:, self.idx, 0])
        return f

    def jac_u(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = self.k
        return J

    def jac_udot(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = self.c
        return J


class CubicJoint(JointElement):
    """
    Cubic hardening spring with optional cubic damping:  f = alpha*u^3 + beta*v^3.

    :param alpha: cubic stiffness [N/m^3] resp. [Nm/rad^3]. Combined with a
        linear spring k on the same DoF, the cubic force equals the linear one
        at |u| = sqrt(k/alpha), so alpha sets the gap amplitude at which the
        nonlinearity takes over.
    :param beta: cubic damping [N s^3/m^3] resp. [Nm s^3/rad^3]; grows with the
        cube of the relative velocity and clips the hardening peak.
    """

    def __init__(self, alpha, beta=0.0, dofs=ALL_DOFS):
        super().__init__(dofs)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def constrained_dofs(self):
        return tuple(self.idx) if self.alpha != 0.0 else ()

    def force(self, u_rel, udot_rel):
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (self.alpha * u_rel[:, self.idx, 0] ** 3
                             + self.beta * udot_rel[:, self.idx, 0] ** 3)
        return f

    def jac_u(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = 3.0 * self.alpha * u_rel[:, self.idx, 0] ** 2
        return J

    def jac_udot(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = 3.0 * self.beta * udot_rel[:, self.idx, 0] ** 2
        return J


class FrictionJoint(JointElement):
    """
    tanh-regularized isotropic Coulomb friction on a TANGENTIAL PAIR of DoFs:

        f_T = 2 mu N tanh(alpha_reg ||v_T||) v_T / ||v_T||

    with v_T the PHYSICAL relative velocity of the two DoFs in ``dofs``. Both
    clamp faces carry mu*N, hence the slip force 2*mu*N. Near sticking the law
    acts like a viscous coupler (c_eff = 2*mu*N*alpha_reg); once the transmitted
    force reaches the slip value the interface slides and the force saturates.
    Being odd in the velocity, it generates only odd harmonics. The force
    depends on the velocity alone, so ``jac_u`` is zero -- the normal and
    rotational springs of a real bolted joint are separate ``linear`` entries on
    their own DoFs, not part of this element.

    :param mu: friction coefficient [-].
    :param N: clamping force [N] (e.g. bolt preload), independent of the
        excitation amplitude F0.
    :param alpha_reg: tanh regularization sharpness [s/m]; ideal Coulomb is
        approached for ||v_T|| >> 1/alpha_reg.
    :param dofs: the tangential pair the isotropic law couples (exactly two).
    :param spin_dof: rotational DoF carrying the torsional friction
        M = 2 mu N G tanh(alpha_reg G rdot). Optional, but it and ``G`` are two
        halves of one term: give both to enable it, or neither for pure
        translational friction.
    :param G: effective contact radius [m] of the torsional friction: G*rdot is
        the sliding speed fed to the same regularization and the slip moment is
        2*mu*N*G.
    """

    def __init__(self, mu, N, alpha_reg, dofs=("ux", "uy"), spin_dof=None, G=0.0):
        super().__init__(dofs)
        assert len(self.idx) == 2, (
            f"FrictionJoint acts on a tangential PAIR of DoFs, got {self.dofs}")
        self.mu = float(mu)
        self.N = float(N)
        self.alpha_reg = float(alpha_reg)
        self.G = float(G)
        self.spin_dof = spin_dof
        # Half a torsional specification would silently do nothing, so it is
        # rejected instead: either both halves or neither.
        if (spin_dof is None) != (self.G == 0.0):
            raise ValueError(
                f"FrictionJoint: the torsional term needs spin_dof AND a "
                f"non-zero G, got spin_dof={spin_dof!r}, G={self.G!r}. Pass "
                f"both to enable it, or neither for pure translational friction.")
        self.spin = None if spin_dof is None else int(resolve_dofs((spin_dof,))[0])

    @property
    def _spin_active(self):
        # the constructor rejects half a specification, so spin_dof alone decides
        return self.spin is not None

    def constrained_dofs(self):
        return tuple(self.idx) + ((self.spin,) if self._spin_active else ())

    def force(self, u_rel, udot_rel):
        vT = udot_rel[:, self.idx, 0]                                # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))
        # s = tanh(alpha_reg*g)/g, continued by its limit alpha_reg at g = 0
        s = np.where(g > 1e-12,
                     np.tanh(self.alpha_reg * g) / np.maximum(g, 1e-30),
                     self.alpha_reg)
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (2.0 * self.mu * self.N * s)[:, None] * vT
        if self._spin_active:
            f[:, self.spin, 0] = (2.0 * self.mu * self.N * self.G
                                  * np.tanh(self.alpha_reg * self.G
                                            * udot_rel[:, self.spin, 0]))
        return f

    def jac_u(self, u_rel, udot_rel):
        # the slip force amplitude is constant -- no displacement dependence.
        return self._zeros_J(u_rel)

    def jac_udot(self, u_rel, udot_rel):
        # df_T/dv_T = 2 mu N [ alpha(1-t^2) vhat vhat^T + (t/g)(I - vhat vhat^T) ]:
        # tanh slope along the sliding direction, direction-turning term across
        # it; both terms tend to alpha_reg as g -> 0 (the viscous stick limit).
        vT = udot_rel[:, self.idx, 0]                                # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))
        gs = np.maximum(g, 1e-30)
        t = np.tanh(self.alpha_reg * g)
        tog = np.where(g > 1e-12, t / gs, self.alpha_reg)
        coef = np.where(g > 1e-12, self.alpha_reg * (1 - t ** 2), self.alpha_reg)
        vhat = vT / gs[:, None]                                      # 0 at g = 0
        vv = vhat[:, :, None] * vhat[:, None, :]
        I2 = np.eye(2)[None]
        JTT = (2.0 * self.mu * self.N) * (coef[:, None, None] * vv
                                          + tog[:, None, None] * (I2 - vv))
        J = self._zeros_J(u_rel)
        J[:, self.idx[:, None], self.idx[None, :]] = JTT
        if self._spin_active:
            t_s = np.tanh(self.alpha_reg * self.G * udot_rel[:, self.spin, 0])
            J[:, self.spin, self.spin] = (2.0 * self.mu * self.N * self.alpha_reg
                                          * self.G ** 2 * (1.0 - t_s ** 2))
        return J


JOINT_TYPES = {"linear": LinearJoint, "cubic": CubicJoint, "friction": FrictionJoint}


def make_joints(specs):
    """Build the element list from a list of dicts, each with a ``"type"`` key
    naming an entry of :data:`JOINT_TYPES`; the remaining keys are the element's
    constructor arguments."""
    joints = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in JOINT_TYPES:
            raise ValueError(f"unknown joint type {kind!r}; valid types are "
                             f"{', '.join(JOINT_TYPES)}")
        joints.append(JOINT_TYPES[kind](**spec))
    return joints


class CoupledJointCB(SecondOrderODE):
    """
    The coupled reduced testbench with an ASSEMBLED joint, as a pyhbm
    second-order system:

        M q'' + C q' + K q + Bc^T sum_e f_e(x, xdot) = f_r F0 cos(tau)

    with x = Bc q the 6 relative virtual-point DoFs (VP_A - VP_B). M, C, K are
    the linearly UNCOUPLED block-diagonal matrices of :func:`assemble_coupled`;
    the ENTIRE joint -- linear springs and dampers included -- enters through
    the nonlinear term as the sum of the :class:`JointElement` contributions.
    Each element writes only its own DoF rows, so the elements are independent
    and freely combinable. qdot handed over by pyhbm is the PHYSICAL velocity,
    so the elements need no extra omega scaling.

    :param joints: list of :class:`JointElement` (see :func:`make_joints`).
    :param polynomial_degree: AFT sampling knob, N_t = (degree+1)*max(|h|)+1.
        It is EXACT for the polynomial laws (linear = degree 1, cubic =
        degree 3), so 3 resolves a linear+cubic joint without aliasing. The
        tanh friction is not a polynomial and has no exact sampling: a run with
        a ``friction`` element needs a large value, chosen so that the aliasing
        of the friction force is negligible. This is the pyhbm analogue of the
        pyFBS ``sample_number``, but not the same knob -- pyFBS takes the
        sample count directly, here it follows from the degree and the highest
        harmonic.
    """
    is_real_valued = True

    def __init__(self, M, C, K, Bc, f_r, F0, joints, polynomial_degree):
        self.mass_matrix = M
        self.damping_matrix = C
        self.stiffness_matrix = K
        self.dimension = M.shape[0]
        self.polynomial_degree = polynomial_degree
        self.Bc = Bc
        self.f_r = np.asarray(f_r, dtype=float)
        self.F0 = float(F0)
        self.joints = list(joints)

        # A DoF with neither stiffness nor friction leaves A and B unconnected
        # there: the coupled residual has (almost) no restoring term on that gap
        # component and the Newton system becomes ill-conditioned.
        held = set()
        for element in self.joints:
            held.update(int(i) for i in element.constrained_dofs())
        loose = [VP_DOFS[i] for i in range(N_IF) if i not in held]
        if loose:
            print(f"warning: interface DoF(s) {', '.join(loose)} carry neither "
                  f"stiffness nor friction -- the coupled problem is "
                  f"unconstrained there and likely ill-conditioned")

    def external_term(self, adimensional_time):
        tau = np.asarray(adimensional_time)
        return (self.F0 * np.cos(tau))[:, None, None] * self.f_r[None, :, None]

    # --- pyhbm interface: the joint acts on x = Bc q, so f_nl = Bc^T f_joint
    # (the matmuls broadcast the (6, d) coupling over the Nt time samples)
    def nonlinear_term(self, q, q_dot, adimensional_time):
        x = self.Bc @ q                                        # (Nt, 6, 1)
        xdot = self.Bc @ q_dot                                 # physical velocity
        f = np.zeros((x.shape[0], N_IF, 1))
        for element in self.joints:
            f += element.force(x, xdot)
        return self.Bc.T @ f

    def jacobian_nonlinear_term(self, q, q_dot, adimensional_time):
        x, xdot = self.Bc @ q, self.Bc @ q_dot
        J = np.zeros((x.shape[0], N_IF, N_IF))
        for element in self.joints:
            J += element.jac_u(x, xdot)
        return self.Bc.T @ J @ self.Bc                         # (Nt, d, d)

    def jacobian_nonlinear_term_qdot(self, q, q_dot, adimensional_time):
        x, xdot = self.Bc @ q, self.Bc @ q_dot
        J = np.zeros((x.shape[0], N_IF, N_IF))
        for element in self.joints:
            J += element.jac_udot(x, xdot)
        return self.Bc.T @ J @ self.Bc
