"""
Dry-friction joint on the RBE2/RBE3 + Craig-Bampton testbench model.

The Ansys import, the RBE2/RBE3 interface condensation, the Craig-Bampton
reduction and the physical-CSV export are shared with the sibling example
``testbench_cubicSpring_CB`` (which also holds the FE data); only the joint law
is new. :class:`CoupledDryFrictionCB` replaces its cubic spring by the
regularized Coulomb friction of the pyFBS ``testbench_dry_friction`` example,
so the same physical problem is solved once through Craig-Bampton (here) and
once through FBS (pyFBS) and the two branches can be overlaid.
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

# re-exported so that main.py imports everything from one module
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

N_IF = 6                 # virtual-point interface DoFs: [ux, uy, uz, rx, ry, rz]


class CoupledDryFrictionCB(SecondOrderODE):
    """
    The coupled reduced testbench with a DRY-FRICTION joint, as a pyhbm
    second-order system:

        M q'' + C q' + K q + Bc^T f_joint(x, xdot) = f_r F0 cos(tau)

    with x = Bc q the 6 relative master DoFs (VP_A - VP_B). M, C, K are the
    linearly UNCOUPLED block-diagonal matrices of :func:`assemble_coupled`; the
    ENTIRE joint enters through the nonlinear term. The law is the one of the
    pyFBS testbench_dry_friction example and mirrors the physical joint (B
    clamps A's plate, contact normal along z, slip in the x-y plane and spin
    about z):

        f_T   = 2 mu N tanh(alpha ||v_T||) v_T/||v_T||   (ux, uy -- friction)
        f_z   = k_trans z                                (uz -- penalty spring)
        M_rxy = k_rot theta                              (rx, ry -- penalty springs)
        M_rz  = 2 mu N G tanh(alpha G rzdot)             (torsional friction)

    Every DoF's term is decoupled from the others. qdot handed over by pyhbm is
    the PHYSICAL velocity, so the friction terms need no extra omega scaling.

    :param mu_trans: friction coefficient [-]
    :param N: bolt clamping force [N]; both clamp faces carry mu_trans*N, so
        the tangential force saturates at the slip force 2*mu_trans*N
    :param alpha: tanh regularization sharpness [s/m]; near sticking the joint
        acts like a viscous damper c_eff = 2*mu_trans*N*alpha
    :param k_trans: penalty stiffness [N/m] tying the normal (z) gap
    :param k_rot: rotational penalty stiffness [Nm/rad] tying rx, ry
    :param G: effective contact radius [m] of the torsional friction
    :param polynomial_degree: AFT sampling knob, N_t = (degree+1)*max(h)+1.
        The tanh law is not a polynomial, so unlike the cubic spring there is
        no exact sampling -- it has to be large enough for the aliasing of the
        friction force to be negligible.
    """
    is_real_valued = True

    def __init__(self, M, C, K, Bc, f_r, F0, mu_trans, N, alpha, k_trans,
                 k_rot, G, polynomial_degree):
        self.mass_matrix = M
        self.damping_matrix = C
        self.stiffness_matrix = K
        self.dimension = M.shape[0]
        self.polynomial_degree = polynomial_degree
        self.Bc = Bc
        self.f_r = np.asarray(f_r, dtype=float)
        self.F0 = float(F0)
        self.mu_trans = float(mu_trans)
        self.N = float(N)
        self.alpha = float(alpha)
        self.k_trans = float(k_trans)
        self.k_rot = float(k_rot)
        self.G = float(G)

    def external_term(self, adimensional_time):
        tau = np.asarray(adimensional_time)
        return (self.F0 * np.cos(tau))[:, None, None] * self.f_r[None, :, None]

    # --- joint law on the 6 relative VP DoFs (as in pyFBS) ------------------
    def interface_force(self, x, xdot, tau):
        """Joint force (Nt, 6, 1) from the gap ``x`` and the PHYSICAL gap
        velocity ``xdot`` (both (Nt, 6, 1))."""
        vT = xdot[:, :2, 0]                                    # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))                     # ||v_T||
        # s = tanh(alpha*g)/g, continued by its limit alpha at g = 0
        s = np.where(g > 1e-12,
                     np.tanh(self.alpha * g) / np.maximum(g, 1e-30), self.alpha)
        f = np.zeros((len(tau), N_IF, 1))
        f[:, :2, 0] = (2.0 * self.mu_trans * self.N * s)[:, None] * vT
        f[:, 2, 0] = self.k_trans * x[:, 2, 0]
        f[:, 3, 0] = self.k_rot * x[:, 3, 0]
        f[:, 4, 0] = self.k_rot * x[:, 4, 0]
        f[:, 5, 0] = (2.0 * self.mu_trans * self.N * self.G
                      * np.tanh(self.alpha * self.G * xdot[:, 5, 0]))
        return f

    def interface_jacobian(self, x, xdot, tau):
        """df_joint/dx (Nt, 6, 6): only the penalty springs depend on the gap
        -- the slip force and moment amplitudes are constant."""
        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, 2, 2] = self.k_trans
        J[:, 3, 3] = self.k_rot
        J[:, 4, 4] = self.k_rot
        return J

    def interface_jacobian_qdot(self, x, xdot, tau):
        """
        df_joint/dxdot (Nt, 6, 6). For the tangential pair,

            df_T/dv_T = 2 mu N [ alpha (1-t^2) vhat vhat^T
                                 + (t/g) (I - vhat vhat^T) ],   t = tanh(alpha g)

        i.e. the tanh slope along the sliding direction and a direction-turning
        term across it; both limits coincide at g -> 0 (viscous stick limit
        2 mu N alpha I). The torsional friction is the 1-D version of the same
        expression, dM_rz/drzdot = 2 mu N alpha G^2 (1 - t_rz^2). The penalty
        springs have no velocity term.
        """
        vT = xdot[:, :2, 0]                                    # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))
        gs = np.maximum(g, 1e-30)
        t = np.tanh(self.alpha * g)
        tog = np.where(g > 1e-12, t / gs, self.alpha)                    # t/g -> alpha
        coef = np.where(g > 1e-12, self.alpha * (1 - t ** 2), self.alpha)  # -> alpha
        vhat = vT / gs[:, None]                                # 0 at g = 0
        vv = vhat[:, :, None] * vhat[:, None, :]               # vhat vhat^T
        I2 = np.eye(2)[None]

        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, :2, :2] = (2.0 * self.mu_trans * self.N) * (
            coef[:, None, None] * vv + tog[:, None, None] * (I2 - vv))
        t_rz = np.tanh(self.alpha * self.G * xdot[:, 5, 0])
        J[:, 5, 5] = (2.0 * self.mu_trans * self.N * self.alpha * self.G ** 2
                      * (1.0 - t_rz ** 2))
        return J

    # --- pyhbm interface: the joint acts on x = Bc q, so f_nl = Bc^T f_joint
    # (the matmuls broadcast the (6, d) coupling over the Nt time samples)
    def nonlinear_term(self, q, q_dot, adimensional_time):
        x = self.Bc @ q                                        # (Nt, 6, 1)
        xdot = self.Bc @ q_dot                                 # physical velocity
        return self.Bc.T @ self.interface_force(x, xdot, adimensional_time)

    def jacobian_nonlinear_term(self, q, q_dot, adimensional_time):
        x, xdot = self.Bc @ q, self.Bc @ q_dot
        J = self.interface_jacobian(x, xdot, adimensional_time)
        return self.Bc.T @ J @ self.Bc                         # (Nt, d, d)

    def jacobian_nonlinear_term_qdot(self, q, q_dot, adimensional_time):
        x, xdot = self.Bc @ q, self.Bc @ q_dot
        J = self.interface_jacobian_qdot(x, xdot, adimensional_time)
        return self.Bc.T @ J @ self.Bc
