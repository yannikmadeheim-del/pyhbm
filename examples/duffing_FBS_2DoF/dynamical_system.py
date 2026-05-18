# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import scipy as sp
import numpy as np
from numpy import cos, array, concatenate
from numpy.typing import ArrayLike
from pyhbm import FBS_System
from pyhbm.dynamical_system import FirstOrderODE


# %%
class System2DoF_1stOrder(FirstOrderODE):
    """
    Class that implements the dynamics

    u1dot = omega u1' = v1
    v1dot = omega v1' = -k2*(u1) -c2*v1 - k3*(u1-u2) - beta*((u1-u2)**3) - alpha*(v1-v2)((u1-u2)**2) + P*cos(tau)
    u2dot = omega u2' = v2
    v2dot = omega v2' = -k2*(u2) -c2*v2 - k3*(u1-u2) - beta*((u2-u1)**3) - alpha*(v1-v2)((u1-u2)**2)

    where:
    - u is the displacement
    - v is the velocity
    - k is the stiffness coefficient per unit mass [T^-2]
    - c is the damping coefficient per unit mass [T^-1]
    - beta is the nonlinearity coefficient per unit mass [L^-2 T^-2]
    - P is the amplitude of the external force per unit mass [L T^-2]
    - tau is the adimensional time, defined as tau = omega * t
    - omega is the frequency of the external force
    - t is the physical time
    - f(z, tau) is the force vector, where z = [u, v] is the state vector
    - zdot = omega z' = f(z, tau)

    """
    is_real_valued = True

    def __init__(self, c1=0.01, c2=0.01, k1=1.0, k2=1.0, k3=1.0, beta=1.0, alpha=0.1, P=20.0):
        """
        Initializes the Duffing oscillator parameters.

        :param c: Damping coefficient per unit mass [T^-1]
        :param k: Stiffness coefficient per unit mass [T^-2]
        :param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
        :param P: Amplitude of the external force per unit mass [L T^-2]
        """
        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.omega_resonance_linear = np.sqrt(k1)
        self.beta = beta
        self.alpha = alpha
        self.P = P
        self.linear_coefficient = array([
            [0.0, 1.0, 0.0, 0.0],
            [-k3-k1, -c1, k3, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [k3, 0.0, -k3-k2, -c2]
        ])
        self.dimension = self.linear_coefficient.shape[
            0]  # 1 dimensional in second order and 2 dimensional in first order
        self.polynomial_degree = 3

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        Calculates the external forcing term.

        :param adimensional_time: Time at which to evaluate the external force
        :return: External force array
        """
        zeros = np.zeros_like(adimensional_time)
        force_ext = self.P * cos(adimensional_time)
        return array([[zeros, force_ext, zeros, zeros]]).transpose()

    def linear_term(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the linear term.

        :param state: State vector
        :return: Linear term array
        """
        return self.linear_coefficient @ state

    def nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """
        Calculates the nonlinear term.

        :param state: State vector
        :return: Nonlinear term array
        """
        u1 = state[..., 0:1, :]  # Select first element along the second-to-last axis
        u2 = state[..., 2:3, :]
        v1 = state[..., 1:2, :]
        v2 = state[..., 3:4, :]
        zeros = np.zeros_like(u1)
        fnl1 = -self.beta * np.power(u1-u2, 3) -self.alpha * np.power(u1-u2, 2) * (v1-v2)
        fnl2 = -self.beta * np.power(u2-u1, 3) -self.alpha * np.power(u2-u1, 2) * (v2-v1)
        return concatenate((zeros, fnl1, zeros, fnl2), axis=-2)

    def jacobian_nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian of the nonlinear term.

        :param state: State vector
        :return: Jacobian of the nonlinear term
        """
        u1 = state[..., 0:1, :]  # Select first element along the second-to-last axis
        u2 = state[..., 2:3, :]
        v1 = state[..., 1:2, :]
        v2 = state[..., 3:4, :]
        zeros = np.zeros_like(u1)
        dfnl1du1 = - 3 * self.beta * np.power(u1-u2, 2) - 2 * self.alpha * (u1-u2) * (v1-v2)
        dfnl1du2 = + 3 * self.beta * np.power(u1-u2, 2) + 2 * self.alpha * (u1-u2) * (v1-v2)
        dfnl2du1 = + 3 * self.beta * np.power(u2-u1, 2) + 2 * self.alpha * (u2-u1) * (v2-v1)
        dfnl2du2 = - 3 * self.beta * np.power(u2-u1, 2) - 2 * self.alpha * (u2-u1) * (v2-v1)
        dfnl1dv1 = - self.alpha * np.power(u1-u2, 2)
        dfnl1dv2 = + self.alpha * np.power(u1-u2, 2)
        dfnl2dv1 = + self.alpha * np.power(u2-u1, 2)
        dfnl2dv2 = - self.alpha * np.power(u2-u1, 2)
        jacobian_zeros = np.concatenate((zeros, zeros, zeros, zeros), axis=-1)
        jacobian1 = np.concatenate((dfnl1du1, dfnl1dv1, dfnl1du2, dfnl1dv2), axis=-1)
        jacobian2 = np.concatenate((dfnl2du1, dfnl2dv1, dfnl2du2, dfnl2dv2), axis=-1)
        return concatenate((jacobian_zeros, jacobian1, jacobian_zeros, jacobian2), axis=-2)

    def jacobian_parameters(self,
                            state: np.ndarray,
                            adimensional_time: np.ndarray,
                            output_c=False,
                            output_k=False,
                            output_beta=False,
                            output_P=False) -> np.ndarray:
        """
        Computes the Jacobian w.r.t the parameters c, k, beta and P.

        :param state: State vector
        :param adimensional_time: Time for evaluating external force
        :param output_c: Boolean to decide whether to compute and output the jacobian w.r.t c
        :param output_k: Boolean to decide whether to compute and output the jacobian w.r.t k
        :param output_beta: Boolean to decide whether to compute and output the jacobian w.r.t beta
        :param output_P: Boolean to decide whether to compute and output the jacobian w.r.t P
        :return: Jacobian w.r.t the parameters
        """
        jacobian_c, jacobian_k, jacobian_beta, jacobian_P = None, None, None, None

        u = state[..., 0:1, :]  # Select first element along the second-to-last axis
        zeros = np.zeros_like(u)

        if output_c:
            # Select second element along the second-to-last axis
            v = state[..., 1:2, :]
            # concatenate along rows to form a column
            jacobian_c = concatenate((zeros, -v), axis=-2)

        if output_k:
            # concatenate along rows to form a column
            jacobian_k = concatenate((zeros, -u), axis=-2)

        if output_beta:
            # concatenate along rows to form a column
            jacobian_beta = concatenate((zeros, -np.power(u, 3)), axis=-2)

        if output_P:
            jacobian_P = array([[np.zeros_like(adimensional_time), cos(adimensional_time)]]).transpose()

        return jacobian_c, jacobian_k, jacobian_beta, jacobian_P

class System2DoF_FBS(FBS_System):
    """
    Implements the coupling of 2 Substructures with the dynamics
        m1*q1'' + c1*q1' + k1*q1 = P*cos(tau)
        m2*q2'' + c2*q2' + k2*q2 = P*cos(tau)
    Coupled by a cubic Spring
        fnl = beta*(q1-q2)**3
    """

    is_real_valued = True

    def __init__(self, c1=0.01, c2=0.01, k1=1.0, k2=1.0, k3=1.0, beta=1.0, alpha=0.1, P=1.0):
        """
		Initializes the linear FBS system.

		:param d: Damping coefficient (nearest-neighbour) [T^-1]
		:param k: Stiffness coefficient (nearest-neighbour) [T^-2]
		:param P: Amplitude of the external force on DOF 0 of subsystem A [L T^-2]
		"""
        M1 = [[1]]
        M2 = [[1]]
        C1 = [[c1]]
        C2 = [[c2]]
        K1 = [[k1]]
        K2 = [[k2]]

        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.beta = beta
        self.alpha = alpha
        self.P = P
        self.mass_matrix = sp.linalg.block_diag(M1, M2)
        self.damping_matrix = sp.linalg.block_diag(C1, C2)
        self.stiffness_matrix = sp.linalg.block_diag(K1, K2)
        self.B_coupling = np.array([[1, -1]])  # u_rel = q1 - q2
        self.total_dimension = 2
        self.dimension = 1
        self.polynomial_degree = 3

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        External force P*cos(tau) on mass 1 only.

        :param adimensional_time: Adimensional time, shape (Nt,)
        :return: External force array, shape (Nt, 2, 1)
        """
        f = np.zeros((len(adimensional_time), self.total_dimension, 1))
        f[:, 0, 0] = self.P * cos(adimensional_time)
        return f

    def interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Cubic interface force: lambda = beta * u_rel^3"""
        return self.k3 * u_rel + self.beta * np.power(u_rel, 3) + self.alpha * np.power(u_rel, 2) * udot_rel

    def jacobian_interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """d(lambda)/d(u_rel) = 3 * beta * u_rel^2, shape (Nt, 1, 1)"""
        J_int = self.k3 + 3 * self.beta * np.power(u_rel, 2) + 2 * self.alpha * u_rel * udot_rel
        return J_int

    def jacobian_interface_force_qdot(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """No velocity dependence."""
        Jdot_int = self.alpha * np.power(u_rel, 2)
        return Jdot_int


class System2DoF_FBS_experimental(System2DoF_FBS):
    """
    Experimental variant of System2DoF_FBS.
    Pre-computes the uncoupled FRF Y(omega) = (-omega^2 * M + i*omega * C + K)^{-1}
    from the subsystem matrices and stores it as self.omega_frf, self.Y_frf
    for interpolation inside FrequencyBasedSubstructuring_experimental.
    """

    def __init__(self, c1=0.01, c2=0.01, k1=1.0, k2=1.0, beta=1.0, P=1.0):
        super().__init__(c1=c1, c2=c2, k1=k1, k2=k2, beta=beta, P=P)
        omega_start = 0.00
        omega_end = 15.0*9*2
        ome_density = 1000.0
        n_points = int((omega_end - omega_start) * ome_density)  # 100 points per rad/s
        omega_frf = np.linspace(omega_start, omega_end, n_points)
        Y_frf = np.zeros((n_points, self.total_dimension, self.total_dimension), dtype=complex)
        for i, w in enumerate(omega_frf):
            Y_frf[i] = np.linalg.solve(
                -w ** 2 * self.mass_matrix + 1j * w * self.damping_matrix + self.stiffness_matrix,
                np.eye(self.total_dimension)
            )
        self.omega_frf = omega_frf
        self.Y_frf = Y_frf
