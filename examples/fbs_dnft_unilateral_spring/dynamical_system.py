import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numpy import cos, sin, array, concatenate
from pyhbm.dynamical_system import FBS_System

class dlft_unilateral(FBS_System):
    """
     System
      - 2-DOF chain: ground — k — M1 — k — M2 — k — ground
      - Unilateral spring on DOF 1: stiffness k_contact = 100, gap δ = 1
      - M = [1, 1], k = [1, 1], d = 0.03·k (proportional damping)
      - Excitation: harmonic force F = 0.1 on DOF 2
      - Frequency sweep: ω ∈ [0.5, 0.8]
    """

    is_real_valued = True

    def __init__(self, P=0.1):
        """
		Initializes the linear FBS system.

		:param d: Damping coefficient (nearest-neighbour) [T^-1]
		:param k: Stiffness coefficient (nearest-neighbour) [T^-2]
		:param P: Amplitude of the external force on DOF 0 of subsystem A [L T^-2]
		"""
        self.P = P
        self.mass_matrix = np.diag([1, 1])
        self.stiffness_matrix = np.array([[1, -1], [-1, 2]])
        self.damping_matrix = 0.15*self.stiffness_matrix
        self.B_coupling = np.array([[1, 0]])
        self.total_dimension = self.mass_matrix.shape[0]
        self.dimension = 1
        self.polynomial_degree = 50

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        External force P*cos(tau) on mass 1 only.

        :param adimensional_time: Adimensional time, shape (Nt,)
        :return: External force array, shape (Nt, 2, 1)
        """
        f = np.zeros((len(adimensional_time), self.total_dimension, 1))
        f[:, 1, 0] = self.P * cos(adimensional_time)
        return f

    def interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        return np.zeros((len(adimensional_time), 1))

    def jacobian_interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        return np.zeros((len(adimensional_time), 1, 1))

    def jacobian_interface_force_qdot(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        return np.zeros((len(adimensional_time), 1, 1))



class aft_unilateral_spring(FBS_System):
    is_real_valued = True

    def __init__(self, P=0.1):
        self.P = P
        self.mass_matrix = np.diag([1, 1])
        self.stiffness_matrix = np.array([[1, -1], [-1, 2]])
        self.damping_matrix = 0.03 * self.stiffness_matrix
        self.B_coupling = np.array([[1, 0]])
        self.total_dimension = self.mass_matrix.shape[0]
        self.dimension = 1
        self.polynomial_degree = 50

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        External force P*cos(tau) on mass 1 only.

        :param adimensional_time: Adimensional time, shape (Nt,)
        :return: External force array, shape (Nt, 2, 1)
        """
        f = np.zeros((len(adimensional_time), self.total_dimension, 1))
        f[:, 1, 0] = self.P * cos(adimensional_time)
        return f

    def interface_force(self, u_rel, udot_rel, adimensional_time):
        k_contact = 100.0
        f = np.where(u_rel > 0.0, k_contact * u_rel, 0.0)  # gap already subtracted by g_zero offset
        return f  # shape (Nt, 1, 1)

    def jacobian_interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        k_contact = 100.0
        f = np.where(u_rel > 0.0, np.eye(self.dimension)*k_contact, 0.0)  # gap already subtracted by g_zero offset
        return f

    def jacobian_interface_force_qdot(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        return np.zeros((len(adimensional_time), 1, 1))