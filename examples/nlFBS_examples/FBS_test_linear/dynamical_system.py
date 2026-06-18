#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import scipy as sp
import numpy as np
from numpy import cos, sin, array, concatenate
from pyhbm.dynamical_system import FBS_System, SecondOrderODE


# %%
class linearFBS_System_numerical(FBS_System):
	"""
	Linear FBS test system: two identical 4-DOF chain subsystems A and B,
	connected at 3 interface DOFs with a linear coupling force.

	Subsystem equations (identical for A and B):
		M u'' + D u' + K u = f_ext + B^T lambda

	where M = I_4, D and K are symmetric tridiagonal matrices with
	nearest-neighbour coupling coefficient d and k respectively.

	Interface compatibility:
		u_rel_1 = u_{2,A} - u_{1,B}
		u_rel_2 = u_{3,A} - u_{2,B}
		u_rel_3 = u_{4,A} - u_{3,B}

	Interface force (linear):
		lambda(u_rel, u_rel_dot) = k * u_rel + d * u_rel_dot

	External forcing: P * cos(tau) applied to DOF 0 of subsystem A only.
	tau = omega * t is the adimensional time.

	Total DOFs: 8 (4 per subsystem), interface DOFs: 3.
	"""

	def __init__(self, d=0.05, k=200.0, P=1.0):
		"""
		Initializes the linear FBS system.

		:param d: Damping coefficient (nearest-neighbour) [T^-1]
		:param k: Stiffness coefficient (nearest-neighbour) [T^-2]
		:param P: Amplitude of the external force on DOF 0 of subsystem A [L T^-2]
		"""
		M = np.diag([1, 1, 1, 1])
		D = np.array([
			[d, -d, 0, 0],
			[-d, 2 * d, -d, 0],
			[0, -d, 2 * d, -d],
			[0, 0, -d, d]])
		K = np.array([
			[k, -k, 0, 0],
			[-k, 2 * k, -k, 0],
			[0, -k, 2 * k, -k],
			[0, 0, -k, k]])

		self.d = d
		self.k = k
		self.omega_resonance_linear = np.sqrt(k)
		self.P = P  # also serves as reference force level
		self.mass_matrix = sp.linalg.block_diag(M, M)
		self.damping_matrix = sp.linalg.block_diag(D, D)
		self.stiffness_matrix = sp.linalg.block_diag(K, K)
		self.B_coupling = np.array([
			[0, 1, 0, 0, -1, 0, 0, 0],  # u_rel_1 = u_{2A} - u_{1B}
			[0, 0, 1, 0, 0, -1, 0, 0],  # u_rel_2 = u_{3A} - u_{2B}
			[0, 0, 0, 1, 0, 0, -1, 0]  # u_rel_3 = u_{4A} - u_{3B}
		])
		self.total_dimension = self.mass_matrix.shape[0]
		self.dimension = self.B_coupling.shape[0]
		self.polynomial_degree = 1

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external forcing term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External force array
		"""
		f = np.zeros((len(adimensional_time), self.total_dimension, 1))
		f[:, 0, 0] = self.P * cos(adimensional_time)
		return f

	def interface_force(self, u_rel: np.ndarray, u_rel_dot: np.ndarray, tau: np.ndarray) -> np.ndarray:
		"""
		Calculates the linear interface coupling force lambda.

		lambda = k * u_rel + d * u_rel_dot

		:param u_rel: Relative interface displacement, shape (Nt, n_int, 1)
		:param u_rel_dot: Relative interface velocity, shape (Nt, n_int, 1)
		:param tau: Adimensional time, shape (Nt,)
		:return: Interface force array, shape (Nt, n_int, 1)
		"""
		return self.k * u_rel + self.d * u_rel_dot

	def jacobian_interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the interface force w.r.t. u_rel: d(lambda)/d(u_rel) = k * I.

		:param u_rel: Relative interface displacement, shape (Nt, n_int, 1)
		:param udot_rel: Relative interface velocity, shape (Nt, n_int, 1)
		:param adimensional_time: Adimensional time, shape (Nt,)
		:return: Jacobian array, shape (Nt, n_int, n_int)
		"""
		return np.tile(self.k * np.eye(self.dimension),(u_rel.shape[0], 1, 1))

	def jacobian_interface_force_qdot(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the interface force w.r.t. u_rel_dot: d(lambda)/d(u_rel_dot) = d * I.

		:param u_rel: Relative interface displacement, shape (Nt, n_int, 1)
		:param udot_rel: Relative interface velocity, shape (Nt, n_int, 1)
		:param adimensional_time: Adimensional time, shape (Nt,)
		:return: Jacobian array, shape (Nt, n_int, n_int)
		"""
		return np.tile(self.d * np.eye(self.dimension),(u_rel.shape[0], 1, 1))



class linearFBS_System_experimental(linearFBS_System_numerical):
	def __init__(self, d=0.05, k=200.0, P=1.0):
		super().__init__(d,k,P)
		f_start = 0.01
		f_end = 20.0
		f_density = 1000
		freq = np.linspace(f_start, f_end, int((f_end - f_start) * f_density))
		ome = 2 * np.pi * freq
		Y_A_B = np.zeros((ome.shape[0], self.mass_matrix.shape[0], self.mass_matrix.shape[0]), dtype=complex)
		for i, w in enumerate(ome):
			Y_A_B[i, :, :] = np.linalg.solve(-w ** 2 * self.mass_matrix + 1j * w * self.damping_matrix + self.stiffness_matrix, np.eye(self.mass_matrix.shape[0]))
		self.Y_frf = Y_A_B
		self.omega_frf = ome


class linearFBS_System_Reference(SecondOrderODE):
	"""
	Full assembled 8-DOF reference system for the linear FBS test.

	Two identical 4-DOF chain subsystems A and B, coupled at 3 interface DOFs.
	The coupling is included directly in the assembled stiffness and damping matrices:

		M_full u'' + D_full u' + K_full u = f_ext

	where:
		M_full = block_diag(I_4, I_4)
		K_full = block_diag(K, K) + k * B^T @ B
		D_full = block_diag(D, D) + d * B^T @ B

	External forcing: P * cos(tau) on DOF 0 of subsystem A only.
	System is linear: no nonlinear term.
	"""

	def __init__(self, d=0.05, k=200.0, P=1.0):
		"""
		:param d: Damping coefficient (nearest-neighbour) [T^-1]
		:param k: Stiffness coefficient (nearest-neighbour) [T^-2]
		:param P: Amplitude of the external force on DOF 0 of subsystem A [L T^-2]
		"""
		self.d = d
		self.k = k
		self.P = P
		self.mass_matrix = np.eye(8)
		self.damping_matrix = np.array([
			[d,  -d,   0,   0,   0,   0,   0,  0],
			[-d, 3*d, -d,   0,  -d,   0,   0,  0],
			[0,  -d,  3*d, -d,   0,  -d,   0,  0],
			[0,   0,  -d,  2*d,  0,   0,  -d,  0],
			[0,  -d,   0,   0,  2*d, -d,   0,  0],
			[0,   0,  -d,   0,  -d,  3*d, -d,  0],
			[0,   0,   0,  -d,   0,  -d,  3*d,-d],
			[0,   0,   0,   0,   0,   0,  -d,  d]])
		self.stiffness_matrix = np.array([
			[k,  -k,   0,   0,   0,   0,   0,  0],
			[-k, 3*k, -k,   0,  -k,   0,   0,  0],
			[0,  -k,  3*k, -k,   0,  -k,   0,  0],
			[0,   0,  -k,  2*k,  0,   0,  -k,  0],
			[0,  -k,   0,   0,  2*k, -k,   0,  0],
			[0,   0,  -k,   0,  -k,  3*k, -k,  0],
			[0,   0,   0,  -k,   0,  -k,  3*k,-k],
			[0,   0,   0,   0,   0,   0,  -k,  k]])
		self.dimension = 8
		self.polynomial_degree = 1

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		External force P * cos(tau) on DOF 0 only.

		:param adimensional_time: Adimensional time, shape (Nt,)
		:return: External force array, shape (Nt, 8, 1)
		"""
		f = np.zeros((len(adimensional_time), self.dimension, 1))
		f[:, 0, 0] = self.P * cos(adimensional_time)
		return f

	def nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""Linear system — no nonlinear term."""
		return np.zeros_like(q)

	def jacobian_nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		return np.zeros((q.shape[0], self.dimension, self.dimension))

	def jacobian_nonlinear_term_qdot(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		return np.zeros((q.shape[0], self.dimension, self.dimension))