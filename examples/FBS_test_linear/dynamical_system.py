#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numpy import cos, sin, array, concatenate
from pyhbm.dynamical_system import FirstOrderODE, SecondOrderODE



# %%
class DuffingForced_SecondOrder(SecondOrderODE):
	"""
	Class that implements the dynamics

	q'' + c*q' + k*q + beta*(q**3) = P*cos(tau)

	where:
	- q is the displacement (' = d/dt)
	- k is the stiffness coefficient per unit mass [T^-2]
	- c is the damping coefficient per unit mass [T^-1]
	- beta is the nonlinearity coefficient per unit mass [L^-2 T^-2]
	- P is the amplitude of the external force per unit mass [L T^-2]
	- tau is the adimensional time, defined as tau = omega * t
	- omega is the frequency of the external force
	- t is the physical time
	"""

	def __init__(self, d=0.05, k=200.0, P=1.0):
		"""
		Initializes the Duffing oscillator parameters.

		:param c: Damping coefficient per unit mass [T^-1]
		:param k: Stiffness coefficient per unit mass [T^-2]
		:param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
		:param P: Amplitude of the external force per unit mass [L T^-2]
		"""
		self.d = d
		self.k = k
		self.omega_resonance_linear = np.sqrt(k)
		self.P = P  # also serves as reference force level
		self.mass_matrix = np.diag([1, 1, 1, 1])
		self.damping_matrix = np.array([
			[ d,    -d,      0,      0  ],
			[-d,  2*d,     -d,      0  ],
			[ 0,    -d,    2*d,    -d  ],
			[ 0,     0,     -d,      d  ]
		])
		self.stiffness_matrix = np.array([
			[ k,   -k,    0,    0],
			[-k,  2*k,   -k,   0],
			[ 0,   -k,   2*k,  -k],
			[ 0,    0,    -k,   k]
		])
		self.dimension = 4  		# 4 DOFs in second-order formulation
		self.polynomial_degree = 1

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external forcing term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External force array
		"""
		Nt = len(adimensional_time)
		f = np.zeros((Nt, self.dimension, 1))
		f[:, 0, 0] = self.P * cos(adimensional_time)
		return f

	def nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the nonlinear term.

		:param q: desplacement vector
		:return: Nonlinear term array
		"""
		return np.zeros_like(q)

	def all_terms(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Combines all terms (linear, nonlinear, external) to compute the total force.

		:param state: State vector
		:param adimensional_time: Time for evaluating external force
		:return: Total force array
		"""
		return (
			- self.stiffness_matrix @ q
			- self.damping_matrix @ q_dot
			- self.nonlinear_term(q, adimensional_time)
			+ self.external_term(adimensional_time)
		)

	def jacobian_nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term. dfnl/dq

		:param q: Displacement vector
		:return: Jacobian of the nonlinear term
		"""
		return np.zeros((q.shape[0], self.dimension, self.dimension))

	def jacobian_nonlinear_term_qdot(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term. dfnl/dqdot

		:param q: Displacement vector
		:return: Jacobian of the nonlinear term
		"""
		return np.zeros((q.shape[0], self.dimension, self.dimension))

	def jacobian_parameters(self,
	                        q: np.ndarray,
	                        q_dot: np.ndarray,
	                        adimensional_time: np.ndarray,
	                        output_c=False,
	                        output_k=False,
	                        output_beta=False,
	                        output_P=False) -> np.ndarray:
		"""
		Computes the Jacobian w.r.t the parameters c, k, beta and P.

		:param q: displacement vector
		:param q_dot: velocity vector
		:param adimensional_time: Time for evaluating external force
		:param output_c: Boolean to decide whether to compute and output the jacobian w.r.t c
		:param output_k: Boolean to decide whether to compute and output the jacobian w.r.t k
		:param output_beta: Boolean to decide whether to compute and output the jacobian w.r.t beta
		:param output_P: Boolean to decide whether to compute and output the jacobian w.r.t P
		:return: Jacobian w.r.t the parameters
		"""
		jacobian_c, jacobian_k, jacobian_beta, jacobian_P = None, None, None, None


		if output_c:
			jacobian_c = -q_dot

		if output_k:
			jacobian_k = -q

		if output_beta:
			jacobian_beta = -np.power(q, 3)

		if output_P:
			jacobian_P = array([cos(adimensional_time)]).transpose()

		return jacobian_c, jacobian_k, jacobian_beta, jacobian_P