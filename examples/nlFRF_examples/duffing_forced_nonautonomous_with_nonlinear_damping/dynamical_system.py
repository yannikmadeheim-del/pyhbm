#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from numpy import cos, sin, array, concatenate
from pyhbm.dynamical_system import FirstOrderODE, SecondOrderODE

#%%
class DuffingForced_nonlin_Damping(FirstOrderODE):
	"""
	Class that implements the dynamics
	
	udot = omega u' = v
	vdot = omega v' = -k*u -c*v - beta*(u**3) - alpha*v*(u**2) + P*cos(tau)
	
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
	def __init__(self, c=0.009, k=1.0, beta=0.1, alpha=0.1, P=1.0):
		"""
		Initializes the Duffing oscillator parameters.

		:param c: Damping coefficient per unit mass [T^-1]
		:param k: Stiffness coefficient per unit mass [T^-2]
		:param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
		:param P: Amplitude of the external force per unit mass [L T^-2]
		"""
		self.c = c
		self.k = k
		self.omega_resonance_linear = np.sqrt(k)
		self.beta = beta
		self.alpha = alpha
		self.P = P # also serves as reference force level
		self.linear_coefficient = array([[0.0, 1.0], [-k, -c]])
		self.dimension = self.linear_coefficient.shape[0] # 1 dimensional in second order and 2 dimensional in first order
		self.polynomial_degree = 3

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external forcing term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External force array
		"""
		zeros = np.zeros_like(adimensional_time)
		force_ext = self.P * cos(adimensional_time)
		return array([[zeros, force_ext]]).transpose()

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
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		v = state[..., 1:2, :]
		zeros = np.zeros_like(u)
		fnl = -self.beta * np.power(u, 3) - self.alpha * v * np.power(u, 2)#* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]
		return concatenate((zeros, fnl), axis=-2)

	def all_terms(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Combines all terms (linear, nonlinear, external) to compute the total force.

		:param state: State vector
		:param adimensional_time: Time for evaluating external force
		:return: Total force array
		"""
		return self.linear_term(state) + \
				self.nonlinear_term(state, adimensional_time) + \
				self.external_term(adimensional_time)

	def jacobian_nonlinear_term(self, state: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term.

		:param state: State vector
		:return: Jacobian of the nonlinear term
		"""
		u = state[..., 0:1, :]  # Select first element along the second-to-last axis
		v = state[..., 1:2, :]
		zeros = np.zeros_like(u)
		dfnldu = -3 * self.beta * np.power(u, 2) - 2 * self.alpha * v * u#* array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
		dfnldv = -self.alpha * np.power(u, 2)
		jacobian1 = np.concatenate((zeros, zeros), axis=-1)
		jacobian2 = np.concatenate((dfnldu, dfnldv), axis=-1)
		return concatenate((jacobian1, jacobian2), axis=-2)
	
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


# %%
class DuffingForced_nonlin_Damping_SecondOrder(SecondOrderODE):
	"""
	Class that implements the dynamics

	q'' + c*q' + k*q + beta*(q**3) + alpha*q'*(q**2) = P*cos(tau)

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

	def __init__(self, c=0.009, k=1.0, beta=0.1, alpha=0.1, P=1.0):
		"""
		Initializes the Duffing oscillator parameters.

		:param c: Damping coefficient per unit mass [T^-1]
		:param k: Stiffness coefficient per unit mass [T^-2]
		:param beta: Nonlinearity coefficient per unit mass [L^-2 T^-2]
		:param P: Amplitude of the external force per unit mass [L T^-2]
		"""
		self.c = c
		self.k = k
		self.omega_resonance_linear = np.sqrt(k)
		self.beta = beta
		self.alpha = alpha
		self.P = P  # also serves as reference force level
		self.mass_matrix = np.eye(1)
		self.damping_matrix = np.array([[c]])
		self.stiffness_matrix = np.array([[k]])
		self.dimension = 1  		# 1 dimensional in second order and 2 dimensional in first order
		self.polynomial_degree = 3

	def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the external forcing term.

		:param adimensional_time: Time at which to evaluate the external force
		:return: External force array
		"""
		force_ext = self.P * cos(adimensional_time)
		return array([[force_ext]]).transpose()

	def nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Calculates the nonlinear term.

		:param q: desplacement vector
		:return: Nonlinear term array
		"""
		fnl = self.beta * np.power(q, 3) + self.alpha * q_dot * np.power(q, 2)# * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]
		return fnl

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
			- self.nonlinear_term(q, q_dot, adimensional_time)
			+ self.external_term(adimensional_time)
		)

	def jacobian_nonlinear_term(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term. dfnl/dq

		:param q: Displacement vector
		:return: Jacobian of the nonlinear term
		"""
		dfnldq = 3 * self.beta * np.power(q,2) + 2* self.alpha * q_dot * q # * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
		return dfnldq

	def jacobian_nonlinear_term_qdot(self, q: np.ndarray, q_dot: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
		"""
		Computes the Jacobian of the nonlinear term. dfnl/dqdot

		:param q: Displacement vector
		:return: Jacobian of the nonlinear term
		"""
		dfnldqdot = self.alpha * np.power(q,2)   # * array(cos(adimensional_time))[...,np.newaxis,np.newaxis]  # Correct coefficient for cubic nonlinearity
		return dfnldqdot

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