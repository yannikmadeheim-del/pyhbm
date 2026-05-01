import numpy as np
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like, vdot, sqrt
from numpy.fft import rfft, irfft, fft, ifft

from .dynamical_system import FirstOrderODE, SecondOrderODE


# %%
class Fourier(object):
    
    harmonics = unique(array([1,3])) # list of relevant harmonics
    polynomial_degree = 3
    
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(abs(harmonics))
    number_of_time_samples = (polynomial_degree+1)*harmonic_truncation_order+1
    adimensional_time_samples = linspace(0, 2*pi, number_of_time_samples, endpoint=False)
    
    @staticmethod
    def update_class_variables(harmonics: array, polynomial_degree: int):
        indexes = sorted(unique(harmonics, return_index=True)[1])
        Fourier.harmonics = array(harmonics)[indexes] # list of relevant harmonics
        Fourier.polynomial_degree = polynomial_degree

        Fourier.number_of_harmonics = len(Fourier.harmonics)
        Fourier.harmonic_truncation_order = max(abs(Fourier.harmonics))
        Fourier.number_of_time_samples = (Fourier.polynomial_degree+1)*Fourier.harmonic_truncation_order+1
        Fourier.adimensional_time_samples = linspace(0, 2*pi, Fourier.number_of_time_samples, endpoint=False)
        
    def __init__(self, coefficients: array) -> None:
        """
        Creates a Fourier instance from a set of Fourier coefficients
        """
        assert coefficients.shape[0] == Fourier.number_of_harmonics, \
            f"Number of harmonics is {Fourier.number_of_harmonics}, but {len(coefficients)} coefficients were provided."

        self.coefficients = coefficients
        # self.real_part = coefficients.real
        # self.imaginary_part = coefficients.imag
        self.time_series = None
        self.adimensional_time_derivative = None

    def compute_adimensional_time_derivative(self):
        self.adimensional_time_derivative = einsum('i,ijk->ijk', Fourier.harmonics, self.coefficients) * 1j
    
    def get_adimensional_time_derivative(self):
        if self.adimensional_time_derivative is None:
            self.compute_adimensional_time_derivative()
        return self.adimensional_time_derivative
    
    def new_from_time_series(time_series: array):
        pass
    
    def compute_time_series(self):
        pass

    def __add__(self, other):
        return Fourier(coefficients = self.coefficients + other.coefficients)
    
    def __sub__(self, other):
        return Fourier(coefficients = self.coefficients - other.coefficients)

    def matmul(self, other):
        return Fourier(coefficients = self.coefficients @ other.coefficients)

    def __mul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)

    def __rmul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)
    
    def __array__(self):
        R = vstack(self.coefficients.real)
        I = vstack(self.coefficients.imag)
        return vstack((R, I))
    
    @staticmethod
    def new_from_RI(RI: array):
        complex_dimension = len(RI) // 2
        fourier_C = RI[:complex_dimension] + 1j*RI[complex_dimension:]
        return Fourier(array(array_split(fourier_C, Fourier.number_of_harmonics)))
    
    @staticmethod
    def coefficients_to_RI(coeffs):
        R = vstack(coeffs.real)
        I = vstack(coeffs.imag)
        return vstack((R, I))
    
    @staticmethod
    def zeros(dimension: int):
        return Fourier(zeros((Fourier.number_of_harmonics, dimension, 1), dtype=complex))
    
    @staticmethod
    def new_from_first_harmonic(first_harmonic: array):
        assert 1 in Fourier.harmonics, "Fourier: Harmonic 1 is not in the list of harmonics"
        z1 = first_harmonic * 0.5 * Fourier.number_of_time_samples
        zz = zeros_like(z1)
        zz_fill = [zz if h !=1 else z1 for h in Fourier.harmonics]
        return Fourier(array(zz_fill))
    
class Fourier_Real(Fourier):
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the Real Fast Fourier transform (rFFT)
        """
        all_coefficients = rfft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        # inverse of Real FFT
        self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
        
class Fourier_Complex(Fourier):
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the Fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        shape = list(self.coefficients.shape)
        shape[0] = 2 * Fourier.harmonic_truncation_order + 1
        new_coeff = zeros(shape, dtype = complex)
        new_coeff[Fourier.harmonics] = self.coefficients
        # inverse of FFT
        self.time_series = ifft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
    
#%%

class FourierOmegaPoint(object):
    def __init__(self, fourier: Fourier, omega: float):
        self.fourier: Fourier = fourier
        self.omega: float = omega
        self.RI = None
        
    @staticmethod
    def new_from_RI_omega(RI_omega: array):
        # in case omega is not included in the array
        if len(RI_omega) % 2 == 0:
            RI_omega = vstack((RI_omega, 0))
            
        omega = RI_omega[-1,0]
        fourier = Fourier.new_from_RI(RI_omega[:-1])
        return FourierOmegaPoint(fourier=fourier, omega=omega)

    def __add__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier + other.fourier, self.omega + other.omega)

    def __sub__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier - other.fourier, self.omega - other.omega)
    
    def __mul__ (self, other: float| complex):
        return FourierOmegaPoint(self.fourier*other, self.omega*other)
    
    def __array__(self):
        if self.RI is None:
            self.RI = vstack((self.fourier.__array__(), self.omega))
        return self.RI
    
    def adimensional_time_derivative_RI(self) -> array:
        adimensional_time_derivative = self.fourier.get_adimensional_time_derivative()
        R = vstack(adimensional_time_derivative.real)
        I = vstack(adimensional_time_derivative.imag)
        return vstack((R, I, 0.0))
    
    def zero_amplitude(dimension: int, omega: float):
        return FourierOmegaPoint(Fourier.zeros(dimension), omega)
    
    def new_from_first_harmonic(first_harmonic: array, omega: float):
        return FourierOmegaPoint(Fourier.new_from_first_harmonic(first_harmonic), omega)

#%%

class JacobianFourier(object):
    
    polynomial_degree = Fourier.polynomial_degree - 1
    harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
    harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
    harmonics = unique(concatenate((unique(harmonics_state), unique(harmonics_state_conj))))
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(abs(harmonics))
    
    @staticmethod
    def update_class_variables():
        JacobianFourier.polynomial_degree = Fourier.polynomial_degree - 1
        JacobianFourier.harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
        JacobianFourier.harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
        JacobianFourier.harmonics = unique(concatenate((unique(JacobianFourier.harmonics_state), unique(JacobianFourier.harmonics_state_conj))))
        JacobianFourier.number_of_harmonics = len(JacobianFourier.harmonics)
        JacobianFourier.harmonic_truncation_order = max(JacobianFourier.harmonics)

    def __init__(self, RR: array, RI: array, IR: array, II: array) -> None:
        self.RR = RR # Derivative of real part wrt real part
        self.RI = RI # Derivative of real part wrt imag part
        self.IR = IR # Derivative of imag part wrt real part
        self.II = II # Derivative of imag part wrt imag part
        
    def new_from_time_series(time_series: array):
        pass
    
    def new_given_all_coefficients(all_coefficients: array):
        pass

class JacobianFourier_Real(JacobianFourier):
    
    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        return JacobianFourier_Real.new_given_all_coefficients(all_coefficients)
    
    def new_given_all_coefficients(all_coefficients: array):

        state = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state]) # row by row
        state_conj = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state_conj])
        state_real = hstack(concatenate(state + state_conj, axis=1)) / Fourier.number_of_time_samples
        state_imag = hstack(concatenate(state - state_conj, axis=1)) / Fourier.number_of_time_samples
        
        return JacobianFourier_Real(RR = state_real.real, RI = -state_imag.imag, IR = state_real.imag, II = state_imag.real)

class JacobianFourier_Complex(JacobianFourier):
    
    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        return JacobianFourier_Complex.new_given_all_coefficients(all_coefficients)
    
    def new_given_all_coefficients(all_coefficients: array):
        state_blocks = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state])
        state = hstack(concatenate(state_blocks, axis=1)) / Fourier.number_of_time_samples
        return JacobianFourier_Complex(RR = state.real, RI = -state.imag, IR = state.imag, II = state.real)

#%%

class FrequencyDomainFirstOrderODE(object):
    def __init__(self, first_order_ode: FirstOrderODE) -> None:
        self.ode = first_order_ode
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2

        self.external_term = self.compute_external_force()
        self.jacobian_linear_term = self.compute_jacobian_linear_term()
        
        self.jacobian_adimensional_time_derivative_term = kron(diag(Fourier.harmonics), eye(self.ode.dimension))

    # Residue in Real-Imaginary Format
    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        nonlinear_term = self.compute_nonlinear_term(state)
        linear_term_coefficients = self.ode.linear_coefficient @ state.coefficients #where A = -w^2M+jwC+K and F_lin = A*Q
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients - self.external_term.coefficients # complex array
        return Fourier.coefficients_to_RI(residue_coefficients) # real array
    
    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        derivative_wrt_omega = -x.fourier.get_adimensional_time_derivative()
        R = vstack(derivative_wrt_omega.real)
        I = vstack(derivative_wrt_omega.imag)
        return vstack((R, I))
    
    """
    # The following methods must be specified separately for real-valued and complex-valued systems.
    """
    
    def compute_jacobian_linear_term(self) -> JacobianFourier:
        pass
    
    def compute_external_force(self) -> Fourier:
        pass

    def compute_nonlinear_term(self, state: Fourier) -> Fourier:
        pass
    
    def compute_jacobian_nonlinear_term(self, state: Fourier) -> JacobianFourier:
        pass

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        pass

class FrequencyDomainFirstOrderODE_Real(FrequencyDomainFirstOrderODE):
    
    """
    # The following methods are for systems where the external force and nonlinear terms are real-valued
    # and the Fourier coefficients are computed using the Real FFT (rFFT).
    """
    
    # Linear Jacobian for Real-Valued Systems
    def compute_jacobian_linear_term(self) -> JacobianFourier_Real:

        state = kron(eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        state_conj = kron(where(JacobianFourier.harmonics_state_conj == 0, 1, 0), self.ode.linear_coefficient)

        RR = state + state_conj
        II = state - state_conj
        # RI = 0
        # IR = -RI = 0
        return JacobianFourier_Real(RR=RR, RI=None, IR=None, II=II)
    
    # External Force for Real-Valued Systems
    def compute_external_force(self) -> Fourier_Real:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(external_term_time_series)

    # Nonlinear Term for Real-Valued Systems
    def compute_nonlinear_term(self, state: Fourier_Real) -> Fourier_Real:
        Fourier_Real.compute_time_series(state)
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(fnl_time_series)
    
    def compute_jacobian_nonlinear_term(self, state: Fourier_Real) -> JacobianFourier_Real:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier_Real.new_from_time_series(dfnldq_time_series)

    # Jacobian of Residue for Real-Valued Systems in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)
        aux = self.jacobian_adimensional_time_derivative_term * x.omega

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + aux
        J_IR = jacobian_nonlinear_term.IR - aux
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.II

        return block([[J_RR, J_RI], [J_IR, J_II]])

class FrequencyDomainFirstOrderODE_Complex(FrequencyDomainFirstOrderODE):
    
    """
    # The following methods are for systems where the external force and nonlinear terms are complex-valued
    # and the Fourier coefficients are computed using the FFT (FFT).
    """
    
    # Linear Jacobian for Complex-Valued Systems
    def compute_jacobian_linear_term(self) -> JacobianFourier_Complex:
        linear = kron(eye(Fourier.number_of_harmonics), self.ode.linear_coefficient)
        # IR = -RI = linear.imag
        # II = RR = linear.real
        return JacobianFourier_Complex(RR = linear.real, RI = -linear.imag, IR = None, II = None)
    
    # External Force for Complex-Valued Systems
    def compute_external_force(self) -> Fourier_Complex:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier_Complex.new_from_time_series(external_term_time_series)

    # Nonlinear Term for Complex-Valued Systems
    def compute_nonlinear_term(self, state: Fourier_Complex) -> Fourier_Complex:
        Fourier_Complex.compute_time_series(state)
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier_Complex.new_from_time_series(fnl_time_series)
    
    # Jacobian of Nonlinear Term for Complex-Valued Systems
    def compute_jacobian_nonlinear_term(self, state: Fourier_Complex) -> JacobianFourier_Complex:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier_Complex.new_from_time_series(dfnldq_time_series)
    
    # Jacobian of Residue for Complex-Valued Systems in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:

        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x.fourier)
        aux = self.jacobian_linear_term.RI + self.jacobian_adimensional_time_derivative_term * x.omega

        J_RR = jacobian_nonlinear_term.RR + self.jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + aux
        J_IR = jacobian_nonlinear_term.IR - aux
        J_II = jacobian_nonlinear_term.II + self.jacobian_linear_term.RR
        
        return block([[J_RR, J_RI], [J_IR, J_II]])

class FrequencyDomainSecondOrderODE(object):
    """
    Base class for the HBM residual of a 2nd-order ODE:
        M*q'' + C*q' + K*q + fnl(q, tau) = fext(tau)

    In the frequency domain, harmonic n has the linear operator:
        A_n = K - n^2*omega^2*M + j*n*omega*C
    """
    def __init__(self, second_order_ode: SecondOrderODE) -> None:
        self.ode = second_order_ode
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2
        self.external_term = self.compute_external_force()

    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        omega = x.omega
        n = Fourier.harmonics          # shape (nh,)
        Q = state.coefficients         # shape (nh, ndof, 1)
        M, C, K = self.ode.mass_matrix, self.ode.damping_matrix, self.ode.stiffness_matrix

        # A_n @ Q_n = (K - n^2*omega^2*M + j*n*omega*C) @ Q_n  for each harmonic n
        linear_coefficients = (
            K @ Q
            + einsum('i,ijk->ijk', -n**2 * omega**2, M @ Q)
            + einsum('i,ijk->ijk',  1j * n * omega,  C @ Q)
        )
        nonlinear_term = self.compute_nonlinear_term(state)
        residue_coefficients = linear_coefficients + nonlinear_term.coefficients - self.external_term.coefficients
        return Fourier.coefficients_to_RI(residue_coefficients)

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        """dR/domega = (-2*n^2*omega*M + j*n*C) @ Q_n  per harmonic n"""
        n = Fourier.harmonics
        Q = x.fourier.coefficients
        M, C = self.ode.mass_matrix, self.ode.damping_matrix
        deriv_coefficients = (
            einsum('i,ijk->ijk', -2 * n**2 * x.omega, M @ Q)
            + einsum('i,ijk->ijk', 1j * n,             C @ Q)
        )
        R = vstack(deriv_coefficients.real)
        I = vstack(deriv_coefficients.imag)
        return vstack((R, I))

    def compute_external_force(self) -> Fourier:
        pass

    def compute_nonlinear_term(self, state: Fourier) -> Fourier:
        pass

    def compute_jacobian_nonlinear_term(self, state: Fourier) -> JacobianFourier:
        pass

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        pass


class FrequencyDomainSecondOrderODE_Real(FrequencyDomainSecondOrderODE):
    """
    HBM frequency-domain residual for real-valued 2nd-order systems.
    Uses the real FFT (rFFT).
    """

    def compute_external_force(self) -> Fourier_Real:
        return Fourier_Real.new_from_time_series(self.ode.external_term(Fourier.adimensional_time_samples))

    def compute_nonlinear_term(self, state: Fourier_Real) -> Fourier_Real:
        Fourier_Real.compute_time_series(state)
        fnl_time_series = self.ode.nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(fnl_time_series)

    def compute_jacobian_nonlinear_term(self, state: Fourier_Real) -> JacobianFourier_Real:
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(state.time_series, Fourier.adimensional_time_samples)
        return JacobianFourier_Real.new_from_time_series(dfnldq_time_series)

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        n = Fourier.harmonics
        omega = x.omega
        M, C, K = self.ode.mass_matrix, self.ode.damping_matrix, self.ode.stiffness_matrix
        nh = Fourier.number_of_harmonics

        jacobian_nl = self.compute_jacobian_nonlinear_term(x.fourier)

        # Block-diagonal linear Jacobian: diag_n(K - n^2*omega^2*M) and diag_n(n*omega*C)
        J_lin_diag  = kron(eye(nh), K) - omega**2 * kron(diag(n**2), M)
        J_lin_cross = omega * kron(diag(n), C)

        J_RR = jacobian_nl.RR + J_lin_diag
        J_RI = jacobian_nl.RI - J_lin_cross
        J_IR = jacobian_nl.IR + J_lin_cross
        J_II = jacobian_nl.II + J_lin_diag

        return block([[J_RR, J_RI], [J_IR, J_II]])


# %% Test
