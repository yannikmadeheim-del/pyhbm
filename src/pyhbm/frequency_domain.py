import numpy as np
import warnings
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like, vdot, sqrt
from numpy.fft import rfft, irfft, fft, ifft

from .dynamical_system import FirstOrderODE, SecondOrderODE, FBS_System
from scipy.interpolate import CubicSpline, make_interp_spline

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

    def compute_time_series_derivative(self, x: FourierOmegaPoint):
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
        if self.time_series is None:
            shape = list(self.coefficients.shape)
            shape[0] = Fourier.harmonic_truncation_order + 1
            new_coeff = zeros(shape, dtype=complex)
            new_coeff[Fourier.harmonics] = self.coefficients
            # inverse of Real FFT
            self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
        return self.time_series

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
        self.time_series_derivative = None
        self.second_adimensional_time_derivative = None
        self.Gdot = None
        self.Y_frf_cache = None
        self.Z_frf_cache = None
        self.nonlinear_term_cache = None
        self.BY_cache = None        # B_fourier @ Y  (complex, reused by residue + Jacobian)
        self.BYBT_RI_cache = None   # FRF_to_RI(B @ Y @ B.T)  (reused by Jacobian + dR/dω)
        self.Yr_cache = None # complex Y_r = B @ Y @ B^T
        self.Zr_rhs = None # solve(Y_r, F_adm − Q_rel)
        self.lambda_corrected = None # corrected λ̃
        self.contact_mask = None # uncorrected lambda prediciton time series
        
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

    def compute_time_series_derivative(self):
        if self.time_series_derivative is None:
            qdot_coefficients = self.omega * self.fourier.get_adimensional_time_derivative()
            qdot_fourier = Fourier_Real(qdot_coefficients)
            Fourier_Real.compute_time_series(qdot_fourier)
            self.time_series_derivative = qdot_fourier.time_series
        return self.time_series_derivative

    def compute_second_adimensional_time_derivative(self):
        if self.second_adimensional_time_derivative is None:
            self.second_adimensional_time_derivative = einsum('i,ijk->ijk', Fourier.harmonics**2, self.fourier.coefficients) * -1
        return self.second_adimensional_time_derivative

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
        linear_term_coefficients = self.ode.linear_coefficient @ state.coefficients - state.get_adimensional_time_derivative() * x.omega
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients + self.external_term.coefficients # complex array
        return Fourier.coefficients_to_RI(residue_coefficients) # real array
    
    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        derivative_wrt_omega = -state.get_adimensional_time_derivative()
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
        return JacobianFourier_Complex(RR=linear.real, RI=-linear.imag, IR=None, II=None)

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

# %%

class FrequencyDomainSecondOrderODE(object):
    def __init__(self, second_order_ode: SecondOrderODE) -> None:
        self.ode = second_order_ode
        self.complex_dimension = Fourier.number_of_harmonics * self.ode.dimension
        self.real_dimension = self.complex_dimension * 2

        self.external_term = self.compute_external_force()

        self.jacobian_adimensional_time_derivative_term = kron(diag(Fourier.harmonics), eye(self.ode.dimension))



    # Residue in Real-Imaginary Format
    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        ome = x.omega
        nonlinear_term = self.compute_nonlinear_term(x)
        linear_term_coefficients = (self.ode.stiffness_matrix @ state.coefficients                                                    # f_lin = (-(omega*n)**2*M + j*(omega*n)*C + K) @ Q for every harmonic n
                + (ome**2) * self.ode.mass_matrix @ x.compute_second_adimensional_time_derivative()
                + ome * self.ode.damping_matrix @ state.get_adimensional_time_derivative())
        residue_coefficients = linear_term_coefficients + nonlinear_term.coefficients - self.external_term.coefficients  # complex array
        return Fourier.coefficients_to_RI(residue_coefficients)  # real array

    # Derivative of Residue with respect to omega in Real-Imaginary Format
    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        ome = x.omega
        qdot_adim = state.get_adimensional_time_derivative()
        derivative_linear = (2 * ome * self.ode.mass_matrix @ x.compute_second_adimensional_time_derivative()
                             + self.ode.damping_matrix @ qdot_adim)
        R = vstack(derivative_linear.real)
        I = vstack(derivative_linear.imag)
        derivative_linear_RI = vstack((R, I))
        Gdot = self.compute_Gdot(x)
        qdot_R = vstack(qdot_adim.real)
        qdot_I = vstack(qdot_adim.imag)
        derivative_nonlinear_RI = vstack((Gdot.RR @ qdot_R + Gdot.RI @ qdot_I,
                                          Gdot.IR @ qdot_R + Gdot.II @ qdot_I))
        return derivative_linear_RI + derivative_nonlinear_RI

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

    def compute_Gdot(self, x: FourierOmegaPoint) -> JacobianFourier_Real:
        pass


class FrequencyDomainSecondOrderODE_Real(FrequencyDomainSecondOrderODE):
    """
    # The following methods are for systems where the external force and nonlinear terms are real-valued
    # and the Fourier coefficients are computed using the Real FFT (rFFT).
    """
    def __init__(self, second_order_ode: SecondOrderODE):
        super().__init__(second_order_ode)
        self.kron_mass = kron(diag(Fourier.harmonics ** 2), self.ode.mass_matrix)
        self.kron_damping = kron(diag(Fourier.harmonics), self.ode.damping_matrix)
        self.kron_stiffness = kron(eye(Fourier.number_of_harmonics), self.ode.stiffness_matrix)

    # Linear Jacobian for Real-Valued Systems
    def compute_jacobian_linear_term(self, omega: float) -> JacobianFourier_Real:
        state =  - omega ** 2 * self.kron_mass \
                 + 1j * omega * self.kron_damping\
                 + self.kron_stiffness

        state_conj = kron(where(JacobianFourier.harmonics_state_conj == 0, 1, 0), -omega**2*self.ode.mass_matrix + 1j*omega*self.ode.damping_matrix + self.ode.stiffness_matrix)

        RR = (state + state_conj).real
        II = (state - state_conj).real
        RI = -(state + state_conj).imag
        IR = (state - state_conj).imag
        return JacobianFourier_Real(RR=RR, RI=RI, IR=IR, II=II)

    # External Force for Real-Valued Systems
    def compute_external_force(self) -> Fourier_Real:
        external_term_time_series = self.ode.external_term(Fourier.adimensional_time_samples)
        return Fourier_Real.new_from_time_series(external_term_time_series)

    # Nonlinear Term for Real-Valued Systems
    def compute_nonlinear_term(self, x: FourierOmegaPoint) -> Fourier_Real:
        if x.nonlinear_term_cache is None:
            Fourier_Real.compute_time_series(x.fourier)
            q = x.fourier.time_series
            qdot = x.compute_time_series_derivative()
            fnl_time_series = self.ode.nonlinear_term(q, qdot, Fourier.adimensional_time_samples)
            x.nonlinear_term_cache = Fourier_Real.new_from_time_series(fnl_time_series)
        return x.nonlinear_term_cache

    def compute_jacobian_nonlinear_term(self, x: FourierOmegaPoint) -> JacobianFourier_Real:
        self.compute_nonlinear_term(x)
        dfnldq_time_series = self.ode.jacobian_nonlinear_term(x.fourier.time_series, x.time_series_derivative, Fourier.adimensional_time_samples)
        G = JacobianFourier_Real.new_from_time_series(dfnldq_time_series)
        Gdot = self.compute_Gdot(x)
        col_scale = x.omega * self.jacobian_adimensional_time_derivative_term
        return JacobianFourier_Real(
            RR=G.RR + Gdot.RI @ col_scale,
            RI=G.RI - Gdot.RR @ col_scale,
            IR=G.IR + Gdot.II @ col_scale,
            II=G.II - Gdot.IR @ col_scale
        )

    # Jacobian of Residue for Real-Valued Systems in Real-Imaginary Format
    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        jacobian_linear_term = self.compute_jacobian_linear_term(x.omega)
        jacobian_nonlinear_term = self.compute_jacobian_nonlinear_term(x)

        J_RR = jacobian_nonlinear_term.RR + jacobian_linear_term.RR
        J_RI = jacobian_nonlinear_term.RI + jacobian_linear_term.RI
        J_IR = jacobian_nonlinear_term.IR + jacobian_linear_term.IR
        J_II = jacobian_nonlinear_term.II + jacobian_linear_term.II

        return block([[J_RR, J_RI], [J_IR, J_II]])

    # compute FFT(df/dqdot):
    def compute_Gdot(self, x: FourierOmegaPoint) -> JacobianFourier_Real:
        if x.Gdot is None:
            dfnldqdot_time_series = self.ode.jacobian_nonlinear_term_qdot(x.fourier.time_series,
                                                                          x.time_series_derivative,
                                                                          Fourier.adimensional_time_samples)
            x.Gdot = JacobianFourier_Real.new_from_time_series(dfnldqdot_time_series)
        return x.Gdot

# class FrequencyDomainSecondOrderODE_Complex MISSING!!!!!!!!!
# %% Test



class FrequencyDomainFRF(FrequencyDomainSecondOrderODE_Real):
    def __init__(self, nonlinear_ode: SecondOrderODE):
        FrequencyDomainSecondOrderODE.__init__(self, nonlinear_ode)

    def FRF_to_RI(self, FRF):
        return block([[FRF.real, -FRF.imag], [FRF.imag, FRF.real]])  # [[Re, -Im], [Im, Re]]

    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        nonlinear_term = self.compute_nonlinear_term(x)
        Q = vstack(state.coefficients)
        Y = self.get_FRF(x)
        Fnl = vstack(nonlinear_term.coefficients)
        Fext = vstack(self.external_term.coefficients)
        R = Q + Y @ (Fnl - Fext)   # R = Q + Y @ Fnl - Y @ Fext
        return vstack((R.real, R.imag))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        Jnl = self.compute_jacobian_nonlinear_term(x)
        Jnl_RI = block([[Jnl.RR, Jnl.RI], [Jnl.IR, Jnl.II]])
        Y = self.get_FRF(x)
        Y_RI = self.FRF_to_RI(Y)
        return eye(self.real_dimension) + Y_RI@Jnl_RI  # J = I + Y_RI @ Jnl_RI

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        Y = self.FRF_to_RI(self.get_FRF(x))
        derivative_FRF = self.compute_FRF_derivative_wrt_omega_RI(x)
        nonlinear_term = self.compute_nonlinear_term(x)
        Fnl = Fourier.coefficients_to_RI(nonlinear_term.coefficients)
        Fext = Fourier.coefficients_to_RI(self.external_term.coefficients)
        qdot_adim = state.get_adimensional_time_derivative()
        Gdot = self.compute_Gdot(x)
        qdot_R = vstack(qdot_adim.real)
        qdot_I = vstack(qdot_adim.imag)
        derivative_nonlinear_RI = vstack((Gdot.RR @ qdot_R + Gdot.RI @ qdot_I,
                                          Gdot.IR @ qdot_R + Gdot.II @ qdot_I))  # dF_nl/dω = G_dot @ q_dot_adim
        return derivative_FRF @ Fnl + Y @ derivative_nonlinear_RI - derivative_FRF @ Fext  # dR/dω = dY/dω @ Fnl + Y @ dF_nl/dω - dY/dω @ Fext


    def get_FRF(self, x):
        if x.Y_frf_cache is None:
            x.Y_frf_cache = self.compute_FRF(x.omega)
        return x.Y_frf_cache

    def compute_FRF(self, omega):
        pass

    def interpolate_FRF(self, omega):
        pass

    def compute_FRF_derivative_wrt_omega_RI(self, x):
        pass


class FrequencyDomainFRF_experimental(FrequencyDomainFRF):
    def __init__(self, nonlinear_ode: SecondOrderODE, omega_frf: array, Y_frf: array,
                 fd_step: float = 1e-6):
        # omega_frf: shape (N_freq,)       — measured frequency points
        # Y_frf:     shape (N_freq, d, d)  — complex FRF matrices
        # fd_step:   step size for dY/dω via finite difference
        super().__init__(nonlinear_ode)
        self.omega_frf = omega_frf
        self.Y_frf = Y_frf
        self.fd_step = fd_step
        self.interp_real = CubicSpline(omega_frf, Y_frf.real)
        self.interp_imag = CubicSpline(omega_frf, Y_frf.imag)

    def compute_FRF(self, omega):
        d = self.ode.dimension
        omega_harmonics = Fourier.harmonics * omega  # shape (Nh,)
        Y_blocks = self.interpolate_FRF(omega_harmonics)  # shape (Nh, d, d)
        Y = zeros((self.complex_dimension, self.complex_dimension), dtype=complex)
        for k in range(len(Fourier.harmonics)):
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = Y_blocks[k]
        return Y

    # def interpolate_FRF(self, omega) -> array:
    #     omega = np.asarray(omega)
    #     if np.any((omega < self.omega_frf[0]) | (omega > self.omega_frf[-1])):
    #         out_of_range = omega[(omega < self.omega_frf[0]) | (omega > self.omega_frf[-1])]
    #         import warnings
    #         warnings.warn(
    #             f"Some omega values outside FRF data range [{self.omega_frf[0]:.4f}, {self.omega_frf[-1]:.4f}]. Extrapolating for {out_of_range}.")
    #     return self.interp_real(omega) + 1j * self.interp_imag(omega)
    #
    # def compute_FRF_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
    #     omega = x.omega
    #     if omega < self.omega_frf[0]+self.fd_step:
    #         dY = (self.compute_FRF(omega + self.fd_step) - self.get_FRF(x)) / (self.fd_step)
    #     else:
    #         dY = (self.compute_FRF(omega + self.fd_step) - self.compute_FRF(omega - self.fd_step)) / (2 * self.fd_step)
    #     return self.FRF_to_RI(dY)


    def interpolate_FRF(self, omega) -> array:
        omega = np.asarray(omega)
        neg_mask = omega < 0
        omega_abs = np.abs(omega)

        if np.any(omega_abs > self.omega_frf[-1]):
            import warnings
            warnings.warn(f"Some omega values outside FRF data range [0, {self.omega_frf[-1]:.4f}]. Extrapolating.")

        result = self.interp_real(omega_abs) + 1j * self.interp_imag(omega_abs)

        # Y(-ω) = conj(Y(ω)) for real systems
        if np.ndim(omega) == 0:
            if bool(neg_mask):
                result = np.conj(result)
        elif np.any(neg_mask):
            result[neg_mask] = np.conj(result[neg_mask])

        return result

    def compute_FRF_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        omega = x.omega
        dY = (self.compute_FRF(omega + self.fd_step) - self.compute_FRF(omega - self.fd_step)) / (2 * self.fd_step)
        return self.FRF_to_RI(dY)


class FrequencyDomainFRF_numerical(FrequencyDomainFRF):
    def compute_FRF(self, omega):
        d = self.ode.dimension
        M, C, K = self.ode.mass_matrix, self.ode.damping_matrix, self.ode.stiffness_matrix
        Y = zeros((self.complex_dimension, self.complex_dimension), dtype=complex)
        for k, n in enumerate(Fourier.harmonics):
            Z_n = -(n * omega) ** 2 * M + 1j * n * omega * C + K
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.linalg.solve(Z_n, eye(d))
        return Y

    def compute_FRF_derivative_wrt_omega_RI(self, x):
        # dY/dω = -Y (dZ/dω) Y,  where dZ/dω = -2n²ω M + jn C
        omega = x.omega
        M, C = self.ode.mass_matrix, self.ode.damping_matrix
        d = self.ode.dimension
        Y_full = self.get_FRF(x)
        dY = zeros((self.complex_dimension, self.complex_dimension), dtype=complex)
        for k, n in enumerate(Fourier.harmonics):
            Y_n  = Y_full[k*d:(k+1)*d, k*d:(k+1)*d]
            dZ_n = -2 * n**2 * omega * M + 1j * n * C
            dY[k*d:(k+1)*d, k*d:(k+1)*d] = - Y_n @ dZ_n @ Y_n
        return self.FRF_to_RI(dY)


class FrequencyBasedSubstructuring(FrequencyDomainFRF):
    def __init__(self, fbs_system: FBS_System):
        super().__init__(fbs_system)
        self.B_fourier = kron(eye(Fourier.number_of_harmonics), fbs_system.B_coupling)
        self.B_RI = block([[self.B_fourier, zeros_like(self.B_fourier)], [zeros_like(self.B_fourier), self.B_fourier]])
        self.F_ext_full = vstack(self.external_term.coefficients)  # (Nh*dTotal, 1)
        self.F_ext_full_RI = Fourier.coefficients_to_RI(self.F_ext_full)  # constant RI form
        self.d_total = self.ode.mass_matrix.shape[0]
        self.total_complex_dimensions = Fourier.number_of_harmonics*self.ode.mass_matrix.shape[0]

    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        nonlinear_term = self.compute_nonlinear_term(x)
        Q_rel = vstack(state.coefficients)
        Fnl = vstack(nonlinear_term.coefficients)
        BY = self.get_BY(x)
        R = Q_rel + BY @ (self.B_fourier.T @ Fnl - self.F_ext_full) # R = Q_rel - B*Y*Fext + B*Y*B^T*Fnl
        return vstack((R.real, R.imag))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        Jnl = self.compute_jacobian_nonlinear_term(x)
        Jnl_RI = block([[Jnl.RR, Jnl.RI], [Jnl.IR, Jnl.II]])
        return eye(self.real_dimension) + self.get_BYBT_RI(x) @ Jnl_RI  # J = I + (B*Y*B^T)_RI @ Jnl_RI

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        state = x.fourier
        dY_RI = self.compute_FRF_derivative_wrt_omega_RI(x)
        nonlinear_term = self.compute_nonlinear_term(x)
        Fnl = Fourier.coefficients_to_RI(nonlinear_term.coefficients)
        qdot_adim = state.get_adimensional_time_derivative()
        Gdot = self.compute_Gdot(x)
        qdot_R = vstack(qdot_adim.real)
        qdot_I = vstack(qdot_adim.imag)
        derivative_nonlinear_RI = vstack((Gdot.RR @ qdot_R + Gdot.RI @ qdot_I,
                                          Gdot.IR @ qdot_R + Gdot.II @ qdot_I))  # dF_nl/dω = G_dot @ q_dot_adim
        BdY = self.B_RI @ dY_RI
        return BdY @ (self.B_RI.T @ Fnl - self.F_ext_full_RI) + self.get_BYBT_RI(x) @ derivative_nonlinear_RI   # dR/dω = dY/dω @ Fnl + Y @ dF_nl/dω - dY/dω @ Fext

    def get_FRF(self, x):
        if x.Y_frf_cache is None:
            x.Y_frf_cache = self.compute_FRF(x.omega)
        return x.Y_frf_cache

    def get_BY(self, x: FourierOmegaPoint) -> array:
        if x.BY_cache is None:
            x.BY_cache = self.B_fourier @ self.get_FRF(x)
        return x.BY_cache

    def get_BYBT_RI(self, x: FourierOmegaPoint) -> array:
        if x.BYBT_RI_cache is None:
            x.BYBT_RI_cache = self.FRF_to_RI(self.get_BY(x) @ self.B_fourier.T)
        return x.BYBT_RI_cache

    def compute_FRF(self, omega):
        pass

    def interpolate_FRF(self, omega):
        pass

    def compute_FRF_derivative_wrt_omega_RI(self, x):
        pass

    def compute_full_response(self, fourier: Fourier, omega: float) -> Fourier:
        """
        Post-processing: compute Fourier coefficients of the full system response
        for all dTotal DOFs from a converged u_rel solution.

        Q_full = Y^{A|B} @ (F_ext - B^T @ F_nl(u_rel))

        :param fourier: Converged Fourier object for u_rel, coefficients shape (Nh, n_int, 1)
        :param omega:   Corresponding angular frequency
        :return:        Fourier object with coefficients shape (Nh, dTotal, 1)
        """
        x = FourierOmegaPoint(fourier, omega)
        nonlinear_term = self.compute_nonlinear_term(x)
        Y = self.get_FRF(x)
        Fnl = vstack(nonlinear_term.coefficients)                      # (Nh*n_int, 1)
        Q_full = Y @ (self.F_ext_full - self.B_fourier.T @ Fnl)       # (Nh*dTotal, 1)
        coefficients = Q_full.reshape(Fourier.number_of_harmonics, self.d_total, 1)
        return Fourier(coefficients)


class FrequencyBasedSubstructuring_numerical(FrequencyBasedSubstructuring):
    def compute_FRF(self, omega):
        M, C, K = self.ode.mass_matrix, self.ode.damping_matrix, self.ode.stiffness_matrix
        d = self.d_total
        Y = zeros((self.total_complex_dimensions, self.total_complex_dimensions), dtype=complex)
        for k, n in enumerate(Fourier.harmonics):
            Z_n = -(n * omega) ** 2 * M + 1j * n * omega * C + K
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.linalg.solve(Z_n, eye(d))
        return Y

    def compute_FRF_derivative_wrt_omega_RI(self, x):
        # dY/dω = -Y (dZ/dω) Y,  where dZ/dω = -2n²ω M + jn C
        omega = x.omega
        M, C = self.ode.mass_matrix, self.ode.damping_matrix
        d = self.d_total
        Y_full = self.get_FRF(x)
        dY = zeros((self.total_complex_dimensions, self.total_complex_dimensions), dtype=complex)
        for k, n in enumerate(Fourier.harmonics):
            Y_n  = Y_full[k*d:(k+1)*d, k*d:(k+1)*d]
            dZ_n = -2 * n**2 * omega * M + 1j * n * C
            dY[k*d:(k+1)*d, k*d:(k+1)*d] = - Y_n @ dZ_n @ Y_n
        return self.FRF_to_RI(dY)

class FrequencyBasedSubstructuring_experimental(FrequencyBasedSubstructuring):
    def __init__(self, fbs_system: FBS_System, fd_step: float = 1e-6):
        # omega_frf: shape (N_freq,)       — measured frequency points
        # Y_frf:     shape (N_freq, d, d)  — complex FRF matrices
        # fd_step:   step size for dY/dω via finite difference
        super().__init__(fbs_system)
        self.fd_step = fd_step
        self.interp_real = CubicSpline(self.ode.omega_frf, self.ode.Y_frf.real)
        self.interp_imag = CubicSpline(self.ode.omega_frf, self.ode.Y_frf.imag)

    def compute_FRF(self, omega):
        d = self.d_total
        omega_harmonics = Fourier.harmonics * omega  # shape (Nh,)
        Y_blocks = self.interpolate_FRF(omega_harmonics)  # shape (Nh, d, d)
        Y = zeros((self.total_complex_dimensions, self.total_complex_dimensions), dtype=complex)
        for k in range(len(Fourier.harmonics)):
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = Y_blocks[k]
        return Y

    def interpolate_FRF(self, omega) -> array:
        omega = np.asarray(omega)
        neg_mask = omega < 0
        omega_abs = np.abs(omega)

        if np.any(omega_abs > self.ode.omega_frf[-1]):
            import warnings
            warnings.warn(f"Some omega values outside FRF data range [0, {self.ode.omega_frf[-1]:.4f}]. Extrapolating.")

        result = self.interp_real(omega_abs) + 1j * self.interp_imag(omega_abs)

        # Y(-ω) = conj(Y(ω)) for real systems
        if np.ndim(omega) == 0:
            if bool(neg_mask):
                result = np.conj(result)
        elif np.any(neg_mask):
            result[neg_mask] = np.conj(result[neg_mask])

        return result

    def compute_FRF_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        omega = x.omega
        dY = (self.compute_FRF(omega + self.fd_step) - self.compute_FRF(omega - self.fd_step)) / (2 * self.fd_step)
        return self.FRF_to_RI(dY)

class FBS_DLFT(FrequencyBasedSubstructuring):
    def __init__(self, fbs_system: FBS_System, epsilon = 1.0, g_zero = 0.0, h=1e-6):
        super().__init__(fbs_system)
        self.epsilon = epsilon
        self.g_zero = g_zero
        self.complex_int_dimension = Fourier.number_of_harmonics*self.ode.dimension
        self.h = h

    def _get_Yr(self, x):
        if x.Yr_cache is None:
            x.Yr_cache = self.get_BY(x) @ self.B_fourier.T # B * Y_A|B * B.T
        return x.Yr_cache

    def _get_Fext_admr(self, x):
        F_admr = self.get_BY(x) @ self.F_ext_full # complex free admittance response to excitation
        return F_admr # shape (Nh*n_int, 1)

    def _get_Zr_rhs(self, x):
        if x.Zr_rhs is None:
            x.Zr_rhs = np.linalg.solve(self._get_Yr(x), self._get_Fext_admr(x)-vstack(x.fourier.coefficients))
        return x.Zr_rhs

    def _get_lambda_corrected(self, x): # DLFT-AFT loop
        if x.lambda_corrected is None:
            zr_fourier = Fourier(self._get_Zr_rhs(x).reshape(Fourier.number_of_harmonics, self.ode.dimension, 1))
            Fourier_Real.compute_time_series(zr_fourier)
            zr_t = zr_fourier.time_series  # shape (Nt, n_int, 1)
            Fourier_Real.compute_time_series(x.fourier)
            q_rel = x.fourier.time_series
            lambda_x = zr_t - self.epsilon * (q_rel+self.g_zero)
            x.contact_mask = lambda_x > 0
            lambda_t_corr = np.where(x.contact_mask, lambda_x, 0.0) # Correction
            lambda_x_corr = Fourier_Real.new_from_time_series(lambda_t_corr)
            x.lambda_corrected = vstack(lambda_x_corr.coefficients)
        return x.lambda_corrected

    def compute_residue_RI(self, x: FourierOmegaPoint) -> array:
        Q_rel = vstack(x.fourier.coefficients)
        R = Q_rel + self._get_Yr(x) @ self._get_lambda_corrected(x) - self._get_Fext_admr(x)
        return vstack((R.real, R.imag))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> array:
        self._get_lambda_corrected(x)
        dlambda_x = np.linalg.solve(self._get_Yr(x), -(np.eye( self.complex_int_dimension) + self.epsilon*self._get_Yr(x))) # Prediction Jacobian shape (Nh*n_int, Nh*n_int)
        dlambda_x_resh = dlambda_x.reshape(Fourier.number_of_harmonics, self.ode.dimension, self.complex_int_dimension) # (Nh, n_int, Nh*n_int)
        dlambda_x_fourier = Fourier(dlambda_x_resh)
        Fourier_Real.compute_time_series(dlambda_x_fourier)
        dlambda_t_corr = np.where(x.contact_mask > 0, dlambda_x_fourier.time_series, 0.0)
        dlambda_x_corr = Fourier_Real.new_from_time_series(dlambda_t_corr) # (Nh, n_int, Nh*n_int)
        dlambda_x_corr_matrix = vstack(dlambda_x_corr.coefficients) # (Nh*n_int, Nh*n_int)
        J_complex = eye(self.complex_int_dimension) + self._get_Yr(x) @ dlambda_x_corr_matrix
        return self.FRF_to_RI(J_complex)

    def _get_dY_complex(self, x) -> array:
        return (self.compute_FRF(x.omega + self.h) - self.compute_FRF(x.omega - self.h)) / (2 * self.h)

    def _get_dY_RI(self, x):
        return self.FRF_to_RI(self._get_dY_complex(x))

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> array:
        dY_complex = self._get_dY_complex(x)
        BdY = self.B_fourier @ dY_complex
        lambda_corrected = self._get_lambda_corrected(x)
        Zr_rhs = self._get_Zr_rhs(x)
        dY_term = BdY @ (self.B_fourier.T @ lambda_corrected - self.F_ext_full) # B*dY/dw*(B^T*lamda - Fext)
        dlambda_pred = np.linalg.solve(self._get_Yr(x), BdY @ (self.F_ext_full - self.B_fourier.T @ Zr_rhs))
        dlambda_pred_resh = dlambda_pred.reshape(Fourier.number_of_harmonics, self.ode.dimension, 1)  # (Nh, n_int, 1)
        dlambda_pred_fourier = Fourier(dlambda_pred_resh)
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        dlambda_t_corr = np.where(x.contact_mask > 0, dlambda_pred_fourier.time_series, 0.0)
        dlambda_x_corr = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_x_corr_vector = vstack(dlambda_x_corr.coefficients)
        dH= dY_term + self._get_Yr(x) @ dlambda_x_corr_vector
        return vstack((dH.real, dH.imag))

class FBS_DLFT_numerical(FBS_DLFT, FrequencyBasedSubstructuring_numerical):
    pass

