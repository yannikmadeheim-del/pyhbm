import numpy as np
from numpy import eye, vstack, block, kron, zeros_like

from .frequency_domain import Fourier, Fourier_Real, FourierOmegaPoint
from .frf_provider import FRFProvider
from .nonlinear_method import NonlinearMethod, AFT


def _FRF_to_RI(Y: np.ndarray) -> np.ndarray:
    return block([[Y.real, -Y.imag], [Y.imag, Y.real]])


class FRFProblem:
    """
    HBM problem for a single subsystem in the FRF formulation.

    Residual:  R = Q + Y (F_nl - F_ext)
    Jacobian:  J = I + Y_RI J_nl_RI
    dR/dω:     dY_RI (F_nl - F_ext)_RI + Y_RI dF_nl/dω
    """

    def __init__(self, ode, frf_provider: FRFProvider):
        self.ode = ode
        self.frf_provider = frf_provider
        self.method = AFT()

        self.d = ode.dimension
        Nh = Fourier.number_of_harmonics
        self.complex_dimension = Nh * self.d
        self.real_dimension = 2 * self.complex_dimension

        external_ts = ode.external_term(Fourier.adimensional_time_samples)
        self.external_term = Fourier_Real.new_from_time_series(external_ts)

        self.method.bind(self)

    def _get_FRF(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.Y_cache is None:
            x.Y_cache = self.frf_provider.compute_FRF(
                x.omega, Fourier.harmonics, self.d
            )
        return x.Y_cache

    def compute_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Q    = vstack(x.fourier.coefficients)
        Fnl  = self.method.compute_F_int(x, self.ode)
        Fext = vstack(self.external_term.coefficients)
        Y    = self._get_FRF(x)
        R    = Q + Y @ (Fnl - Fext)
        return vstack((R.real, R.imag))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Y_RI    = _FRF_to_RI(self._get_FRF(x))
        J_nl_RI = self.method.compute_J_int_RI(x, self.ode)
        return eye(self.real_dimension) + Y_RI @ J_nl_RI

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Y    = self._get_FRF(x)
        dY   = self.frf_provider.compute_FRF_derivative(
                   x.omega, Fourier.harmonics, self.d, Y)
        dY_RI  = _FRF_to_RI(dY)
        Y_RI   = _FRF_to_RI(Y)
        Fnl    = self.method.compute_F_int(x, self.ode)
        Fext   = vstack(self.external_term.coefficients)
        Fnl_RI  = vstack((Fnl.real,  Fnl.imag))
        Fext_RI = vstack((Fext.real, Fext.imag))
        dF_nl_dw_RI = self.method.compute_dF_int_domega_RI(x, self.ode)
        return dY_RI @ (Fnl_RI - Fext_RI) + Y_RI @ dF_nl_dw_RI


class FBSProblem:
    """
    HBM problem for Frequency Based Substructuring (FBS).

    Newton unknown: Q_rel  — interface DOFs, size n_int = fbs.dimension
    FRF:            Y    — assembled system, size (Nh*d_total, Nh*d_total)

    Residual:  R = Q_rel + B Y B^T F_nl - B Y F_ext
    Jacobian:  J = I + (B Y B^T)_RI J_nl_RI
    dR/dω:     B dY_RI (B^T F_nl - F_ext)_RI + (B Y B^T)_RI dF_nl/dω
    """

    def __init__(self, fbs, frf_provider: FRFProvider, method: NonlinearMethod):
        self.ode = fbs
        self.frf_provider = frf_provider
        self.method = method

        self.d_int   = fbs.dimension                # n_int: interface DOFs (Newton unknowns)
        self.d_total = fbs.mass_matrix.shape[0]     # full system DOF count (for FRF)
        Nh = Fourier.number_of_harmonics
        self.complex_dimension = Nh * self.d_int
        self.real_dimension    = 2 * self.complex_dimension

        self.B_fourier = kron(eye(Nh), fbs.B_coupling)
        self.B_RI = block([[self.B_fourier, zeros_like(self.B_fourier)],
                           [zeros_like(self.B_fourier), self.B_fourier]])

        external_ts = fbs.external_term(Fourier.adimensional_time_samples)
        self.external_term   = Fourier_Real.new_from_time_series(external_ts)
        self.F_ext_full      = vstack(self.external_term.coefficients)          # (Nh*d_total, 1)
        self.F_ext_full_RI   = vstack((self.F_ext_full.real, self.F_ext_full.imag))  # (2*Nh*d_total, 1)

        self.method.bind(self)

    def _get_FRF(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.Y_cache is None:
            x.Y_cache = self.frf_provider.compute_FRF(
                x.omega, Fourier.harmonics, self.d_total
            )
        return x.Y_cache

    def _get_BY(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.BY_cache is None:
            x.BY_cache = self.B_fourier @ self._get_FRF(x)
        return x.BY_cache

    def _get_BYBT_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.BYBT_RI_cache is None:
            x.BYBT_RI_cache = _FRF_to_RI(self._get_BY(x) @ self.B_fourier.T)
        return x.BYBT_RI_cache

    def compute_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Q_rel = vstack(x.fourier.coefficients)
        Fnl   = self.method.compute_F_int(x, self.ode)
        BY    = self._get_BY(x)
        R     = Q_rel + BY @ (self.B_fourier.T @ Fnl - self.F_ext_full)
        return vstack((R.real, R.imag))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        J_nl_RI = self.method.compute_J_int_RI(x, self.ode)
        return eye(self.real_dimension) + self._get_BYBT_RI(x) @ J_nl_RI

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Y    = self._get_FRF(x)
        dY   = self.frf_provider.compute_FRF_derivative(
                   x.omega, Fourier.harmonics, self.d_total, Y)
        dY_RI = _FRF_to_RI(dY)
        BdY   = self.B_RI @ dY_RI
        Fnl   = self.method.compute_F_int(x, self.ode)
        Fnl_RI = vstack((Fnl.real, Fnl.imag))
        dF_nl_dw_RI = self.method.compute_dF_int_domega_RI(x, self.ode)
        return BdY @ (self.B_RI.T @ Fnl_RI - self.F_ext_full_RI) + self._get_BYBT_RI(x) @ dF_nl_dw_RI

    def compute_full_response(self, fourier: Fourier, omega: float) -> Fourier:
        """Post-processing: full response for all d_total DOFs from a converged x_r."""
        x     = FourierOmegaPoint(fourier, omega)
        Fnl   = self.method.compute_F_int(x, self.ode)
        Y     = self._get_FRF(x)
        Q_full = Y @ (self.F_ext_full - self.B_fourier.T @ Fnl)
        coefficients = Q_full.reshape(Fourier.number_of_harmonics, self.d_total, 1)
        return Fourier(coefficients)

