import numpy as np
from numpy import eye, vstack, block, kron, diag, zeros_like
from abc import ABC, abstractmethod

from .frequency_domain import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    JacobianFourier_Real,
)


class NonlinearMethod(ABC):
    """
    Strategy that provides F_int, dF_int/dx_r and dF_int/dω in RI form.

    A method that needs problem-level data (e.g. admittance, coupling matrix)
    should override `bind` to capture a reference to its owning problem.
    """

    def bind(self, problem) -> None:
        """Capture problem-level context. Default: no-op."""
        pass

    @abstractmethod
    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """Nonlinear/contact force, shape (Nh*d, 1), complex."""

    @abstractmethod
    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """RI Jacobian dF_int/dx_r, shape (2*Nh*d, 2*Nh*d), real."""

    @abstractmethod
    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """RI omega-derivative of F_int, shape (2*Nh*d, 1), real."""


class AFT(NonlinearMethod):
    """Alternating Frequency-Time (AFT) scheme."""

    def _get_nonlinear_term(self, x: FourierOmegaPoint, ode) -> Fourier_Real:
        if x.nonlinear_term_cache is None:
            Fourier_Real.compute_time_series(x.fourier)
            q    = x.fourier.time_series
            qdot = x.compute_time_series_derivative()
            fnl_ts = ode.nonlinear_term(q, qdot, Fourier.adimensional_time_samples)
            x.nonlinear_term_cache = Fourier_Real.new_from_time_series(fnl_ts)
        return x.nonlinear_term_cache

    def _get_Gdot(self, x: FourierOmegaPoint, ode) -> JacobianFourier_Real:
        if x.Gdot is None:
            self._get_nonlinear_term(x, ode)
            dfnldqdot_ts = ode.jacobian_nonlinear_term_qdot(
                x.fourier.time_series,
                x.time_series_derivative,
                Fourier.adimensional_time_samples,
            )
            x.Gdot = JacobianFourier_Real.new_from_time_series(dfnldqdot_ts)
        return x.Gdot

    def _get_jacobian_nonlinear_term(self, x: FourierOmegaPoint, ode) -> JacobianFourier_Real:
        self._get_nonlinear_term(x, ode)
        dfnldq_ts = ode.jacobian_nonlinear_term(
            x.fourier.time_series,
            x.time_series_derivative,
            Fourier.adimensional_time_samples,
        )
        G    = JacobianFourier_Real.new_from_time_series(dfnldq_ts)
        Gdot = self._get_Gdot(x, ode)
        harmonics_term = kron(diag(Fourier.harmonics), eye(ode.dimension))
        col_scale = x.omega * harmonics_term
        return JacobianFourier_Real(
            RR=G.RR + Gdot.RI @ col_scale,
            RI=G.RI - Gdot.RR @ col_scale,
            IR=G.IR + Gdot.II @ col_scale,
            II=G.II - Gdot.IR @ col_scale,
        )

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        fnl = self._get_nonlinear_term(x, ode)
        return vstack(fnl.coefficients)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        Jnl = self._get_jacobian_nonlinear_term(x, ode)
        return block([[Jnl.RR, Jnl.RI], [Jnl.IR, Jnl.II]])

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        qdot_adim = x.fourier.get_adimensional_time_derivative()
        Gdot   = self._get_Gdot(x, ode)
        qdot_R = vstack(qdot_adim.real)
        qdot_I = vstack(qdot_adim.imag)
        return vstack((
            Gdot.RR @ qdot_R + Gdot.RI @ qdot_I,
            Gdot.IR @ qdot_R + Gdot.II @ qdot_I,
        ))


class DLFTContact(NonlinearMethod):
    r"""
    Dynamic Lagrangian Frequency-Time (DLFT) normal contact.

    Prediction:  λ_p = IDFT[Z_r (F_adm - x_r)] + ε (x_r - g₀)
    Correction:  λ   = max(0, λ_p)
    Force:       λ̃   = DFT[λ]

    Bound to an FBSProblem at construction; reads B_fourier, F_ext_full,
    admittance caches, and the FRF provider through that reference.
    """

    def __init__(self, epsilon: float = 1.0, g_zero: float = 0.0):
        self.epsilon  = epsilon
        self.g_zero   = g_zero
        self._problem = None    # populated by bind()

    def bind(self, problem) -> None:
        self._problem = problem

    # --- internal helpers (use self._problem) ---

    def _get_Yr(self, x):
        if x.Yr_cache is None:
            x.Yr_cache = self._problem._get_BY(x) @ self._problem.B_fourier.T
        return x.Yr_cache

    def _get_Fext_admr(self, x):
        return self._problem._get_BY(x) @ self._problem.F_ext_full

    def _get_Zr_rhs(self, x):
        if x.Zr_rhs is None:
            Yr        = self._get_Yr(x)
            Fext_admr = self._get_Fext_admr(x)
            x.Zr_rhs  = np.linalg.solve(Yr, Fext_admr - vstack(x.fourier.coefficients))
        return x.Zr_rhs

    def _get_lambda_corrected(self, x):
        if x.lambda_corrected is None:
            n_int = self._problem.ode.dimension
            zr_fourier = Fourier(
                self._get_Zr_rhs(x).reshape(Fourier.number_of_harmonics, n_int, 1)
            )
            Fourier_Real.compute_time_series(zr_fourier)
            zr_t   = zr_fourier.time_series
            Fourier_Real.compute_time_series(x.fourier)
            q_rel  = x.fourier.time_series
            lambda_p = zr_t + self.epsilon * (q_rel - self.g_zero)
            x.contact_mask     = lambda_p > 0.0
            lambda_t_corr      = np.where(x.contact_mask, lambda_p, 0.0)
            # ALPHA = 1.0e6
            # soft_mask = 0.5 * (1.0 + np.tanh(ALPHA * lambda_p))
            # lambda_t_corr = soft_mask * lambda_p
            # # keep storing the discrete mask for the Jacobian too, or use the smoothed one
            # x.contact_mask = soft_mask  # was: lambda_p > 0

            lambda_x_corr      = Fourier_Real.new_from_time_series(lambda_t_corr)
            x.lambda_corrected = vstack(lambda_x_corr.coefficients)
        return x.lambda_corrected

    def _invert_Yr_blockwise(self, Yr, n_int):
        Zr = zeros_like(Yr)
        for k in range(Fourier.number_of_harmonics):
            s = slice(k * n_int, (k + 1) * n_int)
            Zr[s, s] = np.linalg.solve(Yr[s, s], eye(n_int))
        return Zr

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask
        n_int = ode.dimension
        contact_tangent = x.contact_mask * eye(n_int)    # (Nt, n_int, n_int)
        J_mask    = JacobianFourier_Real.new_from_time_series(contact_tangent)
        J_mask_RI = block([[J_mask.RR, J_mask.RI], [J_mask.IR, J_mask.II]])
        Yr = self._get_Yr(x)
        Zr = self._invert_Yr_blockwise(Yr, n_int)
        complex_int_dim = Fourier.number_of_harmonics * n_int
        M    = self.epsilon * eye(complex_int_dim) - Zr
        M_RI = block([[M.real, -M.imag], [M.imag, M.real]])
        return J_mask_RI @ M_RI

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        problem = self._problem
        Y   = problem._get_FRF(x)
        dY  = problem.frf_provider.compute_FRF_derivative(
                  x.omega, Fourier.harmonics, problem.d_total, Y)
        BdY    = problem.B_fourier @ dY
        Yr     = self._get_Yr(x)
        Zr_rhs = self._get_Zr_rhs(x)
        n_int  = ode.dimension
        dlambda_pred = np.linalg.solve(
            Yr, BdY @ (problem.F_ext_full - problem.B_fourier.T @ Zr_rhs)
        )
        dlambda_pred_fourier = Fourier(
            dlambda_pred.reshape(Fourier.number_of_harmonics, n_int, 1)
        )
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        dlambda_t_corr = np.where(x.contact_mask, dlambda_pred_fourier.time_series, 0.0)
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_v      = vstack(dlambda_corr.coefficients)
        return vstack((dlambda_v.real, dlambda_v.imag))



class DLFTFriction(NonlinearMethod):
    r"""
    Dynamic Lagrangian Frequency-Time (DLFT) normal contact.

    Prediction:  λ_p = IDFT[Z_r (F_adm - x_r)] + ε (x_r - g₀)
    Correction:  λ   = max(0, λ_p)
    Force:       λ̃   = DFT[λ]

    Bound to an FBSProblem at construction; reads B_fourier, F_ext_full,
    admittance caches, and the FRF provider through that reference.
    """

    def __init__(self, epsilon: float = 1.0, mu: float = 0.00):
        self.epsilon  = epsilon
        self.mu   = mu
        self._problem = None    # populated by bind()

    def bind(self, problem) -> None:
        self._problem = problem

    # --- internal helpers (use self._problem) ---

    def _get_Yr(self, x):
        if x.Yr_cache is None:
            x.Yr_cache = self._problem._get_BY(x) @ self._problem.B_fourier.T
        return x.Yr_cache

    def _get_Fext_admr(self, x):
        return self._problem._get_BY(x) @ self._problem.F_ext_full

    def _get_Zr_rhs(self, x):
        if x.Zr_rhs is None:
            Yr        = self._get_Yr(x)
            Fext_admr = self._get_Fext_admr(x)
            x.Zr_rhs  = np.linalg.solve(Yr, Fext_admr - vstack(x.fourier.coefficients)) + self.epsilon*x.fourier.coefficients
        return x.Zr_rhs

    def _get_lambda_T_corrected(self, x):
        if x.lambda_corrected is None:
            n_int = self._problem.ode.dimension
            lambda_x_opt = Fourier(
                self._get_Zr_rhs(x).reshape(Fourier.number_of_harmonics, n_int, 1)
            )
            Fourier_Real.compute_time_series(lambda_x_opt)
            zr_t   = lambda_x_opt.time_series
            Fourier_Real.compute_time_series(x.fourier)
            q_rel  = x.fourier.time_series
            lambda_t_opt = zr_t
            lambda_corr = self.epsilon * q_rel

            x.contact_mask     = lambda_p > 0.0
            lambda_t_corr      = np.where(x.contact_mask, lambda_p, 0.0)

            lambda_x_corr      = Fourier_Real.new_from_time_series(lambda_t_corr)
            x.lambda_corrected = vstack(lambda_x_corr.coefficients)
        return x.lambda_corrected

    def _invert_Yr_blockwise(self, Yr, n_int):
        Zr = zeros_like(Yr)
        for k in range(Fourier.number_of_harmonics):
            s = slice(k * n_int, (k + 1) * n_int)
            Zr[s, s] = np.linalg.solve(Yr[s, s], eye(n_int))
        return Zr

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask
        n_int = ode.dimension
        contact_tangent = x.contact_mask * eye(n_int)    # (Nt, n_int, n_int)
        J_mask    = JacobianFourier_Real.new_from_time_series(contact_tangent)
        J_mask_RI = block([[J_mask.RR, J_mask.RI], [J_mask.IR, J_mask.II]])
        Yr = self._get_Yr(x)
        Zr = self._invert_Yr_blockwise(Yr, n_int)
        complex_int_dim = Fourier.number_of_harmonics * n_int
        M    = self.epsilon * eye(complex_int_dim) - Zr
        M_RI = block([[M.real, -M.imag], [M.imag, M.real]])
        return J_mask_RI @ M_RI

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        problem = self._problem
        Y   = problem._get_FRF(x)
        dY  = problem.frf_provider.compute_FRF_derivative(
                  x.omega, Fourier.harmonics, problem.d_total, Y)
        BdY    = problem.B_fourier @ dY
        Yr     = self._get_Yr(x)
        Zr_rhs = self._get_Zr_rhs(x)
        n_int  = ode.dimension
        dlambda_pred = np.linalg.solve(
            Yr, BdY @ (problem.F_ext_full - problem.B_fourier.T @ Zr_rhs)
        )
        dlambda_pred_fourier = Fourier(
            dlambda_pred.reshape(Fourier.number_of_harmonics, n_int, 1)
        )
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        dlambda_t_corr = np.where(x.contact_mask, dlambda_pred_fourier.time_series, 0.0)
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_v      = vstack(dlambda_corr.coefficients)
        return vstack((dlambda_v.real, dlambda_v.imag))