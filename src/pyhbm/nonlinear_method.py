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

    def __init__(self):
        self._harmonics_term = None       # cached kron(diag(harmonics), eye(dim))
        self._harmonics_term_key = None

    def _get_harmonics_term(self, dim):
        """Cached kron(diag(Fourier.harmonics), eye(dim))."""
        key = (tuple(Fourier.harmonics), dim)
        if self._harmonics_term_key != key:
            self._harmonics_term = kron(diag(Fourier.harmonics), eye(dim))
            self._harmonics_term_key = key
        return self._harmonics_term

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
        col_scale = x.omega * self._get_harmonics_term(ode.dimension)
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
    Correction:  λ   = c(λ_p)
    Force:       λ̃   = Σ · DFT[λ]      (Σ = Lanczos filter on the forward DFT)

    Two Gibbs-mitigation techniques (Colaïtis & Batailly, JSV 502, 2021, §2.5)
    are available and OFF by default:

    * Regularized correction (``gamma`` > 0): the non-smooth correction
      max(0, λ_p) is replaced by its hyperbolic smoothing
          c(λ_p) = λ_p/2 + sqrt((λ_p/2)² + γ²),
      whose derivative (the per-sample contact mask) is
          c'(λ_p) = ½ (1 + (λ_p/2) / sqrt((λ_p/2)² + γ²)).
      γ = 0 recovers max(0, λ_p) and the Boolean mask exactly.

    * Lanczos filter (``lanczos_m`` > 0): each harmonic block of the forward
      DFT operator is scaled by σ_n = sinc(ρ_n/(H+1))^m, ρ_n = |n| if
      |n| ≥ C_H else 0. m = 0 gives σ_n = 1 (no filtering). The SAME σ is
      applied to the force, its Jacobian and its ω-derivative.

    Bound to an FBSProblem at construction; reads B_fourier, F_ext_full,
    admittance caches, and the FRF provider through that reference.
    """

    def __init__(self, epsilon: float = 1.0, g_zero: float = 0.0,
                 gamma: float = 0.0, lanczos_m: float = 0.0,
                 lanczos_cutoff: int = 1):
        self.epsilon        = epsilon
        self.g_zero         = g_zero
        self.gamma          = gamma           # smoothing parameter [force]; 0 => exact max(0,·)
        self.lanczos_m      = lanczos_m       # Lanczos intensity; 0 => no filtering
        self.lanczos_cutoff = lanczos_cutoff  # C_H: harmonics below it are unfiltered
        self._problem       = None            # populated by bind()
        self._sigma         = None            # cached Lanczos factors σ_n
        self._sigma_key     = None            # (harmonics, H, m, C_H) the cache was built for
        self._sigma_kron     = None           # cached kron(diag(σ), eye(n_int))
        self._sigma_kron_key = None
        self._eye_int        = None           # cached eye(Nh*n_int)

    def bind(self, problem) -> None:
        self._problem = problem

    # --- regularization + Lanczos helpers ---

    def _correct(self, lambda_p):
        """Pointwise correction c(λ_p) and its derivative (the contact mask)."""
        if self.gamma == 0.0:
            mask = lambda_p > 0.0
            return np.where(mask, lambda_p, 0.0), mask
        u = 0.5 * lambda_p
        s = np.sqrt(u * u + self.gamma ** 2)
        lambda_corr = u + s
        mask = 0.5 * (1.0 + u / s)
        return lambda_corr, mask

    def _get_lanczos_sigma(self):
        """Per-harmonic Lanczos factors σ_n, cached for the current harmonics set."""
        harmonics = Fourier.harmonics
        H   = Fourier.harmonic_truncation_order
        key = (tuple(harmonics), H, self.lanczos_m, self.lanczos_cutoff)
        if self._sigma_key != key:
            sigma = np.ones(len(harmonics))
            if self.lanczos_m != 0.0:
                for i, n in enumerate(harmonics):
                    nn  = abs(int(n))
                    rho = nn if nn >= self.lanczos_cutoff else 0
                    X   = rho / (H + 1)
                    if X != 0.0:
                        sigma[i] = (np.sin(np.pi * X) / (np.pi * X)) ** self.lanczos_m
            self._sigma     = sigma
            self._sigma_key = key
        return self._sigma

    def _apply_lanczos_coeffs(self, coefficients):
        """Scale forward-DFT force coefficients (Nh, n_int, 1) by σ_n."""
        if self.lanczos_m == 0.0:
            return coefficients
        return coefficients * self._get_lanczos_sigma()[:, None, None]

    def _get_sigma_kron(self, n_int):
        """Cached Σ = kron(diag(σ_n), eye(n_int)) for the current harmonics set."""
        sigma = self._get_lanczos_sigma()
        key = (self._sigma_key, n_int)
        if self._sigma_kron_key != key:
            self._sigma_kron = kron(diag(sigma), eye(n_int))
            self._sigma_kron_key = key
        return self._sigma_kron

    def _get_eye_complex_int(self, n_int):
        """Cached eye(Nh*n_int)."""
        dim = Fourier.number_of_harmonics * n_int
        if self._eye_int is None or self._eye_int.shape[0] != dim:
            self._eye_int = eye(dim)
        return self._eye_int

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
            lambda_t_corr, x.contact_mask = self._correct(lambda_p)
            lambda_x_corr      = Fourier_Real.new_from_time_series(lambda_t_corr)
            coeffs             = self._apply_lanczos_coeffs(lambda_x_corr.coefficients)
            x.lambda_corrected = vstack(coeffs)
        return x.lambda_corrected

    def _invert_Yr_blockwise(self, Yr, n_int):
        # Yr is block-diagonal (Nh blocks of n_int x n_int); invert all blocks at once.
        Nh = Fourier.number_of_harmonics
        idx = np.arange(Nh)
        diag_blocks = Yr.reshape(Nh, n_int, Nh, n_int)[idx, :, idx, :]   # (Nh, n_int, n_int)
        inv_blocks = np.linalg.solve(
            diag_blocks, np.broadcast_to(eye(n_int), (Nh, n_int, n_int)))
        Zr = zeros_like(Yr)
        Zr.reshape(Nh, n_int, Nh, n_int)[idx, :, idx, :] = inv_blocks
        return Zr

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask
        n_int = ode.dimension
        contact_tangent = x.contact_mask * eye(n_int)    # (Nt, n_int, n_int)
        J_mask    = JacobianFourier_Real.new_from_time_series(contact_tangent)
        if self.lanczos_m != 0.0:
            # filter the forward DFT (output harmonics): Σ Γ⁺ diag(m) Γ
            Sigma = self._get_sigma_kron(n_int)
            J_mask_RI = block([[Sigma @ J_mask.RR, Sigma @ J_mask.RI],
                               [Sigma @ J_mask.IR, Sigma @ J_mask.II]])
        else:
            J_mask_RI = block([[J_mask.RR, J_mask.RI], [J_mask.IR, J_mask.II]])
        Yr = self._get_Yr(x)
        Zr = self._invert_Yr_blockwise(Yr, n_int)
        M    = self.epsilon * self._get_eye_complex_int(n_int) - Zr
        M_RI = block([[M.real, -M.imag], [M.imag, M.real]])
        return J_mask_RI @ M_RI

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        problem = self._problem
        factor = problem._get_factor(x)
        Yr     = self._get_Yr(x)
        Zr_rhs = self._get_Zr_rhs(x)
        n_int  = ode.dimension
        # B dY u, matrix-free: u complex, then B @ (dY @ u).
        u = problem.F_ext_full - problem.B_fourier.T @ Zr_rhs
        BdY_u = problem.B_fourier @ factor.apply_derivative(u)
        dlambda_pred = np.linalg.solve(Yr, BdY_u)
        dlambda_pred_fourier = Fourier(
            dlambda_pred.reshape(Fourier.number_of_harmonics, n_int, 1)
        )
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        # chain rule: ∂λ/∂ω = c'(λ_p) · ∂λ_p/∂ω  (mask is Boolean for γ=0, float for γ>0)
        dlambda_t_corr = x.contact_mask * dlambda_pred_fourier.time_series
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        coeffs         = self._apply_lanczos_coeffs(dlambda_corr.coefficients)
        dlambda_v      = vstack(coeffs)
        return vstack((dlambda_v.real, dlambda_v.imag))