import numpy as np
from numpy import zeros, eye, array
from numpy.fft import rfft, irfft
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from abc import ABC, abstractmethod

from .frequency_domain import Fourier, FourierOmegaPoint


class BlockDiagonalFRF(ABC):
    """
    Block-diagonal admittance applied without ever materializing the dense Y.

    Y is block-diagonal with one (d,d) block Y_n per harmonic, so every product
    acts block-wise on a right-hand side that is stacked along axis 0 in Nh
    blocks of size d (shape (Nh*d, k)). Subclasses provide the per-block apply.
    """

    def __init__(self, d: int, Nh: int):
        self._d = d
        self._Nh = Nh

    @abstractmethod
    def _apply(self, k: int, b: array) -> array:
        """Y_k @ b for harmonic block k."""

    @abstractmethod
    def _apply_transpose(self, k: int, b: array) -> array:
        """Y_k^T @ b for harmonic block k."""

    @abstractmethod
    def _apply_derivative(self, k: int, b: array) -> array:
        """(dY_k/domega) @ b for harmonic block k."""

    def _map_blocks(self, rhs: array, op) -> array:
        d = self._d
        out = np.empty(rhs.shape, dtype=complex)
        for k in range(self._Nh):
            s = slice(k * d, (k + 1) * d)
            out[s] = op(k, rhs[s])
        return out

    def solve(self, rhs: array) -> array:
        """Y @ rhs."""
        return self._map_blocks(rhs, self._apply)

    def solve_transpose(self, rhs: array) -> array:
        """Y^T @ rhs  (used to build B@Y = (Y^T B^T)^T)."""
        return self._map_blocks(rhs, self._apply_transpose)

    def apply_derivative(self, rhs: array) -> array:
        """(dY/domega) @ rhs."""
        return self._map_blocks(rhs, self._apply_derivative)


class _LUFactorizedFRF(BlockDiagonalFRF):
    """Per-harmonic LU factors of Z_n; Y_n = Z_n^{-1} applied via back-substitution."""

    def __init__(self, factors, dZ, d: int, Nh: int):
        super().__init__(d, Nh)
        self._factors = factors   # list of (lu, piv), length Nh
        self._dZ = dZ             # list of dZ_n/domega, length Nh

    def _apply(self, k, b):
        return lu_solve(self._factors[k], b)

    def _apply_transpose(self, k, b):
        return lu_solve(self._factors[k], b, trans=1)

    def _apply_derivative(self, k, b):
        # dY_n = -Y_n dZ_n Y_n  ->  dY_n @ b = -Y_n (dZ_n (Y_n b))
        y_b = lu_solve(self._factors[k], b)
        return -lu_solve(self._factors[k], self._dZ[k] @ y_b)


class _DenseBlockFRF(BlockDiagonalFRF):
    """Block FRF held as dense per-harmonic blocks (e.g. interpolated measurements)."""

    def __init__(self, Y_blocks, dY_blocks, d: int, Nh: int):
        super().__init__(d, Nh)
        self._Y = Y_blocks    # (Nh, d, d)
        self._dY = dY_blocks  # (Nh, d, d)

    def _apply(self, k, b):
        return self._Y[k] @ b

    def _apply_transpose(self, k, b):
        return self._Y[k].T @ b

    def _apply_derivative(self, k, b):
        return self._dY[k] @ b


class FRFProvider(ABC):
    """
    Computes the block-diagonal admittance matrix Y(omega) and its omega-derivative.
    Y has shape (Nh*d, Nh*d) where each (d,d) diagonal block holds Y_n for harmonic n.

    `factorize` returns a BlockDiagonalFRF that applies Y / Y^T / dY without forming
    the dense Y; the substructuring (FBS) path uses it. `compute_FRF` /
    `compute_FRF_derivative` still return the dense Y for the non-substructured path.
    """

    @abstractmethod
    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        """Return block-diagonal Y, shape (Nh*d, Nh*d), complex."""

    @abstractmethod
    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        """Return block-diagonal dY/domega, shape (Nh*d, Nh*d), complex."""

    @abstractmethod
    def factorize(self, omega: float, harmonics, d: int) -> BlockDiagonalFRF:
        """Return a BlockDiagonalFRF for matrix-free Y / Y^T / dY products."""


class NumericalFRF(FRFProvider):
    """
    FRF computed analytically from mass, damping and stiffness matrices.
        Z_n = -(n*omega)^2 * M + i*(n*omega)*C + K
        Y_n = Z_n^{-1}
        dY_n/domega = -Y_n @ dZ_n/domega @ Y_n,  dZ_n/domega = -2*n^2*omega*M + i*n*C
    """

    def __init__(self, M: array, C: array, K: array):
        self.M = M
        self.C = C
        self.K = K
        # Z_0 = K is omega-independent: factorize the static block once and reuse.
        self._K_factor = lu_factor(K)
        self._zero_dZ = zeros(K.shape, dtype=complex)

    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        Nh = len(harmonics)
        Y = zeros((Nh * d,Nh * d), dtype=complex)
        for k, n in enumerate(harmonics):
            Z_n = -(n * omega) ** 2 * self.M + 1j * n * omega * self.C + self.K
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.linalg.solve(Z_n, eye(d))
        return Y

    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        Nh = len(harmonics)
        dY = zeros((Nh * d, Nh * d), dtype=complex)
        for k, n in enumerate(harmonics):
            Y_n = Y_cache[k * d:(k + 1) * d, k * d:(k + 1) * d]
            dZ_n = -2 * n ** 2 * omega * self.M + 1j * n * self.C
            dY[k * d:(k + 1) * d, k * d:(k + 1) * d] = -Y_n @ dZ_n @ Y_n
        return dY

    def factorize(self, omega: float, harmonics, d: int) -> BlockDiagonalFRF:
        factors = []
        dZ = []
        for n in harmonics:
            if n == 0:
                factors.append(self._K_factor)   # Z_0 = K, cached
                dZ.append(self._zero_dZ)          # dY_0/domega = 0
            else:
                Z_n = -(n * omega) ** 2 * self.M + 1j * n * omega * self.C + self.K
                factors.append(lu_factor(Z_n))
                dZ.append(-2 * n ** 2 * omega * self.M + 1j * n * self.C)
        return _LUFactorizedFRF(factors, dZ, d, len(harmonics))


class ExperimentalFRF(FRFProvider):
    """
    FRF interpolated from measured frequency-domain data.

    Parameters
    ----------
    omega_frf : array, shape (N_freq,)
        Measured frequency points (positive, rad/s).
    Y : array, shape (N_freq, d, d)
        Complex admittance matrices at each frequency point.
    fd_step : float
        Step size for central-difference dY/domega.
    """

    def __init__(self, omega_frf: array, Y: array, fd_step: float = 1e-6):
        self.omega_frf = omega_frf
        self.Y = Y
        self.fd_step = fd_step
        self.interp_real = CubicSpline(omega_frf, Y.real)
        self.interp_imag = CubicSpline(omega_frf, Y.imag)

    def interpolate(self, omega) -> array:
        """Interpolate Y at scalar or vector omega. Handles Y(-omega) = conj(Y(omega))."""
        omega = np.asarray(omega)
        neg_mask = omega < 0
        omega_abs = np.abs(omega)

        if np.any(omega_abs > self.omega_frf[-1]):
            import warnings
            warnings.warn(
                f"omega outside FRF data range [0, {self.omega_frf[-1]:.4f}]. Extrapolating."
            )

        result = self.interp_real(omega_abs) + 1j * self.interp_imag(omega_abs)

        if np.ndim(omega) == 0:
            if bool(neg_mask):
                result = np.conj(result)
        elif np.any(neg_mask):
            result[neg_mask] = np.conj(result[neg_mask])

        return result

    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        Nh = len(harmonics)
        omega_harmonics = harmonics * omega          # shape (Nh,)
        Y_blocks = self.interpolate(omega_harmonics) # shape (Nh, d, d)
        Y = zeros((Nh * d, Nh * d), dtype=complex)
        for k in range(Nh):
            Y[k * d:(k + 1) * d, k * d:(k + 1) * d] = Y_blocks[k]
        return Y

    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        h = self.fd_step
        dY_raw = (self.compute_FRF(omega + h, harmonics, d)
                  - self.compute_FRF(omega - h, harmonics, d)) / (2 * h)
        return dY_raw

    def factorize(self, omega: float, harmonics, d: int) -> BlockDiagonalFRF:
        h = self.fd_step
        Y_blocks  = self.interpolate(harmonics * omega)            # (Nh, d, d)
        dY_blocks = (self.interpolate(harmonics * (omega + h))
                     - self.interpolate(harmonics * (omega - h))) / (2 * h)
        return _DenseBlockFRF(Y_blocks, dY_blocks, d, len(harmonics))