"""FRF-provider factories for the two-rod example.

Two ways to feed the coupled rods' LINEAR dynamics into the (DLFT or AFT)
contact solve:

    make_numerical_provider(system)
        Exact  Y(omega) = Z(omega)^-1  computed on the fly from the system's
        (nondimensionalized) M, C, K.

    make_experimental_provider(system, density_per_hz, ...)
        Mimics measured data: pre-sample  Y(omega)  on a uniform grid whose
        spacing is set by a measurement DENSITY in SAMPLES PER HERTZ on the
        physical frequency axis, spanning all harmonic frequencies the solver
        will query, then hand it to :class:`ExperimentalFRF`, which CUBIC-SPLINE
        interpolates between samples.  Coarser grids => interpolation error; this
        is the knob that trades accuracy for a realistic data-driven workflow.

The grid lives in the nondimensional frequency  omega_hat = omega / omega_1 the
continuation runs in (the provider is queried at  harmonics * omega_hat), but the
point count is derived from the PHYSICAL span so that the spacing is a fixed
Delta f = 1 / density_per_hz  [Hz].
"""
import numpy as np
from numpy import eye, zeros

from pyhbm import NumericalFRF, ExperimentalFRF


def make_numerical_provider(system):
    """Exact analytic FRF provider from the system's nondimensional M, C, K."""
    return NumericalFRF(system.mass_matrix,
                        system.damping_matrix,
                        system.stiffness_matrix)


def sample_admittance(system, omega_grid):
    """Y(omega) = (-omega^2 M + i omega C + K)^-1 at each omega on the grid.

    :param omega_grid: (N_freq,) nondimensional frequencies (>= 0).
    :returns: (N_freq, d, d) complex admittance blocks.
    """
    M, C, K = system.mass_matrix, system.damping_matrix, system.stiffness_matrix
    d = M.shape[0]
    Y = zeros((len(omega_grid), d, d), dtype=complex)
    for i, w in enumerate(omega_grid):
        Z = -w ** 2 * M + 1j * w * C + K
        Y[i] = np.linalg.solve(Z, eye(d))
    return Y


def add_measurement_noise(omega_grid, Y, snr_db, rng=None):
    """Inject measurement noise the pyFBS way (pyfbs.mck.MCK.add_noise):

        noise   = |Y| * (n1*randn + 1j*n2*randn)  +  (n3*randn + 1j*n4*randn)
        Y_noisy = Y + noise                        (per frequency bin, per channel)

    The single SNR knob maps onto pyFBS's four coefficients as

        s  = 10^(-snr_db/20)
        n1 = n2 = s                      (proportional, dimensionless)
        n3 = n4 = s * median|Y_ij|       (additive floor, per channel scale)

    so e.g. 40 dB -> 1% proportional jitter + a floor at 1% of the channel's
    median magnitude.

    :param omega_grid: (n_freq,) nondimensional grid (unused, kept for signature).
    :param Y:          (n_freq, d, d) complex admittance blocks.
    :param snr_db:     signal-to-noise ratio in dB; np.inf -> Y returned unchanged.
    :param rng:        optional numpy Generator for reproducible noise.
    :returns:          noisy Y, same shape/dtype as the input.
    """
    if not np.isfinite(snr_db):
        return Y
    rng = np.random.default_rng() if rng is None else rng
    n_freq, d, _ = Y.shape
    s = 10.0 ** (-snr_db / 20.0)
    Y_noisy = Y.copy()
    for i in range(d):                      # per channel: bounds peak memory on
        for j in range(d):                  # the long experimental grids
            y_abs = np.abs(Y[:, i, j])
            floor = np.median(y_abs)        # channel magnitude scale (n3 = n4)
            g = rng.standard_normal((n_freq, 4))
            Y_noisy[:, i, j] += (y_abs * (s * g[:, 0] + 1j * s * g[:, 1])
                                 + floor * (s * g[:, 2] + 1j * s * g[:, 3]))
    return Y_noisy


def make_experimental_provider(system, density_per_hz, harmonics, omega_range,
                               margin=1.05, fd_step=1e-6,
                               snr_db=np.inf, noise_seed=None):
    """Build an ExperimentalFRF by sampling the system's admittance on a grid.

    The grid spacing is a measurement density in SAMPLES PER HERTZ on the
    PHYSICAL axis: Delta f = 1 / density_per_hz.  The point count is derived from
    the physical span [0, f_max], where f_max corresponds to the top queried
    harmonic frequency  H * omega_hi * omega_1.

    :param density_per_hz: samples per Hz on the physical axis
        (e.g. 0.02 => one sample every 50 Hz).
    :param harmonics:      harmonic list (sets the top queried frequency H*omega_hi).
    :param omega_range:    (omega_lo, omega_hi) nondimensional sweep window.
    :param margin:         stretch factor on the top frequency, for safe interpolation.
    :param snr_db:         measurement signal-to-noise ratio in dB applied to the
        sampled admittance (np.inf => noiseless, clean data).
    :param noise_seed:     seed for the noise RNG, for reproducible branches.
    :returns: (provider, n_freq) -- the provider plus the resolved sample count.
    """
    h_max         = int(np.max(harmonics))
    omega_hi      = max(abs(omega_range[0]), abs(omega_range[1]))   # in omega_hat
    omega_max_hat = h_max * omega_hi * margin                       # top queried omega_hat
    f_max = omega_max_hat * system.omega_ref / (2.0 * np.pi)        # physical Hz
    n_freq = max(2, int(round(density_per_hz * f_max)) + 1)
    omega_grid = np.linspace(0.0, omega_max_hat, n_freq)            # nondimensional grid
    Y = sample_admittance(system, omega_grid)
    Y = add_measurement_noise(omega_grid, Y, snr_db,
                              rng=np.random.default_rng(noise_seed))
    return ExperimentalFRF(omega_grid, Y, fd_step=fd_step), n_freq
