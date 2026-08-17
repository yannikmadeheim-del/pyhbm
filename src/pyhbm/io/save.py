import pickle
import json

import numpy as np
from numpy import array

from ..frequency_domain import Fourier


def _save_bifurcation_data(bifurcations: list) -> list:
    """
    Convert bifurcations to serializable dictionary format.
    
    Parameters
    ----------
    bifurcations : list
        List of SpecialPoint objects.
    
    Returns
    -------
    list
        List of dictionaries with bifurcation data.
    """
    data = []
    for bif in bifurcations:
        bif_data = {
            'type': bif.type,
            'index': int(bif.index),
            'omega': float(bif.omega),
        }
        if bif.refined_omega is not None:
            bif_data['refined_omega'] = float(bif.refined_omega)
        
        if bif.type == 'hopf':
            if bif.relative_period is not None:
                bif_data['relative_period'] = float(bif.relative_period)
                bif_data['rational_approx'] = bif.rational_approx
            if bif.hopf_frequency is not None:
                bif_data['hopf_frequency'] = float(bif.hopf_frequency)
            if bif.hopf_multiplier_magnitude is not None:
                bif_data['hopf_multiplier_magnitude'] = float(bif.hopf_multiplier_magnitude)
            if bif.multipliers is not None:
                bif_data['multipliers'] = [
                    {'real': float(m.real), 'imag': float(m.imag)} 
                    for m in bif.multipliers
                ]
        
        data.append(bif_data)
    
    return data


def save_solution_set(solution_set, path, 
                      harmonic_amplitude=True, 
                      amplitude=True, 
                      angular_frequency=True,
                      fourier_coefficients=False, 
                      time_series=False, 
                      adimensional_time_samples=False,
                      iterations=False, 
                      step_length=False,
                      bifurcations=False,
                      MATLAB_compatible=False,
                      freq_domain_ode=None):
    
    solution_data = {}

    if harmonic_amplitude:
        solution_data["harmonic_amplitude"] = \
            abs(array([fourier.coefficients[..., 0] for fourier in solution_set.fourier])) * 2 / Fourier.number_of_time_samples
    
    if amplitude:
        solution_data["amplitude"] = array([np.max(abs(fourier.time_series), axis=0) for fourier in solution_set.fourier])

    if angular_frequency:
        solution_data["angular_frequency"] = solution_set.omega.copy()

    if fourier_coefficients:
        solution_data["fourier_coefficients"] = array([fourier.coefficients.copy() for fourier in solution_set.fourier])

    if time_series:
        solution_data["time_series"] = array([fourier.time_series.copy() for fourier in solution_set.fourier])

    if adimensional_time_samples:
        solution_data["adimensional_time_samples"] = Fourier.adimensional_time_samples

    if iterations:
        solution_data["iterations"] = solution_set.iterations.copy()

    if step_length:
        solution_data["step_length"] = solution_set.step_length.copy()

    if bifurcations:
        if freq_domain_ode is None:
            raise ValueError("freq_domain_ode is required when bifurcations=True")
        
        from ..stability import BifurcationDetector, FloquetAnalyzer
        
        # Compute stability reports
        analyzer = FloquetAnalyzer(freq_domain_ode.ode)
        stability_reports = []
        for fourier in solution_set.fourier:
            if fourier.time_series is None:
                fourier.compute_time_series()
            report = analyzer.analyze(
                fourier.time_series, 
                Fourier.adimensional_time_samples, 
                solution_set.omega[len(stability_reports)]
            )
            stability_reports.append(report)
        
        # Detect bifurcations
        detector = BifurcationDetector()
        bifurcations_data = detector.detect_all(solution_set, stability_reports)
        
        # Save to solution_data
        solution_data['bifurcations'] = _save_bifurcation_data(bifurcations_data)

    if MATLAB_compatible:
        from scipy.io import savemat
        savemat(path, solution_data)
    else:
        with open(path, 'wb') as handle:
            pickle.dump(solution_data, handle)


def save_solution_csv(path, solution_set, labels, out_label,
                      header_lines=(), recovery=None):
    """
    Export a continuation branch as a plain-text CSV, one row per converged point.

    The file is a block of ``#`` comment lines (``header_lines``, meant to record
    the model and every solver setting in readable text), then::

        freq_hz, omega_rad_s, iterations, step_length, uout_h1_abs, uout_time_max,
        re_h<h>_<label>, im_h<h>_<label>, ...        harmonic-major, then channel

    The harmonic block holds the complex amplitudes a_h normalized so that

        u(t)    = Re( sum_h a_h exp(1j h omega t) )
        udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) )

    i.e. |a_h| is the physical amplitude, not the raw rFFT solver coefficient
    (a_h = c_h/N_t for h = 0 and 2 c_h/N_t otherwise). Read back with
    ``pandas.read_csv(path, comment="#")``.

    :param labels: name of every exported channel, in column order.
    :param out_label: the channel summarized by the two ``uout_*`` columns.
    :param recovery: optional (len(labels), dimension) matrix mapping the solver
        coordinates onto the exported channels, applied per harmonic. ``None``
        means the solver coordinates already are the channels.
    :returns: (omega, uout_h1_abs, uout_time_max)
    """
    from numpy.fft import irfft

    labels = list(labels)
    harmonics = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples

    # (n, Nh, dimension) raw rFFT-convention coefficients -> (n, Nh, n_channels)
    raw = np.array([fourier.coefficients[:, :, 0] for fourier in solution_set.fourier])
    if recovery is not None:
        raw = raw @ np.asarray(recovery).T
    n = raw.shape[0]
    assert raw.shape[2] == len(labels), (raw.shape[2], len(labels))

    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])
    amp = raw * scale[None, :, None]                   # physical amplitudes a_h

    # the two summary curves: |a_1| of the output channel and the one-period peak
    # of its time signal -- zero-pad the retained harmonics back onto a dense rFFT
    # bin vector, then one irfft per point gives the whole time series at once
    i_out, i_h1 = labels.index(out_label), harmonics.index(1)
    padded = np.zeros((n, max(harmonics) + 1), dtype=complex)
    padded[:, harmonics] = raw[:, :, i_out]
    uout_time_max = np.abs(irfft(padded, n=n_t, axis=1)).max(axis=1)
    uout_h1_abs = np.abs(amp[:, i_h1, i_out])

    omega = np.asarray(solution_set.omega, dtype=float)
    freq = omega / (2.0 * np.pi)
    reim = np.stack([amp.real, amp.imag], axis=-1)     # re/im adjacent per channel
    data = np.column_stack([
        freq, omega,
        np.asarray(solution_set.iterations, dtype=float),
        np.asarray(solution_set.step_length, dtype=float),
        uout_h1_abs, uout_time_max,
        reim.reshape(n, -1),                           # harmonic-major, then channel
    ])
    columns = (["freq_hz", "omega_rad_s", "iterations", "step_length",
                "uout_h1_abs", "uout_time_max"]
               + [f"{part}_h{h}_{label}" for h in harmonics for label in labels
                  for part in ("re", "im")])
    assert data.shape[1] == len(columns), (data.shape[1], len(columns))

    with open(path, "w", newline="") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        handle.write(",".join(columns) + "\n")
        np.savetxt(handle, data, delimiter=",", fmt="%.10e")
    print(f"solution written: {path}  ({n} points, {len(labels)} channels x"
          f" {len(harmonics)} harmonics)")
    return omega, uout_h1_abs, uout_time_max
