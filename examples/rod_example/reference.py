"""Stored, swappable reference NFRC and the trajectory relative-error metric.

The "ground truth" is a dense DLFT-numerical run (tiny step length) saved to CSV by
``make_reference.py``.  Because it is a plain CSV it can be regenerated once and
swapped for any other solver's solution.

Trajectory relative error (Krack & Gross Eq. 2.46)
--------------------------------------------------
The error quantity is the one Krack & Gross propose for HB convergence/validation
(*Harmonic Balance for Nonlinear Vibration Problems*, Eq. 2.46): the RELATIVE RMS
deviation of a chosen response (here the rod tip u_B(t)), computed from its Fourier
coefficients via Parseval,

    e = sqrt( mean_t [u_B,var - u_B,ref]^2 ) / sqrt( mean_t u_B,ref^2 ).

Two things make this work across HB runs of different harmonic order:
  * PHYSICAL coefficients c_hat_k = c_k / N_time (run.py) so a 21-harmonic variant
    and a 120-harmonic reference are on the same scale, and
  * zero-padding the variant up to the reference order, so the reference's
    truncation tail (harmonics the variant cannot represent) is correctly part of
    the error -- the reason Krack uses a high-H_ref reference in the first place.

Krack evaluates this at a fixed operating point (varying H).  We evaluate it ALONG
the frequency branch to compare methods/parameters; since the NFRC folds, each
variant point is paired to an operating point on the dense reference by **monotonic
nearest projection** (fold-robust: monotonicity stops snapping across the overhang;
sampling-independent: projecting onto the dense reference avoids arc-length drift).
Aggregated as mean / max / RMS (RMS is the headline scalar).

If the reference CSV carries no spectrum columns (e.g. an external amplitude-only
file such as the NLvib references), the metric falls back to the arc-length-matched
peak-amplitude relative error.

CSV format
----------
    omega, A_peak, uB_Re_0, uB_Im_0, uB_Re_1, uB_Im_1, ..., uB_Re_H, uB_Im_H
Extra columns are ignored; the spectrum columns are optional.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent


# ============================ monotonic projection =========================

def _monotonic_match(var_pts, ref_pts):
    """Match each (ordered) variant point to a forward-only nearest reference point.

    Greedy: a pointer over the reference only advances; for each variant point it
    steps forward to the local distance minimum.  Returns (idx, dist) with idx the
    matched reference index per variant point and dist the matched distance.
    """
    nr = len(ref_pts)
    n  = len(var_pts)
    idx  = np.empty(n, int)
    dist = np.empty(n)
    j = 0
    for i in range(n):
        p = var_pts[i]
        d_j = np.hypot(*(p - ref_pts[j]))
        k = j + 1
        while k < nr:
            d_k = np.hypot(*(p - ref_pts[k]))
            if d_k <= d_j:
                d_j, j = d_k, k
                k += 1
            else:
                break                       # first forward local minimum
        idx[i], dist[i] = j, d_j
    return idx, dist


# ============================ reference curve ==============================

@dataclass
class ReferenceCurve:
    """A dense reference NFRC: omega, peak amplitude, and (optionally) the u_B spectrum."""
    omega:        np.ndarray                 # physical angular frequency [rad/s]
    A_peak:       np.ndarray                 # peak ||u_B(t)||_inf [m]
    uB_harmonics: np.ndarray = None          # (n, Nh) complex spectrum of u_B, or None
    k_rel:        float = None
    source:       str = ""

    _w_scale: float = field(default=1.0, repr=False)
    _a_scale: float = field(default=1.0, repr=False)
    _pts:     np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._w_scale = float(np.ptp(self.omega)) or 1.0
        self._a_scale = float(np.ptp(self.A_peak)) or 1.0
        self._pts = self._normalize(self.omega, self.A_peak)

    def _normalize(self, omega, A_peak) -> np.ndarray:
        return np.column_stack((np.asarray(omega) / self._w_scale,
                                np.asarray(A_peak) / self._a_scale))

    @property
    def has_spectrum(self) -> bool:
        return self.uB_harmonics is not None

    def match(self, omega, A_peak):
        """Monotonic projection of an external branch onto this reference.

        :returns: (idx, dist) -- matched reference index and normalized distance
            for each (omega, A_peak) point, in branch order.
        """
        return _monotonic_match(self._normalize(omega, A_peak), self._pts)


# ============================ paths / IO ===================================

def reference_path(k_rel) -> Path:
    """Canonical CSV path for a given obstacle stiffness (mirrors NLvib naming)."""
    return _DIR / f"reference_dlft_numerical_kobs_{k_rel:g}_krod.csv"


def load_reference_csv(path) -> ReferenceCurve:
    """Load a reference NFRC from CSV (header ``omega,A_peak[,uB_Re_h,uB_Im_h,...]``)."""
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    omega  = np.atleast_1d(data["omega"])
    A_peak = np.atleast_1d(data["A_peak"])

    # optional full spectrum: collect uB_Re_h / uB_Im_h columns in harmonic order
    re_cols = sorted((n for n in names if n.startswith("uB_Re_")),
                     key=lambda n: int(n.rsplit("_", 1)[1]))
    uB_harmonics = None
    if re_cols:
        Nh = len(re_cols)
        uB_harmonics = np.empty((len(omega), Nh), complex)
        for h, rname in enumerate(re_cols):
            iname = rname.replace("uB_Re_", "uB_Im_")
            uB_harmonics[:, h] = np.atleast_1d(data[rname]) + 1j * np.atleast_1d(data[iname])

    k_rel = None
    parts = path.stem.split("_")
    if "kobs" in parts:
        try:
            k_rel = float(parts[parts.index("kobs") + 1])
        except (ValueError, IndexError):
            k_rel = None
    return ReferenceCurve(omega=omega, A_peak=A_peak, uB_harmonics=uB_harmonics,
                          k_rel=k_rel, source=str(path))


# ============================ the error metric =============================

_COVERAGE_DIST = 0.02   # normalized distance above which a match is "off the reference"


def _parseval_weights(nh):
    """Parseval weights for two-sided coefficients: DC -> 1, each harmonic -> 2.

    mean_t y^2 = |c_hat_0|^2 + 2 sum_{k>=1} |c_hat_k|^2  (Krack & Gross Eq. 2.18).
    """
    w = np.full(nh, 2.0)
    w[0] = 1.0
    return w


def _pad_cols(X, n):
    """Zero-pad a (rows, k) complex array to (rows, n) columns (k <= n)."""
    if X.shape[1] == n:
        return X
    out = np.zeros((X.shape[0], n), complex)
    out[:, :X.shape[1]] = X
    return out


def relative_error(result, reference: ReferenceCurve):
    """Trajectory relative error of ``result`` vs ``reference``, as Krack proposes.

    Per branch point, the relative RMS deviation of the rod-tip orbit u_B(t) is
    computed from its Fourier coefficients via Parseval (Krack & Gross,
    *Harmonic Balance for Nonlinear Vibration Problems*, Eq. 2.46):

        e = sqrt( mean_t [u_B,var - u_B,ref]^2 ) / sqrt( mean_t u_B,ref^2 )
          = || c_hat_var - c_hat_ref ||_w / || c_hat_ref ||_w ,

    with the Parseval weights w (DC 1, harmonics 2) and PHYSICAL two-sided
    coefficients c_hat_k = c_k / N_time.  The lower-order variant is zero-padded to
    the (high) reference order, so the reference's truncation tail correctly counts
    as error -- exactly why a high-H_ref reference is used.  Operating points are
    paired by monotonic projection onto the reference branch.

    Falls back to peak-amplitude relative error if the reference has no spectrum
    (e.g. an external amplitude-only CSV).

    :returns: (e, mean, max, rms, frac_outside).
    """
    idx, dist = reference.match(result.omega_phys, result.peak)
    frac_out = float(np.mean(dist > _COVERAGE_DIST)) if len(dist) else 0.0

    if reference.has_spectrum and result.uB_harmonics is not None:
        X_ref = reference.uB_harmonics[idx]                      # (n, Hr) complex
        X_var = result.uB_harmonics                             # (n, Hv) complex
        nh = max(X_ref.shape[1], X_var.shape[1])
        X_ref, X_var = _pad_cols(X_ref, nh), _pad_cols(X_var, nh)
        w = _parseval_weights(nh)
        num = np.sqrt((w * np.abs(X_var - X_ref) ** 2).sum(axis=1))
        den = np.sqrt((w * np.abs(X_ref) ** 2).sum(axis=1))
    else:                                                        # amplitude fallback
        A_ref = reference.A_peak[idx]
        num = np.abs(result.peak - A_ref)
        den = np.abs(A_ref)

    e = num / (den + 1e-30)
    if e.size == 0:
        return e, float("nan"), float("nan"), float("nan"), frac_out
    return e, float(e.mean()), float(e.max()), float(np.sqrt(np.mean(e ** 2))), frac_out
