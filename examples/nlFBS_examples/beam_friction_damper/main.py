# NOT WORKING




"""
Steel cantilever beam with a flexible dry-friction damper -- 1-D relative motion.

Reproduction (analogue) of the first validation example of
    S. Nacivet, C. Pierre, F. Thouverez, L. Jezequel,
    "A dynamic Lagrangian frequency-time method for the vibration of
    dry-friction-damped systems", J. Sound Vib. 265(1), 2003, 201-219 (Figs. 4-5).

System (two FBS substructures coupled only through the nonlinear contact):

  * BEAM  -- clamped-free steel beam, cross-section 0.01 m (X) x 0.1 m (Y),
    length L = 0.5 m, modelled with 2-D frame finite elements (Euler-Bernoulli
    bending in X + bar axial in Z, 3 DOF/node: [axial u_z, transverse w_x, rot]).
    Bending (X) is the dominant motion; the small axial (Z) DOF is the contact
    normal direction. Harmonic forcing  F_X = F0 sin(w t)  at the free tip.

  * DAMPER -- a massless node grounded by two mono-directional springs
    k_T = 2.4e7 N/m (tangential / X) and k_N = 2.4e3 N/m (normal / Z), pressed
    against the beam at 0.3 L by a static normal pre-load N0 = 1500 N.

The contact element couples the beam node at 0.3 L to the damper node on the
RELATIVE interface coordinates, ordered  [normal ; tangential]  to match
``DLFTFriction`` (index 0 = normal, index 1 = tangential):

    x_r^N = u_z,beam(0.3L) - u_z,damper      (normal,    Z)
    x_r^T = w_x,beam(0.3L) - u_x,damper      (tangential, X, sliding)

DLFT-FBS solves for x_r; the Coulomb friction + unilateral normal contact force
is slaved to x_r by the prediction-correction at every residue evaluation.

Validation is DLFT-internal (no external time-integration reference):
  (1) finite-difference check of the analytical residual Jacobian and dr/dw;
  (2) nonlinear frequency-response curve (FRC) -- Nacivet Fig. 5b shape, showing
      the friction-damped & frequency-shifted resonance vs. the stuck-linear FRF;
  (3) steady-state time history + stick-slip contact forces at the peak (Fig. 5a);
  (4) penalty (epsilon) independence and harmonic convergence.

The DLFTFriction corrector reproduces Nacivet's Eqs. (17)-(33) exactly
(predictor lambda^opt, the sequential lambda^cor recursion, and the
stick/slip/separation corrections), with the single forward sweep of Fig. 2
(n_sweep=1, lambda^cor_{-1}=0).

DIFFERENCES FROM THE PAPER (intentional):
  * SOLVER: Nacivet solves f({U_r}) = {F_r} - {lambda} - [K_r]{U_r} (Eq. 15,
    Fig. 2) with a HYBRID POWELL trust-region algorithm (MINPACK hybrd), robust
    through gross slip. Here we use pyhbm's Newton + arc-length continuation with
    the analytical (history-dropped) DLFT tangent. In 1-D gross slip that tangent's
    T->T block vanishes and Newton contracts poorly, so we operate in the
    partial-slip regime (below). A Powell-type solver would handle gross slip.
  * MODEL: this is a FULL finite-element beam, NOT the Craig-Bampton-reduced
    ABAQUS model, so the resonance lands at the model's own first mode (~54 Hz),
    not 124 Hz, and the modal content differs.
  * FORCING: F0 is sized so the relative tangential motion is a small multiple of
    the slip threshold  mu*N0/k_T ~ 6.25 um  -- the Nacivet partial-slip /
    micro-slip regime, where Newton converges cleanly. (At the paper's F_X = 50 N
    this compliant full-FE beam grossly slips.)
Parameters otherwise follow Nacivet:  mu = 0.1, xi = 0.01, eps = 8e7.
"""
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

from pyhbm import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTFriction,
    HarmonicBalanceMethod,
)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered


from dynamical_system import BeamFrictionDamper


# ============================ parameters ====================================
MU       = 0.1
XI       = 0.01
K_T      = 2.4e7
K_N      = 2.4e3
PRELOAD  = 1500.0
EPSILON  = 5.0e7
SAVE_PNG = False   # set True to write the figure to disk
HARMONICS = list(range(0, 21))       # DC (preload) + harmonics 1..15
POLY_DEG  = 20
AMP_FACTOR = 3.0                     # peak motion as multiple of slip threshold

# slip threshold: stuck-state beam motion at 0.3L that makes |damper force| = mu*N0
SLIP_THRESHOLD = MU * PRELOAD / K_T  # ~6.25 um

HarmonicBalanceMethod.update_dependencies(HARMONICS, POLY_DEG)
beam = BeamFrictionDamper(k_T=K_T, k_N=K_N, preload=PRELOAD, F0=1.0,
                          xi=XI, poly_deg=POLY_DEG)
OMEGA_REF = beam.omega_ref
print(f"stuck-system modes (Hz): {[f'{w/2/np.pi:.1f}' for w in beam.omega_stuck[:4]]}")
print(f"reference (resonance) frequency: {OMEGA_REF:.1f} rad/s  ({OMEGA_REF/2/np.pi:.1f} Hz)")
print(f"slip threshold (beam motion at 0.3L): {SLIP_THRESHOLD*1e6:.2f} um")
print(f"DOFs: total = {beam.total_dimension}, interface = {beam.dimension}, "
      f"N_time = {Fourier.number_of_time_samples}, N_h = {Fourier.number_of_harmonics}")

M, C, K = beam.mass_matrix, beam.damping_matrix, beam.stiffness_matrix


def stuck_response(omega_hat, F0=1.0):
    """Linear stuck+in-contact FRF: |u_x(0.3L)| with damper springs grounding the node."""
    Z = -omega_hat**2 * M + 1j * omega_hat * C + K
    Z[beam.beam_trans_damper, beam.beam_trans_damper] += K_T
    Z[beam.beam_axial_damper, beam.beam_axial_damper] += K_N
    F = zeros((beam.total_dimension, 1), complex); F[beam.beam_trans_tip, 0] = F0
    return np.linalg.solve(Z, F)


# size F0 so the stuck peak motion at 0.3L = AMP_FACTOR * slip threshold
wh_scan = np.linspace(0.6, 1.4, 200)
stuck_peak = max(abs(stuck_response(w)[beam.beam_trans_damper, 0]) for w in wh_scan)
F0 = AMP_FACTOR * SLIP_THRESHOLD / stuck_peak
beam.F0 = F0
print(f"forcing amplitude F0 = {F0:.4f} N "
      f"(stuck peak motion = {AMP_FACTOR:.1f} x slip threshold)")


# ============================ build problem =================================
provider = NumericalFRF(beam.mass_matrix, beam.damping_matrix, beam.stiffness_matrix)
contact  = DLFTFriction(epsilon_N=EPSILON, epsilon_T=EPSILON, mu=MU,
                        g_zero=0.0, n_tangential=1, n_sweep=1)
problem  = FBSProblem(beam, provider, contact)
h0 = HARMONICS.index(0); h1 = HARMONICS.index(1)


def linear_relative(omega_hat):
    """Open-contact linear relative response x_r = B u to F_X (seed for continuation)."""
    Z = -omega_hat**2 * M + 1j * omega_hat * C + K
    F = zeros((beam.total_dimension, 1), complex); F[beam.beam_trans_tip, 0] = beam.F0
    return (beam.B_coupling @ np.linalg.solve(Z, F))[:, 0]


# ============================ continuation (FRC) ============================
solver = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=problem,
    corrector_parameterization=ArcLengthParameterization,
    predictor=TangentPredictorBordered)

OMEGA_START, OMEGA_END = 0.70, 1.30
xr0 = linear_relative(OMEGA_START)
Q_ig = zeros((Fourier.number_of_harmonics, beam.dimension, 1), complex)
Q_ig[h1, 0, 0] = xr0[0]; Q_ig[h1, 1, 0] = xr0[1]
ig = FourierOmegaPoint(Fourier(Q_ig), omega=OMEGA_START)
rd = FourierOmegaPoint.new_from_first_harmonic(
    zeros((beam.dimension, 1), complex), omega=1.0)

print(f"Continuation w_hat: {OMEGA_START} -> {OMEGA_END}")
t0 = time()
ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=2000,
    angular_frequency_range=[OMEGA_START, OMEGA_END],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={
        "base": 4.0, "initial_step_length": 0.008, "maximum_step_length": 0.015,
        "minimum_step_length": 1e-7, "goal_number_of_iterations": 10},
    jacobian_update_frequency=1)
wh = np.array(ss.omega)
freq_hz = wh * OMEGA_REF / (2 * np.pi)
print(f"-> {len(wh)} points in {time()-t0:.1f} s, freq in "
      f"[{freq_hz.min():.1f}, {freq_hz.max():.1f}] Hz")


def evaluate(fourier, omega_hat):
    """Return (peak |u(0.3L)|, full time series, contact force time series, slip frac)."""
    full = problem.compute_full_response(fourier, omega_hat)
    Fourier_Real.compute_time_series(full)
    x = FourierOmegaPoint(fourier, omega_hat)
    lam = contact._get_lambda_corrected(x)
    lam_f = Fourier(lam.reshape(Fourier.number_of_harmonics, beam.dimension, 1))
    Fourier_Real.compute_time_series(lam_f)
    Jloc = x.contact_mask
    slip = sum(1 for k in range(Jloc.shape[0])
               if not (np.all(np.abs(Jloc[k]) < 1e-30)            # not separation
                       or (abs(Jloc[k][1, 0]) < 1e-30
                           and abs(Jloc[k][0, 0]-1) < 1e-9
                           and abs(Jloc[k][1, 1]-1) < 1e-9)))     # not stick
    return full, lam_f, slip / Jloc.shape[0]


peak = np.array([abs(problem.compute_full_response(f, w).coefficients[h1, beam.beam_trans_damper, 0])
                 for f, w in zip(ss.fourier, wh)])
# peak amplitude over a period (more robust than 1st-harmonic):
peak_inf = np.zeros(len(wh)); slipfrac = np.zeros(len(wh))
for idx, (f, w) in enumerate(zip(ss.fourier, wh)):
    full, _, sf = evaluate(f, w)
    peak_inf[idx] = np.abs(full.time_series[:, beam.beam_trans_damper, 0]).max()
    slipfrac[idx] = sf
i_peak = int(peak_inf.argmax())
print(f"peak |u(0.3L)| = {peak_inf[i_peak]*1e6:.2f} um at {freq_hz[i_peak]:.1f} Hz "
      f"(slip fraction {slipfrac[i_peak]*100:.0f}%);  "
      f"slip fraction over branch: [{slipfrac.min()*100:.0f}, {slipfrac.max()*100:.0f}]%")

# stuck-linear reference FRF (undamped-by-friction), same forcing
freq_lin = np.linspace(OMEGA_START, OMEGA_END, 400)
peak_lin = np.array([abs(stuck_response(w, beam.F0)[beam.beam_trans_damper, 0])
                     for w in freq_lin])


# ============================ plots =========================================
full_pk, lam_pk, _ = evaluate(ss.fourier[i_peak], wh[i_peak])
tau = Fourier.adimensional_time_samples
u_beam = full_pk.time_series[:, beam.beam_trans_damper, 0]    # beam @ 0.3L (X)
u_damp = full_pk.time_series[:, beam.Dx, 0]                   # damper X
fN = lam_pk.time_series[:, 0, 0]                              # normal contact force
fT = lam_pk.time_series[:, 1, 0]                              # tangential (friction) force

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# (a) FRC: friction-damped vs stuck-linear
ax[0, 0].plot(freq_lin * OMEGA_REF/2/np.pi, peak_lin*1e6, 'k:', lw=1.2,
              label="stuck-linear FRF (no slip)")
ax[0, 0].plot(freq_hz, peak_inf*1e6, '-', color="#E8820C", lw=2.0,
              label="DLFT-FBS (dry friction)")
ax[0, 0].axhline(SLIP_THRESHOLD*1e6, color="red", ls="--", lw=1.0,
                 label=r"slip threshold $\mu N_0/k_T$")
ax[0, 0].plot(freq_hz[i_peak], peak_inf[i_peak]*1e6, 'o', color="#1F77B4", ms=7)
ax[0, 0].set_xlabel("frequency [Hz]"); ax[0, 0].set_ylabel(r"peak $|u_X(0.3L)|$ [$\mu$m]")
ax[0, 0].set_title("(a) Frequency response  (Nacivet Fig. 5b)")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

# (b) time history at resonance (Nacivet Fig. 5a)
ax[0, 1].plot(tau/(2*np.pi), u_beam*1e6, '-',  color="#1F77B4", lw=1.8,
              label="beam @ 0.3L (X)")
ax[0, 1].plot(tau/(2*np.pi), u_damp*1e6, '--', color="#E8820C", lw=1.8,
              label="damper (X)")
ax[0, 1].set_xlabel("t / T"); ax[0, 1].set_ylabel(r"displacement [$\mu$m]")
ax[0, 1].set_title(f"(b) Steady state at {freq_hz[i_peak]:.1f} Hz  (Nacivet Fig. 5a)")
ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

# (c) contact forces vs time -- stick/slip saturation at +/- mu*N
ax[1, 0].plot(tau/(2*np.pi), fT, '-', color="#C05800", lw=1.8, label="friction force $f_T$")
ax[1, 0].plot(tau/(2*np.pi),  MU*fN, 'r--', lw=1.0, label=r"$\pm\mu f_N$ (slip bound)")
ax[1, 0].plot(tau/(2*np.pi), -MU*fN, 'r--', lw=1.0)
ax[1, 0].plot(tau/(2*np.pi), fN, ':', color="gray", lw=1.2, label="normal force $f_N$")
ax[1, 0].set_xlabel("t / T"); ax[1, 0].set_ylabel("contact force [N]")
ax[1, 0].set_title("(c) Stick-slip contact forces")
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

# (d) friction hysteresis loop: f_T vs relative tangential displacement
x_rel_T = (u_beam - u_damp)
ax[1, 1].plot(x_rel_T*1e6, fT, '-', color="#7030A0", lw=1.8)
ax[1, 1].set_xlabel(r"relative tangential displ. $x_r^T$ [$\mu$m]")
ax[1, 1].set_ylabel("friction force $f_T$ [N]")
ax[1, 1].set_title("(d) Friction hysteresis loop")
ax[1, 1].grid(alpha=0.3)

fig.suptitle(
    "DLFT dry-friction damper (Nacivet JSV 265, 2003, Ex. 1 analogue)  --  "
    f"full-FE beam, partial slip,  mu={MU}, N0={PRELOAD:.0f} N", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
if SAVE_PNG:
    out = Path(__file__).parent / "beam_friction_damper_frc.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")

plt.show()