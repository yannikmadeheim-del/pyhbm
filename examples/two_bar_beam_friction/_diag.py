"""Is the DLFT Jacobian rank-deficient at a gross-slip point, and is the
analytic (history-dropped) tangent still consistent with the true gradient?

We evaluate at the AFT-regularized solution (a near-true gross-slip state) and
compare the analytic square Jacobian dr/dx_r to a finite-difference one, then
look at the singular-value spectrum (nullity / conditioning)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from numpy import zeros

from pyhbm import (Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem,
                   NumericalFRF, AFT, DLFTFriction, HarmonicBalanceMethod)
from pyhbm.numerical_continuation.corrector_step import ArcLengthParameterization
from pyhbm.numerical_continuation.predictor_step import TangentPredictorBordered
from systems import BarBeamParams, BarBeamFrictionDLFT, BarBeamFrictionAFT

HARM = list(range(0, 16)); POLY = 20
HarmonicBalanceMethod.update_dependencies(HARM, POLY)
h0 = HARM.index(0); h1 = HARM.index(1); Nh = Fourier.number_of_harmonics


def aft_solution_near(p, omega_target):
    """Cold AFT continuation; return the fourier/omega closest to omega_target."""
    sysA = BarBeamFrictionAFT(p)
    M, C, K = sysA.mass_matrix, sysA.damping_matrix, sysA.stiffness_matrix
    prob = FBSProblem(sysA, NumericalFRF(M, C, K), AFT())
    solver = HarmonicBalanceMethod(harmonics=HARM, freq_domain_ode=prob,
            corrector_parameterization=ArcLengthParameterization,
            predictor=TangentPredictorBordered)
    Z = -0.85**2*M + 1j*0.85*C + K
    F = zeros((6,1), complex); F[0,0]=p.P1
    xr0 = (sysA.B_coupling @ np.linalg.solve(Z, F))[:,0]
    Q = zeros((Nh,2,1), complex); Q[h0,0,0]=p.eps; Q[h1,0,0]=xr0[0]; Q[h1,1,0]=xr0[1]
    ig = FourierOmegaPoint(Fourier(Q), 0.85)
    rd = FourierOmegaPoint.new_from_first_harmonic(zeros((2,1),complex), 1.0)
    ss = solver.solve_and_continue(initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=3000, angular_frequency_range=[0.85,1.09],
        solver_kwargs={"maximum_iterations":200,"absolute_tolerance":1e-7},
        step_length_adaptation_kwargs={"base":3.0,"initial_step_length":0.01,
            "maximum_step_length":0.02,"minimum_step_length":1e-8,
            "goal_number_of_iterations":4}, jacobian_update_frequency=1, verbose=False)
    wh=np.array(ss.omega); i=int(np.argmin(np.abs(wh-omega_target)))
    return ss.fourier[i], wh[i]


def analyze(P5, omega_target, eps=1e4):
    p = BarBeamParams(P5=P5, poly_deg=POLY)
    four, w = aft_solution_near(p, omega_target)
    sysD = BarBeamFrictionDLFT(p)
    M,C,K = sysD.mass_matrix, sysD.damping_matrix, sysD.stiffness_matrix
    prob = FBSProblem(sysD, NumericalFRF(M,C,K),
        DLFTFriction(epsilon_N=eps, epsilon_T=eps, mu=p.mu, g_zero=p.eps,
                     n_tangential=1, n_sweep=1))
    x = FourierOmegaPoint(four, w)

    # slip fraction at this state
    prob.method._get_lambda_corrected(x); J = x.contact_mask; Nt=J.shape[0]
    stick = sum(abs(J[k,1,1]-1)<1e-9 for k in range(Nt))
    sep = sum(np.all(np.abs(J[k])<1e-30) for k in range(Nt))
    slipf = 1.0 - (stick+sep)/Nt

    # analytic square Jacobian dr/dx_r (drop omega column)... compute_jacobian gives full square
    Ja = prob.compute_jacobian_of_residue_RI(x)            # (2N, 2N) RI square
    # FD
    r0 = prob.compute_residue_RI(x); n=r0.shape[0]; h=1e-8
    Jf = zeros((n,n)); xa=np.asarray(x).copy()
    for j in range(n):
        xp=xa.copy(); xp[j,0]+=h
        Jf[:,j]=(prob.compute_residue_RI(FourierOmegaPoint(Fourier.new_from_RI(xp[:-1]), x.omega))-r0).ravel()/h
    rel = np.abs(Ja-Jf).max()/(np.abs(Jf).max()+1e-30)
    sva = np.linalg.svd(Ja, compute_uv=False)
    svf = np.linalg.svd(Jf, compute_uv=False)
    print(f"--- P5={P5}, w~={w:.3f}, slip frac={slipf:.2f}, eps={eps:.0e} ---")
    print(f"  analytic vs FD Jacobian rel err : {rel:.3e}")
    print(f"  cond(J_an)={sva[0]/sva[-1]:.2e}  smallest 4 SV (an): {sva[-4:]}")
    print(f"  cond(J_fd)={svf[0]/svf[-1]:.2e}  smallest 4 SV (fd): {svf[-4:]}")


analyze(0.4, 0.99)     # thesis gross slip, near peak
analyze(2.0, 0.99)     # partial-slip variant near peak
analyze(0.4, 0.86)     # thesis gross slip, low freq
