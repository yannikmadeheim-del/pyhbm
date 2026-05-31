"""Diagnostic: watch a single fixed-frequency DLFT Newton solve in the contact region.

Prints, per iteration: ||r||, step norm, number of active contact samples,
how many mask entries flipped vs previous iter, and conditioning of J.
Also reports magnitude of Z_r diagonal blocks vs epsilon across harmonics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from pyhbm import (
    FBS_System, Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact, HarmonicBalanceMethod,
)

class SDOFVibroImpact(FBS_System):
    is_real_valued = True
    def __init__(self, m=1.0, c=0.05, k=1.0, F0=0.02, poly_deg=52):
        self.mass_matrix      = np.array([[m]])
        self.damping_matrix   = np.array([[c]])
        self.stiffness_matrix = np.array([[k]])
        self.B_coupling       = np.array([[1.0]])
        self.total_dimension  = 1
        self.dimension        = 1
        self.polynomial_degree = poly_deg
        self.F0 = F0
    def external_term(self, tau):
        f = np.zeros((len(tau), self.total_dimension, 1))
        f[:, 0, 0] = self.F0 * np.cos(tau)
        return f
    def interface_force(self, u, ud, tau):       return np.zeros((len(tau), self.dimension, 1))
    def jacobian_interface_force(self, u, ud, tau):     return np.zeros((len(tau), self.dimension, self.dimension))
    def jacobian_interface_force_qdot(self, u, ud, tau):return np.zeros((len(tau), self.dimension, self.dimension))

PARAMS = dict(m=1.0, c=0.05, k=1.0, F0=0.02)
GAP = 0.1
HARMONICS = list(range(0, 10))

system = SDOFVibroImpact(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)
provider = NumericalFRF(system.mass_matrix, system.damping_matrix, system.stiffness_matrix)

def linear_guess(omega):
    Z = -omega**2 * system.mass_matrix + 1j*omega*system.damping_matrix + system.stiffness_matrix
    Q1 = np.linalg.solve(Z, np.array([PARAMS['F0']])).reshape(1,1)
    return FourierOmegaPoint.new_from_first_harmonic(Q1, omega=omega)

def contact_guess(omega, amp):
    Q1 = np.array([[amp + 0.0j]])
    return FourierOmegaPoint.new_from_first_harmonic(Q1, omega=omega)

def run_newton(problem, x0, max_iter=60, tol=1e-6, verbose=True):
    x = np.asarray(x0).copy()
    omega = x0.omega
    prev_mask = None
    for it in range(max_iter):
        fourier = Fourier.new_from_RI(x[:-1])
        xp = FourierOmegaPoint(fourier, omega)
        r = problem.compute_residue_RI(xp)
        rn = np.linalg.norm(r)
        mask = xp.contact_mask.ravel().copy() if xp.contact_mask is not None else None
        n_active = int(mask.sum()) if mask is not None else -1
        flips = int((mask != prev_mask).sum()) if (prev_mask is not None and mask is not None) else -1
        if verbose:
            J = problem.compute_jacobian_of_residue_RI(xp)
            cond = np.linalg.cond(J)
            print(f"  it {it:3d}  ||r||={rn:.3e}  active={n_active:4d}/{mask.size if mask is not None else 0}  flips={flips:4d}  cond(J)={cond:.2e}")
        if rn < tol:
            print(f"  CONVERGED in {it} iters")
            return True
        J = problem.compute_jacobian_of_residue_RI(xp)
        delta = np.linalg.solve(J, r)
        x[:-1] = x[:-1] - delta
        prev_mask = mask
    print(f"  FAILED after {max_iter} iters, final ||r||={rn:.3e}")
    return False

print("="*70)
print("Z_r diagonal magnitude vs epsilon across harmonics (omega=1.1)")
print("="*70)
omega = 1.1
x = linear_guess(omega)
problem = FBSProblem(system, provider, DLFTContact(epsilon=1.0, g_zero=GAP))
Yr = problem._get_BY(x) @ problem.B_fourier.T
n_int = 1
for k in [0,1,2,5,10,20,29]:
    s = slice(k*n_int,(k+1)*n_int)
    Zr_block = np.linalg.solve(Yr[s,s], np.eye(n_int))
    print(f"  harmonic n={HARMONICS[k]:2d}  |Z_r|={abs(Zr_block[0,0]):.3e}")

for eps in [0.1, 1.0, 10.0, 100.0]:
    print("="*70)
    print(f"Fixed-freq Newton at omega=1.2, eps={eps}, IN-CONTACT guess amp=0.3")
    print("="*70)
    problem = FBSProblem(system, provider, DLFTContact(epsilon=eps, g_zero=GAP))
    run_newton(problem, contact_guess(1.2, 0.3), max_iter=40)
