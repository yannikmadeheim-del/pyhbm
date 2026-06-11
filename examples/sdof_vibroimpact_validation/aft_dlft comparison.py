from pyhbm import FBS_System


class dlft_system(FBS_System):
    """
    SDOF mass-spring-damper-wall in FBS form.

        m q'' + c q' + k q = F0 cos(tau)
        subject to  q <= g0  (rigid wall, enforced by DLFTContact)
    """
    is_real_valued = True

    def __init__(self, m=1.0, c=0.01, k=1.0, F0=0.02, poly_deg=30, k_rel=20):
        self.mass_matrix       = np.array([[m, 0],[0, 0]])
        self.damping_matrix    = np.array([[c, 0],[0, 0]])
        self.stiffness_matrix  = np.array([[k, 0],[0, k_rel*k]])
        self.B_coupling        = np.array([[1.0, -1.0]])      # 1 interface DOF
        self.total_dimension   = 2
        self.dimension         = 1                       # n_int
        self.polynomial_degree = poly_deg
        self.F0 = F0

    def external_term(self, tau):
        f = np.zeros((len(tau), self.total_dimension, 1))
        f[:, 0, 0] = self.F0 * np.cos(tau)
        return f

    # DLFT computes contact internally; stubs kept so AFT could be plugged in.
    def interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, 1))
    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))


class aft_system(FBS_System):
    is_real_valued = True

    def __init__(self, m=1.0, c=0.01, k=1.0, F0=0.02, poly_deg=30, k_rel=20, alpha=1):
        self.mass_matrix = np.array([[m]])
        self.damping_matrix = np.array([[c]])
        self.stiffness_matrix = np.array([[k]])
        self.B_coupling = np.array([[1.0]])  # 1 interface DOF
        self.total_dimension = 1
        self.dimension = 1  # n_int
        self.polynomial_degree = poly_deg
        self.F0 = F0
        self.g0 = g_zero

    @staticmethod
    def _regularization_tanh(d, alpha, k):
        """tanh-regularized ramp:  f(d) = k * 0.5*d*(1+tanh(alpha*d)) -> k*max(0,d)."""
        if not np.isfinite(alpha):
            r = np.where(d > 0.0, k * d, 0.0)
            dr = np.where(d > 0.0, k, 0.0)
            return r, dr
        t = np.tanh(alpha * d)
        H = 0.5 * (1.0 + t)
        r = k * (d * H)
        dr = k * (H + 0.5 * alpha * d * (1.0 - t * t))
        return r, dr

    @staticmethod
    def _regularization_ln(d, alpha, k):
        """softplus-regularized ramp:  f(d) = k * ln(1+exp(alpha*d))/alpha -> k*max(0,d)."""
        if not np.isfinite(alpha):
            r = np.where(d > 0.0, k * d, 0.0)
            dr = np.where(d > 0.0, k, 0.0)
            return r, dr
        r = k * np.logaddexp(0.0, alpha * d) / alpha  # = k*ln(1+exp(alpha*d))/alpha, overflow-safe
        dr = k * (1.0 / (1.0 + np.exp(-alpha * d)))  # = k*sigmoid(alpha*d)
        return r, dr


    def external_term(self, tau):
        f = np.zeros((len(tau), self.total_dimension, 1))
        f[:, 0, 0] = self.F0 * np.cos(tau)
        return f

    # DLFT computes contact internally; stubs kept so AFT could be plugged in.
    def interface_force(self, u_rel, udot_rel, tau):
        g =
        f = max(0, )
        return np.zeros((len(tau), self.dimension, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))




# ============================ parameters ====================================
EPSILON          = 1.0   # first entry; FD check uses this
PARAMS = dict(m=1.0, c=0.05, k=1.0, F0=0.02, poly_deg=31, k_rel=100)
GAP       = 0.1
HARMONICS = list(range(0, 21))

OMEGA_START = 0.5
OMEGA_END   = 2.5


# ============================ build problem (NEW API) ======================
AFT_system = aft_system(**PARAMS)
DLFT_system = dlft_system(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, AFT_system.polynomial_degree)
HarmonicBalanceMethod.update_dependencies(HARMONICS, DLFT_system.polynomial_degree)

provider = NumericalFRF(AFT_system.mass_matrix, AFT_system.damping_matrix, AFT_system.stiffness_matrix)
DLFT_contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)

AFT_problem  = FBSProblem(AFT_system, provider, AFT())
DLFT_problem = FBSProblem(DLFT_system, provider, DLFT_contact)
