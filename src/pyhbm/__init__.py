__version__ = "2.1"

from .dynamical_system import (
    FirstOrderODE,
    SecondOrderODE,
    FBS_System,
)

from .frequency_domain import (
    Fourier,
    Fourier_Real,
    Fourier_Complex,
    FourierOmegaPoint,
    JacobianFourier,
    JacobianFourier_Real,
    JacobianFourier_Complex,
    FrequencyDomainFirstOrderODE,
    FrequencyDomainFirstOrderODE_Real,
    FrequencyDomainFirstOrderODE_Complex,
    FrequencyDomainFRF,
    FrequencyDomainFRF_experimental,
    FrequencyDomainFRF_numerical,
    FrequencyBasedSubstructuring,
    FrequencyBasedSubstructuring_numerical,
    FrequencyBasedSubstructuring_experimental,
)

from .frf_provider import (
    FRFProvider,
    NumericalFRF,
    ExperimentalFRF,
)

from .nonlinear_method import (
    NonlinearMethod,
    AFT,
    DLFTContact,
)

from .hbm_problems import (
    FRFProblem,
    FBSProblem,
)

from .numerical_continuation.corrector_step import (
    NewtonRaphson,
    CorrectorParameterization,
    OrthogonalParameterization,
)

from .numerical_continuation.predictor_step import (
    Predictor,
    TangentPredictorRobust,
    TangentPredictorOne,
    TangentPredictorTwo,
    StepLengthAdaptation,
    ExponentialAdaptation,
    BiExponentialAdaptation,
)

from .core import (
    SolutionSet,
    HarmonicBalanceMethod,
)

from .validation import (
    TimeDomainValidator,
    ValidationResult,
)

from .io import (
    plot_FRF,
    save_solution_set,
)

from .stability import (
    FloquetAnalyzer,
    StabilityReport,
    BifurcationDetector,
    SpecialPoint,
)

__all__ = [
    "__version__",
    # base systems
    "FirstOrderODE", "SecondOrderODE", "FBS_System",
    # fourier
    "Fourier", "Fourier_Real", "Fourier_Complex", "FourierOmegaPoint",
    "JacobianFourier", "JacobianFourier_Real", "JacobianFourier_Complex",
    # legacy frequency-domain classes (will be removed in cleanup step)
    "FrequencyDomainFirstOrderODE",
    "FrequencyDomainFirstOrderODE_Real",
    "FrequencyDomainFirstOrderODE_Complex",
    "FrequencyDomainFRF",
    "FrequencyDomainFRF_experimental",
    "FrequencyDomainFRF_numerical",
    "FrequencyBasedSubstructuring",
    "FrequencyBasedSubstructuring_numerical",
    "FrequencyBasedSubstructuring_experimental",
    # new restructured API
    "FRFProvider", "NumericalFRF", "ExperimentalFRF",
    "NonlinearMethod", "AFT", "DLFTContact",
    "FRFProblem", "FBSProblem",
    # numerical continuation
    "NewtonRaphson", "CorrectorParameterization", "OrthogonalParameterization",
    "Predictor", "TangentPredictorRobust", "TangentPredictorOne", "TangentPredictorTwo",
    "StepLengthAdaptation", "ExponentialAdaptation", "BiExponentialAdaptation",
    # core
    "SolutionSet", "HarmonicBalanceMethod",
    # IO / validation / stability
    "plot_FRF", "save_solution_set",
    "TimeDomainValidator", "ValidationResult",
    "FloquetAnalyzer", "StabilityReport", "BifurcationDetector", "SpecialPoint",
]