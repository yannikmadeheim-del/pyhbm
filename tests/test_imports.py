import pytest
import numpy as np


class TestPackageImports:
    """Test that all main classes can be imported."""

    def test_import_version(self):
        import pyhbm
        assert pyhbm.__version__ == "2.1"

    def test_import_base_classes(self):
        from pyhbm import FirstOrderODE, SecondOrderODE, FBS_System
        assert FirstOrderODE is not None
        assert SecondOrderODE is not None
        assert FBS_System is not None

    def test_import_fourier_classes(self):
        from pyhbm import (
            Fourier,
            Fourier_Real,
            Fourier_Complex,
            FourierOmegaPoint,
        )
        assert Fourier is not None
        assert Fourier_Real is not None
        assert Fourier_Complex is not None
        assert FourierOmegaPoint is not None

    def test_import_jacobian_classes(self):
        from pyhbm import (
            JacobianFourier,
            JacobianFourier_Real,
            JacobianFourier_Complex,
        )
        assert JacobianFourier is not None
        assert JacobianFourier_Real is not None
        assert JacobianFourier_Complex is not None

    def test_import_frequency_domain_classes(self):
        from pyhbm import (
            FrequencyDomainFirstOrderODE,
            FrequencyDomainFirstOrderODE_Real,
            FrequencyDomainFirstOrderODE_Complex,
        )
        assert FrequencyDomainFirstOrderODE is not None
        assert FrequencyDomainFirstOrderODE_Real is not None
        assert FrequencyDomainFirstOrderODE_Complex is not None

    def test_import_new_api(self):
        from pyhbm import (
            FRFProvider, NumericalFRF, ExperimentalFRF,
            NonlinearMethod, AFT, DLFTContact,
            FRFProblem, FBSProblem,
        )
        assert FRFProvider is not None
        assert NumericalFRF is not None
        assert ExperimentalFRF is not None
        assert NonlinearMethod is not None
        assert AFT is not None
        assert DLFTContact is not None
        assert FRFProblem is not None
        assert FBSProblem is not None

    def test_import_numerical_continuation_classes(self):
        from pyhbm import (
            NewtonRaphson,
            Predictor,
            TangentPredictorOne,
            OrthogonalParameterization,
            ExponentialAdaptation,
        )
        assert NewtonRaphson is not None
        assert Predictor is not None
        assert TangentPredictorOne is not None
        assert OrthogonalParameterization is not None
        assert ExponentialAdaptation is not None

    def test_import_core_classes(self):
        from pyhbm import SolutionSet, HarmonicBalanceMethod
        assert SolutionSet is not None
        assert HarmonicBalanceMethod is not None

    def test_import_all(self):
        import pyhbm
        assert hasattr(pyhbm, "__version__")
        assert hasattr(pyhbm, "FirstOrderODE")
        assert hasattr(pyhbm, "Fourier")
        assert hasattr(pyhbm, "HarmonicBalanceMethod")


class TestFirstOrderODEBase:
    """Test the FirstOrderODE base class."""

    def test_base_class_has_default_values(self):
        from pyhbm import FirstOrderODE

        ode = FirstOrderODE()
        assert ode.dimension == 2
        assert ode.polynomial_degree == 1
        assert ode.is_real_valued is True
        assert ode.linear_coefficient is not None


class TestFourierBasics:
    """Test basic Fourier class functionality."""

    def test_fourier_class_variables(self):
        from pyhbm import Fourier

        assert hasattr(Fourier, "harmonics")
        assert hasattr(Fourier, "polynomial_degree")
        assert hasattr(Fourier, "number_of_harmonics")
        assert hasattr(Fourier, "number_of_time_samples")

    def test_fourier_zeros(self):
        from pyhbm import Fourier

        zeros = Fourier.zeros(dimension=2)
        assert zeros.coefficients is not None
        assert zeros.coefficients.shape[1] == 2