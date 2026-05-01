# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working style

- Do not change anything without explicit permission.
- Keep changes as small as possible.
- Current goal: extend the library to support **2nd-order ODEs** as the system formulation (in addition to the existing 1st-order formulation).

## Commands

```bash
# Install in editable/dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_imports.py::TestFourierBasics::test_fourier_zeros -v

# Run tests with coverage
pytest tests/ --cov=pyhbm --cov-report=term-missing
```

## Architecture

**pyhbm** is a Python library for solving nonlinear dynamical systems via the **Harmonic Balance Method (HBM)** with numerical continuation. Source lives in `src/pyhbm/`.

### Core data flow

1. **User defines a system** by subclassing `FirstOrderODE` (`dynamical_system.py`), implementing `external_term`, `linear_term`, `nonlinear_term`, and `jacobian_nonlinear_term`.
2. **`HarmonicBalanceMethod`** (`core.py`) wraps the system in a `FrequencyDomainFirstOrderODE` (from `frequency_domain.py`), which transforms everything into Fourier coefficient space.
3. **`solve_and_continue()`** runs a predictor–corrector loop:
   - **Corrector** (`numerical_continuation/corrector_step.py`): `NewtonRaphson` with `OrthogonalParameterization` (arc-length / pseudo-arclength constraint).
   - **Predictor** (`numerical_continuation/predictor_step.py`): `TangentPredictorRobust` (adaptive, handles autonomous systems); `StepLengthAdaptation` subclasses control step sizing.
4. Results are collected in a **`SolutionSet`** which holds arrays of `FourierOmegaPoint` objects (Fourier coefficients + frequency).

### Frequency domain module (`frequency_domain.py`)

Key classes:
- `Fourier` / `Fourier_Real` / `Fourier_Complex` — containers for harmonic coefficients.
- `FourierOmegaPoint` — a `Fourier` instance paired with angular frequency ω.
- `FrequencyDomainFirstOrderODE_Real` / `_Complex` — evaluate the HBM residual and its Jacobian for the Newton solver.

### Post-processing

| Module | Class | Purpose |
|--------|-------|---------|
| `stability/stability_analysis.py` | `FloquetAnalyzer` | Floquet exponents → stability per solution point |
| `stability/bifurcation_detection.py` | `BifurcationDetector` | Detect bifurcations from stability sign changes |
| `validation/time_domain.py` | `TimeDomainValidator` | Cross-check HBM solutions via RK45 integration |
| `io/plotting.py` | `plot_FRF` | Frequency-response curve plots |
| `io/save.py` | `save_solution_set` | Persist solution families to disk |

### Known test failures

Three tests in `tests/test_imports.py` fail because they reference `DynamicalSystem`, which does not exist. Users subclass `FirstOrderODE` instead. Do not add a `DynamicalSystem` alias without checking whether the design intent has changed.

### Examples

`examples/` contains six worked problems. The primary reference example is `examples/duffing_forced_nonautonomous/main.py` (forced Duffing oscillator).
