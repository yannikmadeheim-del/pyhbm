# Step 1 — Extract `fourier.py`

Part of the "restructure first" plan for adding DLFT contact to pyhbm (Phase 1).
This file is the precise change list for Step 1 plus a refresher on the Toeplitz HBM
Jacobian math that lives in the moved code. Agreed 2026-05-25.

> Workflow: discuss before editing; ask permission before any file change. The user
> implements; this doc is the guide.

---

## A. Environment facts (verified)

- **Python 3.14.4** in `.venv`. 3.14 evaluates annotations lazily (PEP 649), which is why
  the unquoted forward reference `x: FourierOmegaPoint` in the `Fourier` base class does
  not crash even though `FourierOmegaPoint` is defined later. On Python <= 3.13 that line
  would raise `NameError`; quote it (`x: "FourierOmegaPoint"`) only if portability matters.
- **`pyhbm` is an editable install from `src/`** (`pyhbm.__file__` -> `.../code/pyhbm/src/pyhbm/__init__.py`).
  Editing `src/pyhbm/*.py` takes effect immediately — no reinstall.
- **All `Literatur` PDFs are password-protected** — cannot be opened by tools. Use the
  readable local theory doc `docs/fbs_dlft_admittance.tex` and the section pointers in §D.

---

## B. The change list (exhaustive)

Move these 7 classes out of `frequency_domain.py` into a new `src/pyhbm/fourier.py`,
**keeping this exact order**:

| Class                     | Lines today |
|---------------------------|-------------|
| `Fourier`                 | 10–103      |
| `Fourier_Real`            | 105–123     |
| `Fourier_Complex`         | 125–141     |
| `FourierOmegaPoint`       | 145–216     |
| `JacobianFourier`         | 220–248     |
| `JacobianFourier_Real`    | 250–266     |
| `JacobianFourier_Complex` | 268–280     |

Everything from line 285 (`FrequencyDomainFirstOrderODE`) onward **stays**.

**`fourier.py` header** — copy the numpy imports verbatim (a couple end up unused;
harmless for a pure move):

```python
import numpy as np
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like, vdot, sqrt
from numpy.fft import rfft, irfft, fft, ifft
```

`fourier.py` depends on **numpy only** — no `dynamical_system`, no other pyhbm module —
so there is **no circular import**.

**`frequency_domain.py` edit** — replace the deleted block with a re-export, placed after
the numpy imports and **before** the first class that uses these names (i.e. above the old
line 285):

```python
from .fourier import (
    Fourier, Fourier_Real, Fourier_Complex,
    FourierOmegaPoint,
    JacobianFourier, JacobianFourier_Real, JacobianFourier_Complex,
)
```

**Why nothing else needs editing** — every consumer imports these names *from
`frequency_domain`*, and the re-export keeps that path alive (grepped the whole repo):

| File                                             | Import                                                | After re-export |
|--------------------------------------------------|-------------------------------------------------------|-----------------|
| `src/pyhbm/__init__.py:9`                         | `from .frequency_domain import (Fourier, ...)`        | works           |
| `src/pyhbm/core.py:9`                             | `from .frequency_domain import *`                     | works           |
| `src/pyhbm/io/save.py:7`                          | `from ..frequency_domain import Fourier`              | works           |
| `src/pyhbm/io/plotting.py:6`                      | `from ..frequency_domain import Fourier`              | works           |
| `src/pyhbm/stability/bifurcation_detection.py:253`| `from ..frequency_domain import Fourier`             | works           |
| `examples/fbs_dnft_unilateral_spring/main.py:9`  | `from pyhbm.frequency_domain import ..., Fourier_Real, FourierOmegaPoint` | works |

So Step 1 = **one new file + one re-export line + delete the moved block.** Zero downstream
edits. (A later step may re-point `__init__.py` to `.fourier` directly; optional.)

### Verification (pure move => bit-identical)

1. `& ".\.venv\Scripts\python.exe" -c "import pyhbm; print('ok', pyhbm.__version__)"`
2. `& ".\.venv\Scripts\python.exe" -m pytest tests/ -k "Fourier" -q`
   (Fourier tests pass; the stale `DynamicalSystem` tests are unrelated.)
3. Numeric-equality probe — run BEFORE the move to get a baseline, then AFTER; the printed
   scalar must match exactly. (Wire it to `FBS_test_linear` / `duffing_FBS_2DoF`: build the
   ode, `solve_fixed_frequency` at one omega, print `np.linalg.norm(np.asarray(solution))`.)

Estimate: ~1.5–2 h to move + verify; ~3–4 h with the math read below.

---

## C. Math refresher — the Toeplitz HBM Jacobian (the subtle part of `fourier.py`)

The machinery DLFT reuses later. Code references are to `frequency_domain.py` before the move.

**1. Real Fourier ansatz + rfft convention.** Real T-periodic `q(tau)=sum_n qhat_n e^{i n tau}`
with `qhat_{-n}=conj(qhat_n)`. `Fourier_Real` stores the unnormalized `numpy.rfft` bins for
`n>=0` (`:110`); `irfft(..., n=Nt)` inverts with `1/Nt` (`:122`). Physical amplitude =
`|qhat_n| * 2/Nt` (the 2 is the one-sided rfft convention, c_0=1, c_{n>0}=2).

**2. Force Jacobian is block-Toeplitz.** Let `g(tau_i)=df_nl/dq` sampled in time, and
`Ghat_k = (1/Nt) sum_i g(tau_i) e^{-i k tau_i}`. Treating `qhat_m` as complex:

    dFhat_n / dqhat_m = (1/Nt) sum_i g(tau_i) e^{i m tau_i} e^{-i n tau_i} = Ghat_{n-m}

Depends only on n-m => block-Toeplitz. This is the task's dF_n/dQ_m = G_{n-m}.

**3. Real representation adds a Hankel term.** Newton unknowns are the real/imag parts
`qhat_m = a_m + i b_m`. Since `q(tau_i) = qhat_0 + sum_{m>0} 2(a_m cos m tau_i - b_m sin m tau_i)`:

    dFhat_n / da_m = Ghat_{n-m} + Ghat_{n+m}
    dFhat_n / db_m = i ( Ghat_{n-m} - Ghat_{n+m} )

The `Ghat_{n+m}` piece is Hankel (depends on n+m). Exactly the code:

```python
state      = G[harmonics_state]        # Ghat_{n-m}   (harmonics_state      = n - m)   :261
state_conj = G[harmonics_state_conj]   # Ghat_{n+m}   (harmonics_state_conj = n + m)   :262
state_real = (state + state_conj)/Nt   # dFhat_n/da_m
state_imag = (state - state_conj)/Nt   # dFhat_n/db_m  (up to the i)
JacobianFourier_Real(RR=state_real.real, RI=-state_imag.imag,
                     IR=state_real.imag, II= state_imag.real)   # :266
```

With `Fhat_n = F_n^R + i F_n^I`: RR=dF^R/da, IR=dF^I/da, RI=dF^R/db, II=dF^I/db — matching
the four lines. The `/Nt` is the DFT normalization from step 2.

**4. Velocity dependence -> i*omega*m column scaling.** For `f_nl(q, qdot)` with
`qdot = omega q'`, chain rule adds `dFhat_n/dqhat_m += i m omega * Gdothat_{n-m}`. In code
(`:530–536`) that is `Gdot` plus `col_scale = omega * kron(diag(harmonics), I)`, with the
RR/RI swap encoding multiplication by i. This is the full
dF_n/dQ_m = G_{n-m} + i omega m * Gdot_{n-m}.

**5. Complex case** (`JacobianFourier_Complex` `:277`) keeps only the n-m Toeplitz term =>
the holomorphic block [[Re,-Im],[Im,Re]].

**Why this matters for DLFT later:** the contact Jacobian reuses this exact
`JacobianFourier_Real` path, but the per-sample seed is the contact MASK m_i instead of
df/dq. Because the mask sits in the time domain between Gamma and Gamma+, the result is
real-linear but NOT holomorphic — which is why it must go through `JacobianFourier_Real`
(Hankel term included) and NOT `FRF_to_RI`. See `docs/fbs_dlft_admittance.tex` §4.

---

## D. Reading list (PDFs are locked -> stable section/equation pointers)

Best readable anchor: `docs/fbs_dlft_admittance.tex`.

| For step      | Topic                                                          | Where                                                                 |
|---------------|----------------------------------------------------------------|-----------------------------------------------------------------------|
| 1 (fourier)   | Fourier series for HBM, DFT/IDFT operators, rfft convention    | local `.tex` §1 (Eqs. ansatz, dft, dftentries); Krack & Gross Ch. 2   |
| 1 (fourier)   | Toeplitz analytical Jacobian dF_n/dQ_m                         | Krack & Gross Appendix A; `HarmonicBalanceDerivatives.pdf` (locked)    |
| throughout    | complex differentials, Wirtinger, [Re,-Im;Im,Re] real form    | Hjorungnes (2011) Ch. 2–3                                             |
| 2 (frf)       | dynamic stiffness Z_n, admittance Y_n=Z_n^-1, dY/domega        | local `.tex` §1 (Eq. Zn); Krack Ch. 2                                 |
| 3 (AFT)       | AFT scheme, smooth-force Jacobian                              | Krack Ch. 2 (AFT)                                                     |
| 4 (FBS)       | LM-FBS dual assembly, Y_r=B Y B^T, f_adm                       | local `.tex` §2; van der Seijs Ch. 3 (Eqs. 3.20–3.28, esp. 3.25); Soleimani Eq. 24–26 |
| 5 (DLFT)      | prediction/correction, non-holomorphic mask Jacobian          | local `.tex` §3–4; Vadcard §2.4, Eqs. 38–45                           |
| later         | continuation, parametrizations                                 | Krack §4.5, Fig. 4.6                                                  |

For Step 1 specifically: read `docs/fbs_dlft_admittance.tex` §1 + Krack Ch. 2 & Appendix A,
then re-read §C above against the code.
