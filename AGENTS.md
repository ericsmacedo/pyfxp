# AGENTS.md

Guidance for AI coding agents (and humans) working on this repository.

## What this is

pyfxp is a bit-true fixed-point arithmetic library. Values stay plain floats and NumPy
arrays; `fxp(x, Q(qi, qf, signed, rnd, ovf))` quantizes them. The hot paths are compiled
with Numba and must also work inside user `@njit` code.

## Layout

| Path | Contents |
|---|---|
| `src/pyfxp/_pyfxp.py` | `FxpSpec`, `Q`, `fxp`, `fxpt`, and **every compiled kernel** (`_quantize_int` is the single kernel behind scalars, arrays and raw integers) |
| `src/pyfxp/_raw.py` | Python-level `to_int` / `from_int` / `to_bin` / `to_hex` / `from_bin` / `from_hex` |
| `src/pyfxp/constants.py` | `Rounding` and `Overflow` IntEnums and the module-level mode names |
| `tests/` | unit and property tests; `test_properties.py` has the exact `Fraction` reference model (`ref_fxp`) |
| `benchmarks/` | `bench_pyfxp.py` (`inv bench`) and `compare_libraries.py` (other libraries, `bench` group) |
| `docs/` | Sphinx + Furo + MyST; `docs/conf.py` also writes `llms.txt` / `llms-full.txt` |

## Commands

```bash
uv run inv all                                   # pre-commit, tests (JIT on and off), ty, docs: run before every PR
uv run --frozen pytest -q                        # tests with Numba
NUMBA_DISABLE_JIT=1 uv run --frozen pytest -q    # the same tests as plain Python
uv run inv doc                                   # strict Sphinx build (warnings are errors)
uv run inv bench                                 # pyfxp benchmarks
uv run --group bench python benchmarks/compare_libraries.py   # comparison with other libraries
```

## Rules that are easy to get wrong

- **Code runs twice: compiled and as plain Python.** `inv test` repeats the suite with
  `NUMBA_DISABLE_JIT=1`, and pytest turns warnings into errors, so NumPy overflow /
  invalid-value warnings in the pure-Python path fail the build.
- **Keep compiled kernels in `_pyfxp.py`.** `cache=True` only notices changes to the file a
  function is defined in, not to functions it inlines from other files. If results look
  stale after an edit, delete `src/pyfxp/__pycache__`.
- **Pass plain ints, not IntEnum members, from Python into compiled functions.** Numba types
  IntEnum arguments about 10x slower per call (`_validated_modes` does the conversion).
- **No f-strings in exceptions inside kernels.** Formatting costs about 2 ns per call even
  when nothing is raised; put the value into the message at the Python level instead.
- **Numerical changes need the reference model.** Compare against `ref_fxp` in
  `tests/test_properties.py` (exact rational arithmetic) with Hypothesis, with and without JIT.
- **Performance changes need numbers.** Run `inv bench` before and after; the shared CI
  machines are noisy, so use several repeats.
- **Docstrings are NumPy style** and their examples run as doctests (`testpaths` includes `src`).
- **License headers** are added by pre-commit; copy one from an existing file for new `.py` files.

## Conventions

- One change per pull request; link the issue it fixes.
- Tick the matching item in `ROADMAP.md` when it lands.
- Public API: `pyfxp.__all__` and `pyfxp.constants`. Everything else (`_pyfxp`, `_raw`) is private. Results, mode codes and exception types are covered by SemVer (`docs/api/stability.md`): changing them needs a major release.
