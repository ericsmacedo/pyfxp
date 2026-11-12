# pyfxp Roadmap

Tracked improvements for pyfxp. Tick an item and link its PR when it lands.

## Correctness

- [x] Support formats up to 64 bits signed / 63 bits unsigned and reject wider ones (`qi + qf`); they used to silently return wrong values, e.g. `fxpt(1.0, 40, 30)` → `-5.96e-08`.
- [x] Inputs with `|x| * 2**qf >= 2**63` gave wrong results in any format (#18), e.g. `fxpt(1e19, 8, 4, ovf=SAT)` → `-128.0` instead of `127.9375`.
- [x] HALF_* rounding was wrong for scaled values >= 2**52 (#24).
- [x] Validate the format width: `qi + qf` must be 1–64 (signed) / 1–63 (unsigned) (#15). Negative `qi` or `qf` alone stay allowed on purpose: `sQ-2.10` is a fraction-only format and `sQ4.-2` a coarse one with `lsb = 4`.
- [x] Support arrays of any shape and layout (#26).
- [x] NaN propagates; ±inf saturates with SAT, raises with ERROR, gives NaN with WRAP (#27, #28).
- [x] 64-bit signed formats: huge inputs saturated to `2**63 − 1024` and passed ERROR silently (found while adding raw integers, #49).
- [x] Accept NumPy scalars (`np.integer`, `np.floating`) and array-likes (lists, tuples) as input (#19).
- [ ] Make overflow `ERROR` messages useful: include the value, the format and the valid range (today: `OverflowError: Overflow!`).

## Performance

Baseline numbers come from `uv run inv bench`; the comparison with other libraries from `uv run --group bench python benchmarks/compare_libraries.py`.

- [x] Speed up the per-sample path inside `@njit` code: 15–18 → ~5 ns/sample (#30). A hand-written single-mode loop is 0.7 ns; the rest is the run-time method dispatch.
- [x] Find out why `WRAP` costs ~7 ns/sample more than `SAT` in the scalar path (fixed by the new mask computation).
- [x] Convert large arrays in a single loop instead of chained NumPy temporaries: 10–40 → ~5 ns/elem at 1M elements (#31).
- [x] Remove the redundant `HALF_EVEN` special case (`np.round` already rounds ties to even) (#32, #33).
- [x] Merge the duplicated scalar and array implementations (#31).
- [ ] Vectorize the array loop (fast pass without checks, then fix up rare out-of-range / NaN elements) to approach 1 ns/elem.

## API / user experience

- [x] Spec arithmetic helpers for bit growth: `a.add(b)`, `a.sub(b)`, `a.mul(b)`, `a.neg()`, `a.acc(n)` (#41).
- [x] `FxpSpec` helpers: `min`, `max`, `min_int`, `max_int`, `lsb`, `width`, and a readable repr such as `sQ3.13(HALF_EVEN, SAT)` (#40).
- [ ] Format check `check(x, spec)` that raises if `x` isn't exactly representable in `spec`, behind a global switch that costs nothing when off. Catches forgotten quantization.
- [x] Raw integer conversion `to_int(x, spec)` / `from_int(n, spec)`, plus `to_bin` / `to_hex` / `from_bin` / `from_hex` for HDL test vectors (#43).
- [x] Rounding and overflow modes as `IntEnum`s: readable when printed, still usable inside Numba (#42).
- [ ] Complex number support (quantize real and imaginary parts separately).
- [ ] Optional thin Python wrapper type that tracks its spec, for interactive exploration only. The core stays on plain floats (#44).

## Documentation

- [x] Fix the broken docstring examples (`Q(3, 1)` labelled "Q3.5", stray `>>> 3.14159…` lines) and run them as doctests (#20).
- [x] Fix the `fxpt` "Returns" section (it returns floats, not integers) (#36).
- [x] Move the docs from MkDocs to Sphinx + Furo, Markdown via MyST (#50).
- [x] Front page for engineers and a comparison with other fixed-point libraries (#51).
- [x] AI-friendly docs and repo: `llms.txt` / `llms-full.txt` with the full API reference, and `AGENTS.md` for coding agents (#53).
- [ ] Use one docstring style throughout: the docstrings and Sphinx (napoleon) use NumPy style, but `[tool.ruff.lint.pydocstyle]` says `google` (not enforced today, as no `D` rules are selected).
- [x] Fix the `pyproject.toml` description typo ("library..") (#34).
- [ ] Tell users to call pyfxp on arrays or inside `@njit` code: a pure-Python per-sample loop is ~200× slower (the front page and comparison in #51 cover it; the user guide should too).

## Testing / CI

- [x] Fix `gen_ref_data.py`, stop skipping the first reference case, seed RNGs (#10).
- [x] Pin Python in the coveralls finish job (#13).
- [x] Property-based tests against an exact reference model (#11).
- [x] Benchmarks with `inv bench` (#12).
- [x] Add tests for wide formats.
- [x] Fix the empty `COVERALLS_FLAG_NAME`: each upload now gets a distinct flag name (#35).
- [x] Docstring examples run as doctests (#20).

## 1.0 release

What still separates the package from a stable 1.0 (checked against `main`):

- [ ] Write a `CHANGELOG.md`. There is none, and 0.2.1 → 1.0 has user-visible changes: modes are `IntEnum`s (#42), the `FxpSpec` repr is now `sQ3.13(...)` (#40), `fxp` / `Q` are plain functions with Numba overloads (#29, #42), formats wider than 64 bits now raise instead of returning wrong values (#15), NaN / inf behavior (#28).
- [ ] Define the public API: everything in `pyfxp.__all__` and `pyfxp.constants`; `pyfxp._pyfxp` / `pyfxp._raw` are private. State the SemVer promise (the README has a SemVer badge) in the docs.
- [x] Ship a `py.typed` marker, so type checkers use the annotations (#53).
- [x] Package metadata: `Intended Audience`, `Topic` and `Typing :: Typed` classifiers, and `keywords` (#53).
- [x] Set `Development Status :: 5 - Production/Stable`.
- [ ] Python 3.14: add it to the CI matrix and classifiers once the minimum supported Numba version supports it.
- [ ] Docs URLs: `pyproject.toml` points to `/en/stable/`, the README to `/en/latest/`. Pick one, and check the first Read the Docs build with the new Sphinx config (#50).
- [ ] Release dry run: build the sdist and wheel, and check the README on PyPI's renderer (GitHub alerts like `> [!NOTE]` show as plain quotes there).
- [ ] Close or defer the remaining open items above, so 1.0 doesn't ship with a known-wrong error message.
