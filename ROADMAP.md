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
- [x] Make overflow `ERROR` messages useful: the value, its index, the format and the valid range (`5.0 is out of range: the range of sQ3.4(TRUNC, ERROR) is [-4.0, 3.9375]`). Compiled code keeps a constant message.

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
- [x] Use one docstring style throughout: the docstrings and Sphinx (napoleon) use NumPy style, but `[tool.ruff.lint.pydocstyle]` says `google` (not enforced today, as no `D` rules are selected). Fixed in #54.
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

- [x] Write a `CHANGELOG.md` (Keep a Changelog format; also in the docs).
- [x] Define the public API: everything in `pyfxp.__all__` and `pyfxp.constants`; `pyfxp._pyfxp` / `pyfxp._raw` are private. State the SemVer promise (the README has a SemVer badge) in the docs. See `docs/api/stability.md`.
- [x] Ship a `py.typed` marker, so type checkers use the annotations (#53).
- [x] Package metadata: `Intended Audience`, `Topic` and `Typing :: Typed` classifiers, and `keywords` (#53).
- [x] Set `Development Status :: 5 - Production/Stable`.
- [x] Docs URLs: everything (README, `pyproject.toml`, llms.txt) points to `/en/stable/`, the docs of the latest release. The Read the Docs `latest` build of the new Sphinx config works (Furo, `dirhtml` URLs, `llms.txt`); `stable` switches to it with the 1.0 tag.
- [x] Overflow `ERROR` messages name the value, the format and the range (#56).
- [ ] README as a front page like the docs, with the comparison (#60).
- [ ] User guide: call pyfxp on arrays or inside `@njit` code (see Documentation).
- [ ] Release dry run: build the sdist and wheel, and check the README on PyPI's renderer (GitHub alerts like `> [!NOTE]` show as plain quotes there; the chart shows only its light version).

### Release steps

- [ ] Replace `Unreleased` in `CHANGELOG.md` with the release date.
- [ ] Tag `v1.0.0` on `main` and publish to PyPI.
- [ ] Check that Read the Docs builds `stable` from the tag with the Sphinx docs, so the `/en/stable/` links work (`llms.txt`, `comparison/`, `api/stability/` return 404 until then).
- [ ] Check the PyPI page: README, classifiers, project links.

### Deferred to after 1.0

New features are minor releases under SemVer, so these don't block 1.0:

- Vectorize the array loop (Performance).
- `check(x, spec)`, complex numbers, the spec-tracking wrapper type (#44) (API / user experience).
- Python 3.14: add it to the CI matrix and classifiers once the minimum supported Numba version supports it.
