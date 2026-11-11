# pyfxp Roadmap

Tracked improvements for pyfxp. Tick an item and link its PR when it lands.

## Correctness

- [x] Support formats up to 64 bits signed / 63 bits unsigned and reject wider ones (`qi + qf`); they used to silently return wrong values, e.g. `fxpt(1.0, 40, 30)` → `-5.96e-08`.
- [x] Inputs with `|x| * 2**qf >= 2**63` gave wrong results in any format (#18), e.g. `fxpt(1e19, 8, 4, ovf=SAT)` → `-128.0` instead of `127.9375`.
- [x] HALF_* rounding was wrong for scaled values >= 2**52 (#24).
- [ ] Validate `qi` / `qf` (no negative values, total width ≥ 1).
- [ ] Accept NumPy scalars (`np.integer`, `np.floating`) and array-likes (lists, tuples) as input.
- [ ] Make overflow `ERROR` messages useful: include the value, the format and the valid range.

## Performance

Baseline numbers come from `uv run inv bench`.

- [ ] Speed up the per-sample path inside `@njit` code: 19–26 ns/sample today vs 0.7 ns for a hand-written loop (~30×).
- [x] Find out why `WRAP` costs ~7 ns/sample more than `SAT` in the scalar path (fixed by the new mask computation).
- [ ] Convert large arrays in a single loop instead of chained NumPy temporaries (3–5× slower than NumPy at 1M elements).
- [ ] Remove the redundant `HALF_EVEN` special case (`np.round` already rounds ties to even).
- [ ] Merge the duplicated scalar and array implementations.

## API / user experience

- [ ] Spec arithmetic helpers for bit growth, e.g. `Q.add(a, b)`, `Q.mul(a, b)`, `Q.acc(spec, n)`.
- [ ] `FxpSpec` helpers: `min`, `max`, `lsb`, `width`, and a readable repr such as `sQ3.13 SAT/HALF_EVEN`.
- [ ] Format check `check(x, spec)` that raises if `x` isn't exactly representable in `spec`, behind a global switch that costs nothing when off. Catches forgotten quantization.
- [ ] Raw integer conversion `to_int(x, spec)` / `from_int(n, spec)`, plus bin/hex helpers for HDL test vectors.
- [ ] Rounding and overflow modes as `IntEnum`s: readable when printed, still usable inside Numba.
- [ ] Complex number support (quantize real and imaginary parts separately).
- [ ] Optional thin Python wrapper type that tracks its spec, for interactive exploration only. The core stays on plain floats.

## Documentation

- [ ] Fix the broken docstring examples (`Q(3, 1)` labelled "Q3.5", stray `>>> 3.14159…` lines).
- [ ] Fix the `fxpt` "Returns" section (it returns floats, not integers).
- [ ] Use one docstring style throughout: mkdocs is set up for NumPy style, ruff for Google style.
- [ ] Fix the `pyproject.toml` description typo ("library..").
- [ ] Tell users to call pyfxp on arrays or inside `@njit` code: a pure-Python per-sample loop is ~180× slower.

## Testing / CI

- [x] Fix `gen_ref_data.py`, stop skipping the first reference case, seed RNGs (#10).
- [x] Pin Python in the coveralls finish job (#13).
- [ ] Property-based tests against an exact reference model (#11).
- [ ] Benchmarks with `inv bench` (#12).
- [x] Add tests for wide formats.
- [ ] Fix the empty `COVERALLS_FLAG_NAME` (`matrix.test-name` does not exist).
