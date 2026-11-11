# pyfxp Roadmap

Tracked improvements for pyfxp. Tick an item and link its PR when it lands.

## Correctness

- [x] Support formats up to 64 bits signed / 63 bits unsigned and reject wider ones (`qi + qf`); they used to silently return wrong values, e.g. `fxpt(1.0, 40, 30)` → `-5.96e-08`.
- [x] Inputs with `|x| * 2**qf >= 2**63` gave wrong results in any format (#18), e.g. `fxpt(1e19, 8, 4, ovf=SAT)` → `-128.0` instead of `127.9375`.
- [x] HALF_* rounding was wrong for scaled values >= 2**52 (#24).
- [ ] Validate `qi` / `qf` (no negative values, total width ≥ 1).
- [x] Support arrays of any shape and layout (#26).
- [x] Accept NumPy scalars (`np.integer`, `np.floating`) and array-likes (lists, tuples) as input (#19).
- [ ] Make overflow `ERROR` messages useful: include the value, the format and the valid range.

## Performance

Baseline numbers come from `uv run inv bench`.

- [x] Speed up the per-sample path inside `@njit` code: 15–18 → ~5 ns/sample (#30). A hand-written single-mode loop is 0.7 ns; the rest is the run-time method dispatch.
- [x] Find out why `WRAP` costs ~7 ns/sample more than `SAT` in the scalar path (fixed by the new mask computation).
- [x] Convert large arrays in a single loop instead of chained NumPy temporaries: 10–40 → ~5 ns/elem at 1M elements (#31).
- [x] Remove the redundant `HALF_EVEN` special case (`np.round` already rounds ties to even) (#32, #33).
- [x] Merge the duplicated scalar and array implementations (#31).
- [ ] Vectorize the array loop (fast pass without checks, then fix up rare out-of-range / NaN elements) to approach 1 ns/elem.

## API / user experience

- [ ] Spec arithmetic helpers for bit growth, e.g. `Q.add(a, b)`, `Q.mul(a, b)`, `Q.acc(spec, n)`.
- [ ] `FxpSpec` helpers: `min`, `max`, `lsb`, `width`, and a readable repr such as `sQ3.13 SAT/HALF_EVEN`.
- [ ] Format check `check(x, spec)` that raises if `x` isn't exactly representable in `spec`, behind a global switch that costs nothing when off. Catches forgotten quantization.
- [ ] Raw integer conversion `to_int(x, spec)` / `from_int(n, spec)`, plus bin/hex helpers for HDL test vectors.
- [x] Rounding and overflow modes as `IntEnum`s: readable when printed, still usable inside Numba (#42).
- [ ] Complex number support (quantize real and imaginary parts separately).
- [ ] Optional thin Python wrapper type that tracks its spec, for interactive exploration only. The core stays on plain floats.

## Documentation

- [x] Fix the broken docstring examples (`Q(3, 1)` labelled "Q3.5", stray `>>> 3.14159…` lines) and run them as doctests (#20).
- [x] Fix the `fxpt` "Returns" section (it returns floats, not integers) (#36).
- [ ] Use one docstring style throughout: mkdocs is set up for NumPy style, ruff for Google style.
- [x] Fix the `pyproject.toml` description typo ("library..") (#34).
- [ ] Tell users to call pyfxp on arrays or inside `@njit` code: a pure-Python per-sample loop is ~180× slower.

## Testing / CI

- [x] Fix `gen_ref_data.py`, stop skipping the first reference case, seed RNGs (#10).
- [x] Pin Python in the coveralls finish job (#13).
- [x] Property-based tests against an exact reference model (#11).
- [x] Benchmarks with `inv bench` (#12).
- [x] Add tests for wide formats.
- [x] Fix the empty `COVERALLS_FLAG_NAME`: each upload now gets a distinct flag name (#35).
