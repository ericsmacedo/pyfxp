# Changelog

All notable changes to pyfxp. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and from 1.0 on the project follows [Semantic Versioning](https://semver.org/): see
[Public API and versioning](https://pyfxp.readthedocs.io/en/stable/api/stability/).

## [1.0.0] - 2026-11-12

First stable release. Code written for 0.2.1 keeps working, except for the points under
**Changed**, which change results or behavior that was wrong or undefined before.

### Added

- `FxpSpec` properties `width`, `lsb`, `min_int`, `max_int`, `min` and `max`, also inside `@njit` code (#40, #46).
- Bit-growth helpers `FxpSpec.add`, `sub`, `mul`, `neg` and `acc`: the smallest format that holds an exact result (#41, #49).
- Raw integer conversion: `to_int`, `from_int`, `to_bin`, `to_hex`, `from_bin` and `from_hex` (#43, #49).
- `Rounding` and `Overflow` `IntEnum`s in `pyfxp.constants`; `TRUNC`, `SAT`, ... are their members (#42, #45).
- NumPy scalars, lists and tuples are accepted as input, from Python and inside `@njit` code (#19, #29).
- Arrays of any shape and memory layout (N-D, non-contiguous views, Fortran order, 0-d, empty). Before, anything but 1-D failed with a Numba `TypingError` (#26, #38).
- Formats up to 64 bits signed and 63 bits unsigned (#15).
- A `py.typed` marker, and `llms.txt` / `llms-full.txt` with the docs as plain text for AI assistants (#53).
- Documentation: Sphinx with the Furo theme (#50), a comparison with other fixed-point libraries (#51), the public API and SemVer policy (#55), and a README that works as a front page, with the comparison (#60).

### Changed

- **The `FxpSpec` repr** is Q notation, `sQ3.13(HALF_EVEN, SAT)`, instead of `FxpSpec(qi=3, qf=13, ...)` (#40, #46). Use `spec._asdict()` for the fields.
- **Modes are `IntEnum` members** and print by name (`<Rounding.HALF_EVEN: 6>`). They still compare equal to their integer codes, and integer codes are still accepted. `Q()` converts codes to members (#42, #45).
- **Swapped modes raise `TypeError`**: an `Overflow` member passed as `rnd`, or a `Rounding` member as `ovf`. Before, `TRUNC` / `WRAP` (both 0) and `CEIL` / `SAT` (both 1) were silently accepted (#45).
- **`fxp` and `Q` are plain Python functions** with Numba overloads, instead of `@njit` dispatchers. They work the same from Python and inside `@njit` code, but dispatcher attributes such as `fxp.py_func` are gone (#29, #45).
- **NaN and ±inf**: NaN is returned unchanged in every mode. ±inf saturates with `SAT`, raises `OverflowError` with `ERROR`, and gives NaN with `WRAP`. Before, they gave arbitrary values, e.g. `fxpt(nan, 8, 4, ovf=SAT)` returned `-128.0` (#27, #28).
- **Formats wider than 64 bits** (63 unsigned) raise `ValueError`. Before, any format with `qi + qf >= 64` returned wrong values, e.g. `fxpt(1.0, 40, 30)` gave `-5.96e-08` (#15).
- **Development Status** is `5 - Production/Stable` (#59).
- **Docs links** point to `/en/stable/`, the docs of the latest release (#57).
- **Overflow messages** with `ERROR` name the value, its index, the format and its range: `5.0 is out of range: the range of sQ3.4(TRUNC, ERROR) is [-4.0, 3.9375]` (was `Overflow!`) (#56).

### Fixed

- Wrong results for inputs with `|x| * 2**qf >= 2**63`, e.g. `fxpt(1e19, 8, 4, ovf=SAT)` returned `-128.0` instead of `127.9375` (#18, #25).
- `HALF_UP`, `HALF_DOWN`, `HALF_ZERO` and `HALF_AWAY` rounded wrongly for scaled values of `2**52` and above (#24, #25).

### Performance

- One compiled kernel shared by scalars, arrays and raw integers (#38):
  - per-sample `fxp` inside `@njit`: 15–18 → ~5 ns;
  - arrays: 10–40 → ~5 ns per element;
  - first-call compile: ~5.8 → ~1 s.
- `HALF_EVEN` uses `np.rint`: arrays ~3× faster (#32, #33).

## [0.2.1] and earlier

See the [GitHub releases](https://github.com/ericsmacedo/pyfxp/releases).

[1.0.0]: https://github.com/ericsmacedo/pyfxp/compare/v0.2.1...v1.0.0
[0.2.1]: https://github.com/ericsmacedo/pyfxp/releases/tag/v0.2.1
