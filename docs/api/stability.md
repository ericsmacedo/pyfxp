# Public API and versioning

pyfxp follows [Semantic Versioning 2.0.0](https://semver.org/) from 1.0 on:
`MAJOR.MINOR.PATCH`, where only a major release may break code that uses the public API.

## Public API

The public API is exactly:

| Name | What |
|---|---|
| `pyfxp.__all__` | [`fxp`](fxp.md), [`fxpt`](fxpt.md), [`Q`](Q.md), [`FxpSpec`](FxpSpec.md), [`to_int`, `from_int`, `to_bin`, `to_hex`, `from_bin`, `from_hex`](raw.md) |
| [`pyfxp.constants`](constants.md) | `Rounding`, `Overflow` and the module-level mode names (`TRUNC` … `HALF_AWAY`, `WRAP`, `SAT`, `ERROR`) |

This covers their documented signatures, return types, attributes and methods, and their use
from Python and inside Numba `@njit` code.

Everything else is private and may change in any release: the modules `pyfxp._pyfxp` and
`pyfxp._raw`, the compiled kernels, and any name that starts with an underscore. The exception
is the standard `NamedTuple` methods of `FxpSpec` (`_asdict`, `_replace`, `_fields`), which are public.

## What counts as a breaking change

These change only in a **major** release:

- **Numerical results.** For a given input and spec, `fxp` / `fxpt` and the raw-integer
  functions return the same bits. Results are defined by the exact rational reference model
  in the tests, so they don't depend on the platform or on whether Numba JIT is enabled.
- **Mode codes.** The integer value of each mode (`TRUNC = 0` … `HALF_AWAY = 8`,
  `WRAP = 0`, `SAT = 1`, `ERROR = 2`) never changes, so stored codes keep their meaning.
- **Q-format conventions.** `qi` includes the sign bit (ARM style); the defaults are
  `signed=True`, `rnd=TRUNC`, `ovf=WRAP`.
- **Exception types.** Which exception is raised (`OverflowError`, `ValueError`, `TypeError`)
  and when.
- Removing or renaming a public name, parameter or `FxpSpec` field, or changing a parameter's
  position.

## What is not a breaking change

These can happen in a **minor** release:

- New functions, new parameters with defaults, new `FxpSpec` properties or methods, new
  rounding or overflow modes (with new codes).
- Accepting inputs that used to raise.
- Dropping a Python or NumPy/Numba version that has reached end of life. The supported range is
  in `pyproject.toml` (`requires-python`, `dependencies`).

And in a **patch** release:

- **Bug fixes**, including ones that change a result, when the old result disagreed with the
  documentation or the reference model. The changelog lists every such fix.
- Exception message text, `repr` text and warnings. Don't parse them.
- Performance, compile time and Numba caching behavior.

## Deprecations

A public name or behavior is deprecated for at least one minor release before it is removed in
the next major release. Deprecated calls emit a `DeprecationWarning` that names the replacement.

## Before 1.0

Versions 0.x make no stability promise; the changes since 0.2.1 are listed in the changelog.
