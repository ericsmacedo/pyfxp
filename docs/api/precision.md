# Precision of float results

Formats can be up to **64 bits signed** or **63 bits unsigned**: the stored integer `n`
(the value with `fxp(x, spec) == n * 2**-qf`) is an `int64` and is always exact.

The real value is returned as a float64, which has a 53-bit significand. The float result is
**exact if and only if `n` has at most 53 significant bits** (its bit length after dropping
trailing zero bits). Every format of 53 bits or less is therefore always exact. In wider
formats, a result can be rounded to the nearest float64. This applies to:

- `fxp` and `fxpt`,
- `from_int`, `from_bin` and `from_hex`,
- `FxpSpec.min` and `FxpSpec.max` (`min_int` and `max_int` stay exact).

Float inputs carry at most 53 significant bits, so `fxp` of a float only loses bits when the
stored integer comes from **WRAP or SAT of a large value**. Integer inputs are converted to
float64 first, so **integers above `2**53`** are rounded before they are quantized.

```python
>>> from pyfxp import Q, fxp, to_int, to_hex
>>> from pyfxp.constants import SAT
>>> spec = Q(64, 0, ovf=SAT)          # sQ64.0: max_int = 2**63 - 1
>>> fxp(1e30, spec)                   # 2**63: rounded, and outside the format
9.223372036854776e+18
>>> to_int(1e30, spec)                # the stored integer is exact
9223372036854775807
>>> to_hex(1e30, spec)
'7fffffffffffffff'
>>> fxp(2**58 + 1, Q(60, 0)) == 2**58 # the input is rounded to float64 first
True

```

For bit-exact work above 53 bits (test vectors, comparison with RTL), use the stored integers
and bit patterns from [`to_int`, `to_bin` and `to_hex`](raw.md), and compare in that domain
rather than on floats.
