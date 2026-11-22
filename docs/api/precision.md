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
stored integer comes from **WRAP or SAT of a large value**. Integer inputs (Python ints of any
size, NumPy integer scalars and arrays, integer lists) are quantized exactly, in integer
arithmetic: for an **integer above `2**53`**, the stored integer is exact and only the returned
float is rounded. Inside `@njit` code, uint64 inputs are the exception: they are converted to
float64 first.

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
>>> fxp(2**58 + 1, Q(60, 0)) == 2**58 # the float result is rounded
True
>>> to_int(2**58 + 1, Q(60, 0))       # the stored integer is exact
288230376151711745

```

For bit-exact work above 53 bits (test vectors, comparison with RTL), use the stored integers
and bit patterns from [`to_int`, `to_bin` and `to_hex`](raw.md), and compare in that domain
rather than on floats.

## Complex values

`fxp` of a complex value quantizes the real and imaginary parts separately, each a float64, so
the rules above apply to each part. Products computed before quantizing are a separate limit:
for `fxp(a * b, spec)`, NumPy computes the real part `a.real * b.real - a.imag * b.imag` in
float64, which is exact only while each product and their difference fit in 53 bits, i.e.
roughly `a_spec.width + b_spec.width + 1 <= 53`. The format of that exact real part is
`a_spec.mul(b_spec).sub(a_spec.mul(b_spec))`.
