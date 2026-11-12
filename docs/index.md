# pyfxp

:::{container} hero-tagline
**Bit-true fixed-point arithmetic for Python, at compiled speed.**
:::

Model the datapath of your FPGA, ASIC or DSP firmware in Python. Pick the word
length, the rounding and the overflow behavior, and get exactly the numbers the
hardware will produce: on plain floats and NumPy arrays, from Python or inside
Numba `@njit` loops.

::::{grid} auto
:gutter: 2

:::{grid-item}
```{button-ref} getting-started
:color: primary
:shadow:

Get started
```
:::

:::{grid-item}
```{button-ref} api/index
:color: secondary
:outline:

API reference
```
:::

:::{grid-item}
```{button-link} https://github.com/ericsmacedo/pyfxp
:color: secondary
:outline:

GitHub
```
:::
::::

```bash
pip install pyfxp
```

## A fixed-point FIR filter, end to end

```python
import numpy as np
from numba import njit
from pyfxp import Q, fxp, to_hex
from pyfxp.constants import HALF_EVEN, SAT

x_spec = Q(1, 15, rnd=HALF_EVEN, ovf=SAT)   # ADC samples:   sQ1.15
h_spec = Q(1, 15, rnd=HALF_EVEN, ovf=SAT)   # filter taps:   sQ1.15
acc    = x_spec.mul(h_spec).acc(32)         # accumulator:   sQ7.30, can't overflow
y_spec = Q(1, 15, rnd=HALF_EVEN, ovf=SAT)   # output register

@njit                                       # pyfxp runs inside compiled loops
def fir(x, h, acc, y_spec):
    y = np.zeros_like(x)
    for n in range(h.size - 1, x.size):
        s = 0.0
        for k in range(h.size):
            s = fxp(s + x[n - k] * h[k], acc)
        y[n] = fxp(s, y_spec)
    return y

h = fxp(np.hanning(32) / np.hanning(32).sum(), h_spec)
x = fxp(0.9 * np.sin(np.linspace(0, 20, 10_000)), x_spec)
y = fir(x, h, acc, y_spec)

to_hex(y, y_spec)                           # expected values for your RTL testbench
```

## Why pyfxp

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Bit-true
Nine rounding modes, wrap / saturate / error on overflow, formats up to 64 bits.
Every mode is property-tested against an exact rational-arithmetic reference.
:::

:::{grid-item-card} Fast where it counts
**~5 ns per sample** inside `@njit` loops, and as fast as the fastest library on
NumPy arrays. The only Python fixed-point library that works inside Numba.
:::

:::{grid-item-card} Plain floats and NumPy
No custom number type: results are floats and float64 arrays, so SciPy, Matplotlib
and the model you already have keep working.
:::

:::{grid-item-card} Hardware-ready
`to_int`, `to_hex` and `to_bin` give the stored integers and two's-complement
patterns for `$readmemh` test vectors, exact at every width.
:::

:::{grid-item-card} Bit growth, computed
`a.mul(b)`, `a.add(b)` and `a.acc(n)` return the smallest format that holds the
exact result, so intermediate values never overflow by accident.
:::

:::{grid-item-card} Readable formats
`Q(3, 13, True, HALF_EVEN, SAT)` prints as `sQ3.13(HALF_EVEN, SAT)`, and swapping
the rounding and overflow arguments is an error, not a silent bug.
:::
::::

## Performance

Quantizing to `sQ8.12` with floor rounding and saturation, compared with the other
Python fixed-point libraries (log scale, lower is better):

```{image} _static/compare-light.svg
:class: only-light
:alt: Bar chart. Array of one million values, time per value: pyfxp 5 ns, numfi 5 ns, APyTypes 32 ns, fxpmath 396 ns, spfpm 4.2 µs, fixedpoint 9.3 µs. One value at a time: pyfxp inside @njit 5 ns, APyTypes 387 ns, pyfxp from Python 1 µs, spfpm 3.8 µs, fixedpoint 9.2 µs, numfi 19 µs, fxpmath 85 µs.
```

```{image} _static/compare-dark.svg
:class: only-dark
:alt: Bar chart. Array of one million values, time per value: pyfxp 5 ns, numfi 5 ns, APyTypes 32 ns, fxpmath 396 ns, spfpm 4.2 µs, fixedpoint 9.3 µs. One value at a time: pyfxp inside @njit 5 ns, APyTypes 387 ns, pyfxp from Python 1 µs, spfpm 3.8 µs, fixedpoint 9.2 µs, numfi 19 µs, fxpmath 85 µs.
```

On arrays, pyfxp and numfi are the fastest, within run-to-run noise of each other.
The difference that matters for DSP models is per-sample code: inside an `@njit`
loop pyfxp takes about 5 ns per value, while the other libraries can only be
called from Python, at 0.4 µs or more per call. See the
[full comparison](comparison.md) for the method, the numbers and what each library
supports.

```{toctree}
:hidden:
:caption: User guide

getting-started
examples/example_int
comparison
```

```{toctree}
:hidden:
:caption: API reference

api/index
```

```{toctree}
:hidden:
:caption: Development

contributing
changelog
```
