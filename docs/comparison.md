# Comparison with other libraries

pyfxp is compared with the other Python fixed-point libraries on PyPI:
[numfi](https://pypi.org/project/numfi/), [APyTypes](https://pypi.org/project/apytypes/),
[fxpmath](https://pypi.org/project/fxpmath/), [fixedpoint](https://pypi.org/project/fixedpoint/)
and [spfpm](https://pypi.org/project/spfpm/).

## What each library offers

|                                              | pyfxp | numfi | APyTypes | fxpmath | fixedpoint | spfpm |
|----------------------------------------------|:-----:|:-----:|:--------:|:-------:|:----------:|:-----:|
| Result is a plain float / NumPy array        | ✔     | — ¹   | —        | —       | —          | —     |
| Vectorized over NumPy arrays                 | ✔     | ✔     | ✔        | ✔       | —          | —     |
| Quantize inside Numba `@njit` code           | ✔     | —     | —        | —       | —          | —     |
| Floor rounding with saturation               | ✔     | ✔     | ✔        | ✔       | ✔          | — ²   |
| Formats wider than 64 bits                   | —     | ✔     | ✔        | ✔       | ✔          | ✔     |

¹ numfi returns an `ndarray` subclass holding the stored integers; the real values
are in `.double`.

² spfpm rounds to nearest and raises on overflow.

Every cell was checked by running the library (see the method below).

If you need formats wider than 64 bits, use one of the other libraries: pyfxp
supports up to 64 bits (signed) / 63 bits (unsigned).

## Speed

```{image} _static/compare-light.svg
:class: only-light
:alt: Bar chart of the timings in the tables below.
```

```{image} _static/compare-dark.svg
:class: only-dark
:alt: Bar chart of the timings in the tables below.
```

**Quantize an array of 1,000,000 values**, time per value:

| Library | Time per value |
|---------|---------------:|
| pyfxp | 5.0 ns |
| numfi | 5.1 ns |
| APyTypes | 32 ns |
| fxpmath | 396 ns |
| spfpm (Python loop) | 4.2 µs |
| fixedpoint (Python loop) | 9.3 µs |

**Quantize one value at a time**, time per value. "pyfxp in @njit" is a compiled
loop calling `fxp` per sample; every other row is one call from Python per value:

| Library | Time per value |
|---------|---------------:|
| pyfxp in @njit | 5.0 ns |
| APyTypes | 387 ns |
| pyfxp | 997 ns |
| spfpm | 3.8 µs |
| fixedpoint | 9.2 µs |
| numfi | 19 µs |
| fxpmath | 85 µs |

## Method

- Every library quantizes the same values to signed Q8.12 (20 bits, 12 fractional),
  rounding toward negative infinity and saturating on overflow.
- Before timing, the results of every library are checked against pyfxp, on random
  values, values beyond the range and rounding ties. All match except spfpm, which has
  no floor rounding or saturation: it is timed on the same in-range values and marked
  with `*` in the chart.
- Arrays: 1,000,000 values for vectorized libraries. fixedpoint and spfpm
  have no array type, so they are timed on a Python loop over
  20,000 values.
- One value at a time: 5,000 calls (1,000 for fxpmath).
- Best of 11 runs. The first call is excluded, so Numba's
  compilation time is not counted.
- Measured on a shared cloud machine (Python 3.13.18, Linux-6.18.44-fc-v378-x86_64-with-glibc2.39),
  with pyfxp 0.2.2.dev20+g8ba956a.d20261112, numpy 2.3.4, numba 0.62.1, fxpmath 0.4.10, numfi 0.5.0, apytypes 0.5.1, fixedpoint 1.0.1, spfpm 1.8.0. Absolute times vary between runs on such a machine (up to about 2× for
  single calls from Python), so compare orders of magnitude rather than exact ratios.

To reproduce, or to measure on your own machine:

```bash
uv run --group bench python benchmarks/compare_libraries.py
```

It writes `benchmarks/results/compare_libraries.json` and redraws the charts.
