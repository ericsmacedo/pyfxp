# Bit-exact integer datapath

Formats wider than 53 bits can't be modeled on float64 values without rounding (see
[Precision of float results](../api/precision.md)). This example models a datapath of that kind
on the **stored integers** instead, so every intermediate result is exact. The datapath is
an 8-tap FIR filter with a 56-bit product and a 59-bit accumulator.

## Formats

```python
>>> import numpy as np
>>> from pyfxp import Q, fxp, to_int, to_hex
>>> from pyfxp.constants import HALF_EVEN, SAT
>>> x_spec = Q(1, 31)                    # input samples
>>> h_spec = Q(1, 23)                    # coefficients
>>> p_spec = x_spec.mul(h_spec)          # full-precision product
>>> a_spec = p_spec.acc(8)               # sum of 8 products
>>> y_spec = Q(2, 30, rnd=HALF_EVEN, ovf=SAT)
>>> p_spec, a_spec
(sQ2.54(TRUNC, WRAP), sQ5.54(TRUNC, WRAP))
>>> p_spec.width, a_spec.width
(56, 59)

```

## Quantize the inputs once

`to_int` returns the stored integers as an int64 array. From here on, everything is integer
arithmetic.

```python
>>> t = np.linspace(0, 20, 64)
>>> x = 0.9 * np.sin(t) + 0.05 * np.sin(37 * t)
>>> h = np.hanning(8) / np.hanning(8).sum()
>>> nx = to_int(x, x_spec)
>>> nh = to_int(h, h_spec)
>>> nx.dtype, nh.dtype
(dtype('int64'), dtype('int64'))

```

## Multiply and accumulate on integers

The product of an `sQ1.31` and an `sQ1.23` stored integer is the `sQ2.54` stored integer of
the exact product: the fractional bits add up. `a_spec` guarantees that the sum of 8 such
products fits, so int64 arithmetic never overflows and no bit is lost.

```python
>>> n_acc = np.convolve(nx, nh)[: len(x)]   # stored integers of sQ5.54
>>> int(np.abs(n_acc).max()).bit_length()   # more than float64's 53 bits
54

```

## Requantize to the output format

To round a stored integer `n` of format `src` to format `dst`, quantize `n` as an integer
input with a format of `dst`'s width whose fractional bits are `dst.qf - src.qf`. It computes
`n * 2**(dst.qf - src.qf)` with `dst`'s rounding and overflow modes. A negative `qf`
is an exact right shift. Integer inputs are quantized in integer arithmetic, so this step is
exact at every width.

```python
>>> def requantize(n, src, dst):
...     """Stored integers of format `src` -> stored integers of format `dst`."""
...     return to_int(n, Q(dst.qi + src.qf, dst.qf - src.qf, dst.signed, dst.rnd, dst.ovf))
>>> n_y = requantize(n_acc, a_spec, y_spec)  # sQ2.30, HALF_EVEN, SAT
>>> n_y[2:6]  # the first coefficient of the Hann window is 0
array([ 14113243,  73773752, 202538934, 394024858])

```

## Test vectors

For the bit patterns, pass the stored integers with a `qf = 0` format of the same width and
signedness. The stored integer is then unchanged.

```python
>>> to_hex(n_y[2:6], Q(32, 0))
array(['00d759db', '0465b2b8', '0c127fb6', '177c579a'], dtype='<U8')
>>> to_hex(n_acc[2:4], Q(59, 0))            # 59-bit accumulator, exact
array(['000d759dac38bf8', '00465b2b84188a9'], dtype='<U15')

```

Reading patterns back works the same way. `int(s, 16)` gives the unsigned pattern, and WRAP
in a `qf = 0` format turns it into the signed stored integer. Use this, not `from_hex`, for
formats wider than 53 bits: `from_hex` returns a float.

```python
>>> rtl_acc = ["7ffffffffffffff", "400000000000001"]  # e.g. read from an RTL dump
>>> to_int([int(s, 16) for s in rtl_acc], Q(59, 0))
array([                 -1, -288230376151711743])

```

## Comparison with the float flow

The same filter on float64 values rounds the 56–59-bit intermediate results:

```python
>>> acc_float = np.convolve(fxp(x, x_spec), fxp(h, h_spec))[: len(x)]
>>> int((to_int(acc_float, a_spec) != n_acc).sum())  # accumulator values that differ
26

```

Here, the output still matches, because the float error is far below the output LSB. It can
differ, though, whenever an accumulator value is close to a rounding boundary of the output
format. The integer flow gives the bits the hardware gives, for every input.

## Inside `@njit`

`to_int` and `Q` work inside `@njit` code, and integer arrays use the same exact kernels:

```python
>>> from numba import njit
>>> @njit
... def fir(nx, nh, src, dst):
...     n_acc = np.convolve(nx, nh)[: len(nx)]
...     return to_int(n_acc, Q(dst.qi + src.qf, dst.qf - src.qf, dst.signed, dst.rnd, dst.ovf))
>>> bool(np.array_equal(fir(nx, nh, a_spec, y_spec), n_y))
True

```
