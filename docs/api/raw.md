# Raw integers and test vectors

`fxp` returns the quantized **real value**. Hardware works with the **stored integer**:
the value `n` such that `fxp(x, spec) == n * 2**-qf`. These functions convert between
the two, and to the two's-complement bit patterns used in HDL test vectors.

```python
import numpy as np
from pyfxp import Q, fxp, to_int, from_int, to_hex, from_hex

spec = Q(1, 15)                       # sQ1.15

to_int(0.5, spec)                     # 16384
to_hex(-1.0, spec)                    # '8000'
from_hex('4000', spec)                # 0.5

# Stimulus file for $readmemh, and reading back the RTL's output
x = np.sin(np.linspace(0, 2 * np.pi, 64)) * 0.9
np.savetxt("stimulus.hex", to_hex(x, spec), fmt="%s")
y_rtl = from_hex(np.loadtxt("rtl_output.hex", dtype=str), spec)
```

Unlike the float result, the integer from `to_int` is exact for every supported
width (up to 64 bits signed / 63 bits unsigned).

```{eval-rst}
.. autofunction:: pyfxp.to_int
```
```{eval-rst}
.. autofunction:: pyfxp.from_int
```
```{eval-rst}
.. autofunction:: pyfxp.to_bin
```
```{eval-rst}
.. autofunction:: pyfxp.to_hex
```
```{eval-rst}
.. autofunction:: pyfxp.from_bin
```
```{eval-rst}
.. autofunction:: pyfxp.from_hex
```
