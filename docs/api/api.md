# API Overview

`pyfxp` offers two ways to convert numbers to fixed-point:

- **[fxpt](fxpt.md)** → “typed” (parameter-based)
    Pass the format directly as function parameters.
- **[fxp](fxp.md)** → “spec-based” (explicit FxpSpec)
    Build a reusable spec with [`Q(...)`](Q.md), then pass that spec.


## Parameter-based (fxpt)

Use when the format is simple or ad-hoc.

```python
from pyfxp import fxpt
from pyfxp.constants import SAT, HALF_EVEN  # overflow & rounding modes

y = fxpt(
    x=3.14159,
    qi=3,            # integer bits (including sign when signed=True)
    qf=13,           # fractional bits
    signed=True,     # signed or unsigned format
    ovf=SAT,         # overflow behavior: WRAP | SAT | ERROR
    rnd=HALF_EVEN,   # rounding: TRUNC, CEIL, TO_ZERO, AWAY, HALF_UP, ...
)
```

!!! note "ARM-style Q-format"
    In ARM-style Q-format notation (used by `pyfxp`), the **sign bit is included** in the integer bit count `qi`.
    Refer to [Q (number format)](https://en.wikipedia.org/wiki/Q_(number_format)) for more information.

---

## Spec-based (fxp)

Use when you want a reusable, readable definition.

```python
import numpy as np
from pyfxp import fxp, Q
from pyfxp.constants import SAT, HALF_EVEN

# Define the format once
Q3_13 = Q(qi=3, qf=13, signed=True, ovf=SAT, rnd=HALF_EVEN)

# Reuse it across your codebase
y1 = fxp(3.14159, Q3_13)
y2 = fxp(np.array([0.1, 0.2, 0.3]), Q3_13)
```

[`Q(...)`](Q.md) returns an [FxpSpec](FxpSpec.md) instance (the explicit spec object used by fxp).

---

## The [`pyfxp.constants`](constants.md) module

The `constants` module defines the rounding and overflow modes as the `IntEnum`s `Rounding` and `Overflow`.
They are used by both `fxpt` and `fxp` to specify fixed-point arithmetic behavior.

### Rounding modes

| Constant | Description |
|-----------|--------------|
| `TRUNC` | Bit truncation (rounds toward negative infinity) |
| `CEIL` | Round toward positive infinity |
| `TO_ZERO` | Round toward zero |
| `AWAY` | Round away from zero |
| `HALF_UP` | Round to nearest; ties toward positive infinity |
| `HALF_DOWN` | Round to nearest; ties toward negative infinity |
| `HALF_EVEN` | Round to nearest; ties to even |
| `HALF_ZERO` | Round to nearest; ties toward zero |
| `HALF_AWAY` | Round to nearest; ties away from zero |

### Overflow modes

| Constant | Description |
|-----------|--------------|
| `WRAP` | Wrap around on overflow (modular arithmetic) |
| `SAT` | Saturate at the maximum/minimum representable value |
| `ERROR` | Raise an exception on overflow |

!!! tip "Usage example"
    The modes are `IntEnum` members: they print by name (`<Rounding.HALF_EVEN: 6>`) but are plain integers everywhere else,
    so the numeric codes (`rnd=6`) keep working and the modes can be used inside `@njit` code.
    `Q(...)` converts integer codes to members, and passing a `Rounding` member as `ovf` (or the reverse) raises a `TypeError`.

    ```python
    from pyfxp.constants import SAT, HALF_EVEN
    fxpt(1.25, qi=3, qf=13, ovf=SAT, rnd=HALF_EVEN)
    ```
