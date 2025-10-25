# MIT License
#
# Copyright (c) 2025 ericsmacedo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import NamedTuple

import numpy as np
from numba import njit, types
from numba.extending import overload

from .constants import AWAY, CEIL, ERROR, HALF_AWAY, HALF_DOWN, HALF_EVEN, HALF_UP, HALF_ZERO, SAT, TO_ZERO, TRUNC, WRAP

RND_MIDPOINT = 0.5  # Threshold for half-way rounding


class FxpSpec(NamedTuple):
    qi: int
    qf: int
    signed: bool
    rnd: int
    ovf: int


@njit
def Q(qi: int, qf: int, signed: bool = True, rnd: int = TRUNC, ovf: int = WRAP) -> FxpSpec:
    """Convert a numeric value to fixed-point representation using Q-format notation.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    qi : int
        Number of integer bits (excluding sign bit if signed=True).
    qf : int
        Number of fractional bits.
    signed : bool, optional
        Whether the fixed-point format is signed (default is True).
    rnd : int, optional
        Rounding method to apply (default is TRUNC (0)).

        Supported methods:

        - TRUNC (0): Bit Truncation. Rounds towards negative infinity.
        - CEIL (1): Round toward positive infinity.
        - TO_ZERO (2): Round toward zero.
        - AWAY (3): Round away from zero.
        - HALF_UP (4): Round to nearest; ties round towards positive infinity.
        - HALF_DOWN (5): Round to nearest; ties round toward negative infinity.
        - HALF_EVEN (6): Round to nearest; ties round to even.
        - HALF_ZERO (7): Round to nearest; ties round toward zero.
        - HALF_AWAY (8): Round to nearest; ties round away from zero.

    ovf : int, optional
        Overflow handling method (default is WRAP (0)).

        Supported methods:

        - WRAP (0): Wrap around on overflow (modulo behavior).
        - SAT (1): Saturate to maximum/minimum representable value.
        - ERROR (2): Raise an error if overflow occurs.

    Returns:
    -------
    float or ndarray
        Fixed-point representation of the input, as integer(s).

    Notes:
    -----
    Uses ARM-style Q-format notation where a Qm.n format has:
        - m integer bits (qi)
        - n fractional bits (qf)
        - Optional sign bit if `signed` is True
    """
    return FxpSpec(qi, qf, signed, rnd, ovf)


# --- usage ---
# Q1_15  = Q(1, 15)            # signed by default
# UQ0_8  = Q(0, 8, False)      # unsigned 0.8
# Q3_12R = Q(3, 12, True, TRUNC, WRAP)


# Option that uses simple tuples for the specs
# @njit
# def fxpt(x, spec):
#     qi, qf, signed, rnd, ovf = spec
#     return fxp(x, qi, qf, signed, rnd, ovf)


@njit
def fxpt(x: float | np.ndarray, spec: FxpSpec) -> float | np.ndarray:
    """
    Convert a numeric value to fixed-point representation using a pre-defined
    fixed-point specification.

    This function behaves like `fxp`, but instead of requiring multiple
    arguments (`qi`, `qf`, `signed`, `rnd`, `ovf`), it accepts a single
    `FxpSpec` named tuple (or tuple) that encapsulates all format parameters.
    This makes it easier to reuse and pass fixed-point type definitions in
    a compact, Numba-friendly form.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.

    spec : FxpSpec or tuple
        A fixed-point specification tuple or named tuple with the following fields:
            - qi : int
              Number of integer bits (excluding sign bit if `signed=True`).
            - qf : int
              Number of fractional bits.
            - signed : bool
              Whether the fixed-point format is signed.
            - rnd : int
              Rounding method to apply.
              Supported methods (same as in `fxp`):
                  * TRUNC (0): Bit truncation; rounds toward negative infinity.
                  * CEIL (1): Round toward positive infinity.
                  * TO_ZERO (2): Round toward zero.
                  * AWAY (3): Round away from zero.
                  * HALF_UP (4): Round to nearest; ties toward +∞.
                  * HALF_DOWN (5): Round to nearest; ties toward −∞.
                  * HALF_EVEN (6): Round to nearest; ties to even.
                  * HALF_ZERO (7): Round to nearest; ties toward zero.
                  * HALF_AWAY (8): Round to nearest; ties away from zero.
            - ovf : int
              Overflow handling method.
              Supported methods (same as in `fxp`):
                  * WRAP (0): Wrap around (modulo behavior).
                  * SAT (1): Saturate to max/min representable value.
                  * ERROR (2): Raise an error on overflow.

    Returns
    -------
    float or ndarray
        Fixed-point representation of the input value(s).

    Notes
    -----
    - Equivalent to:
          >>> fxpt(x, Q(qi, qf, signed, rnd, ovf))
          == fxp(x, qi, qf, signed, rnd, ovf)
    - Uses ARM-style Q-format notation (Qm.n), where:
        * m = `qi`  → number of integer bits
        * n = `qf`  → number of fractional bits
        * Optional sign bit if `signed` is True
    - Designed to be fully compatible with Numba `@njit` mode when `spec`
      is a tuple or `NamedTuple` of primitive types.
    """
    return fxp(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)


@njit
def _rnd_scalar(x, method=TRUNC):  # noqa: PLR0911, PLR0912, C901
    if method == TRUNC:
        return int(np.floor(x))
    if method == CEIL:
        return int(np.ceil(x))
    if method == TO_ZERO:
        return int(x)
    if method == AWAY:
        if x >= 0:
            return int(np.ceil(x))
        return int(-np.ceil(np.abs(x)))
    if method == HALF_UP:
        return int(np.floor(x + RND_MIDPOINT))
    if method == HALF_DOWN:
        return int(np.ceil(x - RND_MIDPOINT))
    if method == HALF_ZERO:
        if x >= 0:
            return int(np.ceil(x - RND_MIDPOINT))
        return int(-np.ceil(np.abs(x) - RND_MIDPOINT))
    if method == HALF_AWAY:
        if x >= 0:
            return int(np.floor(x + RND_MIDPOINT))
        return int(-np.floor(np.abs(x) + RND_MIDPOINT))
    if method == HALF_EVEN:
        floor_x = np.floor(x)
        frac = x - floor_x
        is_half = frac == RND_MIDPOINT
        if is_half:
            return int(floor_x + (floor_x % 2 == 1))
        return int(np.round(x))
    raise ValueError(f"invalid method: {method}")


@njit
def _rnd_array(x, method=TRUNC):
    if method == TRUNC:  # Round towards -inf
        x = np.floor(x)
    elif method == CEIL:  # Round towards +inf
        x = np.ceil(x)
    elif method == TO_ZERO:
        pass
    elif method == AWAY:
        x = np.where(x >= 0, np.ceil(np.abs(x)), -np.ceil(np.abs(x)))
    elif method == HALF_UP:
        x = np.floor(x + RND_MIDPOINT)
    elif method == HALF_DOWN:
        x = np.ceil(x - RND_MIDPOINT)
    elif method == HALF_ZERO:
        x = np.where(x >= 0, np.ceil(np.abs(x) - RND_MIDPOINT), -np.ceil(np.abs(x) - RND_MIDPOINT))
    elif method == HALF_AWAY:
        x = np.where(x >= 0, np.floor(np.abs(x) + RND_MIDPOINT), -np.floor(np.abs(x) + RND_MIDPOINT))
    elif method == HALF_EVEN:
        floor_x = np.floor(x)
        frac = x - floor_x
        is_half = frac == RND_MIDPOINT
        even_correction = floor_x % 2 == 1  # if odd, add 1 to make even
        x = np.where(
            is_half,
            floor_x + even_correction,
            np.round(x),  # normal round otherwise (to nearest)
        )
    else:
        raise ValueError(f"invalid method: {method}")

    return x.astype(np.int64)


@njit
def _overflow_scalar(x: int, signed: bool = True, w: int = 16, method: int = WRAP):
    # Maximum and minimum values with w bits representation
    if signed:
        upper = (1 << (w - 1)) - 1
        lower = -(1 << (w - 1))
    else:
        upper = (1 << w) - 1
        lower = 0

    if method == WRAP:
        mask = 1 << w
        x = x & (mask - 1)
        if signed:
            if x >= (1 << (w - 1)):
                return x | (-mask)
    elif method == SAT:
        if x > upper:
            return upper
        if x < lower:
            return lower
    elif method == ERROR:
        if x > upper or x < lower:
            raise OverflowError("Overflow!")
    else:
        raise ValueError(f"invalid method: {method}")

    return x


@njit
def _overflow_array(x, signed: bool = True, w: int = 16, method: int = WRAP):
    x = np.asarray(x, dtype=np.int64)

    # Maximum and minimum values with w bits representation
    if signed:
        upper = (1 << (w - 1)) - 1
        lower = -(1 << (w - 1))
    else:
        upper = (1 << w) - 1
        lower = 0

    if method == WRAP:
        mask = 1 << w
        x = x & (mask - 1)
        if signed:
            x = np.where(x < (1 << (w - 1)), x, x | (-mask))
    elif method == SAT:
        x[x > upper] = upper
        x[x < lower] = lower
    elif method == ERROR:
        up = x > upper
        low = x < lower
        if np.any(up | low):
            raise OverflowError("Overflow!")
    else:
        raise ValueError(f"invalid method: {method}")

    return x


@njit
def _fxp_array(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    x = x * 2.0**qf

    x = _rnd_array(x, method=rnd)
    x = _overflow_array(x, signed=signed, w=(qi + qf), method=ovf)

    return x / 2.0**qf


@njit
def _fxp_scalar(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    x *= 2.0**qf

    x = _rnd_scalar(x, method=rnd)
    x = _overflow_scalar(x, signed=signed, w=(qi + qf), method=ovf)

    return x / 2.0**qf


def fxp(  # noqa: PLR0913
    x: float | np.ndarray, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP
) -> float | np.ndarray:
    """Convert a numeric value to fixed-point representation using Q-format notation.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    qi : int
        Number of integer bits (excluding sign bit if signed=True).
    qf : int
        Number of fractional bits.
    signed : bool, optional
        Whether the fixed-point format is signed (default is True).
    rnd : int, optional
        Rounding method to apply (default is TRUNC (0)).

        Supported methods:

        - TRUNC (0): Bit Truncation. Rounds towards negative infinity.
        - CEIL (1): Round toward positive infinity.
        - TO_ZERO (2): Round toward zero.
        - AWAY (3): Round away from zero.
        - HALF_UP (4): Round to nearest; ties round towards positive infinity.
        - HALF_DOWN (5): Round to nearest; ties round toward negative infinity.
        - HALF_EVEN (6): Round to nearest; ties round to even.
        - HALF_ZERO (7): Round to nearest; ties round toward zero.
        - HALF_AWAY (8): Round to nearest; ties round away from zero.

    ovf : int, optional
        Overflow handling method (default is WRAP (0)).

        Supported methods:

        - WRAP (0): Wrap around on overflow (modulo behavior).
        - SAT (1): Saturate to maximum/minimum representable value.
        - ERROR (2): Raise an error if overflow occurs.

    Returns:
    -------
    float or ndarray
        Fixed-point representation of the input, as integer(s).

    Notes:
    -----
    Uses ARM-style Q-format notation where a Qm.n format has:
        - m integer bits (qi)
        - n fractional bits (qf)
        - Optional sign bit if `signed` is True
    """
    if isinstance(x, np.ndarray):
        return _fxp_array(x, qi, qf, signed, rnd, ovf)

    if isinstance(x, float | int):
        return _fxp_scalar(x, qi, qf, signed, rnd, ovf)

    raise TypeError(f"Unsupported type: {x}")


# Numba's overload is used to decide when to call the array or the scalar function.
@overload(fxp)
def fxp_overload(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913, ARG001
    # Array case
    if isinstance(x, types.Array):

        def impl(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
            return _fxp_array(x, qi, qf, signed, rnd, ovf)  # pragma: no cover

        return impl

    if isinstance(x, types.Integer | types.Float):

        def impl(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
            return _fxp_scalar(x, qi, qf, signed, rnd, ovf)  # pragma: no cover

        return impl
    raise TypeError(f"Unsupported type: {x}")
