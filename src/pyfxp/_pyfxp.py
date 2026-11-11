# MIT License
#
# Copyright (c) 2025-2026 ericsmacedo
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


from collections.abc import Sequence
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numba import njit, types
from numba.extending import overload

from .constants import AWAY, CEIL, ERROR, HALF_AWAY, HALF_DOWN, HALF_EVEN, HALF_UP, HALF_ZERO, SAT, TO_ZERO, TRUNC, WRAP

RND_MIDPOINT = 0.5  # Threshold for half-way rounding

Scalar: TypeAlias = float | int | np.integer | np.floating
"""Scalar inputs: Python and NumPy integers and floats. They return a float."""

ArrayLike: TypeAlias = np.ndarray | Sequence[float]
"""Array inputs: NumPy arrays, lists and tuples. They return a float64 ndarray."""
_TWO_63 = 2.0**63  # first float64 outside the int64 range
_MAX_BELOW_TWO_63 = 2.0**63 - 1024.0  # largest float64 below 2**63


class FxpSpec(NamedTuple):
    """Fixed-point specification."""

    qi: int
    qf: int
    signed: bool
    rnd: int
    ovf: int


@njit(cache=True, nogil=True)
def Q(qi: int, qf: int, signed: bool = True, rnd: int = TRUNC, ovf: int = WRAP) -> FxpSpec:
    """Create a fixed-point format specification using ARM-style Q-format notation.

    The `Q` function defines a fixed-point representation by specifying
    the number of integer and fractional bits, whether the format is signed,
    and how rounding and overflow are handled. It returns an `FxpSpec` object
    that can be passed to other functions (e.g., `fxp`) for actual numeric conversion.

    Parameters
    ----------
    qi : int
        Number of integer bits.
    qf : int
        Number of fractional bits.
    signed : bool, optional
        Whether the fixed-point format is signed (default is True).
    rnd : int, optional
        Rounding method to apply (default is TRUNC (0)).

        Supported methods:

        - TRUNC (0): Bit truncation (rounds toward negative infinity)
        - CEIL (1): Round toward positive infinity
        - TO_ZERO (2): Round toward zero
        - AWAY (3): Round away from zero
        - HALF_UP (4): Round to nearest; ties toward positive infinity
        - HALF_DOWN (5): Round to nearest; ties toward negative infinity
        - HALF_EVEN (6): Round to nearest; ties to even
        - HALF_ZERO (7): Round to nearest; ties toward zero
        - HALF_AWAY (8): Round to nearest; ties away from zero

    ovf : int, optional
        Overflow handling method (default is WRAP (0)).

        Supported methods:

        - WRAP (0): Wrap around on overflow (modulo behavior)
        - SAT (1): Saturate to maximum/minimum representable value
        - ERROR (2): Raise an error if overflow occurs

    Returns
    -------
    FxpSpec
        A specification object describing the fixed-point format.

    Notes
    -----
    This function uses ARM-style Q-format notation:

    - `Qm.n` indicates `m` integer bits and `n` fractional bits (ARM style).
    - The returned `FxpSpec` can be reused for consistent conversions across multiple values.

    Examples
    --------
    >>> spec = Q(3, 1)  # Q3.5 format (3 integer bits, 5 fractional bits)
    >>> spec.qi, spec.qf
    (3, 1)
    >>> fxp(1.25, spec)
    1.0
    """
    return FxpSpec(qi, qf, signed, rnd, ovf)


def fxp(x: Scalar | ArrayLike, spec: FxpSpec) -> Any:  # float for scalars, ndarray for arrays (see Returns)
    """Convert a numeric value to fixed-point representation using a pre-defined fixed-point specification.

    This function behaves like `fxpt`, but instead of requiring multiple
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
          Number of integer bits (including sign bit if `signed=True`).
        - qf : int
          Number of fractional bits.
        - signed : bool
          Whether the fixed-point format is signed.
        - rnd : int
          Rounding method to apply.
          Supported methods (same as in `fxpt`):
              * TRUNC (0): Bit truncation; rounds toward negative infinity.
              * CEIL (1): Round toward positive infinity.
              * TO_ZERO (2): Round toward zero.
              * AWAY (3): Round away from zero.
              * HALF_UP (4): Round to nearest; ties toward positive infinity.
              * HALF_DOWN (5): Round to nearest; ties toward negative infinity.
              * HALF_EVEN (6): Round to nearest; ties to even.
              * HALF_ZERO (7): Round to nearest; ties toward zero.
              * HALF_AWAY (8): Round to nearest; ties away from zero.
        - ovf : int
          Overflow handling method.
          Supported methods (same as in `fxpt`):
              * WRAP (0): Wrap around.
              * SAT (1): Saturate to max/min representable value.
              * ERROR (2): Raise an error on overflow.

    Returns
    -------
    float or ndarray
        Fixed-point representation of the input value(s).

    Notes
    -----
    - Equivalent to:
          >>> fxp(x, Q(qi, qf, signed, rnd, ovf))
          == fxpt(x, qi, qf, signed, rnd, ovf)
    - Uses ARM-style Q-format notation (Qm.n), where:
        * m = `qi`  → number of integer bits (including signed bit if `signed=True`)
        * n = `qf`  → number of fractional bits
    - Designed to be fully compatible with Numba `@njit` mode

    Examples
    --------
    >>> from pyfxp import Q, fxp
    >>> from pyfxp.constants import TRUNC, WRAP
    >>> import numpy as np
    >>> np.pi
    >>> 3.14159265358979
    >>> Q3_4T = Q(3, 4, signed=True, rnd=TRUNC, ovf=WRAP)      # Q3.4 format (3 integer bits, 4 fractional bits)
    >>> fxp(np.pi, Q3_4T)
    3.125
    """
    return fxpt(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)


@njit(cache=True, nogil=True)
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
    # Round-to-nearest modes decide from floor(x) and frac = x - floor(x), which are
    # both exact; x +/- 0.5 is not exact once |x| >= 2**52.
    if method == HALF_UP:
        floor_x = np.floor(x)
        return int(floor_x + (x - floor_x >= RND_MIDPOINT))
    if method == HALF_DOWN:
        floor_x = np.floor(x)
        return int(floor_x + (x - floor_x > RND_MIDPOINT))
    if method == HALF_ZERO:
        floor_x = np.floor(x)
        frac = x - floor_x
        return int(floor_x + (frac > RND_MIDPOINT or (frac == RND_MIDPOINT and floor_x < 0)))
    if method == HALF_AWAY:
        floor_x = np.floor(x)
        frac = x - floor_x
        return int(floor_x + (frac > RND_MIDPOINT or (frac == RND_MIDPOINT and floor_x >= 0)))
    if method == HALF_EVEN:
        floor_x = np.floor(x)
        frac = x - floor_x
        is_half = frac == RND_MIDPOINT
        if is_half:
            return int(floor_x + (floor_x % 2 == 1))
        return int(np.round(x))
    raise ValueError(f"invalid method: {method}")


@njit(cache=True, nogil=True)
def _rnd_array(x, method=TRUNC):
    if method == TRUNC:  # Round towards -inf
        x = np.floor(x)
    elif method == CEIL:  # Round towards +inf
        x = np.ceil(x)
    elif method == TO_ZERO:
        pass
    elif method == AWAY:
        ax = np.abs(x)
        x = np.where(x >= 0, np.ceil(ax), -np.ceil(ax))
    # See _rnd_scalar: decide from floor(x) and the exact fraction, not x +/- 0.5.
    elif method == HALF_UP:
        floor_x = np.floor(x)
        x = floor_x + (x - floor_x >= RND_MIDPOINT)
    elif method == HALF_DOWN:
        floor_x = np.floor(x)
        x = floor_x + (x - floor_x > RND_MIDPOINT)
    elif method == HALF_ZERO:
        floor_x = np.floor(x)
        frac = x - floor_x
        x = floor_x + ((frac > RND_MIDPOINT) | ((frac == RND_MIDPOINT) & (floor_x < 0)))
    elif method == HALF_AWAY:
        floor_x = np.floor(x)
        frac = x - floor_x
        x = floor_x + ((frac > RND_MIDPOINT) | ((frac == RND_MIDPOINT) & (floor_x >= 0)))
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


MAX_WIDTH_SIGNED = 64
"""Widest supported signed format (`qi + qf`), limited by int64."""

MAX_WIDTH_UNSIGNED = 63
"""Widest supported unsigned format (`qi + qf`), limited by int64."""

_INT64_MAX = np.int64(2**63 - 1)
_INT64_MIN = np.int64(-(2**63))


@njit(cache=True, nogil=True)
def _width_params(signed: bool, w: int):
    """Return (half, low, upper, lower) for a w-bit format without ever computing 1 << w.

    `half` is 2**(w-1) and `low` is a mask of w ones. For w <= 63 they are built from
    `1 << (w - 1)` (at most 2**62), so no step overflows int64. A 64-bit signed format
    spans all of int64 and is returned as constants.
    """
    max_w = MAX_WIDTH_SIGNED if signed else MAX_WIDTH_UNSIGNED
    if w < 1 or w > max_w:
        raise ValueError("invalid format width (qi + qf): must be between 1 and 64 (signed) or 63 (unsigned)")
    if w == MAX_WIDTH_SIGNED:
        return _INT64_MIN, np.int64(-1), _INT64_MAX, _INT64_MIN
    half = np.int64(1) << (w - 1)
    low = half - 1 + half
    if signed:
        return half, low, half - 1, -half
    return half, low, low, np.int64(0)


@njit(cache=True, nogil=True)
def _overflow_scalar(x: int, signed: bool = True, w: int = 16, method: int = WRAP):
    half, low, upper, lower = _width_params(signed, w)

    if method == WRAP:
        if w == MAX_WIDTH_SIGNED:
            return x  # every int64 is already a valid 64-bit signed value
        x = x & low
        if signed:
            x = (x ^ half) - half  # sign-extend bit w-1
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


@njit(cache=True, nogil=True)
def _overflow_array(x, signed: bool = True, w: int = 16, method: int = WRAP):
    x = np.asarray(x, dtype=np.int64)
    half, low, upper, lower = _width_params(signed, w)

    if method == WRAP:
        if w == MAX_WIDTH_SIGNED:
            return x  # every int64 is already a valid 64-bit signed value
        x = x & low
        if signed:
            x = (x ^ half) - half  # sign-extend bit w-1
    elif method == SAT:
        x[x > upper] = upper
        x[x < lower] = lower
    elif method == ERROR:
        up = x > upper
        low_ = x < lower
        if np.any(up | low_):
            raise OverflowError("Overflow!")
    else:
        raise ValueError(f"invalid method: {method}")

    return x


@njit(cache=True, nogil=True)
def _into_int64_range_scalar(x: float, w: int, method: int = WRAP) -> float:
    """Bring a scaled value into the int64 range so the int64 cast in rounding is defined.

    Values with |x| >= 2**63 are already integers and lie outside every supported
    format, so:

    - WRAP: reduce modulo 2**w (exact for floats), keeping the bits WRAP keeps.
    - SAT / ERROR: clamp to the int64 range, which still saturates / raises
      in the overflow step.

    NaN is returned unchanged.
    """
    if not (x >= _TWO_63 or x < -_TWO_63):
        return x
    if method == WRAP:
        if np.isinf(x):
            return np.nan  # an infinite value has no defined wrap
        m = 2.0**w
        x = np.fmod(x, m)  # exact; |x| < 2**w afterwards
        if x >= _TWO_63:  # only possible for w == 64
            x -= m
        elif x < -_TWO_63:
            x += m
        return x
    if x > 0:
        return _MAX_BELOW_TWO_63
    return -_TWO_63


@njit(cache=True, nogil=True)
def _into_int64_range_array(x, w: int, method: int = WRAP):
    """Array version of `_into_int64_range_scalar`; only touches out-of-range elements.

    NaN elements (including ±inf under WRAP, which becomes NaN) are set to 0 so the
    integer steps stay defined. Returns `(x, nan_mask, has_nan)`; `nan_mask` is only
    allocated when a NaN is found, so the common case costs one compare per element.
    """
    flat = x.reshape(-1)  # view: x is a fresh contiguous array
    has_nan = False
    nan_mask = np.zeros(0, dtype=np.bool_)
    for i in range(flat.size):
        v = flat[i]
        if v >= _TWO_63 or v < -_TWO_63:
            v = _into_int64_range_scalar(v, w, method)
            flat[i] = v
        if v != v:  # NaN
            if not has_nan:
                has_nan = True
                nan_mask = np.zeros(flat.size, dtype=np.bool_)
            nan_mask[i] = True
            flat[i] = 0.0
    return x, nan_mask, has_nan


@njit(cache=True, nogil=True)
def _fxpt_array(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    x = x * 2.0**qf
    x, nan_mask, has_nan = _into_int64_range_array(x, w=(qi + qf), method=ovf)

    x = _rnd_array(x, method=rnd)
    x = _overflow_array(x, signed=signed, w=(qi + qf), method=ovf)

    out = x / 2.0**qf
    if has_nan:
        out.reshape(-1)[nan_mask] = np.nan  # NaN in, NaN out
    return out


@njit(cache=True, nogil=True)
def _fxpt_scalar(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    x *= 2.0**qf
    x = _into_int64_range_scalar(x, w=(qi + qf), method=ovf)
    if x != x:  # NaN in, NaN out (also ±inf under WRAP)
        return x

    x = _rnd_scalar(x, method=rnd)
    x = _overflow_scalar(x, signed=signed, w=(qi + qf), method=ovf)

    return x / 2.0**qf


def fxpt(  # noqa: PLR0913
    x: Scalar | ArrayLike, qi: int, qf: int, signed: bool = True, rnd: int = TRUNC, ovf: int = WRAP
) -> Any:  # float for scalars, ndarray for arrays (see Returns)
    """Convert a numeric value to fixed-point representation using Q-format notation.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    qi : int
        Number of integer bits (including sign bit if signed=True).
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

        - WRAP (0): Wrap around on overflow.
        - SAT (1): Saturate to maximum/minimum representable value.
        - ERROR (2): Raise an error if overflow occurs.

    Returns:
    -------
    float or ndarray
        Fixed-point representation of the input, as integer(s).

    Notes:
    -----
    Uses ARM-style Q-format notation where a Qm.n format has:
        - m integer bits (qi) (including sign bit if signed=True)
        - n fractional bits (qf)

    Non-finite inputs:
        - NaN is returned unchanged, for every rounding and overflow mode.
        - ±inf saturates to the max/min value with SAT, raises with ERROR,
          and returns NaN with WRAP (an infinite value has no defined wrap).

    Examples
    --------
    >>> from pyfxp import fxpt
    >>> from pyfxp.constants import TRUNC, WRAP
    >>> import numpy as np
    >>> np.pi
    >>> 3.14159265358979
    >>> fxpt(np.pi, 3, 4, signed=True, rnd=TRUNC, ovf=WRAP)  # Q3.4 format (3 integer bits, 4 fractional bits)
    3.125
    """
    if isinstance(x, np.ndarray):
        return _fxpt_array(x, qi, qf, signed, rnd, ovf)

    if isinstance(x, float | int | np.integer | np.floating):
        # float() gives one compiled specialization for every scalar type (and handles
        # float16, which Numba can't type); the scaling step converts to float anyway.
        return _fxpt_scalar(float(x), qi, qf, signed, rnd, ovf)

    if isinstance(x, list | tuple):
        return _fxpt_array(np.asarray(x, dtype=np.float64), qi, qf, signed, rnd, ovf)

    raise TypeError(f"Unsupported type: {x}")


# Numba's overload is used to decide when to call the array or the scalar function.
@overload(fxpt)
def fxp_overload(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913, ARG001
    # Array case
    if isinstance(x, types.Array):

        def impl(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
            return _fxpt_array(x, qi, qf, signed, rnd, ovf)  # pragma: no cover

        return impl

    if isinstance(x, types.Integer | types.Float):

        def impl(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
            return _fxpt_scalar(x, qi, qf, signed, rnd, ovf)  # pragma: no cover

        return impl

    if isinstance(x, types.List | types.UniTuple):

        def impl(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
            return _fxpt_array(np.asarray(x, dtype=np.float64), qi, qf, signed, rnd, ovf)  # pragma: no cover

        return impl
    raise TypeError(f"Unsupported type: {x}")


@overload(fxp)
def _fxp_overload(x, spec):  # noqa: ARG001
    """Make `fxp` callable from user @njit code (it is a plain function at Python level)."""

    def impl(x, spec):
        return fxpt(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)  # pragma: no cover

    return impl
