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


_MAX_SHIFT_QF = 62  # largest qf whose scale 2**qf can be built with an int64 shift


@njit(cache=True, nogil=True)
def _check_modes(rnd: int, ovf: int):
    """Raise ValueError for an unknown rounding or overflow method.

    The messages are constants on purpose: formatting the value into the message
    (an f-string) costs about 2 ns per call even when nothing is raised. Calls from
    Python get the value in the message from `fxpt` itself.
    """
    if rnd < TRUNC or rnd > HALF_AWAY:
        raise ValueError("invalid method: unknown rounding method (see pyfxp.constants)")
    if ovf < WRAP or ovf > ERROR:
        raise ValueError("invalid method: unknown overflow method (see pyfxp.constants)")


@njit(cache=True, nogil=True)
def _scale(qf: int) -> float:
    """Return 2.0**qf. An integer shift is much cheaper than pow() and lets LLVM vectorize callers."""
    if 0 <= qf <= _MAX_SHIFT_QF:
        return float(np.int64(1) << qf)
    return 2.0**qf


@njit(cache=True, nogil=True)
def _quantize(x, scale: float, w: int, signed: bool, rnd: int, ovf: int):  # noqa: C901, PLR0911, PLR0912, PLR0913
    """Quantize one value: the single kernel behind the scalar and the array paths.

    `scale` is 2.0**qf and `w` is qi + qf. `rnd` and `ovf` must have been validated
    with `_check_modes`. Returns a float on the 2**-qf grid (NaN for NaN input).
    """
    v = x * scale

    # Rare cases first: NaN, +-inf and values outside the int64 range.
    if not (-_TWO_63 <= v < _TWO_63):
        v = _into_int64_range_scalar(v, w, ovf)
        if v != v:  # NaN in, NaN out (also +-inf under WRAP)
            return v

    # Rounding, in floating point. floor(v) and v - floor(v) are both exact, which
    # v +/- 0.5 is not once |v| >= 2**52.
    if rnd == TRUNC:
        r = np.floor(v)
    elif rnd == CEIL:
        r = np.ceil(v)
    elif rnd == TO_ZERO:
        r = np.trunc(v)
    elif rnd == HALF_EVEN:
        r = np.rint(v)  # IEEE round-to-nearest-even
    else:
        r = np.floor(v)
        frac = v - r
        # Bitwise & and | keep these branch-free; the data-dependent branches of
        # `and` / `or` cost several ns per sample.
        if rnd == AWAY:
            r += (frac > 0.0) & (r >= 0.0)
        elif rnd == HALF_UP:
            r += frac >= RND_MIDPOINT
        elif rnd == HALF_DOWN:
            r += frac > RND_MIDPOINT
        elif rnd == HALF_ZERO:
            r += (frac > RND_MIDPOINT) | ((frac == RND_MIDPOINT) & (r < 0.0))
        else:  # HALF_AWAY
            r += (frac > RND_MIDPOINT) | ((frac == RND_MIDPOINT) & (r >= 0.0))

    # r can be exactly 2**63 here (rounding up from just below it); keep the cast defined.
    n = _INT64_MAX if r >= _TWO_63 else np.int64(r)

    # Overflow, in int64.
    half, low, upper, lower = _width_params(signed, w)
    if ovf == SAT:
        if n > upper:
            n = upper
        elif n < lower:
            n = lower
    elif ovf == WRAP:
        if w != MAX_WIDTH_SIGNED:  # every int64 is already a valid 64-bit signed value
            n = n & low
            if signed:
                n = (n ^ half) - half  # sign-extend bit w-1
    elif n > upper or n < lower:  # ERROR
        raise OverflowError("Overflow!")

    return n / scale


@njit(cache=True, nogil=True)
def _fxpt_array(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    """Quantize an array of any shape in one pass: one read and one write per element."""
    _check_modes(rnd, ovf)
    w = qi + qf
    _width_params(signed, w)  # validate the width once, also for empty arrays
    scale = _scale(qf)

    out = np.empty(x.shape, dtype=np.float64)
    flat = out.reshape(-1)
    i = 0
    for v in x.flat:
        flat[i] = _quantize(v, scale, w, signed, rnd, ovf)
        i += 1
    return out


@njit(cache=True, nogil=True, inline="always")
def _fxpt_scalar(x, qi: int, qf: int, signed: bool = True, rnd=TRUNC, ovf=WRAP):  # noqa: PLR0913
    _check_modes(rnd, ovf)
    return _quantize(x, _scale(qf), qi + qf, signed, rnd, ovf)


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
    if not TRUNC <= rnd <= HALF_AWAY:
        raise ValueError(f"invalid method: {rnd}")
    if not WRAP <= ovf <= ERROR:
        raise ValueError(f"invalid method: {ovf}")

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
@overload(fxpt, inline="always")
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


@overload(fxp, inline="always")
def _fxp_overload(x, spec):  # noqa: ARG001
    """Make `fxp` callable from user @njit code (it is a plain function at Python level)."""

    def impl(x, spec):
        return fxpt(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)  # pragma: no cover

    return impl
