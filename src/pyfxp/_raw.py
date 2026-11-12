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
"""Raw integer conversion: the stored integer of a fixed-point value, and bin/hex strings."""

import math

import numpy as np
from numba import types
from numba.extending import overload

from ._pyfxp import (
    ArrayLike,
    FxpSpec,
    Scalar,
    _check_width,
    _from_int_array,
    _from_int_scalar,
    _to_int_array,
    _to_int_scalar,
    _validated_modes,
)
from .constants import SAT, WRAP


def _from_int_py(n: int, spec: FxpSpec, ovf: int) -> float:
    """Scalar from_int with exact Python ints (any size)."""
    lo, hi = spec.min_int, spec.max_int
    if n < lo or n > hi:
        if ovf == WRAP:
            n = (n - lo) % (1 << spec.width) + lo
        elif ovf == SAT:
            n = min(max(n, lo), hi)
        else:
            raise OverflowError("Overflow!")
    return math.ldexp(n, -spec.qf)


# ----------------------------------------------------------------------------- public API


def to_int(x: Scalar | ArrayLike, spec: FxpSpec):
    """Quantize like `fxp` and return the stored integer instead of the real value.

    The stored integer `n` and the real value `y = fxp(x, spec)` are related by
    `y = n * 2**-qf`. Unlike the float result, `n` is exact for every supported
    width (up to 64 bits signed / 63 bits unsigned).

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    spec : FxpSpec
        The fixed-point format, rounding and overflow modes (see `Q`).

    Returns
    -------
    int or ndarray
        A Python `int` for scalar inputs, an int64 array (same shape) for array,
        list and tuple inputs. Signed formats give negative integers; use `to_bin`
        or `to_hex` for the two's-complement bit pattern.

    Raises
    ------
    ValueError
        For NaN input (and +-inf with WRAP), which has no integer representation.

    Examples
    --------
    >>> from pyfxp import Q, fxp, to_int
    >>> spec = Q(3, 4)  # sQ3.4, lsb = 1/16
    >>> fxp(-1.3, spec)
    -1.3125
    >>> to_int(-1.3, spec)
    -21
    >>> to_int([0.5, -0.5], spec)
    array([ 8, -8])
    """
    rnd, ovf = _validated_modes(spec.rnd, spec.ovf)
    if isinstance(x, np.ndarray):
        return _to_int_array(x, spec.qi, spec.qf, spec.signed, rnd, ovf)
    if isinstance(x, float | int | np.integer | np.floating):
        return int(_to_int_scalar(float(x), spec.qi, spec.qf, spec.signed, rnd, ovf))
    if isinstance(x, list | tuple):
        return _to_int_array(np.asarray(x, dtype=np.float64), spec.qi, spec.qf, spec.signed, rnd, ovf)
    raise TypeError(f"Unsupported type: {x}")


def from_int(n, spec: FxpSpec):
    """Convert stored integer(s) back to the real value(s): `n * 2**-qf`.

    Integers outside the format's range are handled with the spec's overflow
    mode, so `from_int(to_int(x, spec), spec) == fxp(x, spec)`.

    Parameters
    ----------
    n : int or array-like of int
        Stored integer(s). Floats are rejected.
    spec : FxpSpec
        The fixed-point format and overflow mode (`rnd` is not used).

    Returns
    -------
    float or ndarray
        A float for scalar inputs, a float64 array (same shape) otherwise.

    Examples
    --------
    >>> from pyfxp import Q, from_int
    >>> from_int(-21, Q(3, 4))
    -1.3125
    >>> from_int([8, -8], Q(3, 4))
    array([ 0.5, -0.5])
    """
    _, ovf = _validated_modes(spec.rnd, spec.ovf)
    _check_width(spec.signed, spec.width)
    if isinstance(n, int | np.integer):
        return _from_int_py(int(n), spec, ovf)
    if isinstance(n, np.ndarray | list | tuple):
        arr = np.asarray(n)
        if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"from_int expects integers, got an array of {arr.dtype}")
        if arr.dtype == np.uint64 and arr.size and arr.max() > np.iinfo(np.int64).max:
            raise ValueError("from_int: array values above 2**63 - 1 are not supported")
        return _from_int_array(arr, spec.qi, spec.qf, spec.signed, ovf)
    raise TypeError(f"from_int expects integers, got {type(n).__name__}")


def _pattern_strings(x, spec: FxpSpec, base: str, digits: int):
    n = to_int(x, spec)
    mask = (1 << spec.width) - 1
    if isinstance(n, int):
        return format(n & mask, f"0{digits}{base}")
    out = [format(v & mask, f"0{digits}{base}") for v in n.ravel().tolist()]
    return np.array(out, dtype=f"<U{digits}").reshape(n.shape)


def to_bin(x: Scalar | ArrayLike, spec: FxpSpec):
    """Quantize like `fxp` and return the two's-complement bit pattern as a binary string.

    The string has exactly `spec.width` digits.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    spec : FxpSpec
        The fixed-point format, rounding and overflow modes (see `Q`).

    Returns
    -------
    str or ndarray
        A string for scalar inputs, a string array of the same shape otherwise.

    Examples
    --------
    >>> from pyfxp import Q, to_bin
    >>> to_bin(-1.3, Q(3, 4))
    '1101011'
    >>> to_bin([0.5, -0.5], Q(3, 4))
    array(['0001000', '1111000'], dtype='<U7')
    """
    return _pattern_strings(x, spec, "b", spec.width)


def to_hex(x: Scalar | ArrayLike, spec: FxpSpec):
    """Quantize like `fxp` and return the two's-complement bit pattern as a hex string.

    The string is zero-padded to `ceil(width / 4)` lowercase digits, the format
    `$readmemh` and VHDL `hread` expect.

    Parameters
    ----------
    x : int, float or array-like
        The input value(s) to convert.
    spec : FxpSpec
        The fixed-point format, rounding and overflow modes (see `Q`).

    Returns
    -------
    str or ndarray
        A string for scalar inputs, a string array of the same shape otherwise.

    Examples
    --------
    >>> from pyfxp import Q, to_hex
    >>> to_hex(-1.3, Q(3, 4))
    '6b'
    >>> to_hex(-1.0, Q(1, 15))
    '8000'
    """
    return _pattern_strings(x, spec, "x", (spec.width + 3) // 4)


def _from_pattern(s, spec: FxpSpec, base: int):
    w = spec.width
    _check_width(spec.signed, w)

    def one(text: str) -> float:
        v = int(text.strip().replace("_", ""), base)
        if v < 0 or v >> w:
            raise ValueError(f"{text!r} does not fit in {w} bits")
        if spec.signed and v >> (w - 1):
            v -= 1 << w  # sign bit set: negative two's-complement value
        return math.ldexp(v, -spec.qf)

    if isinstance(s, str):
        return one(s)
    arr = np.asarray(s)
    return np.array([one(t) for t in arr.ravel().tolist()], dtype=np.float64).reshape(arr.shape)


def from_bin(s, spec: FxpSpec):
    """Real value(s) of two's-complement binary string(s) in the given format.

    Parameters
    ----------
    s : str or array-like of str
        Two's-complement bit pattern(s). Underscores and a `0b` prefix are allowed.
    spec : FxpSpec
        The fixed-point format (`rnd` and `ovf` are not used).

    Returns
    -------
    float or ndarray
        A float for a single string, a float64 array (same shape) otherwise.

    Raises
    ------
    ValueError
        If a string has more than `spec.width` significant bits.

    Examples
    --------
    >>> from pyfxp import Q, from_bin
    >>> from_bin("1101011", Q(3, 4))
    -1.3125
    """
    return _from_pattern(s, spec, 2)


def from_hex(s, spec: FxpSpec):
    """Real value(s) of two's-complement hex string(s) in the given format.

    Parameters
    ----------
    s : str or array-like of str
        Two's-complement bit pattern(s). Underscores and a `0x` prefix are allowed.
    spec : FxpSpec
        The fixed-point format (`rnd` and `ovf` are not used).

    Returns
    -------
    float or ndarray
        A float for a single string, a float64 array (same shape) otherwise.

    Raises
    ------
    ValueError
        If a string has more than `spec.width` significant bits.

    Examples
    --------
    >>> from pyfxp import Q, from_hex
    >>> from_hex("6b", Q(3, 4))
    -1.3125
    >>> from_hex(["0008", "fff8"], Q(12, 4))
    array([ 0.5, -0.5])
    """
    return _from_pattern(s, spec, 16)


# ----------------------------------------------------------------------------- @njit support


@overload(to_int, inline="always")
def _to_int_overload(x, spec):
    """Make `to_int` callable from user @njit code."""
    if isinstance(x, types.Array):

        def impl(x, spec):
            return _to_int_array(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)  # pragma: no cover

        return impl
    if isinstance(x, types.Integer | types.Float):

        def impl(x, spec):
            return _to_int_scalar(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)  # pragma: no cover

        return impl
    if isinstance(x, types.List | types.UniTuple):

        def impl(x, spec):
            a = np.asarray(x, dtype=np.float64)  # pragma: no cover
            return _to_int_array(a, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)  # pragma: no cover

        return impl
    raise TypeError(f"Unsupported type: {x}")


@overload(from_int, inline="always")
def _from_int_overload(n, spec):
    """Make `from_int` callable from user @njit code (int64 inputs)."""
    if isinstance(n, types.Array) and isinstance(n.dtype, types.Integer | types.Boolean):

        def impl(n, spec):
            return _from_int_array(n, spec.qi, spec.qf, spec.signed, spec.ovf)  # pragma: no cover

        return impl
    if isinstance(n, types.Integer | types.Boolean):

        def impl(n, spec):
            return _from_int_scalar(n, spec.qi, spec.qf, spec.signed, spec.ovf)  # pragma: no cover

        return impl
    raise TypeError(f"from_int expects integers, got {n}")


__all__ = ["from_bin", "from_hex", "from_int", "to_bin", "to_hex", "to_int"]
