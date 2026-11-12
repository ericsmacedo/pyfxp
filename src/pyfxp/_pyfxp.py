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


import math
from collections.abc import Sequence
from typing import Any, NamedTuple, TypeAlias

import numpy as np
from numba import njit, types
from numba.extending import overload, overload_attribute

from .constants import (
    AWAY,
    CEIL,
    ERROR,
    HALF_AWAY,
    HALF_DOWN,
    HALF_EVEN,
    HALF_UP,
    HALF_ZERO,
    SAT,
    TO_ZERO,
    TRUNC,
    WRAP,
    Overflow,
    Rounding,
    overflow_modes_inv,
    rounding_modes_inv,
)

RND_MIDPOINT = 0.5  # Threshold for half-way rounding

Scalar: TypeAlias = float | int | np.integer | np.floating
"""Scalar inputs: Python and NumPy integers and floats. They return a float."""

ArrayLike: TypeAlias = np.ndarray | Sequence[float]
"""Array inputs: NumPy arrays, lists and tuples. They return a float64 ndarray."""
_TWO_63 = 2.0**63  # first float64 outside the int64 range


class FxpSpec(NamedTuple):
    """Fixed-point specification.

    Usually created with `Q(...)`. Besides the five fields it has read-only
    properties describing the format, which also work inside `@njit` code:

    - `width`: total number of bits, `qi + qf`
    - `lsb`: step between representable values, `2**-qf`
    - `min_int`, `max_int`: range of the stored integer (exact Python ints)
    - `min`, `max`: range of the real value, `min_int * lsb` and `max_int * lsb`

    For formats wider than 53 bits, `min` and `max` are rounded to the nearest
    float64; `min_int` and `max_int` stay exact.

    The repr uses Q notation: `s` or `u` for signed/unsigned, `Qqi.qf`, then the
    rounding and overflow modes. Use `spec._asdict()` for the field values.

    Examples
    --------
    >>> from pyfxp import Q
    >>> from pyfxp.constants import HALF_EVEN, SAT
    >>> spec = Q(3, 13, True, HALF_EVEN, SAT)
    >>> spec
    sQ3.13(HALF_EVEN, SAT)
    >>> spec.width, spec.lsb
    (16, 0.0001220703125)
    >>> spec.min_int, spec.max_int
    (-32768, 32767)
    >>> spec.min, spec.max
    (-4.0, 3.9998779296875)
    >>> Q(8, 0, signed=False)
    uQ8.0(TRUNC, WRAP)
    """

    qi: int
    qf: int
    signed: bool
    rnd: int
    ovf: int

    def __repr__(self) -> str:
        rnd = rounding_modes_inv.get(self.rnd, repr(self.rnd))
        ovf = overflow_modes_inv.get(self.ovf, repr(self.ovf))
        return f"{'s' if self.signed else 'u'}Q{self.qi}.{self.qf}({rnd}, {ovf})"

    @property
    def width(self) -> int:
        """Total number of bits, `qi + qf` (the sign bit is part of `qi`)."""
        return self.qi + self.qf

    @property
    def lsb(self) -> float:
        """Step between representable values, `2**-qf`."""
        return math.ldexp(1.0, -self.qf)

    @property
    def min_int(self) -> int:
        """Smallest stored integer: `-2**(width-1)` if signed, else `0`."""
        _check_width(self.signed, self.width)
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max_int(self) -> int:
        """Largest stored integer: `2**(width-1) - 1` if signed, else `2**width - 1`."""
        _check_width(self.signed, self.width)
        return (1 << (self.width - 1)) - 1 if self.signed else (1 << self.width) - 1

    @property
    def min(self) -> float:
        """Smallest representable value, `min_int * lsb`."""
        return math.ldexp(self.min_int, -self.qf)

    @property
    def max(self) -> float:
        """Largest representable value, `max_int * lsb` (rounded to float64 above 53 bits)."""
        return math.ldexp(self.max_int, -self.qf)

    # Result formats of arithmetic (bit growth). Each method returns the smallest
    # format that holds every possible exact result, so converting the result
    # with it never rounds or overflows. rnd/ovf are taken from `self`.

    def add(self, other: "FxpSpec") -> "FxpSpec":
        """Full-precision format of `a + b` (a value of `self` plus a value of `other`).

        >>> from pyfxp import Q
        >>> Q(4, 5).add(Q(7, 5))
        sQ8.5(TRUNC, WRAP)
        >>> Q(4, 5, signed=False).add(Q(4, 5, signed=False))
        uQ5.5(TRUNC, WRAP)
        """
        qf, (lo_a, hi_a), (lo_b, hi_b) = _aligned(self, other)
        return _fit("add", lo_a + lo_b, hi_a + hi_b, qf, self)

    def sub(self, other: "FxpSpec") -> "FxpSpec":
        """Full-precision format of `a - b`. Always signed unless the result can't be negative.

        >>> from pyfxp import Q
        >>> Q(4, 5).sub(Q(4, 5))
        sQ5.5(TRUNC, WRAP)
        >>> Q(4, 5, signed=False).sub(Q(4, 5, signed=False))
        sQ5.5(TRUNC, WRAP)
        """
        qf, (lo_a, hi_a), (lo_b, hi_b) = _aligned(self, other)
        return _fit("sub", lo_a - hi_b, hi_a - lo_b, qf, self)

    def mul(self, other: "FxpSpec") -> "FxpSpec":
        """Full-precision format of `a * b`: the fractional bits add up.

        >>> from pyfxp import Q
        >>> Q(4, 5).mul(Q(7, 5))
        sQ11.10(TRUNC, WRAP)
        >>> Q(1, 15).mul(Q(1, 15))
        sQ2.30(TRUNC, WRAP)
        """
        corners = [x * y for x in (self.min_int, self.max_int) for y in (other.min_int, other.max_int)]
        return _fit("mul", min(corners), max(corners), self.qf + other.qf, self)

    def neg(self) -> "FxpSpec":
        """Full-precision format of `-a`. Needs one more bit, because `-min` is not representable.

        >>> from pyfxp import Q
        >>> Q(4, 5).neg()
        sQ5.5(TRUNC, WRAP)
        """
        return _fit("neg", -self.max_int, -self.min_int, self.qf, self)

    def acc(self, n: int) -> "FxpSpec":
        """Full-precision format of a sum of `n` values of this format (an accumulator).

        >>> from pyfxp import Q
        >>> Q(4, 5).acc(1024)
        sQ14.5(TRUNC, WRAP)
        """
        if n < 1:
            raise ValueError(f"acc needs n >= 1, got {n}")
        return _fit("acc", n * self.min_int, n * self.max_int, self.qf, self)


def _aligned(a: FxpSpec, b: FxpSpec) -> tuple[int, tuple[int, int], tuple[int, int]]:
    """Common qf of `a` and `b`, and both integer ranges expressed in units of 2**-qf."""
    qf = max(a.qf, b.qf)
    sa, sb = qf - a.qf, qf - b.qf
    return qf, (a.min_int << sa, a.max_int << sa), (b.min_int << sb, b.max_int << sb)


def _fit(op: str, lo: int, hi: int, qf: int, like: FxpSpec) -> FxpSpec:
    """Smallest format with `qf` fractional bits whose integer range contains [lo, hi]."""
    signed = lo < 0
    if signed:
        w = max((-lo - 1).bit_length(), hi.bit_length()) + 1
    else:
        w = max(hi.bit_length(), 1)
    limit = MAX_WIDTH_SIGNED if signed else MAX_WIDTH_UNSIGNED
    if w > limit:
        kind = "signed" if signed else "unsigned"
        raise ValueError(
            f"{op}: the full-precision result needs {w} bits, more than the {limit}-bit limit for {kind} "
            "formats; quantize an operand to a narrower format first"
        )
    return FxpSpec(w - qf, qf, signed, like.rnd, like.ovf)


def _check_mode_kinds(rnd, ovf) -> None:
    """Raise TypeError when a rounding and an overflow mode are swapped.

    TRUNC and WRAP are both 0 and CEIL and SAT are both 1, so a swap would
    otherwise be accepted silently. Plain integers are not checked here.
    """
    if isinstance(rnd, Overflow):
        raise TypeError(f"rnd={rnd!r} is an overflow mode; did you swap rnd and ovf?")
    if isinstance(ovf, Rounding):
        raise TypeError(f"ovf={ovf!r} is a rounding mode; did you swap rnd and ovf?")


def _validated_modes(rnd, ovf) -> tuple[int, int]:
    """Validate rnd/ovf at Python level and return them as plain ints.

    Plain ints for the compiled kernels: Numba's dispatcher types IntEnum
    arguments ~10x slower than ints, which dominated scalar calls from Python.
    """
    _check_mode_kinds(rnd, ovf)
    if not TRUNC <= rnd <= HALF_AWAY:
        raise ValueError(f"invalid method: {rnd}")
    if not WRAP <= ovf <= ERROR:
        raise ValueError(f"invalid method: {ovf}")
    return int(rnd), int(ovf)


def Q(qi: int, qf: int, signed: bool = True, rnd: int | np.integer = TRUNC, ovf: int | np.integer = WRAP) -> FxpSpec:
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
    rnd : Rounding or int, optional
        Rounding method to apply (default is TRUNC (0)). Integer codes are
        converted to `Rounding` members, so the returned spec prints mode names.

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

    ovf : Overflow or int, optional
        Overflow handling method (default is WRAP (0)). Integer codes are
        converted to `Overflow` members.

        Supported methods:

        - WRAP (0): Wrap around on overflow (modulo behavior)
        - SAT (1): Saturate to maximum/minimum representable value
        - ERROR (2): Raise an error if overflow occurs

    Returns
    -------
    FxpSpec
        A specification object describing the fixed-point format.

    Raises
    ------
    ValueError
        If `rnd` or `ovf` is not a valid mode code.
    TypeError
        If an `Overflow` member is passed as `rnd` or a `Rounding` member as `ovf`
        (a likely swap: TRUNC and WRAP are both 0, CEIL and SAT are both 1).

    Notes
    -----
    This function uses ARM-style Q-format notation:

    - `Qm.n` indicates `m` integer bits and `n` fractional bits (ARM style).
    - The returned `FxpSpec` can be reused for consistent conversions across multiple values.

    Examples
    --------
    >>> from pyfxp import Q, fxp
    >>> spec = Q(3, 1)  # Q3.1 format (3 integer bits, 1 fractional bit)
    >>> spec.qi, spec.qf
    (3, 1)
    >>> fxp(1.25, spec)
    1.0
    """
    _check_mode_kinds(rnd, ovf)
    try:
        rnd = Rounding(rnd)
    except ValueError:
        raise ValueError(f"invalid method: {rnd}") from None
    try:
        ovf = Overflow(ovf)
    except ValueError:
        raise ValueError(f"invalid method: {ovf}") from None
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
        A fixed-point specification (usually created with `Q`) with the fields:

        - `qi` (int): number of integer bits, including the sign bit if `signed=True`.
        - `qf` (int): number of fractional bits.
        - `signed` (bool): whether the format is signed.
        - `rnd` (Rounding): rounding method, as in `fxpt`.
        - `ovf` (Overflow): overflow handling method, as in `fxpt`.

    Returns
    -------
    float or ndarray
        Fixed-point representation of the input value(s).

    Notes
    -----
    - `fxp(x, Q(qi, qf, signed, rnd, ovf))` is equivalent to
      `fxpt(x, qi, qf, signed, rnd, ovf)`.
    - Uses ARM-style Q-format notation (Qm.n), where:
        * m = `qi`  → number of integer bits (including signed bit if `signed=True`)
        * n = `qf`  → number of fractional bits
    - Designed to be fully compatible with Numba `@njit` mode

    Examples
    --------
    >>> from pyfxp import Q, fxp
    >>> from pyfxp.constants import TRUNC, WRAP
    >>> import math
    >>> math.pi
    3.141592653589793
    >>> Q3_4T = Q(3, 4, signed=True, rnd=TRUNC, ovf=WRAP)  # Q3.4 format (3 integer bits, 4 fractional bits)
    >>> fxp(math.pi, Q3_4T)
    3.125
    """
    return fxpt(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf)


MAX_WIDTH_SIGNED = 64
"""Widest supported signed format (`qi + qf`), limited by int64."""

MAX_WIDTH_UNSIGNED = 63
"""Widest supported unsigned format (`qi + qf`), limited by int64."""

_INT64_MAX = np.int64(2**63 - 1)
_INT64_MIN = np.int64(-(2**63))


_WIDTH_ERROR = "invalid format width (qi + qf): must be between 1 and 64 (signed) or 63 (unsigned)"


def _check_width(signed: bool, w: int) -> None:
    """Python-level version of the width check in `_width_params`."""
    if w < 1 or w > (MAX_WIDTH_SIGNED if signed else MAX_WIDTH_UNSIGNED):
        raise ValueError(_WIDTH_ERROR)


@njit(cache=True, nogil=True)
def _width_params(signed: bool, w: int):
    """Return (half, low, upper, lower) for a w-bit format without ever computing 1 << w.

    `half` is 2**(w-1) and `low` is a mask of w ones. For w <= 63 they are built from
    `1 << (w - 1)` (at most 2**62), so no step overflows int64. A 64-bit signed format
    spans all of int64 and is returned as constants.
    """
    max_w = MAX_WIDTH_SIGNED if signed else MAX_WIDTH_UNSIGNED
    if w < 1 or w > max_w:
        raise ValueError(_WIDTH_ERROR)
    if w == MAX_WIDTH_SIGNED:
        return _INT64_MIN, np.int64(-1), _INT64_MAX, _INT64_MIN
    half = np.int64(1) << (w - 1)
    low = half - 1 + half
    if signed:
        return half, low, half - 1, -half
    return half, low, low, np.int64(0)


@njit(cache=True, nogil=True)
def _wrap_huge(x: float, w: int) -> float:
    """WRAP a scaled value with |x| >= 2**63 into the int64 range, keeping the bits WRAP keeps.

    Such values are already integers and lie outside every supported format, so
    reducing them modulo 2**w (exact for floats) gives the same result as wrapping
    the exact integer. Only called by `_quantize_int`, for non-NaN values outside the
    int64 range under WRAP. +-inf gives NaN (an infinite value has no defined wrap).
    """
    if np.isinf(x):
        return np.nan
    m = 2.0**w
    x = np.fmod(x, m)  # exact; |x| < 2**w afterwards
    if x >= _TWO_63:  # only possible for w == 64
        x -= m
    elif x < -_TWO_63:
        x += m
    return x


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


@njit(cache=True, nogil=True, inline="always")
def _apply_overflow(n, w: int, signed: bool, ovf: int):
    """Apply WRAP / SAT / ERROR to an int64 for a `w`-bit format."""
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
    return n


@njit(cache=True, nogil=True, inline="always")
def _quantize_int(x, scale: float, w: int, signed: bool, rnd: int, ovf: int):  # noqa: C901, PLR0912, PLR0913
    """Quantize one value to its stored integer: the single kernel behind every conversion.

    `scale` is 2.0**qf and `w` is qi + qf. `rnd` and `ovf` must have been validated
    with `_check_modes`. Returns `(n, ok)`: the int64 stored integer, and False if
    the input was NaN (or +-inf under WRAP), in which case `n` is meaningless.
    """
    v = x * scale

    # Rare cases first: NaN, +-inf and values outside the int64 range. Those values
    # are beyond every supported format (at most 64 bits), so SAT / ERROR are decided
    # here: the int64 range check below could not see them for a 64-bit format.
    if not (-_TWO_63 <= v < _TWO_63):
        if v != v:  # NaN
            return np.int64(0), False
        if ovf == WRAP:
            v = _wrap_huge(v, w)
            if v != v:  # +-inf under WRAP
                return np.int64(0), False
        elif ovf == SAT:
            _, _, upper, lower = _width_params(signed, w)
            return (upper if v > 0 else lower), True
        else:  # ERROR
            raise OverflowError("Overflow!")

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
    return _apply_overflow(n, w, signed, ovf), True


@njit(cache=True, nogil=True)
def _quantize(x, scale: float, w: int, signed: bool, rnd: int, ovf: int):  # noqa: PLR0913
    """Quantize one value: a float on the 2**-qf grid (NaN in, NaN out)."""
    n, ok = _quantize_int(x, scale, w, signed, rnd, ovf)
    if not ok:
        return np.nan
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


# Raw-integer kernels (used by pyfxp._raw). They live in this file because Numba's
# cache only notices changes to the file a cached function is defined in, not to
# functions it inlines from other files.

_NAN_ERROR = "NaN (or +-inf with WRAP) has no fixed-point integer representation"


@njit(cache=True, nogil=True)
def _to_int_scalar(x, qi: int, qf: int, signed: bool, rnd: int, ovf: int):  # noqa: PLR0913
    _check_modes(rnd, ovf)
    n, ok = _quantize_int(x, _scale(qf), qi + qf, signed, rnd, ovf)
    if not ok:
        raise ValueError(_NAN_ERROR)
    return n


@njit(cache=True, nogil=True)
def _to_int_array(x, qi: int, qf: int, signed: bool, rnd: int, ovf: int):  # noqa: PLR0913
    _check_modes(rnd, ovf)
    w = qi + qf
    _width_params(signed, w)  # validate the width once, also for empty arrays
    scale = _scale(qf)
    out = np.empty(x.shape, dtype=np.int64)
    flat = out.reshape(-1)
    i = 0
    for v in x.flat:
        n, ok = _quantize_int(v, scale, w, signed, rnd, ovf)
        if not ok:
            raise ValueError(_NAN_ERROR)
        flat[i] = n
        i += 1
    return out


@njit(cache=True, nogil=True)
def _from_int_array(n, qi: int, qf: int, signed: bool, ovf: int):
    _check_modes(0, ovf)
    w = qi + qf
    _width_params(signed, w)
    scale = _scale(qf)
    out = np.empty(n.shape, dtype=np.float64)
    flat = out.reshape(-1)
    i = 0
    for v in n.flat:
        flat[i] = _apply_overflow(np.int64(v), w, signed, ovf) / scale
        i += 1
    return out


@njit(cache=True, nogil=True)
def _from_int_scalar(n, qi: int, qf: int, signed: bool, ovf: int):
    _check_modes(0, ovf)
    return _apply_overflow(np.int64(n), qi + qf, signed, ovf) / _scale(qf)


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
    rnd : Rounding or int, optional
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

    ovf : Overflow or int, optional
        Overflow handling method (default is WRAP (0)).

        Supported methods:

        - WRAP (0): Wrap around on overflow.
        - SAT (1): Saturate to maximum/minimum representable value.
        - ERROR (2): Raise an error if overflow occurs.

    Returns
    -------
    float or ndarray
        The quantized value(s), as float(s) on the `2**-qf` grid: a float for
        scalar inputs, a float64 array for array, list and tuple inputs.

    Notes
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
    >>> import math
    >>> math.pi
    3.141592653589793
    >>> fxpt(math.pi, 3, 4, signed=True, rnd=TRUNC, ovf=WRAP)  # Q3.4 format (3 integer bits, 4 fractional bits)
    3.125
    """
    rnd, ovf = _validated_modes(rnd, ovf)

    if isinstance(x, np.ndarray):
        return _fxpt_array(x, qi, qf, signed, rnd, ovf)

    if isinstance(x, float | int | np.integer | np.floating):
        # float() gives one compiled specialization for every scalar type (and handles
        # float16, which Numba can't type); the scaling step converts to float anyway.
        # float(): with NUMBA_DISABLE_JIT=1 the kernel returns np.float64
        return float(_fxpt_scalar(float(x), qi, qf, signed, rnd, ovf))

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


@overload(Q, inline="always")
def _q_overload(qi, qf, signed=True, rnd=TRUNC, ovf=WRAP):  # noqa: ARG001
    """Make `Q` callable from user @njit code (it is a plain function at Python level)."""

    def impl(qi, qf, signed=True, rnd=TRUNC, ovf=WRAP):
        return FxpSpec(qi, qf, signed, rnd, ovf)  # pragma: no cover

    return impl


# FxpSpec properties inside @njit code. Plain Python properties are invisible to
# Numba, so each one is registered again for both NamedTuple flavors Numba may
# infer (NamedUniTuple when all five fields share a type).
def _is_fxpspec(spec) -> bool:
    return getattr(spec, "instance_class", None) is FxpSpec


for _tuple_type in (types.NamedTuple, types.NamedUniTuple):

    @overload_attribute(_tuple_type, "width", inline="always")
    def _spec_width(spec):
        if _is_fxpspec(spec):
            return lambda spec: spec.qi + spec.qf  # pragma: no cover
        return None

    @overload_attribute(_tuple_type, "lsb", inline="always")
    def _spec_lsb(spec):
        if _is_fxpspec(spec):
            return lambda spec: 1.0 / _scale(spec.qf)  # pragma: no cover
        return None

    @overload_attribute(_tuple_type, "min_int", inline="always")
    def _spec_min_int(spec):
        if _is_fxpspec(spec):
            return lambda spec: _width_params(spec.signed, spec.qi + spec.qf)[3]  # pragma: no cover
        return None

    @overload_attribute(_tuple_type, "max_int", inline="always")
    def _spec_max_int(spec):
        if _is_fxpspec(spec):
            return lambda spec: _width_params(spec.signed, spec.qi + spec.qf)[2]  # pragma: no cover
        return None

    @overload_attribute(_tuple_type, "min", inline="always")
    def _spec_min(spec):
        if _is_fxpspec(spec):
            return lambda spec: _width_params(spec.signed, spec.qi + spec.qf)[3] / _scale(spec.qf)  # pragma: no cover
        return None

    @overload_attribute(_tuple_type, "max", inline="always")
    def _spec_max(spec):
        if _is_fxpspec(spec):
            return lambda spec: _width_params(spec.signed, spec.qi + spec.qf)[2] / _scale(spec.qf)  # pragma: no cover
        return None
