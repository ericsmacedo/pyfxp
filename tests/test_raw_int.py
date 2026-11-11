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
"""Raw integer conversion: to_int / from_int / to_bin / to_hex / from_bin / from_hex (issue #43)."""

import math
import os

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numba import njit

from pyfxp import FxpSpec, Q, from_bin, from_hex, from_int, fxp, to_bin, to_hex, to_int
from pyfxp._pyfxp import _from_int_scalar
from pyfxp._raw import _from_int_py
from pyfxp.constants import ERROR, HALF_EVEN, SAT, WRAP

from .test_properties import _ref_round

JIT = os.environ.get("NUMBA_DISABLE_JIT") != "1"


def ref_int(x: float, spec) -> int:
    """Exact reference for the stored integer (Python ints, any width)."""
    from fractions import Fraction  # noqa: PLC0415

    n = _ref_round(Fraction(x) * Fraction(2) ** spec.qf, spec.rnd)
    lo, hi = spec.min_int, spec.max_int
    if spec.ovf == WRAP:
        n = (n - lo) % (1 << spec.width) + lo
    elif spec.ovf == SAT:
        n = min(max(n, lo), hi)
    elif not lo <= n <= hi:
        raise OverflowError
    return n


@st.composite
def spec_and_value(draw, ovf_modes=(WRAP, SAT), max_width=None):
    """Any width up to the 64/63-bit limits, and a value around its range."""
    signed = draw(st.booleans())
    w = draw(st.integers(1, max_width or (64 if signed else 63)))
    qf = draw(st.integers(-4, w + 4))
    spec = Q(w - qf, qf, signed, draw(st.integers(0, 8)), draw(st.sampled_from(ovf_modes)))
    span = math.ldexp(1.0, spec.qi + 1)
    plain = st.floats(-span, span, allow_nan=False, allow_infinity=False)
    ties = st.integers(spec.min_int, spec.max_int).map(lambda k: math.ldexp(k + 0.5, -spec.qf))
    return spec, draw(st.one_of(plain, ties))


@given(spec_and_value())
def test_to_int_matches_exact_reference(case):
    """to_int equals the exact integer, at every width up to 64 bits."""
    spec, x = case
    n = to_int(x, spec)
    assert isinstance(n, int)
    assert n == ref_int(x, spec)
    assert to_int(np.array([x]), spec).tolist() == [n]


@given(spec_and_value())
def test_from_int_inverts_to_int(case):
    """from_int(to_int(x)) is fxp(x), for scalars and arrays."""
    spec, x = case
    assert from_int(to_int(x, spec), spec) == fxp(x, spec)
    xs = np.array([x, -x, 0.0])
    assert np.array_equal(from_int(to_int(xs, spec), spec), fxp(xs, spec))


@given(spec_and_value(ovf_modes=(ERROR,)))
def test_error_mode(case):
    """With ERROR, to_int raises exactly when the reference overflows."""
    spec, x = case
    try:
        expected = ref_int(x, spec)
    except OverflowError:
        with pytest.raises(OverflowError):
            to_int(x, spec)
    else:
        assert to_int(x, spec) == expected


def test_wide_format_is_exact_as_integer():
    """Above 53 bits the float result is rounded, the integer is not."""
    spec = Q(20, 40, ovf=SAT)
    assert to_int(1e10, spec) == 2**59 - 1
    assert fxp(1e10, spec) == 524288.0
    assert to_int(1e30, Q(40, 24, ovf=SAT)) == 2**63 - 1
    assert to_int(-1e30, Q(40, 24, ovf=SAT)) == -(2**63)


@pytest.mark.parametrize("spec", [Q(4, 3), Q(4, 3, signed=False), Q(8, 0), Q(1, 15)])
@pytest.mark.parametrize("ovf", [WRAP, SAT])
def test_from_int_out_of_range_uses_ovf(spec, ovf):
    """Out-of-range integers follow the overflow mode, like fxp does."""
    spec = spec._replace(ovf=ovf)
    for n in (spec.max_int + 1, spec.min_int - 1, 2**70, -(2**70), spec.max_int + 12345):
        expected = fxp(math.ldexp(n, -spec.qf), spec) if abs(n) < 2**52 else None
        out = from_int(n, spec)
        if expected is not None:
            assert out == expected
        assert spec.min <= out <= spec.max


def test_from_int_error_mode():
    """With ERROR, an out-of-range integer raises."""
    spec = Q(4, 3, ovf=ERROR)
    assert from_int(spec.max_int, spec) == spec.max
    with pytest.raises(OverflowError):
        from_int(spec.max_int + 1, spec)
    with pytest.raises(OverflowError):
        from_int(np.array([0, spec.min_int - 1]), spec)


@pytest.mark.parametrize("bad", [1.5, 2.0, np.array([1.5]), [1.0, 2.0], "3", None])
def test_from_int_rejects_non_integers(bad):
    """from_int needs integers."""
    with pytest.raises(TypeError):
        from_int(bad, Q(8, 4))


def test_from_int_uint64_above_int64():
    """uint64 values that do not fit int64 are rejected instead of wrapping silently."""
    with pytest.raises(ValueError, match="above 2"):
        from_int(np.array([2**63], dtype=np.uint64), Q(32, 0, signed=False))
    assert from_int(np.array([5], dtype=np.uint64), Q(8, 0, signed=False)).tolist() == [5.0]


@pytest.mark.parametrize("x", [math.nan, np.array([1.0, math.nan])])
def test_nan_has_no_integer(x):
    """NaN can't be stored as an integer."""
    with pytest.raises(ValueError, match="NaN"):
        to_int(x, Q(8, 4))


def test_inf_saturates_or_raises():
    """+-inf follow the overflow mode: SAT gives the limits, WRAP has no integer."""
    assert to_int(math.inf, Q(8, 4, ovf=SAT)) == Q(8, 4).max_int
    assert to_int(-math.inf, Q(8, 4, ovf=SAT)) == Q(8, 4).min_int
    with pytest.raises(ValueError, match="NaN"):
        to_int(math.inf, Q(8, 4, ovf=WRAP))


def test_shapes_and_dtypes():
    """Arrays keep their shape; to_int gives int64, from_int gives float64."""
    x = np.linspace(-3, 3, 12).reshape(3, 4)
    n = to_int(x, Q(4, 4))
    assert n.shape == (3, 4)
    assert n.dtype == np.int64
    y = from_int(n, Q(4, 4))
    assert y.shape == (3, 4)
    assert y.dtype == np.float64
    assert to_int([], Q(4, 4)).shape == (0,)
    assert from_int(np.array([], dtype=np.int32), Q(4, 4)).shape == (0,)
    assert to_int((0.5, 1.5), Q(4, 4)).tolist() == [8, 24]
    assert to_int(np.int32(2), Q(4, 4)) == 32
    assert from_int(np.int16(-16), Q(4, 4)) == -1.0


@pytest.mark.parametrize(
    ("x", "spec", "b", "h"),
    [
        (-1.3, Q(3, 4), "1101011", "6b"),
        (-1.0, Q(1, 15), "1000000000000000", "8000"),
        (0.5, Q(1, 15), "0100000000000000", "4000"),
        (255.0, Q(8, 0, signed=False), "11111111", "ff"),
        (-1.0, Q(1, 0), "1", "1"),
        (-(2.0**63), Q(64, 0), "1" + "0" * 63, "8" + "0" * 15),
        (2.0**62, Q(63, 0, signed=False), "1" + "0" * 62, "4" + "0" * 15),
    ],
)
def test_bin_hex_known_values(x, spec, b, h):
    """Two's-complement patterns, zero-padded to the format width."""
    assert to_bin(x, spec) == b
    assert to_hex(x, spec) == h
    assert from_bin(b, spec) == fxp(x, spec)
    assert from_hex(h, spec) == fxp(x, spec)


@given(spec_and_value())
def test_bin_hex_round_trip(case):
    """from_bin(to_bin(x)) and from_hex(to_hex(x)) give fxp(x), at every width."""
    spec, x = case
    b, h = to_bin(x, spec), to_hex(x, spec)
    assert len(b) == spec.width
    assert len(h) == (spec.width + 3) // 4
    assert from_bin(b, spec) == fxp(x, spec)
    assert from_hex(h, spec) == fxp(x, spec)
    assert int(b, 2) == int(h, 16) == to_int(x, spec) % (1 << spec.width)


def test_bin_hex_arrays():
    """Arrays give string arrays of the same shape, and parse back."""
    spec = Q(4, 4, True, HALF_EVEN, SAT)
    x = np.array([[0.5, -0.5], [7.99, -8.0]])
    hx = to_hex(x, spec)
    assert hx.shape == (2, 2)
    assert hx.tolist() == [["08", "f8"], ["7f", "80"]]
    assert np.array_equal(from_hex(hx, spec), fxp(x, spec))
    assert np.array_equal(from_bin(to_bin(x, spec), spec), fxp(x, spec))
    assert np.array_equal(from_hex(["0x08", "f_8"], spec), [0.5, -0.5])


@pytest.mark.parametrize(("text", "spec"), [("100", Q(2, 0)), ("1ff", Q(4, 4)), ("-1", Q(4, 4))])
def test_patterns_wider_than_format_raise(text, spec):
    """A pattern with more significant bits than the format is an error."""
    func = from_bin if set(text) <= set("01-") else from_hex
    with pytest.raises(ValueError, match="does not fit"):
        func(text, spec)


def test_invalid_spec_raises():
    """Width and mode checks apply as in fxp."""
    with pytest.raises(ValueError, match="invalid format width"):
        to_int(1.0, FxpSpec(40, 30, True, 0, 0))
    with pytest.raises(ValueError, match="invalid format width"):
        from_int(1, FxpSpec(40, 30, True, 0, 0))
    with pytest.raises(ValueError, match="invalid format width"):
        from_hex("1", FxpSpec(0, 0, True, 0, 0))
    with pytest.raises(ValueError, match="invalid method"):
        to_int(1.0, FxpSpec(8, 4, True, 20, 0))
    with pytest.raises(TypeError, match="Unsupported type"):
        to_int("1.0", Q(8, 4))


@njit
def _njit_raw(x, xs, spec):
    n = to_int(x, spec)
    ns = to_int(xs, spec)
    ls = to_int([x, -x], spec)
    return n, ns, ls, from_int(n, spec), from_int(ns, spec)


def test_inside_njit():
    """to_int and from_int also work inside @njit code."""
    spec = Q(3, 4, True, HALF_EVEN, SAT)
    xs = np.array([[0.5, -1.3], [100.0, -100.0]])
    n, ns, ls, y, ys = _njit_raw(-1.3, xs, spec)
    assert n == to_int(-1.3, spec)
    assert np.array_equal(ns, to_int(xs, spec))
    assert list(ls) == [to_int(-1.3, spec), to_int(1.3, spec)]
    assert y == fxp(-1.3, spec)
    assert np.array_equal(ys, fxp(xs, spec))


@njit
def _njit_nan(x, spec):
    return to_int(x, spec)


@pytest.mark.skipif(not JIT, reason="checks the compiled error path")
def test_nan_inside_njit():
    """NaN raises from @njit code too."""
    with pytest.raises(ValueError, match="NaN"):
        _njit_nan(math.nan, Q(8, 4))
    with pytest.raises(ValueError, match="NaN"):
        _njit_nan(np.array([math.nan]), Q(8, 4))


@pytest.mark.parametrize("spec", [Q(4, 3), Q(4, 3, signed=False), Q(1, 15), Q(40, 24)])
@pytest.mark.parametrize("ovf", [WRAP, SAT])
def test_compiled_from_int_matches_python(spec, ovf):
    """The int64 kernel behind from_int in @njit code agrees with the exact Python version."""
    spec = spec._replace(ovf=ovf)
    for n in (0, 1, -1, spec.max_int, spec.min_int, 12345, -12345, 2**62, -(2**62)):
        assert _from_int_scalar(n, spec.qi, spec.qf, spec.signed, int(ovf)) == _from_int_py(n, spec, int(ovf))


@njit
def _njit_to_int_any(x, spec):
    return to_int(x, spec)


@njit
def _njit_from_int_any(n, spec):
    return from_int(n, spec)


@pytest.mark.skipif(not JIT, reason="checks Numba typing")
def test_unsupported_types_inside_njit():
    """Unsupported argument types are rejected when the @njit function is compiled."""
    with pytest.raises(TypeError, match="Unsupported type"):
        _njit_to_int_any("1.0", Q(8, 4))
    with pytest.raises(TypeError, match="expects integers"):
        _njit_from_int_any(1.5, Q(8, 4))
