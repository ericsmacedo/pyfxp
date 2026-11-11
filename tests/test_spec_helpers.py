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
"""FxpSpec helpers and readable repr (issue #40)."""

import math
import os
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from numba import njit
from numba.core.errors import TypingError

from pyfxp import FxpSpec, Q, fxp
from pyfxp.constants import ERROR, HALF_EVEN, SAT, TRUNC, WRAP


@st.composite
def any_spec(draw):
    """Any valid spec, up to the 64-bit (signed) / 63-bit (unsigned) limits."""
    signed = draw(st.booleans())
    w = draw(st.integers(1, 64 if signed else 63))
    qf = draw(st.integers(-8, w + 8))
    return Q(w - qf, qf, signed, draw(st.integers(0, 8)), draw(st.integers(0, 2)))


@njit
def _njit_props(spec):
    return spec.width, spec.lsb, spec.min_int, spec.max_int, spec.min, spec.max


@given(any_spec())
def test_properties_match_definition(spec):
    """Each property matches its definition, computed with exact Python ints."""
    w = spec.qi + spec.qf
    lo, hi = (-(2 ** (w - 1)), 2 ** (w - 1) - 1) if spec.signed else (0, 2**w - 1)
    assert spec.width == w
    assert spec.lsb == 2.0**-spec.qf
    assert (spec.min_int, spec.max_int) == (lo, hi)
    assert spec.min == math.ldexp(float(lo), -spec.qf)
    assert spec.max == math.ldexp(float(hi), -spec.qf)


@given(any_spec())
def test_properties_inside_njit(spec):
    """The same properties are available inside @njit code, with the same values."""
    assert _njit_props(spec) == (spec.width, spec.lsb, spec.min_int, spec.max_int, spec.min, spec.max)


@pytest.mark.parametrize(
    "spec",
    [FxpSpec(4, 3, True, 0, 1), FxpSpec(4, 3, 1, 0, 1), FxpSpec(8, 0, False, 6, 2)],  # type: ignore[invalid-argument-type]
)
def test_properties_on_plain_fxpspec(spec):
    """Specs built directly (ints, or all-int fields) work too, in Python and in @njit."""
    assert _njit_props(spec) == (spec.width, spec.lsb, spec.min_int, spec.max_int, spec.min, spec.max)


@given(any_spec().filter(lambda s: s.width <= 53))
def test_min_max_are_saturation_limits(spec):
    """min/max are representable and are what SAT saturates to."""
    sat = spec._replace(ovf=SAT)
    assert fxp(spec.min, sat) == spec.min
    assert fxp(spec.max, sat) == spec.max
    assert fxp(1e300, sat) == spec.max
    assert fxp(-1e300, sat) == spec.min


@pytest.mark.parametrize(
    ("spec", "text"),
    [
        (Q(3, 13, True, HALF_EVEN, SAT), "sQ3.13(HALF_EVEN, SAT)"),
        (Q(8, 0, signed=False), "uQ8.0(TRUNC, WRAP)"),
        (Q(1, 15, True, TRUNC, ERROR), "sQ1.15(TRUNC, ERROR)"),
        (FxpSpec(4, 3, True, 6, 1), "sQ4.3(HALF_EVEN, SAT)"),
        (FxpSpec(4, -2, False, 0, 0), "uQ4.-2(TRUNC, WRAP)"),
        (FxpSpec(4, 3, True, 42, 7), "sQ4.3(42, 7)"),
    ],
)
def test_repr(spec, text):
    """The repr uses Q notation and mode names; unknown codes are shown as numbers."""
    assert repr(spec) == text
    assert str(spec) == text


def test_fields_unchanged():
    """FxpSpec is still a NamedTuple with the same fields."""
    spec = Q(3, 13, True, HALF_EVEN, SAT)
    qi, qf, signed, rnd, ovf = spec
    assert (qi, qf, signed, rnd, ovf) == (3, 13, True, HALF_EVEN, SAT)
    assert spec._asdict() == {"qi": 3, "qf": 13, "signed": True, "rnd": HALF_EVEN, "ovf": SAT}
    assert spec == FxpSpec(3, 13, True, 6, 1)


@pytest.mark.parametrize("spec", [FxpSpec(0, 0, True, 0, 0), FxpSpec(40, 25, True, 0, 0), FxpSpec(40, 24, False, 0, 0)])
def test_invalid_width(spec):
    """Range properties raise for widths outside 1..64 (signed) / 1..63 (unsigned)."""
    for prop in ("min_int", "max_int", "min", "max"):
        with pytest.raises(ValueError, match="invalid format width"):
            getattr(spec, prop)
    with pytest.raises(ValueError, match="invalid format width"):
        _njit_props(spec)


def test_wide_max_is_rounded_but_max_int_exact():
    """Above 53 bits, max is the nearest float while max_int stays exact."""
    spec = Q(20, 40)
    assert spec.max_int == 2**59 - 1
    assert spec.max == 524288.0
    assert np.float64(spec.max) == spec.max


class _Other(NamedTuple):
    qi: int
    qf: int
    signed: bool
    rnd: int
    ovf: int


_GETTERS = [
    lambda t: t.width,
    lambda t: t.lsb,
    lambda t: t.min_int,
    lambda t: t.max_int,
    lambda t: t.min,
    lambda t: t.max,
]


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="checks Numba typing")
@pytest.mark.parametrize("other", [_Other(4, 3, True, 0, 1), _Other(4, 3, 1, 0, 1)])  # type: ignore[invalid-argument-type]
@pytest.mark.parametrize("getter", _GETTERS)
def test_properties_only_on_fxpspec(other, getter):
    """The @njit attributes are registered for FxpSpec only, not for other named tuples."""
    with pytest.raises(TypingError):
        njit(getter)(other)
