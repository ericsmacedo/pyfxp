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
"""Rounding and overflow modes as IntEnums (issue #42)."""

import numpy as np
import pytest
from numba import njit

from pyfxp import FxpSpec, Q, fxp, fxpt
from pyfxp.constants import (
    ERROR,
    HALF_EVEN,
    SAT,
    TRUNC,
    WRAP,
    Overflow,
    Rounding,
    overflow_modes,
    overflow_modes_inv,
    rounding_modes,
    rounding_modes_inv,
)


def test_members_are_the_old_integer_codes():
    """Every mode keeps its numeric code and its module-level name."""
    assert [m.value for m in Rounding] == list(range(9))
    assert [m.value for m in Overflow] == [0, 1, 2]
    assert HALF_EVEN is Rounding.HALF_EVEN
    assert SAT is Overflow.SAT
    assert HALF_EVEN == 6
    assert isinstance(SAT, int)


def test_mapping_dicts():
    """The name/code dictionaries still map both ways."""
    assert rounding_modes["HALF_EVEN"] == 6
    assert overflow_modes["SAT"] == 1
    assert rounding_modes_inv[6] == "HALF_EVEN"
    assert overflow_modes_inv[2] == "ERROR"
    assert len(rounding_modes) == 9
    assert len(overflow_modes) == 3


def test_q_converts_codes_to_members():
    """Q stores enum members, whether it gets members or integer codes."""
    for spec in (Q(8, 4, True, 6, 1), Q(8, 4, True, HALF_EVEN, SAT), Q(8, 4, rnd=np.int64(6), ovf=np.int32(1))):
        assert spec.rnd is Rounding.HALF_EVEN
        assert spec.ovf is Overflow.SAT
    assert Q(8, 4).rnd is TRUNC
    assert Q(8, 4).ovf is WRAP
    assert "HALF_EVEN" in repr(Q(8, 4, True, 6, 1))


@pytest.mark.parametrize(("rnd", "ovf"), [(9, 0), (-1, 0), (0, 3), (0, -1)])
def test_q_rejects_invalid_codes(rnd, ovf):
    """Q validates the modes when the spec is created."""
    with pytest.raises(ValueError, match="invalid method"):
        Q(8, 4, True, rnd, ovf)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rnd": SAT}, "is an overflow mode"),
        ({"rnd": WRAP}, "is an overflow mode"),
        ({"ovf": HALF_EVEN}, "is a rounding mode"),
        ({"ovf": TRUNC}, "is a rounding mode"),
        ({"rnd": ERROR, "ovf": HALF_EVEN}, "did you swap"),
    ],
)
@pytest.mark.parametrize("func", ["Q", "fxpt", "fxpt-array"])
def test_swapped_modes_raise(kwargs, match, func):
    """A Rounding member as ovf (or the reverse) is rejected instead of silently reinterpreted."""
    with pytest.raises(TypeError, match=match):
        if func == "Q":
            Q(8, 4, **kwargs)
        elif func == "fxpt":
            fxpt(1.0, 8, 4, **kwargs)
        else:
            fxpt(np.ones(3), 8, 4, **kwargs)


def test_integer_codes_still_work():
    """Plain integers are accepted everywhere and give the same results as members."""
    x = np.linspace(-20, 20, 101)
    assert np.array_equal(fxpt(x, 4, 3, True, 6, 1), fxpt(x, 4, 3, True, HALF_EVEN, SAT))
    assert np.array_equal(fxp(x, FxpSpec(4, 3, True, 6, 1)), fxp(x, Q(4, 3, True, HALF_EVEN, SAT)))


@njit
def _njit_modes(x, spec):
    inner = Q(4, 3, True, HALF_EVEN, SAT)
    return fxp(x, spec), fxp(x, inner), fxpt(x, 4, 3, True, HALF_EVEN, SAT), fxpt(x, 4, 3, True, 6, 1)


@pytest.mark.parametrize("as_array", [False, True])
def test_modes_inside_njit(as_array):
    """Members, integer codes and Q() itself all work inside @njit code."""
    x = np.array([1.3, -20.0]) if as_array else 1.3
    expected = fxpt(x, 4, 3, True, 6, 1)
    for spec in (Q(4, 3, True, HALF_EVEN, SAT), FxpSpec(4, 3, True, 6, 1)):
        for out in _njit_modes(x, spec):
            assert np.array_equal(out, expected)
