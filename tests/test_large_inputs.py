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
"""Inputs with a large scaled value |x| * 2**qf: beyond 2**52 (issue #24) and beyond 2**63 (issue #18)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyfxp import FxpSpec, fxp, fxpt
from pyfxp._pyfxp import _wrap_huge
from pyfxp.constants import ERROR, HALF_AWAY, HALF_DOWN, HALF_UP, HALF_ZERO, SAT, WRAP

from pyfxp.constants import rounding_modes

from .test_properties import ref_fxp, specs

# Large enough to exceed 2**63 after scaling, small enough that x * 2**qf stays finite.
huge = st.one_of(
    st.floats(2.0**63, 1e280, allow_nan=False, allow_infinity=False),
    st.floats(-1e280, -(2.0**63), allow_nan=False, allow_infinity=False),
    st.floats(-1e280, 1e280, allow_nan=False, allow_infinity=False),
)


@given(specs(), huge)
def test_huge_scalar_matches_reference(spec, x):
    """Scalar results match the exact reference for any finite input magnitude."""
    assert fxp(x, spec) == ref_fxp(x, *spec)


@given(specs(), st.lists(huge, min_size=1, max_size=16))
def test_huge_array_matches_reference(spec, xs):
    """Array results match the exact reference element-wise, mixed with in-range values."""
    xs = [*xs, 0.5, -0.25]
    out = np.asarray(fxp(np.array(xs), spec)).tolist()
    assert out == [ref_fxp(x, *spec) for x in xs]


@given(specs(ovf_modes=(ERROR,)), huge)
def test_huge_error_mode(spec, x):
    """ERROR mode raises exactly when the reference overflows."""
    try:
        expected = ref_fxp(x, *spec)
    except OverflowError:
        with pytest.raises(OverflowError):
            fxp(x, spec)
    else:
        assert fxp(x, spec) == expected


@pytest.mark.parametrize(
    ("x", "qi", "qf", "ovf", "expected"),
    [
        (1e19, 8, 4, SAT, 127.9375),
        (-1e19, 8, 4, SAT, -128.0),
        (1e10, 20, 40, SAT, 524288.0),  # max of Q20.40, rounded to float64
        (1e10, 20, 40, WRAP, -269312.0),
        (-1e10, 20, 40, WRAP, 269312.0),
    ],
)
@pytest.mark.parametrize("as_array", [False, True])
def test_issue_18_examples(x, qi, qf, ovf, expected, as_array):  # noqa: PLR0913
    """The examples from issue #18."""
    y = fxpt(np.array([x]) if as_array else x, qi, qf, True, 0, ovf)
    assert np.asarray(y).tolist() == ([expected] if as_array else expected)


def test_unsigned_sat_huge_negative():
    """A huge negative input saturates to 0 in an unsigned format."""
    assert fxp(-1e19, FxpSpec(8, 4, False, 0, SAT)) == 0.0


@pytest.mark.parametrize("x", [3e19, -3e19, 1.5e19, -1.5e19, 2.0**63, -(2.0**64) + 2048.0])
def test_wrap_reduction_64_bits(x):
    """For a 64-bit format, WRAP reduction lands in the int64 range and is exact modulo 2**64."""
    y = _wrap_huge(x, 64)
    assert -(2.0**63) <= y < 2.0**63
    assert (int(x) - int(y)) % 2**64 == 0


@st.composite
def spec_and_value_above_2_52(draw):
    """A 60-bit format and a value whose scaled magnitude is in [2**52, 2**59), so no overflow occurs."""
    qf = draw(st.integers(0, 40))
    spec = FxpSpec(60 - qf, qf, True, draw(st.sampled_from(sorted(rounding_modes.values()))), SAT)
    k = draw(st.integers(2**52, 2**59 - 1)) * draw(st.sampled_from([1, -1]))
    return spec, float(k) / 2**qf


@given(spec_and_value_above_2_52())
def test_rounding_above_2_52_matches_reference(case):
    """Scaled values >= 2**52 are integers; no rounding mode may change them (issue #24)."""
    spec, x = case
    expected = ref_fxp(x, *spec)
    assert fxp(x, spec) == expected
    assert np.asarray(fxp(np.array([x]), spec)).tolist() == [expected]


@pytest.mark.parametrize("rnd", [HALF_UP, HALF_DOWN, HALF_ZERO, HALF_AWAY])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("as_array", [False, True])
def test_issue_24_examples(rnd, sign, as_array):
    """The examples from issue #24: 2**52 + 1 is unchanged by every half-rounding mode."""
    x = sign * (2.0**52 + 1)
    y = fxpt(np.array([x]) if as_array else x, 60, 0, True, rnd, SAT)
    assert np.asarray(y).tolist() == ([x] if as_array else x)
    # same scaled value via fractional bits
    x2 = sign * (2.0**12 + 2.0**-40)
    y2 = fxpt(np.array([x2]) if as_array else x2, 20, 40, True, rnd, SAT)
    assert np.asarray(y2).tolist() == ([x2] if as_array else x2)


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("x", [1e30, -1e30, math.inf, -math.inf, 2.0**63, -(2.0**64)])
def test_64_bit_format_saturates_and_raises(x, as_array):
    """For a 64-bit signed format, huge inputs saturate to the exact limits and raise with ERROR.

    The int64 overflow check can't see them for this width (its range is all of int64),
    so they used to saturate to 2**63 - 1024 and pass ERROR silently.
    """
    spec = FxpSpec(40, 24, True, 0, SAT)
    xin = np.array([x]) if as_array else x
    expected = spec.max if x > 0 else spec.min
    assert np.asarray(fxp(xin, spec)).tolist() == ([expected] if as_array else expected)
    with pytest.raises(OverflowError):
        fxp(xin, spec._replace(ovf=ERROR))
