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
"""Non-finite inputs (issue #27): NaN propagates; ±inf saturates (SAT), raises (ERROR) or gives NaN (WRAP)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyfxp import FxpSpec, fxp
from pyfxp.constants import ERROR, SAT, WRAP, rounding_modes

from .test_properties import ref_fxp, specs

ALL_RND = sorted(rounding_modes.values())
SIGNED = [True, False]


def _limits(spec):
    w = spec.qi + spec.qf
    lo, hi = (-(2 ** (w - 1)), 2 ** (w - 1) - 1) if spec.signed else (0, 2**w - 1)
    return lo / 2**spec.qf, hi / 2**spec.qf


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("ovf", [WRAP, SAT, ERROR])
@pytest.mark.parametrize("signed", SIGNED)
def test_nan_propagates(rnd, ovf, signed):
    """NaN in, NaN out, for every mode, scalar and array."""
    spec = FxpSpec(8, 4, signed, rnd, ovf)
    assert math.isnan(fxp(math.nan, spec))
    assert np.isnan(fxp(np.array([math.nan]), spec)).all()


@pytest.mark.parametrize("rnd", ALL_RND)
@pytest.mark.parametrize("signed", SIGNED)
def test_inf_saturates(rnd, signed):
    """With SAT, +inf gives the maximum and -inf the minimum."""
    spec = FxpSpec(8, 4, signed, rnd, SAT)
    lo, hi = _limits(spec)
    assert fxp(math.inf, spec) == hi
    assert fxp(-math.inf, spec) == lo
    assert np.asarray(fxp(np.array([math.inf, -math.inf]), spec)).tolist() == [hi, lo]


@pytest.mark.parametrize("x", [math.inf, -math.inf])
@pytest.mark.parametrize("as_array", [False, True])
def test_inf_error_raises(x, as_array):
    """With ERROR, ±inf raises OverflowError."""
    spec = FxpSpec(8, 4, True, 0, ERROR)
    with pytest.raises(OverflowError):
        fxp(np.array([x]) if as_array else x, spec)


@pytest.mark.parametrize("x", [math.inf, -math.inf])
@pytest.mark.parametrize("signed", SIGNED)
def test_inf_wrap_is_nan(x, signed):
    """With WRAP, ±inf has no defined result and gives NaN."""
    spec = FxpSpec(8, 4, signed, 0, WRAP)
    assert math.isnan(fxp(x, spec))
    assert np.isnan(fxp(np.array([x]), spec)).all()


finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@given(specs(), st.lists(st.one_of(finite, st.just(math.nan)), min_size=1, max_size=32))
def test_nan_mixed_array(spec, xs):
    """NaN positions are kept; every other element matches the reference."""
    out = np.asarray(fxp(np.array(xs), spec)).tolist()
    for x, y in zip(xs, out, strict=True):
        if math.isnan(x):
            assert math.isnan(y)
        else:
            assert y == ref_fxp(x, *spec)


def test_nan_does_not_raise_in_error_mode():
    """NaN is not an overflow: ERROR mode returns NaN and still checks other elements."""
    spec = FxpSpec(8, 4, True, 0, ERROR)
    out = fxp(np.array([math.nan, 1.0]), spec)
    assert math.isnan(out[0])
    assert out[1] == 1.0
    with pytest.raises(OverflowError):
        fxp(np.array([math.nan, 1e6]), spec)
