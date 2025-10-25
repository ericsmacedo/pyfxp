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
"""Checks if scalar, array, njin and non njit functions match."""

import numpy as np
import pytest
from numba import njit

from pyfxp import fxp
from pyfxp.constants import (
    AWAY,
    CEIL,
    HALF_AWAY,
    HALF_DOWN,
    HALF_EVEN,
    HALF_UP,
    HALF_ZERO,
    SAT,
    TO_ZERO,
    TRUNC,
    WRAP,
)


def _fxp_arr(a, qi, qf, signed, rnd, ovf):  # noqa: PLR0913
    return fxp(a, qi, qf, signed, rnd, ovf)


def _fxp_scalar(a, qi, qf, signed, rnd, ovf):  # noqa: PLR0913
    n = len(a)
    out = np.zeros(n)
    for i in range(n):
        out[i] = fxp(a[i], qi, qf, signed=signed, rnd=rnd, ovf=ovf)
    return out


@njit
def _fxp_arr_njit(a, qi, qf, signed, rnd, ovf):  # noqa: PLR0913
    return fxp(a, qi, qf, signed=signed, rnd=rnd, ovf=ovf)


@njit
def _fxp_scalar_njit(a, qi, qf, signed, rnd, ovf):  # noqa: PLR0913
    n = len(a)
    out = np.zeros(n)
    for i in range(n):
        out[i] = fxp(a[i], qi, qf, signed=signed, rnd=rnd, ovf=ovf)
    return out


# Test configurations
QI_VALUES = [4, 8, 16, 9]
QF_VALUES = [0, 4, 8, 10, 1]
ROUNDING_METHODS = [TRUNC, CEIL, TO_ZERO, AWAY, HALF_UP, HALF_DOWN, HALF_EVEN, HALF_ZERO, HALF_AWAY]
OVERFLOW_METHODS = [WRAP, SAT]
SIGNED = [True, False]
ARR_NJIT = [True, False]
SCALAR_NJIT = [True, False]


@pytest.mark.parametrize("qi", QI_VALUES)
@pytest.mark.parametrize("qf", QF_VALUES)
@pytest.mark.parametrize("rnd", ROUNDING_METHODS)
@pytest.mark.parametrize("ovf", OVERFLOW_METHODS)
@pytest.mark.parametrize("signed", SIGNED)
@pytest.mark.parametrize("arr_njit", ARR_NJIT)
@pytest.mark.parametrize("scalar_njit", SCALAR_NJIT)
def test_array_vs_scalar_consistency(qi, qf, rnd, ovf, signed, arr_njit, scalar_njit):  # noqa: PLR0913
    """Test that array and scalar implementations produce identical results."""
    rng = np.random.default_rng()  # Create a Generator instance
    test_data = rng.uniform(low=-1000.0, high=1000.0, size=2**4)

    # Array implementation
    if arr_njit:
        arr_result = _fxp_arr_njit(test_data, qi=qi, qf=qf, signed=signed, rnd=rnd, ovf=ovf)
    else:
        arr_result = _fxp_arr(test_data, qi=qi, qf=qf, signed=signed, rnd=rnd, ovf=ovf)

    if scalar_njit:
        scalar_result = _fxp_scalar_njit(test_data, qi=qi, qf=qf, signed=signed, rnd=rnd, ovf=ovf)
    else:
        scalar_result = _fxp_scalar(test_data, qi=qi, qf=qf, signed=signed, rnd=rnd, ovf=ovf)

    # Compare results
    assert np.array_equal(arr_result, scalar_result), (
        f"Inconsistent results for Q{qi}.{qf} with "
        f"rnd={rnd}, ovf={ovf}\n"
        f"Input: {test_data}\n"
        f"Array result: {arr_result}\n"
        f"Scalar result: {scalar_result}"
    )
