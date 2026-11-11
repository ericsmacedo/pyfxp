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
"""Accepted input types (issue #19): NumPy scalars, lists and tuples, from Python and from @njit code."""

import numpy as np
import pytest
from numba import njit

from pyfxp import Q, fxp, fxpt

SPEC = Q(8, 4)

NUMPY_SCALARS = [
    np.int8(-3),
    np.int16(-3),
    np.int32(3),
    np.int64(3),
    np.uint8(3),
    np.uint32(3),
    np.float16(1.3),
    np.float32(1.3),
    np.float64(1.3),
]


@pytest.mark.parametrize("x", NUMPY_SCALARS, ids=lambda x: type(x).__name__)
def test_numpy_scalars(x):
    """NumPy integer/float scalars behave like the equivalent Python float."""
    expected = fxpt(float(x), 8, 4)
    assert fxpt(x, 8, 4) == expected
    assert fxp(x, SPEC) == expected
    assert isinstance(fxpt(x, 8, 4), float)


@pytest.mark.parametrize("seq", [[1.3, -2.7, 0], (1.3, -2.7, 0), [1, 2, 3], (1, 2.5), []], ids=repr)
def test_sequences(seq):
    """Lists and tuples are converted to float arrays."""
    expected = fxpt(np.asarray(seq, dtype=np.float64), 8, 4)
    for out in (fxpt(seq, 8, 4), fxp(seq, SPEC)):
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.tolist() == expected.tolist()


def test_scalar_from_int32_array():
    """Elements indexed from a non-float64 array are accepted."""
    arr = np.array([1, 2, 3], dtype=np.int32)
    assert fxpt(arr[0], 8, 4) == 1.0


@njit
def _njit_inputs(spec):
    return (
        fxpt([1.3, 2.6], 8, 4),
        fxpt((1.3, 2.6), 8, 4),
        fxp([1.3, 2.6], spec),
        fxp((1.3, 2.6), spec),
        fxp(np.int32(3), spec),
        fxp(np.float32(1.3), spec),
    )


def test_inputs_inside_njit():
    """The same input types work inside user @njit code."""
    a, b, c, d, e, f = _njit_inputs(SPEC)
    for arr in (a, b, c, d):
        assert arr.tolist() == [1.25, 2.5625]
    assert e == 3.0
    assert f == 1.25


@pytest.mark.parametrize("x", ["1.0", None, {"a": 1}, b"1"], ids=repr)
def test_unsupported_types_still_raise(x):
    """Other types still raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported type"):
        fxpt(x, 8, 4)
    with pytest.raises(TypeError, match="Unsupported type"):
        fxp(x, SPEC)
