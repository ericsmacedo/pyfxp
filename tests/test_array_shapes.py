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
"""Array shapes and layouts (issue #26): N-D, non-contiguous, 0-d and empty arrays."""

import numpy as np
import pytest
from numba import njit

from pyfxp import Q, fxp, fxpt
from pyfxp._pyfxp import _check_modes
from pyfxp.constants import ERROR, SAT, WRAP, rounding_modes

from .test_properties import ref_fxp

RNG = np.random.default_rng(0)
SHAPES = [(7,), (3, 4), (2, 3, 4), (2, 1, 3, 2), (1, 1), (5, 1)]


def _expected(x, qi, qf, signed, rnd, ovf):  # noqa: PLR0913
    flat = [ref_fxp(float(v), qi, qf, signed, rnd, ovf) for v in x.reshape(-1)]
    return np.array(flat).reshape(x.shape)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ovf", [WRAP, SAT])
@pytest.mark.parametrize("rnd", sorted(rounding_modes.values()))
def test_nd_arrays(shape, ovf, rnd):
    """Arrays of any dimension keep their shape and match the reference element-wise."""
    x = RNG.uniform(-20, 20, shape)
    out = fxpt(x, 4, 3, True, rnd, ovf)
    assert out.shape == x.shape
    assert out.dtype == np.float64
    assert np.array_equal(out, _expected(x, 4, 3, True, rnd, ovf))


@pytest.mark.parametrize(
    "view",
    [
        lambda a: a[::2],
        lambda a: a[::-1],
        lambda a: a.T,
        lambda a: a[:, 1::2],
        lambda a: np.asfortranarray(a),
    ],
    ids=["step", "reversed", "transposed", "column-slice", "fortran"],
)
def test_non_contiguous_arrays(view):
    """Views and Fortran-ordered arrays give the same result as a contiguous copy."""
    x = view(RNG.uniform(-20, 20, (6, 8)))
    out = fxpt(x, 4, 3, True, 0, SAT)
    assert np.array_equal(out, fxpt(np.ascontiguousarray(x), 4, 3, True, 0, SAT))
    assert np.array_equal(out, _expected(x, 4, 3, True, 0, SAT))


def test_input_is_not_modified():
    """The input array is left untouched."""
    x = RNG.uniform(-20, 20, (3, 4))
    before = x.copy()
    fxpt(x, 4, 3, True, 0, SAT)
    assert np.array_equal(x, before)


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 3)])
def test_empty_arrays(shape):
    """Empty arrays return an empty float64 array of the same shape."""
    out = fxpt(np.zeros(shape), 8, 4)
    assert out.shape == shape
    assert out.dtype == np.float64


def test_zero_dim_array():
    """A 0-d array is quantized like its value."""
    assert float(fxpt(np.array(1.3), 8, 2)) == 1.25


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.int32, np.int64, np.uint16, np.bool_])
def test_array_dtypes(dtype):
    """Non-float64 arrays are accepted and return float64."""
    x = np.array([[0, 1], [1, 0]], dtype=dtype)
    out = fxpt(x, 8, 4)
    assert out.dtype == np.float64
    assert np.array_equal(out, x.astype(np.float64))


def test_nd_error_mode():
    """ERROR mode raises for an overflow anywhere in an N-D array."""
    x = np.zeros((3, 4))
    assert np.array_equal(fxpt(x, 4, 3, True, 0, ERROR), x)
    x[2, 1] = 100.0
    with pytest.raises(OverflowError):
        fxpt(x, 4, 3, True, 0, ERROR)


@njit
def _njit_2d(x, spec):
    return fxp(x, spec)


def test_nd_inside_njit():
    """N-D arrays also work from user @njit code."""
    x = RNG.uniform(-20, 20, (3, 4))
    spec = Q(4, 3, True, 0, SAT)
    assert np.array_equal(_njit_2d(x, spec), _expected(x, *spec))


@njit
def _njit_modes(x, rnd, ovf):
    return fxpt(x, 8, 4, True, rnd, ovf)


@pytest.mark.parametrize(("rnd", "ovf"), [(20, 0), (-1, 0), (0, 20), (0, -1)])
@pytest.mark.parametrize("as_array", [False, True])
def test_invalid_modes_inside_njit(rnd, ovf, as_array):
    """Unknown rounding/overflow methods raise ValueError from @njit code too."""
    x = np.ones(3) if as_array else 1.0
    with pytest.raises(ValueError, match="invalid method"):
        _njit_modes(x, rnd, ovf)


@pytest.mark.parametrize(("rnd", "ovf"), [(20, 0), (-1, 0), (0, 20), (0, -1)])
def test_check_modes(rnd, ovf):
    """The compiled-code validator rejects unknown methods and accepts valid ones."""
    with pytest.raises(ValueError, match="invalid method"):
        _check_modes(rnd, ovf)
    _check_modes(0, 0)
    _check_modes(8, 2)
