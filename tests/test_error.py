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

"""Test errors."""

import numpy as np
import pytest

from pyfxp import fxpt
from numba import njit


@njit
def wrapper():
    fxpt(x="string", qi=8, qf=0, ovf=2)  # type: ignore


def test_func_raises_typeerror():
    with pytest.raises(TypeError, match=r"Unsupported type: .*"):
        fxpt(x="string", qi=8, qf=0, ovf=2)  # type: ignore


def test_func_raises_typeerror_numba():
    with pytest.raises(TypeError, match=r"Unsupported type: .*"):
        wrapper()


def test_func_raises_valueerror_scalar():
    with pytest.raises(ValueError, match=r"invalid method: 20"):
        fxpt(x=5, qi=8, qf=0, ovf=20)


def test_func_raises_valueerror_array():
    with pytest.raises(ValueError, match=r"invalid method: 20"):
        fxpt(x=np.arange(10), qi=8, qf=0, ovf=20)


def test_overflow_error():
    """Test overflow error."""
    with pytest.raises(OverflowError):
        # Code that should raise OverflowError
        fxpt(x=512, qi=8, qf=0, ovf=2)


def test_overflow_error_array():
    """Test overflow error."""
    with pytest.raises(OverflowError):
        x = np.ones(100) * 1000
        # Code that should raise OverflowError
        fxpt(x=x, qi=8, qf=0, ovf=2)


def test_invalid_method_rnd_array():
    """Test invalid method error."""
    with pytest.raises(ValueError):
        # Code that should raise OverflowError
        fxpt(x=np.array([512]), qi=10, qf=0, rnd=20)


def test_invalid_method_rnd_scalar():
    """Test invalid method error."""
    with pytest.raises(ValueError):
        # Code that should raise OverflowError
        fxpt(x=512, qi=10, qf=0, rnd=20)
