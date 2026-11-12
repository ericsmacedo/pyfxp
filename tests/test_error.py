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


# Overflow messages (ovf=ERROR) name the value, its position, the format and the range.


def test_overflow_message_scalar():
    from pyfxp import Q, fxp

    with pytest.raises(
        OverflowError, match=r"^5\.0 is out of range: the range of sQ3\.4\(TRUNC, ERROR\) is \[-4\.0, 3\.9375\]$"
    ):
        fxp(5.0, Q(3, 4, ovf=2))


def test_overflow_message_array_index_and_count():
    from pyfxp import Q, fxp

    x = np.array([[1.0, np.nan], [9.0, -30.0]])
    with pytest.raises(OverflowError, match=r"^x\[1, 0\] = 9\.0 is out of range \(2 values out of range\): "):
        fxp(x, Q(3, 4, ovf=2))


def test_overflow_message_rounding_and_inf():
    from pyfxp import Q, fxp, to_bin, to_int

    # 3.99 fits as a real number, but CEIL rounds it up to 4.0
    with pytest.raises(OverflowError, match=r"^3\.99 is out of range: the range of sQ3\.4\(CEIL, ERROR\)"):
        to_int(3.99, Q(3, 4, rnd=1, ovf=2))
    with pytest.raises(OverflowError, match=r"^x\[1\] = inf is out of range"):
        fxp([1.0, np.inf], Q(3, 4, ovf=2))
    with pytest.raises(OverflowError, match=r"^9\.0 is out of range"):
        to_bin(9.0, Q(3, 4, ovf=2))


def test_overflow_message_64_bit():
    with pytest.raises(OverflowError, match=r"^x\[1\] = -1e\+300 is out of range: the range of sQ64\.0"):
        fxpt([-(2.0**63), -1e300], 64, 0, ovf=2)


def test_overflow_message_from_int():
    from pyfxp import Q, from_int

    with pytest.raises(
        OverflowError, match=r"^200 is out of range: the stored integers of sQ3\.4\(TRUNC, ERROR\) are \[-64, 63\]$"
    ):
        from_int(200, Q(3, 4, ovf=2))
    with pytest.raises(OverflowError, match=r"^n\[2\] = 500 is out of range"):
        from_int(np.array([1, 2, 500], dtype=np.uint16), Q(3, 4, ovf=2))


def test_overflow_message_in_njit():
    """Compiled code raises a constant message (no formatting cost in the kernel).

    With NUMBA_DISABLE_JIT=1, `f` is plain Python and gets the detailed message.
    """

    @njit
    def f(x):
        return fxpt(x, 3, 4, True, 0, 2)

    with pytest.raises(
        OverflowError, match=r"out of range of the fixed-point format \(ovf=ERROR\)|^5\.0 is out of range"
    ):
        f(5.0)
