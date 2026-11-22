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

"""Complex inputs to fxp / fxpt: the real and imaginary parts are quantized separately."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numba import njit

from pyfxp import FxpSpec, Q, fxp, fxpt, to_bin, to_hex, to_int
from pyfxp.constants import ERROR, HALF_EVEN, SAT, TRUNC, WRAP, rounding_modes

from .test_properties import ref_fxp, specs, values


def _ref(z: complex, spec: FxpSpec) -> complex:
    return complex(ref_fxp(z.real, *spec), ref_fxp(z.imag, *spec))


def complex_values(spec: FxpSpec):
    return st.builds(complex, values(spec), values(spec))


@st.composite
def spec_and_complex(draw, ovf_modes=(WRAP, SAT)):
    spec = draw(specs(ovf_modes))
    return spec, draw(complex_values(spec))


@given(spec_and_complex())
def test_scalar_matches_reference(case):
    spec, z = case
    out = fxp(z, spec)
    assert type(out) is complex
    assert out == _ref(z, spec)
    assert fxpt(z, *spec) == out


@given(spec_and_complex(ovf_modes=(ERROR,)))
def test_error_mode_matches_reference(case):
    spec, z = case
    try:
        expected = _ref(z, spec)
    except OverflowError:
        with pytest.raises(OverflowError):
            fxp(z, spec)
    else:
        assert fxp(z, spec) == expected


@given(specs().flatmap(lambda s: st.tuples(st.just(s), st.lists(complex_values(s), min_size=1, max_size=32))))
def test_array_matches_reference(case):
    spec, zs = case
    expected = [_ref(z, spec) for z in zs]
    out = fxp(np.array(zs), spec)
    assert out.dtype == np.complex128
    assert out.tolist() == expected
    assert fxp(zs, spec).tolist() == expected  # list input
    assert fxp(tuple(zs), spec).tolist() == expected  # tuple input


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_numpy_complex_types(dtype):
    spec = Q(3, 4, rnd=HALF_EVEN)
    z = dtype(1.3 - 0.7j)
    expected = _ref(complex(z), spec)
    assert fxp(z, spec) == expected
    out = fxp(np.array([z, z], dtype=dtype), spec)
    assert out.dtype == np.complex128
    assert out.tolist() == [expected, expected]


def test_mixed_list_is_complex():
    """A list mixing complex, float and int values is quantized as complex."""
    out = fxp([1 + 1j, 0.3, 2], Q(3, 4))
    assert out.dtype == np.complex128
    assert out.tolist() == [1 + 1j, 0.25 + 0j, 2 + 0j]


def test_shapes_and_layouts():
    spec = Q(3, 4, ovf=SAT)
    rng = np.random.default_rng(0)
    z = (rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))) * 3
    ref = np.vectorize(lambda v: _ref(complex(v), spec), otypes=[complex])
    for view in (z, z[:, ::2], z.transpose(2, 0, 1), np.asfortranarray(z), z[1, 2, 0]):
        assert np.array_equal(fxp(view, spec), ref(view))
    assert fxp(np.empty((0, 3), dtype=complex), spec).shape == (0, 3)


@pytest.mark.parametrize("rnd", sorted(rounding_modes.values()))
@pytest.mark.parametrize("ovf", [WRAP, SAT, ERROR])
def test_nan_in_one_part(rnd, ovf):
    """NaN propagates in its own part only."""
    spec = Q(3, 4, rnd=rnd, ovf=ovf)
    for z, nan_re in ((complex(math.nan, 0.5), True), (complex(0.5, math.nan), False)):
        for out in (fxp(z, spec), fxp(np.array([z]), spec)[0]):
            assert math.isnan(out.real) == nan_re
            assert math.isnan(out.imag) != nan_re
            assert (out.imag if nan_re else out.real) == 0.5


def test_inf_parts():
    assert fxp(complex(math.inf, -math.inf), Q(3, 4, ovf=SAT)) == complex(3.9375, -4.0)
    out = fxp(complex(math.inf, 0.5), Q(3, 4, ovf=WRAP))
    assert math.isnan(out.real)
    assert out.imag == 0.5
    with pytest.raises(OverflowError, match=r"\(imaginary part\)"):
        fxp(complex(0.5, math.inf), Q(3, 4, ovf=ERROR))


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (
            5 + 0j,
            r"^\(5\+0j\) is out of range \(real part\): the range of sQ3\.4\(TRUNC, ERROR\) is \[-4\.0, 3\.9375\]$",
        ),
        (1 + 7j, r"^\(1\+7j\) is out of range \(imaginary part\): "),
        (9 - 9j, r"^\(9-9j\) is out of range \(real and imaginary parts\): "),
        (np.complex64(5 + 0j), r"^\(5\+0j\) is out of range \(real part\): "),
        (
            np.array([[0j, complex(math.nan, 1)], [2 + 8j, 9j]]),
            r"^x\[1, 0\] = \(2\+8j\) is out of range \(imaginary part; 2 values out of range\): ",
        ),
        ([0j, 1 + 0j, 2 + 8j], r"^x\[2\] = \(2\+8j\) is out of range \(imaginary part\): "),
    ],
)
def test_overflow_messages(x, message):
    with pytest.raises(OverflowError, match=message):
        fxp(x, Q(3, 4, ovf=ERROR))


def test_rounding_rounds_up_into_overflow():
    """3.99 fits as a real number, but CEIL rounds the imaginary part up to 4.0."""
    with pytest.raises(OverflowError, match=r"\(imaginary part\)"):
        fxpt(1 + 3.99j, 3, 4, rnd=1, ovf=ERROR)


def test_invalid_width_and_modes():
    with pytest.raises(ValueError, match="width"):
        fxpt(1j, 40, 30)
    with pytest.raises(ValueError, match="width"):
        fxpt(np.array([1j]), 40, 30)
    with pytest.raises(ValueError, match="width"):
        fxpt(np.empty(0, dtype=complex), 40, 30)
    with pytest.raises(ValueError):
        fxpt(1j, 3, 4, rnd=99)


@pytest.mark.parametrize("func", [to_int, to_bin, to_hex])
@pytest.mark.parametrize("x", [1j, np.complex64(1j), np.array([1j]), [1j, 0.5], (1j,)])
def test_raw_functions_reject_complex(func, x):
    with pytest.raises(TypeError, match="complex inputs are not supported"):
        func(x, Q(3, 4))


def test_inside_njit():
    spec = Q(3, 4, rnd=HALF_EVEN, ovf=SAT)

    @njit
    def run(z, zs, spec):
        acc = 0j
        for v in zs:
            acc += fxp(v, spec)
        return fxp(z, spec), fxp(zs, spec), acc, fxp([1.3 + 1j, 2j], spec), fxpt(z, 3, 4, True, TRUNC, WRAP)

    zs = np.array([0.33 + 0.66j, -0.1j, 9 + 9j])
    one, arr, acc, lst, trunc = run(1.3 + 0.7j, zs, spec)
    assert one == _ref(1.3 + 0.7j, spec)
    assert arr.tolist() == [_ref(z, spec) for z in zs]
    assert acc == sum(_ref(z, spec) for z in zs)
    assert lst.tolist() == [_ref(1.3 + 1j, spec), 2j]
    assert trunc == _ref(1.3 + 0.7j, Q(3, 4))
    # complex64 inside @njit
    one32, arr32, _, _, _ = run(np.complex64(1.3 + 0.7j), zs.astype(np.complex64), spec)
    assert one32 == _ref(complex(np.complex64(1.3 + 0.7j)), spec)
    assert arr32.dtype == np.complex128


def test_to_int_inside_njit_rejects_complex():
    from numba.core.errors import TypingError

    @njit
    def run(z, spec):
        return to_int(z, spec)

    with pytest.raises((TypingError, TypeError)):
        run(1j, Q(3, 4))
