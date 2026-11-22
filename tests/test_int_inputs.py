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
"""Integer inputs are quantized exactly, in integer arithmetic, at every supported width."""

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numba import njit

from pyfxp import FxpSpec, Q, fxp, fxpt, to_hex, to_int
from pyfxp.constants import ERROR, HALF_EVEN, SAT, WRAP, rounding_modes

from .test_properties import _limits, _ref_round, ref_fxp

INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def ref_int(x: int, qi: int, qf: int, signed: bool, rnd: int, ovf: int) -> int:  # noqa: PLR0913
    """Exact reference stored integer (same model as `ref_fxp`)."""
    n = _ref_round(Fraction(x) * Fraction(2) ** qf, rnd)
    lower, upper = _limits(qi, qf, signed)
    w = qi + qf
    if ovf == WRAP:
        n &= (1 << w) - 1
        if signed and n >= 1 << (w - 1):
            n -= 1 << w
    elif ovf == SAT:
        n = min(max(n, lower), upper)
    elif n < lower or n > upper:
        raise OverflowError
    return n


def ref_float(x: int, spec: FxpSpec) -> float:
    return float(Fraction(ref_int(x, *spec)) * Fraction(2) ** -spec.qf)


@st.composite
def wide_specs(draw, ovf_modes=(WRAP, SAT)):
    """Any width up to the 64 / 63-bit limit, with negative and large fractional bit counts."""
    signed = draw(st.booleans())
    w = draw(st.integers(1, 64 if signed else 63))
    qf = draw(st.one_of(st.integers(-8, 8), st.integers(-80, 80)))
    return FxpSpec(
        w - qf,
        qf,
        signed,
        draw(st.sampled_from(sorted(rounding_modes.values()))),
        draw(st.sampled_from(ovf_modes)),
    )


int64s = st.integers(INT64_MIN, INT64_MAX)
ints = st.one_of(
    int64s,
    st.integers(-(2**80), 2**80),  # also beyond int64: the Python-int path
    st.integers(-1000, 1000),
    st.sampled_from([0, 1, -1, INT64_MIN, INT64_MAX, 2**53 + 1, -(2**53) - 1, 2**63, 2**64 - 1]),
)


@given(wide_specs(), ints)
def test_scalar_matches_reference(spec, x):
    assert to_int(x, spec) == ref_int(x, *spec)
    assert fxp(x, spec) == ref_float(x, spec)


@given(wide_specs(), st.lists(int64s, max_size=12))
def test_int64_array_matches_reference(spec, xs):
    arr = np.array(xs, dtype=np.int64)
    assert to_int(arr, spec).tolist() == [ref_int(x, *spec) for x in xs]
    assert fxp(arr, spec).tolist() == [ref_float(x, spec) for x in xs]


@given(wide_specs(), st.lists(ints, min_size=1, max_size=12))
def test_list_matches_reference(spec, xs):
    """Lists of Python ints, including ones outside int64."""
    assert np.asarray(to_int(xs, spec)).tolist() == [ref_int(x, *spec) for x in xs]
    assert np.asarray(fxp(tuple(xs), spec)).tolist() == [ref_float(x, spec) for x in xs]


@given(wide_specs(ovf_modes=(ERROR,)), ints)
def test_error_mode_matches_reference(spec, x):
    try:
        expected = ref_int(x, *spec)
    except OverflowError:
        with pytest.raises(OverflowError, match="is out of range"):
            to_int(x, spec)
        with pytest.raises(OverflowError, match="is out of range"):
            fxp(np.array([0, x], dtype=np.int64) if INT64_MIN <= x <= INT64_MAX else [0, x], spec)
    else:
        assert to_int(x, spec) == expected


@given(wide_specs(), int64s)
def test_float_representable_ints_unchanged(spec, x):
    """For integers exactly representable as float64 the result is the same as for the float."""
    if float(x) == x and abs(x) < 2**63:
        assert fxp(x, spec) == fxp(float(x), spec)
        assert to_int(x, spec) == to_int(float(x), spec)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]
)
@given(spec=wide_specs(), data=st.data())
def test_integer_dtypes(dtype, spec, data):
    info = np.iinfo(dtype) if dtype is not np.bool_ else None
    lo, hi = (0, 1) if info is None else (int(info.min), int(info.max))
    xs = data.draw(st.lists(st.integers(lo, hi), max_size=8))
    arr = np.array(xs, dtype=dtype)
    assert to_int(arr, spec).tolist() == [ref_int(x, *spec) for x in xs]
    assert fxp(arr, spec).tolist() == [ref_float(x, spec) for x in xs]
    if xs and dtype is not np.bool_:
        assert to_int(dtype(xs[0]), spec) == ref_int(xs[0], *spec)


@njit
def _jit_fxp_to_int(x, spec):
    return fxp(x, spec), to_int(x, spec)


@given(wide_specs(), int64s)
def test_njit_scalar_and_array(spec, x):
    y, n = _jit_fxp_to_int(x, spec)
    assert (y, n) == (ref_float(x, spec), ref_int(x, *spec))
    ya, na = _jit_fxp_to_int(np.array([x, 1]), spec)
    assert na.tolist() == [ref_int(x, *spec), ref_int(1, *spec)]
    assert ya.tolist() == [ref_float(x, spec), ref_float(1, spec)]


def test_examples():
    """Integers above 2**53 are no longer rounded to float64 before quantization."""
    assert to_int(2**58 + 1, Q(60, 0)) == 2**58 + 1
    assert to_int(np.array([2**58 + 1]), Q(60, 0)).tolist() == [2**58 + 1]
    assert to_hex(2**63 - 1, Q(64, 0)) == "7fffffffffffffff"  # was '8000000000000000' (rounded to 2**63, wrapped)
    assert to_int(2**63 - 1, Q(64, 0, ovf=ERROR)) == 2**63 - 1  # was an OverflowError
    assert to_int(np.uint64(2**64 - 1), Q(64, 0, ovf=SAT)) == 2**63 - 1
    assert to_int(np.array([2**64 - 1], dtype=np.uint64), Q(63, 0, False)).tolist() == [2**63 - 1]
    assert to_int(2**100 + 3, Q(4, 0)) == 3
    assert to_int(7, Q(8, -2, rnd=HALF_EVEN)) == 2
    assert fxpt(10**30, 8, 0, True, 0, SAT) == 127.0


def test_reference_agrees_with_float_reference():
    """`ref_int` is the same model as `ref_fxp` in test_properties."""
    spec = Q(20, 4)
    for x in range(-(2**20), 2**20, 997):
        assert ref_float(x, spec) == ref_fxp(x, *spec)
