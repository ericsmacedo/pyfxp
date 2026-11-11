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
"""Property-based tests against an exact (rational arithmetic) reference model."""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numba import njit

from pyfxp import FxpSpec, Q, fxp, fxpt
from pyfxp.constants import (
    AWAY,
    CEIL,
    ERROR,
    HALF_AWAY,
    HALF_DOWN,
    HALF_EVEN,
    HALF_UP,
    HALF_ZERO,
    SAT,
    TO_ZERO,
    TRUNC,
    WRAP,
    overflow_modes,
    rounding_modes,
)

# Numba compiles on first call, which can exceed hypothesis' default deadline.
settings.register_profile("pyfxp", deadline=None, max_examples=200)
settings.load_profile("pyfxp")

MAX_QF = 20
MAX_WIDTH = 48  # well below the 53-bit float mantissa, so the reference is exact


def _ref_round(v: Fraction, rnd: int) -> int:  # noqa: PLR0911
    """Round an exact rational to an integer."""
    half = Fraction(1, 2)
    if rnd == TRUNC:
        return math.floor(v)
    if rnd == CEIL:
        return math.ceil(v)
    if rnd == TO_ZERO:
        return math.trunc(v)
    if rnd == AWAY:
        return math.ceil(v) if v >= 0 else math.floor(v)
    if rnd == HALF_UP:
        return math.floor(v + half)
    if rnd == HALF_DOWN:
        return math.ceil(v - half)
    if rnd == HALF_EVEN:
        return round(v)  # Fraction.__round__ ties to even
    if rnd == HALF_ZERO:
        return math.ceil(v - half) if v >= 0 else math.floor(v + half)
    if rnd == HALF_AWAY:
        return math.floor(v + half) if v >= 0 else math.ceil(v - half)
    raise AssertionError(rnd)


def _limits(qi: int, qf: int, signed: bool) -> tuple[int, int]:
    w = qi + qf
    if signed:
        return -(1 << (w - 1)), (1 << (w - 1)) - 1
    return 0, (1 << w) - 1


def ref_fxp(x: float, qi: int, qf: int, signed: bool, rnd: int, ovf: int) -> float:  # noqa: PLR0913
    """Exact reference implementation of fxpt."""
    n = _ref_round(Fraction(x) * 2**qf, rnd)
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
    return float(Fraction(n, 2**qf))


@st.composite
def specs(draw, ovf_modes=(WRAP, SAT)):
    """Draw a valid fixed-point specification."""
    qf = draw(st.integers(0, MAX_QF))
    qi = draw(st.integers(0, MAX_WIDTH - qf))
    if qi + qf == 0:
        qi = 1
    return FxpSpec(
        qi=qi,
        qf=qf,
        signed=draw(st.booleans()),
        rnd=draw(st.sampled_from(sorted(rounding_modes.values()))),
        ovf=draw(st.sampled_from(ovf_modes)),
    )


def values(spec: FxpSpec):
    """Floats around the representable range of `spec`, plus exact ties."""
    span = 2.0 ** (spec.qi + 2)
    lsb = 2.0**-spec.qf
    plain = st.floats(-span, span, allow_nan=False, allow_infinity=False)
    ties = st.integers(-(2 ** (spec.qi + spec.qf + 1)), 2 ** (spec.qi + spec.qf + 1)).map(lambda k: (k + 0.5) * lsb)
    return st.one_of(plain, ties)


@st.composite
def spec_and_value(draw, ovf_modes=(WRAP, SAT)):
    """Draw a spec together with a value for it."""
    spec = draw(specs(ovf_modes))
    return spec, draw(values(spec))


@given(spec_and_value())
def test_scalar_matches_reference(case):
    """Scalar fxp equals the exact reference for every rounding/overflow mode."""
    spec, x = case
    assert fxp(x, spec) == ref_fxp(x, *spec)


@given(spec_and_value(ovf_modes=(ERROR,)))
def test_error_mode_matches_reference(case):
    """ERROR mode raises exactly when the reference overflows."""
    spec, x = case
    try:
        expected = ref_fxp(x, *spec)
    except OverflowError:
        with pytest.raises(OverflowError):
            fxp(x, spec)
    else:
        assert fxp(x, spec) == expected


@given(specs().flatmap(lambda s: st.tuples(st.just(s), st.lists(values(s), min_size=1, max_size=32))))
def test_array_matches_reference(case):
    """Array fxp equals the exact reference element-wise."""
    spec, xs = case
    out = fxp(np.array(xs), spec)
    assert out.tolist() == [ref_fxp(x, *spec) for x in xs]


@given(spec_and_value())
def test_output_is_representable(case):
    """Output lies on the 2**-qf grid and inside the format's range."""
    spec, x = case
    y = fxp(x, spec)
    n = Fraction(y) * 2**spec.qf
    lower, upper = _limits(spec.qi, spec.qf, spec.signed)
    assert n.denominator == 1
    assert lower <= n <= upper


@given(spec_and_value())
def test_idempotent(case):
    """Quantizing an already quantized value is a no-op."""
    spec, x = case
    y = fxp(x, spec)
    assert fxp(y, spec) == y


@given(spec_and_value(), st.floats(0, 4, allow_nan=False))
def test_sat_is_monotonic(case, delta):
    """With saturation, quantization preserves ordering."""
    spec, x = case
    spec = spec._replace(ovf=SAT)
    assert fxp(x, spec) <= fxp(x + delta, spec)


@given(spec_and_value())
def test_fxp_equals_fxpt(case):
    """fxp(x, Q(...)) and fxpt(x, ...) agree."""
    spec, x = case
    assert fxp(x, Q(*spec)) == fxpt(x, *spec)


@given(spec_and_value(), st.integers(-(2**20), 2**20))
def test_integer_input(case, k):
    """Python ints are handled like the equivalent float."""
    spec, _ = case
    assert fxp(k, spec) == fxp(float(k), spec)


def test_constants_are_complete():
    """Every mode exported by constants is covered by these tests."""
    assert sorted(rounding_modes.values()) == list(range(9))
    assert sorted(overflow_modes.values()) == [WRAP, SAT, ERROR]


@njit
def _user_kernel(x, spec_in, spec_out):
    acc = 0.0
    out = np.empty_like(x)
    xq = fxp(x, spec_in)
    for i in range(x.size):
        acc = fxp(acc + xq[i], spec_out)
        out[i] = acc
    return out


def _py_kernel(x, spec_in, spec_out):
    acc = 0.0
    out = []
    for v in x:
        acc = ref_fxp(acc + ref_fxp(v, *spec_in), *spec_out)
        out.append(acc)
    return out


@given(st.lists(st.floats(-8, 8, allow_nan=False), min_size=1, max_size=64))
def test_fxp_inside_user_njit_function(xs):
    """Q/fxp work inside user @njit code (scalar and array overloads)."""
    spec_in = Q(4, 5, True, TRUNC, SAT)
    spec_out = Q(7, 5, True, TRUNC, WRAP)
    assert _user_kernel(np.array(xs), spec_in, spec_out).tolist() == _py_kernel(xs, spec_in, spec_out)


@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("rnd", [HALF_UP, HALF_DOWN, HALF_EVEN, HALF_ZERO, HALF_AWAY])
def test_ties_exhaustive(rnd, signed):
    """All ties of a small format, scalar and array, against the reference."""
    qi, qf = 4, 2
    xs = [(k + 0.5) / 2**qf for k in range(-(2 ** (qi + qf)), 2 ** (qi + qf))]
    expected = [ref_fxp(x, qi, qf, signed, rnd, SAT) for x in xs]
    assert [fxpt(x, qi, qf, signed, rnd, SAT) for x in xs] == expected
    assert np.asarray(fxpt(np.array(xs), qi, qf, signed, rnd, SAT)).tolist() == expected


@st.composite
def wide_spec_and_value(draw):
    """Formats wider than the float mantissa (54..64 bits), with values whose scaled magnitude fits in int64."""
    signed = draw(st.booleans())
    w = draw(st.integers(54, 64 if signed else 63))
    qf = draw(st.integers(0, w))
    spec = FxpSpec(
        qi=w - qf,
        qf=qf,
        signed=signed,
        rnd=draw(st.sampled_from(sorted(rounding_modes.values()))),
        ovf=draw(st.sampled_from((WRAP, SAT))),
    )
    limit = 2.0 ** (62 - qf)  # keep x * 2**qf below 2**62
    x = draw(st.floats(-limit, limit, allow_nan=False, allow_infinity=False))
    return spec, x


@given(wide_spec_and_value())
def test_wide_formats_match_reference(case):
    """Formats up to 64 bits (signed) / 63 bits (unsigned) are exact, scalar and array."""
    spec, x = case
    expected = ref_fxp(x, *spec)
    assert fxp(x, spec) == expected
    assert np.asarray(fxp(np.array([x]), spec)).tolist() == [expected]


@pytest.mark.parametrize(
    ("qi", "qf", "signed", "expected"),
    [
        (40, 24, True, 1.0),  # 64-bit signed
        (40, 23, False, 1.0),  # 63-bit unsigned
        (4, 60, True, -(2.0**-30)),  # 64-bit signed, mostly fractional
    ],
)
@pytest.mark.parametrize("ovf", [WRAP, SAT])
@pytest.mark.parametrize("as_array", [False, True])
def test_wide_format_examples(qi, qf, signed, expected, ovf, as_array):
    """Known values at the maximum widths, which used to return garbage."""
    x = np.array([expected]) if as_array else expected
    y = fxpt(x, qi, qf, signed, TRUNC, ovf)
    assert np.asarray(y).tolist() == ([expected] if as_array else expected)


@pytest.mark.parametrize(
    ("qi", "qf", "signed"),
    [(40, 25, True), (40, 24, False), (70, 0, True), (0, 0, True), (0, 0, False)],
)
@pytest.mark.parametrize("as_array", [False, True])
def test_invalid_width_raises(qi, qf, signed, as_array):
    """Widths outside 1..64 (signed) / 1..63 (unsigned) raise instead of returning garbage."""
    x = np.array([1.0]) if as_array else 1.0
    with pytest.raises(ValueError, match="invalid format width"):
        fxpt(x, qi, qf, signed)
