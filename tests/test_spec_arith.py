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
"""Result formats of arithmetic: FxpSpec.add / sub / mul / neg / acc (issue #41)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyfxp import Q, fxp
from pyfxp.constants import ERROR, HALF_EVEN, SAT


@st.composite
def specs(draw, max_width=30):
    """Specs small enough that every result fits in 64 bits."""
    signed = draw(st.booleans())
    w = draw(st.integers(1, max_width))
    qf = draw(st.integers(-4, w + 4))
    return Q(w - qf, qf, signed, draw(st.integers(0, 8)), draw(st.integers(0, 2)))


def _int_range(spec, qf):
    """Integer range of `spec` in units of 2**-qf (qf >= spec.qf)."""
    return spec.min_int << (qf - spec.qf), spec.max_int << (qf - spec.qf)


def _fits(values, spec):
    return all(spec.min_int <= v <= spec.max_int for v in values)


def _is_minimal(values, spec):
    """No format with one bit less (same qf, same signedness rules) holds all values."""
    if spec.width == 1:
        return True
    smaller = spec._replace(qi=spec.qi - 1)
    narrower_unsigned = spec._replace(qi=spec.qi - 1, signed=False)
    return not _fits(values, smaller) and (spec.signed is False or not _fits(values, narrower_unsigned))


def _corners(a, b, op):
    if op == "mul":
        return [x * y for x in (a.min_int, a.max_int) for y in (b.min_int, b.max_int)]
    qf = max(a.qf, b.qf)
    (la, ha), (lb, hb) = _int_range(a, qf), _int_range(b, qf)
    if op == "add":
        return [la + lb, ha + hb]
    return [la - hb, ha - lb]  # sub


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@given(a=specs(), b=specs())
def test_binary_result_is_exact_and_minimal(op, a, b):
    """Every exact result fits the returned format, and one bit less would not be enough."""
    r = getattr(a, op)(b)
    values = _corners(a, b, op)
    assert r.qf == (a.qf + b.qf if op == "mul" else max(a.qf, b.qf))
    assert _fits(values, r)
    assert _is_minimal(values, r)
    assert r.signed == (min(values) < 0)
    assert (r.rnd, r.ovf) == (a.rnd, a.ovf)


@given(a=specs())
def test_neg(a):
    """-a fits, minimally; it needs one more bit for signed formats."""
    r = a.neg()
    values = [-a.max_int, -a.min_int]
    assert r.qf == a.qf
    assert _fits(values, r)
    assert _is_minimal(values, r)


@given(a=specs(), n=st.integers(1, 10_000))
def test_acc(a, n):
    """A sum of n values fits, minimally."""
    r = a.acc(n)
    values = [n * a.min_int, n * a.max_int]
    assert _fits(values, r)
    assert _is_minimal(values, r)


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@given(a=specs(max_width=20), b=specs(max_width=20), data=st.data())
def test_converting_the_result_is_exact(op, a, b, data):
    """fxp(x op y, result_spec) never overflows and never changes the exact result."""
    pick = st.sampled_from
    x = data.draw(st.one_of(pick([a.min, a.max]), st.integers(a.min_int, a.max_int).map(lambda k: k * a.lsb)))
    y = data.draw(st.one_of(pick([b.min, b.max]), st.integers(b.min_int, b.max_int).map(lambda k: k * b.lsb)))
    exact = {"add": x + y, "sub": x - y, "mul": x * y}[op]
    r = getattr(a, op)(b)._replace(ovf=ERROR)
    assert fxp(exact, r) == exact


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (lambda: Q(4, 5).add(Q(7, 5)), "sQ8.5(TRUNC, WRAP)"),
        (lambda: Q(4, 5).mul(Q(7, 5)), "sQ11.10(TRUNC, WRAP)"),
        (lambda: Q(4, 5).acc(1024), "sQ14.5(TRUNC, WRAP)"),
        (lambda: Q(4, 5).neg(), "sQ5.5(TRUNC, WRAP)"),
        (lambda: Q(1, 15).mul(Q(1, 15)), "sQ2.30(TRUNC, WRAP)"),
        (lambda: Q(4, 5, signed=False).add(Q(4, 5, signed=False)), "uQ5.5(TRUNC, WRAP)"),
        (lambda: Q(4, 5, signed=False).sub(Q(4, 5, signed=False)), "sQ5.5(TRUNC, WRAP)"),
        (lambda: Q(4, 5, signed=False).add(Q(4, 5)), "sQ6.5(TRUNC, WRAP)"),
        (lambda: Q(4, 5, signed=False).mul(Q(4, 5, signed=False)), "uQ8.10(TRUNC, WRAP)"),
        (lambda: Q(3, 13, True, HALF_EVEN, SAT).mul(Q(2, 6)), "sQ5.19(HALF_EVEN, SAT)"),
        (lambda: Q(4, -2).add(Q(3, 4)), "sQ5.4(TRUNC, WRAP)"),
    ],
)
def test_known_results(expr, expected):
    """Hand-checked examples, including mixed signedness and negative qf."""
    assert repr(expr()) == expected


@pytest.mark.parametrize(
    "expr",
    [lambda: Q(32, 0).mul(Q(33, 0)), lambda: Q(64, 0).add(Q(64, 0)), lambda: Q(40, 0).acc(2**30)],
)
def test_result_wider_than_64_bits_raises(expr):
    """A result that can't be represented says which operation and how many bits."""
    with pytest.raises(ValueError, match="more than the 64-bit limit"):
        expr()


def test_acc_needs_positive_n():
    """acc(0) has no meaning."""
    with pytest.raises(ValueError, match="n >= 1"):
        Q(4, 5).acc(0)
