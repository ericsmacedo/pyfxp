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
"""Fixed-point arithmetic constants.

Rounding and overflow modes are `IntEnum` members: they print by name
(`<Rounding.HALF_EVEN: 6>`) but are plain integers everywhere else, so the
numeric codes (`rnd=6`) keep working and they can be used inside `@njit` code.
"""

from enum import IntEnum


class Rounding(IntEnum):
    """Rounding methods (the `rnd` argument)."""

    TRUNC = 0
    """Bit truncation. Rounds toward negative infinity."""

    CEIL = 1
    """Round toward positive infinity."""

    TO_ZERO = 2
    """Round toward zero."""

    AWAY = 3
    """Round away from zero."""

    HALF_UP = 4
    """Round to nearest; ties round toward positive infinity."""

    HALF_DOWN = 5
    """Round to nearest; ties round toward negative infinity."""

    HALF_EVEN = 6
    """Round to nearest; ties to even."""

    HALF_ZERO = 7
    """Round to nearest; ties toward zero."""

    HALF_AWAY = 8
    """Round to nearest; ties away from zero."""


class Overflow(IntEnum):
    """Overflow handling methods (the `ovf` argument)."""

    WRAP = 0
    """Overflow wraps around."""

    SAT = 1
    """Overflow saturates to max/min representable value."""

    ERROR = 2
    """Overflow raises an error."""


TRUNC = Rounding.TRUNC
"""Bit truncation. Rounds toward negative infinity."""

CEIL = Rounding.CEIL
"""Round toward positive infinity."""

TO_ZERO = Rounding.TO_ZERO
"""Round toward zero."""

AWAY = Rounding.AWAY
"""Round away from zero."""

HALF_UP = Rounding.HALF_UP
"""Round to nearest; ties round toward positive infinity."""

HALF_DOWN = Rounding.HALF_DOWN
"""Round to nearest; ties round toward negative infinity."""

HALF_EVEN = Rounding.HALF_EVEN
"""Round to nearest; ties to even."""

HALF_ZERO = Rounding.HALF_ZERO
"""Round to nearest; ties toward zero."""

HALF_AWAY = Rounding.HALF_AWAY
"""Round to nearest; ties away from zero."""

WRAP = Overflow.WRAP
"""Overflow wraps around."""

SAT = Overflow.SAT
"""Overflow saturates to max/min representable value."""

ERROR = Overflow.ERROR
"""Overflow raises an error."""

rounding_modes = {m.name: m for m in Rounding}
"""Dictionary mapping rounding mode names to their members."""

overflow_modes = {m.name: m for m in Overflow}
"""Dictionary mapping overflow mode names to their members."""

rounding_modes_inv = {m.value: m.name for m in Rounding}
"""Inverse mapping of rounding mode codes to names."""

overflow_modes_inv = {m.value: m.name for m in Overflow}
"""Inverse mapping of overflow mode codes to names."""
