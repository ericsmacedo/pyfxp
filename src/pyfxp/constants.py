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
"""Fixed-point arithmetic constants."""

# Integer codes (for Numba performance)
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

WRAP = 0
"""Overflow wraps around."""

SAT = 1
"""Overflow saturates to max/min representable value."""

ERROR = 2
"""Overflow raises an error."""

rounding_modes = {
    "TRUNC": TRUNC,
    "CEIL": CEIL,
    "TO_ZERO": TO_ZERO,
    "AWAY": AWAY,
    "HALF_UP": HALF_UP,
    "HALF_DOWN": HALF_DOWN,
    "HALF_EVEN": HALF_EVEN,
    "HALF_ZERO": HALF_ZERO,
    "HALF_AWAY": HALF_AWAY,
}
"""Dictionary mapping rounding mode names to their numeric codes."""

overflow_modes = {
    "WRAP": WRAP,
    "SAT": SAT,
    "ERROR": ERROR,
}
"""Dictionary mapping overflow mode names to their numeric codes."""

rounding_modes_inv = {v: k for k, v in rounding_modes.items()}
"""Inverse mapping of rounding mode codes to names."""

overflow_modes_inv = {v: k for k, v in overflow_modes.items()}
"""Inverse mapping of overflow mode codes to names."""
