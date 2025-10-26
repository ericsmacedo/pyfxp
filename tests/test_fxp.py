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
"""Tests package against reference data."""

from pathlib import Path

import numpy as np
import pytest

from pyfxp import Q, fxp, fxpt
from pyfxp._pyfxp import _rnd_array, _rnd_scalar

from pyfxp.constants import HALF_EVEN

PRJ_PATH = Path(__file__).parent.parent


def parse_test_line(line):
    """Parse a line of test data into individual parameters."""
    parts = line.strip().split(", ")
    return {
        "x": float(parts[0]),
        "qi": int(parts[1]),
        "qf": int(parts[2]),
        "rnd": int(parts[3]),
        "ovf": int(parts[4]),
        "signed": parts[5] == "True",
        "expected": float(parts[6]),
    }


def pytest_generate_tests(metafunc):
    """Generate tests from test_data.txt."""
    if "test_case" in metafunc.fixturenames:
        test_file = PRJ_PATH / "tests" / "test_data.txt"
        lines = test_file.read_text().splitlines()[1:]  # Skip header
        test_cases = [parse_test_line(line) for line in lines if line.strip()]
        metafunc.parametrize("test_case", test_cases)


def test_fxp_temp_with_file_data(test_case):
    """Compares output of fxp against test data."""
    spec = Q(
        qi=test_case["qi"],
        qf=test_case["qf"],
        rnd=test_case["rnd"],
        ovf=test_case["ovf"],
        signed=test_case["signed"],
    )
    result = fxp(
        x=test_case["x"],  # Use dictionary keys
        spec=spec,
    )
    assert result == test_case["expected"], f"Failed for test_case: {test_case}"


def test_func_raises_valueerror_scalar():
    """Checks the half-even rnd method."""
    x_fxp = fxpt(x=5.5, qi=8, qf=0, rnd=HALF_EVEN)

    assert x_fxp == 6
