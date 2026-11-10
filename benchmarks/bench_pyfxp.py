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

"""Benchmarks for pyfxp.

Run with::

    uv run inv bench
    uv run python benchmarks/bench_pyfxp.py --sizes 1000 100000 --repeat 7

Each case reports the best time per call and per element over ``--repeat`` runs.
Numba compilation is excluded by warming up every case first.
"""

import argparse
import platform
import timeit
from collections.abc import Callable

import numba
import numpy as np
from numba import njit

from pyfxp import Q, fxp, fxpt
from pyfxp.constants import HALF_EVEN, SAT, TRUNC, WRAP

QI, QF = 8, 12
SPECS = {
    "TRUNC/WRAP": Q(QI, QF, True, TRUNC, WRAP),
    "HALF_EVEN/SAT": Q(QI, QF, True, HALF_EVEN, SAT),
}


def numpy_baseline(x, qi, qf):
    """Plain NumPy TRUNC + SAT, as a reference point."""
    scale = 2.0**qf
    lim = 2.0 ** (qi + qf - 1)
    return np.clip(np.floor(x * scale), -lim, lim - 1) / scale


def py_loop(x, spec):
    """Element-wise fxp calls from Python."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = fxp(float(x[i]), spec)
    return out


@njit(cache=True)
def njit_loop(x, spec):
    """Element-wise fxp calls inside @njit (typical DSP-loop usage)."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = fxp(x[i], spec)
    return out


@njit(cache=True)
def njit_array(x, spec):
    """Whole-array fxp call inside @njit."""
    return fxp(x, spec)


@njit(cache=True)
def njit_integrator(x, spec_in, spec_out):
    """Fixed-point integrator, as in examples/integrator."""
    xq = fxp(x, spec_in)
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):
        acc = fxp(acc + xq[i], spec_out)
        out[i] = acc
    return out


def cases(spec) -> dict[str, tuple[Callable, bool]]:
    """Benchmark cases: name -> (function of x, is_slow)."""
    return {
        "fxp(array)            ": (lambda x: fxp(x, spec), False),
        "fxpt(array)           ": (lambda x: fxpt(x, spec.qi, spec.qf, spec.signed, spec.rnd, spec.ovf), False),
        "njit: fxp(array)      ": (lambda x: njit_array(x, spec), False),
        "njit: loop fxp(scalar)": (lambda x: njit_loop(x, spec), False),
        "njit: integrator      ": (lambda x: njit_integrator(x, spec, spec), False),
        "python: loop fxp      ": (lambda x: py_loop(x, spec), True),
    }


def bench(func, x, repeat):
    """Return the best time of one call, in seconds."""
    func(x)  # warm-up / JIT compile
    timer = timeit.Timer(lambda: func(x))
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def fmt(seconds):
    """Format a duration with a suitable unit."""
    for unit, scale in (("s", 1), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:8.2f} {unit} "
    return f"{seconds / 1e-9:8.2f} ns "


def main():
    """Run all benchmarks and print a table."""
    parser = argparse.ArgumentParser(description="Benchmarks for pyfxp.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 1_000_000])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-slow-size", type=int, default=100_000, help="skip slow Python loops above this size")
    args = parser.parse_args()

    print(f"python {platform.python_version()}, numpy {np.__version__}, numba {numba.__version__}")
    print(f"format: Q{QI}.{QF} signed, best of {args.repeat}\n")
    rng = np.random.default_rng(0)

    for n in args.sizes:
        x = rng.normal(0, 50, n)
        print(f"n = {n:,}")
        base = bench(lambda x: numpy_baseline(x, QI, QF), x, args.repeat)
        print(f"  {'numpy baseline (TRUNC/SAT)':<38}{fmt(base)}{fmt(base / n)}/elem")
        for spec_name, spec in SPECS.items():
            for name, (func, slow) in cases(spec).items():
                label = f"{name} {spec_name}"
                if slow and n > args.max_slow_size:
                    print(f"  {label:<38}{'skipped':>12}")
                    continue
                t = bench(func, x, args.repeat)
                print(f"  {label:<38}{fmt(t)}{fmt(t / n)}/elem  x{t / base:6.1f} vs numpy")
        print()


if __name__ == "__main__":
    main()
