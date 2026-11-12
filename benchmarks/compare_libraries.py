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
"""Compare pyfxp with other Python fixed-point libraries: same quantization, measured speed.

Every library quantizes the same values to signed Q8.12 (20 bits, 12 fractional),
rounding toward negative infinity (floor) and saturating on overflow. Before timing,
each library's results are checked against pyfxp; spfpm has no floor rounding or
saturation, so it is timed on in-range values and marked in the chart.

Run with the optional `bench` dependency group:

    uv run --group bench python benchmarks/compare_libraries.py            # measure + plot
    uv run --group bench python benchmarks/compare_libraries.py --quick    # smaller sizes
    uv run --group bench python benchmarks/compare_libraries.py --plot-only

Results go to benchmarks/results/compare_libraries.json; charts to
docs/_static/compare-light.svg and docs/_static/compare-dark.svg.
"""

import argparse
import json
import platform
import timeit
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numba import njit

from pyfxp import Q, fxp
from pyfxp.constants import SAT, TRUNC

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results" / "compare_libraries.json"
STATIC = ROOT / "docs" / "_static"

QI, QF = 8, 12
SPEC = Q(QI, QF, True, TRUNC, SAT)

# ----------------------------------------------------------------------------- one adapter per library


def _adapters() -> dict[str, dict]:
    """Array and scalar functions per library, all computing floor + saturate to sQ8.12."""
    import apytypes as apy  # ty: ignore[unresolved-import]  # noqa: PLC0415
    from fixedpoint import FixedPoint  # ty: ignore[unresolved-import]  # noqa: PLC0415
    from FixedPoint import FXfamily, FXnum  # ty: ignore[unresolved-import]  # noqa: PLC0415
    from fxpmath import Fxp  # ty: ignore[unresolved-import]  # noqa: PLC0415
    from numfi import numfi  # ty: ignore[unresolved-import]  # noqa: PLC0415

    fam = FXfamily(n_bits=QF, n_intbits=QI)

    def fxpmath_q(v):
        return Fxp(v, True, QI + QF, QF, rounding="floor", overflow="saturate").get_val()

    def numfi_q(v):
        return numfi(v, 1, QI + QF, QF, RoundingMethod="Floor", OverflowAction="Saturate").double

    def apy_array(x):
        # from_float rounds to nearest and wraps: convert exactly, then cast with TRN + SAT
        wide = apy.APyFixedArray.from_float(x, int_bits=QI + 12, frac_bits=64)
        return wide.cast(QI, QF, apy.QuantizationMode.TRN, apy.OverflowMode.SAT).to_numpy()

    def apy_scalar(v):
        wide = apy.APyFixed.from_float(v, int_bits=QI + 12, frac_bits=64)
        return float(wide.cast(QI, QF, apy.QuantizationMode.TRN, apy.OverflowMode.SAT))

    def fixedpoint_scalar(v):
        return float(FixedPoint(v, True, QI, QF, overflow="clamp", rounding="down", overflow_alert="ignore"))

    def spfpm_scalar(v):
        return float(FXnum(v, fam))

    def loop(scalar: Callable) -> Callable:
        return lambda x: np.array([scalar(v) for v in x.tolist()])

    return {
        "pyfxp": {"array": lambda x: fxp(x, SPEC), "scalar": lambda v: fxp(v, SPEC), "vectorized": True},
        "fxpmath": {"array": fxpmath_q, "scalar": lambda v: float(fxpmath_q(v)), "vectorized": True},
        "numfi": {"array": numfi_q, "scalar": lambda v: float(numfi_q(v).reshape(-1)[0]), "vectorized": True},
        "APyTypes": {"array": apy_array, "scalar": apy_scalar, "vectorized": True},
        "fixedpoint": {"array": loop(fixedpoint_scalar), "scalar": fixedpoint_scalar, "vectorized": False},
        "spfpm": {"array": loop(spfpm_scalar), "scalar": spfpm_scalar, "vectorized": False, "exact": False},
    }


@njit(cache=True)
def _pyfxp_njit_loop(x, spec):
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = fxp(x[i], spec)
    return out


# ----------------------------------------------------------------------------- measurement


def _verify(libs: dict) -> None:
    """Every library must give pyfxp's results (spfpm: only checked to run, see module docstring)."""
    rng = np.random.default_rng(1)
    in_range = rng.uniform(-127, 127, 2000)
    edge = np.array([127.99, 128.5, -128.0, -128.3, 300.0, -300.0, *((np.arange(-20, 20) + 0.5) / 4096)])
    for name, lib in libs.items():
        cases = (in_range,) if lib.get("exact") is False else (in_range, edge)
        for x in cases:
            ref = fxp(x, SPEC)
            arr = np.asarray(lib["array"](x), dtype=float)
            sca = np.array([lib["scalar"](v) for v in x[:200].tolist()], dtype=float)
            if lib.get("exact") is False:
                continue
            assert np.array_equal(arr, ref), f"{name}: array results differ from pyfxp"
            assert np.array_equal(sca, ref[:200]), f"{name}: scalar results differ from pyfxp"
        print(f"  {name:10s} {'matches pyfxp' if lib.get('exact', True) else 'runs (different semantics)'}")


def _best(func: Callable, repeat: int) -> float:
    func()  # warm-up (and Numba compilation)
    return min(timeit.repeat(func, number=1, repeat=repeat))


def measure(quick: bool = False) -> dict:
    """Time every library; returns ns per value for the array and the per-call scenarios."""
    libs = _adapters()
    print("verifying results against pyfxp ...")
    _verify(libs)

    n_array = 100_000 if quick else 1_000_000
    n_loop = 2_000 if quick else 20_000  # scalar-only libraries: Python loop, extrapolated per value
    n_calls = 500 if quick else 5_000
    repeat = 3 if quick else 11
    rng = np.random.default_rng(0)
    x = rng.uniform(-127, 127, n_array)  # in range, so spfpm runs too

    array, per_call = {}, {}
    for name, lib in libs.items():
        n = n_array if lib["vectorized"] else n_loop
        xa = x[:n]
        array[name] = _best(lambda lib=lib, xa=xa: lib["array"](xa), repeat) / n * 1e9
        m = min(n_calls, 1_000) if name == "fxpmath" else n_calls
        vals = x[:m].tolist()
        per_call[name] = _best(lambda lib=lib, vals=vals: [lib["scalar"](v) for v in vals], repeat) / m * 1e9
        print(f"  {name:10s} array {array[name]:10.1f} ns/value   one value per call {per_call[name]:10.0f} ns")
    per_call["pyfxp in @njit"] = _best(lambda: _pyfxp_njit_loop(x, SPEC), repeat) / n_array * 1e9
    print(f"  {'pyfxp @njit':10s} per value inside a compiled loop {per_call['pyfxp in @njit']:.1f} ns")

    return {
        "format": "signed Q8.12, floor rounding, saturation",
        "sizes": {"array": n_array, "scalar_only_loop": n_loop, "calls": n_calls, "repeat": repeat},
        "array_ns_per_value": array,
        "per_call_ns": per_call,
        "vectorized": {name: lib["vectorized"] for name, lib in libs.items()},
        "same_results_as_pyfxp": {name: lib.get("exact", True) for name, lib in libs.items()},
        "machine": {"python": platform.python_version(), "system": platform.platform(), "cpu": platform.processor()},
        "versions": {
            p: version(p) for p in ("pyfxp", "numpy", "numba", "fxpmath", "numfi", "apytypes", "fixedpoint", "spfpm")
        },
    }


# ----------------------------------------------------------------------------- chart

# One accent for pyfxp, gray for everything else (an emphasis chart). Validated with the
# dataviz palette checker against Furo's light (#ffffff) and dark (#131416) surfaces.
THEMES = {
    "light": {"accent": "#2a78d6", "other": "#8d8c86", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9"},
    "dark": {"accent": "#3987e5", "other": "#75746e", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2c2c2a"},
}


def _fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns" if ns >= 10 else f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1e3:.0f} µs" if ns >= 10_000 else f"{ns / 1e3:.1f} µs"
    return f"{ns / 1e6:.1f} ms"


def _panel(ax, rows: list[tuple[str, float, bool]], title: str, ref_ns: float, theme: dict) -> None:
    from matplotlib.ticker import NullLocator  # noqa: PLC0415

    rows = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [theme["accent"] if r[2] else theme["other"] for r in rows][::-1]
    y = np.arange(len(rows))
    ax.barh(y, values, height=0.56, color=colors, zorder=2)
    ax.set_xscale("log")
    lo, hi = min(values), max(values)
    ax.set_xlim(lo / 3, hi * 60)
    for yi, v, c in zip(y, values, colors, strict=True):
        ratio = v / ref_ns
        if c == theme["accent"]:
            note = ""
        elif ratio < 1.2:  # within run-to-run noise on a shared machine
            note = "  ·  about the same"
        else:
            note = f"  ·  {ratio:,.1f}× slower" if ratio < 10 else f"  ·  {ratio:,.0f}× slower"
        ax.text(v * 1.15, yi, _fmt(v) + note, va="center", ha="left", fontsize=9.5, color=theme["ink2"], zorder=3)
    ax.set_yticks(y, labels)
    for tick, c in zip(ax.get_yticklabels(), colors, strict=True):
        tick.set_color(theme["ink"])
        tick.set_fontweight("bold" if c == theme["accent"] else "normal")
    ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=8)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", colors=theme["ink2"], labelsize=8.5, length=0)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(lambda v, _pos: _fmt(v))
    ax.xaxis.set_minor_locator(NullLocator())


def plot(results: dict, out_dir: Path = STATIC) -> list[Path]:
    """Draw the comparison chart, once per Furo theme (transparent background)."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    exact = results["same_results_as_pyfxp"]
    vec = results["vectorized"]

    def name(lib: str, show_loop: bool) -> str:
        suffix = " (loop)" if show_loop and not vec.get(lib, True) else ""
        return lib + suffix + ("*" if not exact.get(lib, True) else "")

    arr = [(name(k, True), v, k == "pyfxp") for k, v in results["array_ns_per_value"].items()]
    call = [(name(k, False), v, k.startswith("pyfxp")) for k, v in results["per_call_ns"].items()]
    arr_ref = results["array_ns_per_value"]["pyfxp"]
    call_ref = results["per_call_ns"]["pyfxp in @njit"]
    n = results["sizes"]["array"]

    paths = []
    for mode, theme in THEMES.items():
        mpl.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), gridspec_kw={"hspace": 0.55})
        _panel(axes[0], arr, f"Quantize an array of {n:,} values: time per value", arr_ref, theme)
        _panel(axes[1], call, "Quantize one value at a time: time per value", call_ref, theme)
        fig.text(
            0.01,
            0.005,
            "Signed Q8.12, floor rounding, saturation. Lower is better. Best of "
            f"{results['sizes']['repeat']} runs.  * spfpm rounds to nearest and has no saturation.",
            fontsize=8,
            color=theme["ink2"],
        )
        fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.08)
        path = out_dir / f"compare-{mode}.svg"
        fig.savefig(path, transparent=True)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    """Measure (unless --plot-only), save the results and draw the charts."""
    parser = argparse.ArgumentParser(description="Compare pyfxp with other fixed-point libraries.")
    parser.add_argument("--quick", action="store_true", help="smaller sizes, fewer repeats")
    parser.add_argument("--plot-only", action="store_true", help=f"redraw the charts from {RESULTS.name}")
    args = parser.parse_args()

    if args.plot_only:
        results = json.loads(RESULTS.read_text())
    else:
        results = measure(quick=args.quick)
        RESULTS.write_text(json.dumps(results, indent=2) + "\n")
        print(f"results: {RESULTS.relative_to(ROOT)}")
    for p in plot(results):
        print(f"chart:   {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
