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

"""Sphinx configuration for the pyfxp documentation."""

import importlib
import inspect
import re
from importlib.metadata import version as package_version
from pathlib import Path

project = "pyfxp"
author = "ericsmacedo"
copyright = "2025-2026, ericsmacedo"  # noqa: A001
release = package_version("pyfxp")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

# Docstrings are NumPy style; `single backticks` render as inline code.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
default_role = "code"
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme = "furo"
html_title = "pyfxp"
html_theme_options = {
    "source_repository": "https://github.com/ericsmacedo/pyfxp",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ----------------------------------------------------------------------------- README as the getting-started page

_README = Path(__file__).parent.parent / "README.md"
_ALERT = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\n((?:>.*\n?)*)", re.MULTILINE)


def _alert_to_admonition(match: re.Match) -> str:
    """Turn a GitHub alert (``> [!NOTE]``) into a MyST admonition, so it renders in both places."""
    kind = match.group(1).lower()
    body = re.sub(r"^> ?", "", match.group(2), flags=re.MULTILINE).strip("\n")
    return f":::{{{kind}}}\n{body}\n:::\n"


def _readme_for_docs() -> str:
    text = _ALERT.sub(_alert_to_admonition, _README.read_text(encoding="utf-8"))
    # The getting-started page has its own title: start at "## Installation", and skip the
    # README sections the landing page already shows (between the two "docs:" markers).
    text = text[text.index("\n## Installation") + 1 :]
    text = re.sub(r"<!-- docs: the landing page.*?<!-- docs: end -->\n", "", text, flags=re.DOTALL)
    # Links into docs/ are written relative to the repository root for GitHub.
    return text.replace("](docs/", "](")


def _insert_readme(app, docname, source):  # noqa: ARG001
    if docname == "getting-started":
        source[0] = source[0].replace("<!-- README -->", _readme_for_docs())


# ----------------------------------------------------------------------------- llms.txt for AI assistants

# https://llmstxt.org/: llms.txt is a short index of the docs, llms-full.txt the whole text in one
# file. Both are written next to the HTML. Autodoc directives are replaced by the docstrings
# themselves, so llms-full.txt carries the complete API reference.

_BASE_URL = "https://pyfxp.readthedocs.io/en/stable/"
_SUMMARY = (
    "pyfxp is a bit-true fixed-point arithmetic library for Python. Values stay plain floats and "
    "NumPy arrays: fxp(x, Q(qi, qf, signed, rnd, ovf)) quantizes them to an ARM-style Q format (the sign "
    "bit counts in qi) with one of nine rounding modes and wrap / saturate / error on overflow. It is "
    "compiled with Numba and also works inside user @njit code."
)
_KEY_FACTS = """\
- Install: `pip install pyfxp`. Import: `from pyfxp import Q, fxp, fxpt, FxpSpec, to_int, from_int, to_hex`.
- `Q(qi, qf, signed=True, rnd=TRUNC, ovf=WRAP)` builds an `FxpSpec`; `fxp(x, spec)` quantizes;
  `fxpt(x, qi, qf, signed, rnd, ovf)` does the same without a spec.
- Q format is ARM style: `qi` includes the sign bit. `Q(1, 15)` is a 16-bit signed fraction in [-1, 1).
- Results are floats / float64 arrays on the 2**-qf grid, not a custom type. Arrays of any shape work.
- Modes live in `pyfxp.constants`: rounding `TRUNC, CEIL, TO_ZERO, AWAY, HALF_UP, HALF_DOWN, HALF_EVEN,
  HALF_ZERO, HALF_AWAY`; overflow `WRAP, SAT, ERROR`. They are IntEnums (`Rounding`, `Overflow`).
- Widths up to 64 bits (signed) / 63 bits (unsigned). NaN propagates; +-inf saturates with SAT.
- Bit growth: `a.add(b)`, `a.sub(b)`, `a.mul(b)`, `a.neg()`, `a.acc(n)` return the full-precision result format.
- Raw integers for HDL: `to_int`, `from_int`, `to_bin`, `to_hex`, `from_bin`, `from_hex`.
- Fast paths: call `fxp` on whole arrays, or call it per sample inside a Numba `@njit` function.
  Calling it per value from plain Python costs about 1 us per call."""

_AUTODOC = re.compile(r"```\{eval-rst\}\n\.\. auto(function|class|module):: (\S+)\n((?:   :.*\n)*)```")


_SIGNATURE_NAMES = {
    "float | int | numpy.integer | numpy.floating | numpy.ndarray | collections.abc.Sequence[float]": "Scalar | ArrayLike",
    "int | numpy.integer": "int",
    "pyfxp._pyfxp.FxpSpec": "FxpSpec",
    "<Rounding.TRUNC: 0>": "TRUNC",
    "<Overflow.WRAP: 0>": "WRAP",
    "'FxpSpec'": "FxpSpec",
    "(self, ": "(",
    "(self)": "()",
}


def _signature(obj) -> str:
    """inspect.signature with the type aliases and enum defaults written as in the docs."""
    text = str(inspect.signature(obj))
    for long, short in _SIGNATURE_NAMES.items():
        text = text.replace(long, short)
    return text


def _object(path: str):
    module, _, name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return importlib.import_module(path)


def _doc(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "")


def _render_autodoc(match: re.Match) -> str:
    """Plain-text stand-in for an autodoc directive: signatures and docstrings."""
    kind, path, options = match.group(1), match.group(2), match.group(3)
    obj = _object(path)
    if kind == "function":
        return f"### `{path}{_signature(obj)}`\n\n{_doc(obj)}\n"
    if kind == "class":
        members = re.search(r":members: (.*)", options)
        out = [f"### `{path}{_signature(obj)}`\n\n{_doc(obj)}\n"]
        for name in [m.strip() for m in members.group(1).split(",")] if members else []:
            member = inspect.getattr_static(obj, name)
            func = member.fget if isinstance(member, property) else member
            sig = "" if isinstance(member, property) else _signature(func)
            out.append(f"#### `{obj.__name__}.{name}{sig}`\n\n{_doc(func)}\n")
        return "\n".join(out)
    # automodule: module docstring, then each class and its documented members
    from sphinx.pycode import ModuleAnalyzer  # noqa: PLC0415

    attr_docs = ModuleAnalyzer.for_module(path).find_attr_docs()
    out = [_doc(obj) + "\n"]
    for name, cls in inspect.getmembers(obj, inspect.isclass):
        if cls.__module__ != path:
            continue
        out.append(f"### `{path}.{name}`\n\n{_doc(cls)}\n")
        for (scope, attr), lines in attr_docs.items():
            if scope == name:
                out.append(f"- `{name}.{attr}` = {int(getattr(cls, attr))}: {' '.join(lines).strip()}")
        out.append("")
    return "\n".join(out)


def _page_text(app, docname: str) -> str:
    text = Path(app.env.doc2path(docname)).read_text(encoding="utf-8")
    text = text.replace("<!-- README -->", _readme_for_docs())
    doc_dir = Path(app.env.doc2path(docname)).parent
    text = re.sub(  # inline MyST {include} directives
        r"```\{include\} (\S+)\n```", lambda m: (doc_dir / m.group(1)).read_text(encoding="utf-8"), text
    )
    text = _AUTODOC.sub(_render_autodoc, text)
    text = re.sub(r"```\{toctree\}.*?```\n?", "", text, flags=re.DOTALL)  # navigation only
    text = re.sub(r"^%.*\n", "", text, flags=re.MULTILINE)  # MyST comments
    return _plain_layout(text).strip() + "\n"


_ADMONITIONS = {"note", "tip", "important", "warning", "caution"}


def _plain_layout(text: str) -> str:
    """Turn layout-only MyST / sphinx-design markup into plain text (cards, buttons, images, fences)."""

    def image(match: re.Match) -> str:
        options = match.group(1)
        if "only-dark" in options:  # the same chart is described once, by its light version
            return ""
        alt = re.search(r":alt: (.*)", options)
        return f"[Image: {alt.group(1)}]\n" if alt else ""

    text = re.sub(r"```\{image\} \S+\n((?::[\w-]+:.*\n)*)```\n?", image, text)
    text = re.sub(r"```\{button-(?:ref|link)\}.*?```\n?", "", text, flags=re.DOTALL)

    def fence(match: re.Match) -> str:
        name, title = match.group(1), match.group(2).strip()
        if name == "grid-item-card":
            return f"**{title}**\n"
        if name in _ADMONITIONS:
            return f"**{name.title()}{': ' + title if title else ''}**\n"
        return ""  # grid, grid-item, container: layout only

    # opening colon fences with their option lines, then the closing fences
    text = re.sub(r"^:{3,}\{([\w-]+)\}(.*)\n(?::[\w-]+:.*\n)*", fence, text, flags=re.MULTILINE)
    text = re.sub(r"^:{3,}[ \t]*\n", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text)


def _doc_order(app) -> list[str]:
    """Documents in navigation order (depth-first through the toctrees)."""
    order, seen = [], set()

    def visit(doc):
        if doc in seen:
            return
        seen.add(doc)
        order.append(doc)
        for child in app.env.toctree_includes.get(doc, []):
            visit(child)

    visit(app.config.root_doc)
    return order


def _url(docname: str) -> str:
    if docname == "index":
        return _BASE_URL
    return _BASE_URL + docname.removesuffix("/index").removesuffix("index") + "/"


def _write_llms_txt(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    out = Path(app.outdir)
    docs = _doc_order(app)
    titles = {d: app.env.titles[d].astext() for d in docs}
    index = [f"# {project}", "", f"> {_SUMMARY}", "", _KEY_FACTS, "", "## Docs", ""]
    index += [f"- [{titles[d]}]({_url(d)})" for d in docs]
    index += ["", "## Optional", "", f"- [Full documentation as one text file]({_BASE_URL}llms-full.txt)", ""]
    (out / "llms.txt").write_text("\n".join(index), encoding="utf-8")

    full = [f"# {project} {release}", "", f"> {_SUMMARY}", "", _KEY_FACTS, ""]
    for d in docs:
        full += [f"<!-- page: {_url(d)} -->", "", _page_text(app, d)]
    (out / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")


def setup(app):
    app.connect("source-read", _insert_readme)
    app.connect("build-finished", _write_llms_txt)
