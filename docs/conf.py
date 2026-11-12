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
    # The getting-started page has its own title: start at the first "##" section,
    # after the README's badges, title and intro (the landing page covers those).
    text = text[text.index("\n## ") + 1 :]
    # Links into docs/ are written relative to the repository root for GitHub.
    return text.replace("](docs/", "](")


def _insert_readme(app, docname, source):  # noqa: ARG001
    if docname == "getting-started":
        source[0] = source[0].replace("<!-- README -->", _readme_for_docs())


def setup(app):
    app.connect("source-read", _insert_readme)
