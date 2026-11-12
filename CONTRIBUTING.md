# CONTRIBUTING

Please follow the github workflow. Create a ticket and/or branch. Create a pull-request.

See [github Documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for further information.
We follow the [Semantic Version Scheme](https://semver.org/).

## Local Development

### Installation

Please install these tools:

* [`uv` Installation](https://docs.astral.sh/uv/getting-started/installation/)
* [`git`](https://git-scm.com/)


### Testing

Run auto-formatting, linting, tests and documentation build:

```bash
uv run inv all
```

See `uv run inv --list` for any further details.

### Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and the
[Furo](https://pradyunsg.me/furo/) theme. Pages are Markdown (via MyST), and the
API reference is generated from the docstrings (NumPy style).

```bash
uv run inv doc          # build into docs/_build/dirhtml (warnings are errors)
uv run inv doc-serve    # live-reloading preview at http://127.0.0.1:8000
```

The landing page is the `README.md`. Use GitHub alerts (`> [!NOTE]`) for callouts
there: they render on GitHub and are turned into admonitions in the docs.

### Benchmarks

Run the benchmarks (best time per call and per element, compared to a plain NumPy baseline):

```bash
uv run inv bench
uv run python benchmarks/bench_pyfxp.py --sizes 1000 100000 --repeat 7
```

Run them before and after performance-related changes and include the numbers in the pull request.


## Project Structure

The project contains these files and directories:

| File/Directory | Description |
|---|---|
| `src/` | Python Package Sources - the files this is all about |
| `docs/` | Documentation sources (Sphinx, `docs/conf.py`) |
| `benchmarks/` | Performance benchmarks (`uv run inv bench`) |
| `pyproject.toml` | Python Package Meta File. Also contains all tool settings |
| `task.py` | Invoke tasks.py file |
| `.gitignore` | Lists of files and directories ignored by version control system |
| `.github/` | Github Settings |
| `.readthedocs.yaml` | Documentation Server Configuration |
| `.pre-commit-config.yaml` | Pre-Commit Check Configuration |
| `uv.lock` | File with resolved python package dependencies |

Next to that, there are some temporary files ignored by version control system.

| File/Directory | Description |
|---|---|
| `htmlcov/` | Test Execution Code Coverage Report in HTML format |
| `report.xml` | Test Execution Report |
| `.venv` | Virtual Environments |
