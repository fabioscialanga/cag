# Release Checklist

Use this before tagging a public release.

## Local Checks

```bash
pip install -e ".[dev,eval,api]"
pytest
python -m build
cd frontend
npm run build
```

## Smoke Checks

```bash
cag ingest --data-dir ./data/benchmark_corpus --reset
cag query "What is the minimum RAM required to run Nexus Platform?" --json
cag demo --reset --json
cag eval --system cag --limit 3 --judge-mode off
```

## Versioning

- update `pyproject.toml`
- update `src/cag/__init__.py`
- update `CHANGELOG.md`
- tag as `vX.Y.Z`

## Package Plan

Current status:

- local install is supported with `pip install -e .`
- GitHub install can use `pip install "git+https://github.com/fabioscialanga/cag.git"`
- PyPI name `cag` is already occupied by another project, so this repo should not publish under that name
- `experiments/`, generated artifacts, and local Chroma data are excluded from release packages via `MANIFEST.in`

Recommended first public package path:

- keep the import package as `cag` for now
- choose a distinct distribution name before PyPI release, for example `cag-reasoning`, `cag-ai`, or `cognitive-augmented-generation`
- use first PyPI version `0.3.0` after the GitHub preview stabilizes
- publish only after local `python -m build` and CI package build pass

Release commands once the distribution name is final:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## GitHub Release

Include:

- short summary
- upgrade notes
- quickstart command
- benchmark note
- known limits
