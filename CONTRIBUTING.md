# Contributing to CAG

Thanks for your interest in improving CAG.

This repository is currently a **GitHub preview** focused on AI builders exploring grounded document reasoning, retrieval orchestration, and evaluation.

## Before You Start

- read [README.md](README.md)
- check existing issues before opening a new one
- prefer small, focused pull requests
- avoid unrelated cleanup in the same change

## Local Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Frontend:

```bash
cd frontend
npm install
```

## Expected Checks

Before opening a pull request, run:

```bash
pytest
```

```bash
cd frontend
npm run build
```

## Contribution Priorities

High-signal contributions for this preview:
- reproducibility improvements
- benchmark clarity and evaluation rigor
- grounded-answer reliability
- onboarding and docs improvements
- issue repro fixes with tight diffs

Lower priority for this preview:
- major product pivots
- unrelated refactors
- hosted deployment assumptions
- package publishing work

## Pull Request Guidelines

- keep changes scoped
- explain the user-visible or benchmark-visible impact
- mention any API or schema changes clearly
- update docs when the workflow or public behavior changes

## Reporting Sensitive Issues

For security-related reports, see [SECURITY.md](SECURITY.md).
