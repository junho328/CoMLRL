---
title: Contributing
weight: 2
---
## Overview

Contributions are welcome. Please start with an issue or draft PR describing the change and motivation.

## How to contribute

- Fork the repo and create a feature branch.
- Keep changes focused; add tests where practical.
- Open a PR early for feedback.

## Code style

- Follow PEP8; run `pre-commit` hooks locally if possible.
- Keep docs and examples up to date when APIs change.

## Testing

- Add minimal tests for new functionality.
- Prefer deterministic examples and seeds.

## Pre-commit

- Configure and run `pre-commit` to format/lint.

## PR checklist

- Description and motivation
- Tests (or rationale if not applicable)
- Docs updated

## Local setup

Use a fresh environment and install in editable mode:

```bash
cd CoMLRL
conda create -n comlrl python=3.10
conda activate comlrl
pip install -r requirements.txt # torch need to be compatible
pip install -r requirements.txt  # ensure torch wheel is compatible
pip install -e .
```
