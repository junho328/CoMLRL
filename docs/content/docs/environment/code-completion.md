---
title: Code Completion
weight: 3
---

## Overview

Agents complete partial code with context (e.g., ClassEval). Collaboration helps propose alternatives and reconcile constraints.

## Repository

https://github.com/OpenMLRL/LLM_Collab_Code_Completion

## Design goals

- Fill in a target region while preserving compilation and semantics.
- Leverage surrounding context (imports, types, identifiers) to constrain choices.
- Coordinate roles (e.g., draft vs. reviewer) to reduce regressions.

## Datasets

- ClassEval and internal completion suites.

## Data format

- Context block(s) and a `prompt` describing the target snippet.
- Provide minimal reproducible context for determinism.

## Reward signals

- Completion compiles, unit tests pass, semantic checks succeed.

## Example pipeline

- Single‑Turn for quick completions; Multi‑Turn to iterate on failing tests.

## How it differs from Code Generation

- Operates inside a fixed scaffold; must obey existing signatures and imports.
- Lower degrees of freedom than generation; stronger emphasis on compilation and style consistency.
- Evaluation often reuses the project’s tests rather than standalone specs.

## Tips

- Favor shorter generations; steer sampling with temperature/top‑p.
