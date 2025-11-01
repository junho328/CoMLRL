---
title: Code Generation
weight: 2
---
## Overview

Agents produce code solutions from natural language prompts (e.g., MBPP, HumanEval). Collaboration can improve reasoning, test‑driven repair, or modular design.

## Repository

https://github.com/OpenMLRL/LLM_Collab_Code_Generation

## Design goals

- Synthesize complete functions/programs from a textual spec and examples.
- Encourage correctness via unit tests and safe sandbox execution.
- Support cooperative roles (e.g., planner + implementer, solver + tester).

## Datasets

- MBPP, HumanEval, CoopHumanEval; internal task suites.

## Data format

- Expected field: `prompt` (string). Some tasks include tests for evaluation.

## Reward signals

- Unit‑test pass ratio, compilation success, style/lint signals, heuristic rewards.

## Example pipeline

- Multi‑Turn with External feedback (`level_feedback`) to attach failing tests and diagnostics between turns.

## How it differs from Code Completion

- Starts from a description/spec; output can define new functions/classes freely.
- Fewer structural constraints; can rename or choose APIs as long as tests pass.
- Completion, by contrast, must fit into a fixed scaffold and compile alongside existing code.

## Tips

- Sandboxed execution is essential for safety; limit test slices (`external.sandbox_slice`).
- Cache tool outputs to avoid flaky signals.
