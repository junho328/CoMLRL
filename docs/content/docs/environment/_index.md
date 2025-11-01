---
title: Environments
weight: 5
bookCollapseSection: false
bookFlatSection: true
---

CoMLRL environments define the task interface (inputs, feedback, and rewards) that multi‑agent trainers interact with. They differ in constraints, evaluation, and suitable reward signals.

## Purpose

- Provide consistent data shapes (e.g., a `prompt` field) and optional feedback hooks between turns.
- Support both single‑turn (one shot) and multi‑turn (iterative refinement) pipelines.
- Enable safe, test‑driven evaluation for code tasks and heuristic metrics for open‑ended text.

## Families

- Writing Collaboration: open‑ended text (summaries, stories, articles). Rewards can be heuristic (length/diversity) or model‑/metric‑based (e.g., ROGUE proxies).
- Code Generation: create a full solution from a specification (e.g., MBPP, HumanEval). Rewards are typically test‑pass rates from a sandbox.
- Code Completion: fill in a missing region within existing code (e.g., ClassEval). Rewards emphasize compilation and tests while respecting surrounding context and style.

## Repositories

- Writing Collaboration: https://github.com/OpenMLRL/LLM_Collab_Writing
- Code Generation: https://github.com/OpenMLRL/LLM_Collab_Code_Generation
- Code Completion: https://github.com/OpenMLRL/LLM_Collab_Code_Completion

## Key differences

- Generation vs. Completion: generation synthesizes a solution from a spec; completion must integrate into a given scaffold. Completion is more constrained and sensitive to context, identifiers, and imports.
- Feedback: code tasks often have executable tests (deterministic signals). Writing tasks rely on proxies or external critique.
- Multi‑turn: code tasks benefit from attaching failing tests as feedback; writing tasks benefit from role critiques and revision prompts.

See the pages below for dataset notes, reward options, and pipeline suggestions.
