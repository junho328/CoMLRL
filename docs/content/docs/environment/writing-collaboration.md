---
title: Writing Collaboration
weight: 1
---
## Overview

Multiple agents co‑author content (e.g., summaries, stories, articles). Agents may assume complementary roles (outline vs. expansion) and collaborate across turns.

## Repository

https://github.com/OpenMLRL/LLM_Collab_Writing

## Design goals

- Improve coherence and coverage via role specialization.
- Support iterative critique and revision across turns.
- Allow heuristic or model‑based rewards when tests are unavailable.

## Datasets

- TLDR (summarization), custom prompt lists (story prompts), or domain corpora.

## Data format

- Expected field: `prompt` (string). You may extend with metadata fields as needed.

## Reward signals

- Length ratio (see Quick Demo), ROGUE/BLEU‑like metrics, lexical diversity, task‑specific heuristics.

## Example pipeline

- Single‑Turn with Joint Mode align to validate role prompts.
- Multi‑Turn with External feedback (e.g., critique) to refine drafts.

## Tips

- Keep role prompts concise and consistent.
- Start with small models and short outputs; scale up after sanity checks.
- Consider lightweight, measurable proxies (length/diversity) early; switch to stronger metrics later.
