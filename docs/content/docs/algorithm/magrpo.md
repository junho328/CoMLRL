---
title: MAGRPO
weight: 2
math: true
---

## Overview

Multi‑Agent Group‑Relative Policy Optimization (MAGRPO) optimizes each agent with a group‑relative baseline computed among sibling joint actions at the same node/turn.

## Objective

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}\left[ \frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}}
\Big(R^{(g)}_t - \operatorname{mean}(R^{\mathcal{G}}_t)\Big)
\cdot \log \pi_{\theta_i}\big(a^{(g)}_{i,t} \mid h_{i,t}\big) \right].
{{< /katex >}}

- ## Siblings and baseline

- Sibling set size depends on Joint Mode and Multi‑Turn (see User Guide): align ⇒ \(G\), cross ⇒ \(G^N\).
- Group baseline is the mean over siblings at the same node/turn; this keeps the estimator unbiased and provides stable credit assignment.

## Configuration tips

- Prefer `align` initially for speed; try `cross` for more accurate estimates.
- Use modest `G` (e.g., 2–4) and small `max_new_tokens` to control cost.
- Pair with a simple reward processor (e.g., scaling) to keep signals in a convenient range.

## Cost and scalability

Runtime scales with the number of siblings per node and the number of leaves. Monitor GPU memory and iteration time; reduce T, G, or token lengths as needed.

## References

- GRPO: https://arxiv.org/pdf/2402.03300
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- TreeRPO: https://arxiv.org/abs/2506.05183
