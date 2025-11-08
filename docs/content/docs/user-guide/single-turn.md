---
title: Single-Turn Training
linkTitle: Single-Turn Training
weight: 2
math: true
---

## Overview

Single‑turn setups optimize agents for one round of generation per sample. The key control is how to form joint actions from each agent’s {{< katex inline=true >}}G{{< /katex >}} generations.

## Joint Mode

- align (default): pairs the g‑th generation of every agent → {{< katex inline=true >}}G{{< /katex >}} joint actions per node.
- cross: Cartesian product within a node → {{< katex inline=true >}}G^N{{< /katex >}} joint actions per node (N agents).

{{% hint %}}
Choosing align vs. cross: Align uses fewer sibling evaluations per node, leading to faster wall‑time and is a good default. Cross compares more siblings per node for better value estimation while using the same VRAM, since it reuses the same {{< katex inline=true >}}G{{< /katex >}} generations and only crosses them within the node.
{{% /hint %}}

We never cross across different nodes; this maintains causal consistency and correct credit assignment.

## Practical tips

- Start small: {{< katex inline=true >}}G \in [2, 4]{{< /katex >}} and ensure throughput is acceptable.
- Keep prompts consistent across agents if you intend symmetric roles.
- Add light diversity to agent prompts if specialization helps the task.

## Limitations

- Single‑turn ignores iterative refinement. For tasks benefiting from feedback cycles, use Multi‑Turn instead.
