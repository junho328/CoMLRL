---
title: MAREINFORCE
weight: 1
math: true
---

## Overview

Multi‑Agent REINFORCE without a baseline.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} R^{(g)}_t \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t})\Bigg];
{{< /katex >}}

## Variants

- MARLOO: Multi‑Agent REINFORCE Leave‑One‑Out (RLOO / Revisiting REINFORCE). Baseline is the mean return of other agents (leave‑one‑out) at the same step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \sum_{k\in \mathcal{G},\, k\neq g}\tfrac{R^{(k)}_t}{|\mathcal{G}|-1} \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

- MAReMax: Multi‑Agent REINFORCE with Group Max (ReMax). Baseline is the maximum group return at the step.

{{< katex display=true >}}
J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
\Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big( R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big) \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}\mid h_{i,t}) \Bigg];
{{< /katex >}}

## When to use

- Simple baseline‑free training; good for small problems with dense rewards.
- Use as a reference point to compare baseline variants (MARLOO/MAReMax).

## Notes

- For sparse/noisy rewards, a baseline often stabilizes training (see variants).

## References

- RLOO (Leave‑One‑Out): https://openreview.net/forum?id=r1lgTGL5DE
- Revisiting REINFORCE: https://arxiv.org/abs/2402.14740
- ReMax: https://arxiv.org/abs/2310.10505
