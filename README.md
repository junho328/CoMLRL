# CoMLRL

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml)
[![pre-commit.ci](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face](https://img.shields.io/badge/huggingface-CoMLRL-yellow.svg)](https://huggingface.co/CoMLRL)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04652-b31b1b.svg)](https://arxiv.org/pdf/2508.04652)

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning.

<details>
<summary>LLM Collaboration</summary>
<p>
LLM collaboration refers to the problems where LLMs cooperatively solve a class of tasks in MAS. The tasks are specified in natural language and provided to the agents as prompts. Each decentralized LLM agent generates a response synchronously based on its individual instructions. The set of responses jointly forms a solution to the task. Most tasks cannot be resolved in one turn. Users, external models, or systems validate the solutions and provide additional requirements or suggestions for LLMs. These components also serve as part of the environment for LLM collaboration, whose states may change based on the agents’ outputs. The updates are embedded into prompts for subsequent turns. This iterative process continues until the task is successfully completed or a predefined turn limit is reached.
</p>
</details>

<details>
<summary>Dec-POMDP</summary>
<p>
The cooperative MARL problem can be generally represented as a Decentralized Partially Observable MDP (Dec-POMDP). In the decentralized fully observable MDP cases, the methods can also fit by replacing histories with states. A <strong>Dec-POMDP</strong> is a tuple ⟨𝕀, 𝕊, {𝔸ᵢ}, T, R, {𝒪ᵢ}, O, 𝒯, γ⟩ where 𝕀 is a finite set of agents; 𝕊 is a set of states with designated initial state distribution b₀; 𝔸ᵢ is a set of actions for agent i with 𝔸 = ×ᵢ𝔸ᵢ being the set of joint actions; T is the state transition probability function, T: 𝕊 × 𝔸 × 𝕊 → [0,1], that specifies the probability of transitioning from state s ∈ 𝕊 to s′ ∈ 𝕊 when actions a ∈ 𝔸 are taken by agents (i.e., T(s, a, s′) = P(s′|s, a)); R is the joint reward function, where R: 𝕊 × 𝔸 → ℝ; 𝒪ᵢ is a set of observations for each agent i, with 𝒪 = ×ᵢ𝒪ᵢ as the set of joint observations; O is an observation probability function, O: 𝔸 × 𝕊 × 𝒪, that specifies the probability of seeing observation o′ ∈ 𝒪 given actions a ∈ 𝔸 are taken and state s′ ∈ 𝕊 is observed (i.e., O(a, s′, o′) = P(o′|a, s′)); 𝒯 is the finite time horizon; and γ is the discount factor. A solution to a Dec-POMDP is a joint policy π with πᵢ as the local policy. In a Dec-POMDP, since the state is usually not observed directly, it is typically beneficial for each agent to save the history of its observations, τᵢ = {aᵢ,₀, oᵢ,₁, ⋯, oᵢ,ₜ₋₁}, and τ = ⟨τᵢ⟩ᵢ∈𝕀.
</p>
</details>

## Setup

```bash
cd CoMLRL
conda create -n comlrl python=3.10
conda activate comlrl
pip install -r requirements.txt # torch must be compatible with device
pip install -e .
```

## Usage

See scripts in `examples/` for usage examples.

## Algorithms

- **MAGRPO:** Multi-Agent Group-Relative Policy Optimization, credits to [GRPO](https://arxiv.org/pdf/2402.03300),[Dr. GRPO](https://arxiv.org/abs/2503.20783), and [TreeRPO](https://arxiv.org/abs/2506.05183):

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|B|}\frac{1}{|\mathcal{G}|}\sum_{h_i^\mathcal{G} \in B}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \text{mean}(R^{\mathcal{G}}_t)\Big)\Bigg];
$$

- **MARLOO:** Multi-Agent REINFORCE Leave-One-Out, credits to [RLOO](https://openreview.net/forum?id=r1lgTGL5DE) and [Revisiting REINFORCE](https://arxiv.org/abs/2402.14740):

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|B|}\frac{1}{|\mathcal{G}|}\sum_{h_i^\mathcal{G} \in B}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \sum_{k\in \mathcal{G}, k\neq g}\frac{R^{(k)}_t}{|\mathcal{G}|-1}\Big)\Bigg];
$$

- **MAReMax:** Multi-Agent REINFORCE with Group Max, credits to [ReMax](https://arxiv.org/abs/2310.10505):

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|B|}\frac{1}{|\mathcal{G}|}\sum_{h_i^\mathcal{G} \in B}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big)\Bigg];
$$

- More algs are coming soon!

## Environments

This library supports LLM collaboration in various environments:

- [Code Generation](https://github.com/OpenMLRL/LLM_Collaboration_Code_Generation)
  - MBPP
  - HumanEval
  - CoopHumanEval
- [Code Completion](https://github.com/OpenMLRL/LLM_Collaboration_Code_Completion)
  - ClassEval
- [Bug Fix](https://github.com/OpenMLRL/LLM_Collaboration_Software_Engineering)


## Contributors

We would like to thank all contributors to this project.
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=80" width="60px;" alt=""/><br /><sub><b>Shuo Liu</b></sub></a><br /><a href="#ideas" title="Ideas">🤔</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a></td>
    <td align="center"><a href="https://github.com/Tenshi0x0"><img src="https://avatars.githubusercontent.com/u/105730496?v=4?s=80" width="60px;" alt=""/><br /><sub><b>Tianle Chen</b></sub></a><br /><a href="https://github.com/OpenMLRL/CoMLRL/commits?author=Tenshi0x0" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
<td align="center"><a href="https://github.com/ryankamiri"><img src="https://avatars.githubusercontent.com/u/44690200?v=4?s=80" width="60px;" alt=""/><br /><sub><b>Ryan Amiri</b></sub></a><br /><a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=ryankamiri" title="Code">💻</a> </td>
    <td align="center"><a href="https://github.com/zedyelllion"><img src="https://avatars.githubusercontent.com/u/111674669?v=4?s=80" width="60px;" alt=""/><br /><sub><b>Zeyu Liang</b></sub></a><br /> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs"></a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
 </tr>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->
