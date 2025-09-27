# CoMLRL

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml)
[![pre-commit.ci](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face](https://img.shields.io/badge/huggingface-CoMLRL-yellow.svg)](https://huggingface.co/CoMLRL)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04652-b31b1b.svg)](https://arxiv.org/pdf/2508.04652)

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning.

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

- **MAGRPO:** Multi-Agent Group-Relative Policy Optimization
- More algs are coming soon!

## Environments

- [LLM Collaboration for Code Generation](https://github.com/OpenMLRL/LLM_Collaboration_Code_Generation)
  - MBPP
  - HumanEval
  - CoopHumanEval
- [LLM Collaboration for Module Completion](https://github.com/OpenMLRL/LLM_Collaboration_Module_Completion)
  - ClassEval
- [LLM Collaboration for Software Engineering](https://github.com/OpenMLRL/LLM_Collaboration_Software_Engineering)


## Contributors

We would like to thank all contributors to this project.
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Shuo Liu</b></sub></a><br /><a href="#ideas" title="Ideas">🤔</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a></td>
    <td align="center"><a href="https://github.com/Tenshi0x0"><img src="https://avatars.githubusercontent.com/u/105730496?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Tianle Chen</b></sub></a><br /><a href="https://github.com/OpenMLRL/CoMLRL/commits?author=Tenshi0x0" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
<td align="center"><a href="https://github.com/ryankamiri"><img src="https://avatars.githubusercontent.com/u/44690200?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Ryan Amiri</b></sub></a><br /><a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=ryankamiri" title="Code">💻</a> </td>
    <td align="center"><a href="https://github.com/zedyelllion"><img src="https://avatars.githubusercontent.com/u/111674669?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Zeyu Liang</b></sub></a><br /> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs"></a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
 </tr>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->
