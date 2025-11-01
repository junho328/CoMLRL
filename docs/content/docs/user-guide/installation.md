---
title: Installation
weight: 1
---

## Overview

Install a stable release from PyPI or the latest source from GitHub. Ensure your PyTorch wheel matches your CUDA/OS stack.

## Build From PyPI

```bash
pip install comlrl
```

## Build From Source

```bash
git clone https://github.com/OpenMLRL/CoMLRL
cd CoMLRL
pip install -r requirements.txt  # ensure torch is a compatible wheel
pip install -e .
```

## Requirements

- Python 3.10+
- PyTorch compatible with your GPU/CUDA (or CPU‑only wheel)
- Optional: Hugging Face account/tokens if pulling gated models

## Verify

```python
import comlrl
print(comlrl.__version__)
```

## Troubleshooting

- Torch install fails: select a wheel from https://pytorch.org matching your CUDA/OS.
- OOM/VRAM errors: try a smaller model, reduce `max_new_tokens`, or lower `num_generations`.
- Tokenizer/model trust: set `trust_remote_code=True` only for trusted repos.
