# Shared Parameter Support for MAGRPO

## Overview

This branch (`koh-dev/shared`) adds native shared parameter support to the `MAGRPOTrainer`. When multiple agents share the same model instance, the trainer automatically detects this and handles gradient accumulation correctly.

## How It Works

### 1. Automatic Detection

When `MAGRPOTrainer` is initialized, it checks if all agents are the same model instance:

```python
unique_agent_ids = set(id(agent) for agent in self.agents)
self._shared_params = len(unique_agent_ids) == 1 and self.num_agents > 1
```

If all agents have the same `id()`, shared parameter mode is activated automatically.

### 2. Single Optimizer

Instead of creating separate optimizers per agent, a single optimizer is created:

```python
if self._shared_params:
    self._single_optimizer = torch.optim.AdamW(
        self.agents[0].parameters(),
        lr=self.args.learning_rate,
        weight_decay=self.args.weight_decay,
    )
    self.optimizers = [self._single_optimizer] * self.num_agents
```

### 3. Gradient Accumulation

The `_update_from_samples` method handles gradient accumulation across agents:

- **First agent (agent_idx=0)**: Zeros gradients
- **All agents**: Compute loss and accumulate gradients (scaled by `1 / (num_samples * num_agents)`)
- **Last agent**: Performs optimizer step

```python
if self._shared_params:
    if agent_idx == 0:
        self._single_optimizer.zero_grad()
        self._shared_grad_step_counter = 0

    scale = 1.0 / (len(samples) * self.num_agents)
    for sample in samples:
        loss = self._compute_loss_with_gradients(...)
        (loss * scale).backward()

    self._shared_grad_step_counter += 1
    if self._shared_grad_step_counter >= self.num_agents:
        self._single_optimizer.step()
```

### 4. Model Saving

When saving, shared models are saved once to `shared_model/` directory instead of duplicating:

```python
if self._shared_params:
    agent_dir = f"{output_dir}/shared_model"
    self.agents[0].save_pretrained(agent_dir)
```

## Usage

### In Training Script

Simply pass the same model instance for all agents:

```python
from transformers import AutoModelForCausalLM
from comlrl.trainers.magrpo import MAGRPOTrainer, MAGRPOConfig

# Load model ONCE
shared_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Create agents list with same instance
num_agents = 2
agents = [shared_model] * num_agents

# Trainer auto-detects shared params
trainer = MAGRPOTrainer(
    agents=agents,
    num_agents=num_agents,
    # ... other config
)

# Console output: "[SharedParams] Detected shared model across 2 agents"
trainer.train()
```

### Benefits

1. **~50% Memory Reduction**: Single model instead of multiple copies
2. **Automatic**: No manual optimizer wrapping needed
3. **Correct Gradients**: Proper accumulation and scaling
4. **Backward Compatible**: Standard multi-agent training unchanged

## Files Modified

- `comlrl/trainers/magrpo.py`:
  - `__init__`: Added shared parameter detection and single optimizer creation
  - `_update_from_samples`: Added gradient accumulation logic
  - `save_model`: Added efficient single-save for shared models

## Testing

Run with shared parameters:
```bash
cd /workspace/LLM_Collab_Writing
python sharedparam_train_magrpo.py --config configs/magrpo_arxiv_config.yaml
```

Expected console output:
```
[SharedParam] Loading single shared model: Qwen/Qwen3-0.6B
[SharedParam] Model loaded once, shared by 2 agents
[SharedParams] Detected shared model across 2 agents
```

## Branch Information

- Repository: `junho328/CoMLRL`
- Branch: `koh-dev/shared`
- Commit: `95bac0b`
