"""
HAVPPO: Heterogeneous-Agent Value-based Proximal Policy Optimization

This implements HAVPPO, which combines:
- HAPPO's sequential update scheme with importance ratio accumulation (M factor)
- PPO clip objective for policy optimization
- Centralized value network for advantage estimation (CTDE paradigm)

Key features:
1. Agents are updated sequentially, not simultaneously
2. Each agent's update uses M = cumulative product of previous agents' importance ratios
3. M is clipped to prevent extreme values
4. Advantage is estimated using a centralized value network (not GRPO's group normalization)
5. Policy models use LoRA adapters, value network uses a 2-layer MLP value head
"""

import inspect
import os
import random
from dataclasses import dataclass, field
import itertools
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    AutoModelForCausalLM,
)

from comlrl.models.actor_critic import CausalLMWithValueHead


@dataclass
class HAVPPOConfig(TrainingArguments):
    """
    Configuration for HAVPPO training, inheriting from TrainingArguments.
    Supports both single-turn and multi-turn training modes.
    """

    # Core setup
    num_train_epochs: float = field(
        default=20,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Per-device batch size (must be 1 for HAVPPO)."},
    )
    learning_rate: float = field(
        default=5.0e-6,
        metadata={"help": "Learning rate for policy optimizer."},
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "Log every N steps."},
    )
    save_steps: int = field(
        default=200,
        metadata={"help": "Save every N steps."},
    )
    num_agents: int = field(
        default=2,
        metadata={"help": "Number of agents; set to 1 for single-agent."},
    )

    # Sampling/generation
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations to sample per prompt for each agent."},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate after the prompt."},
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=0.6,
        metadata={"help": "Top-p for sampling."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "Top-k for sampling (set to None to disable)."},
    )

    # Multi-turn / tree rollout
    num_turns: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of turns per episode (set >1 for multi-turn with external transitions)."
        },
    )
    discount: float = field(
        default=0.9,
        metadata={"help": "Discount factor (gamma) over turns for returns."},
    )
    joint_mode: str = field(
        default="aligned",
        metadata={
            "help": "Joint action composition: 'cross' (Cartesian product) or 'aligned' (index-aligned)."
        },
    )
    termination_threshold: Optional[float] = field(
        default=-0.2,
        metadata={
            "help": "Early stop a branch if mean reward at a node exceeds this threshold."
        },
    )
    external_prompt_passthrough: bool = field(
        default=False,
        metadata={
            "help": "Use external prompts directly in multi-turn (skip formatter wrapping)."
        },
    )

    # HAPPO-specific parameters
    ppo_clip_eps: float = field(
        default=0.2,
        metadata={"help": "PPO clipping epsilon for importance ratio."},
    )
    m_clip_min: float = field(
        default=0.1,
        metadata={"help": "Minimum value for M factor clipping."},
    )
    m_clip_max: float = field(
        default=2.0,
        metadata={"help": "Maximum value for M factor clipping."},
    )
    shuffle_agent_order: bool = field(
        default=False,
        metadata={"help": "Whether to randomly permute agent update order each batch."},
    )
    reverse_agent_order: bool = field(
        default=False,
        metadata={
            "help": "Whether to reverse agent update order (main first, helper second). "
            "Ignored if shuffle_agent_order is True."
        },
    )
    use_ppo_clip: bool = field(
        default=True,
        metadata={
            "help": "Whether to use PPO-clip objective for current agent's update. "
            "If False, uses simple policy gradient (MAGRPO-style): loss = -log_prob * M. "
            "M factor accumulation from previous agents is still applied."
        },
    )

    # Value network parameters (HAVPPO-specific)
    value_head_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for value head MLP."},
    )
    value_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for value head optimizer."},
    )
    value_loss_coef: float = field(
        default=0.5,
        metadata={"help": "Coefficient for value loss."},
    )
    advantage_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages."},
    )

    # Evaluation
    eval_interval: int = field(
        default=16,
        metadata={"help": "Run evaluation every N training batches."},
    )
    eval_num_samples: int = field(
        default=4,
        metadata={"help": "Number of samples to evaluate per evaluation run."},
    )
    rollout_buffer_size: int = field(
        default=2,
        metadata={"help": "Number of node samples to buffer before an update."},
    )


@dataclass
class NodeSample:
    """Data structure for storing rollout samples per agent."""
    agent_idx: int
    turn_idx: int
    completions_data: Dict[str, Any]
    returns: List[float]
    node_mean_reward: float
    node_mean_return: float
    node_env_step: int
    # HAVPPO specific: store old log probs for importance ratio calculation
    old_log_probs: Optional[List[float]] = None
    # HAVPPO specific: store value estimates
    values: Optional[List[float]] = None


class HAVPPOTrainer:
    """
    Heterogeneous-Agent Value-based PPO Trainer (HAVPPO).
    
    Key features:
    - Sequential agent updates following HAPPO's scheme
    - Importance ratio accumulation (M factor) across agents
    - Centralized value network for advantage estimation (CTDE)
    - PPO clip objective for policy optimization
    - Supports both single-turn and multi-turn training
    """

    def __init__(
        self,
        # Model/tokenizer setup
        model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[List[PreTrainedModel]] = None,
        num_agents: int = 2,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_config: Optional[Dict[str, Any]] = None,
        # Data
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        dataset_type: Optional[str] = None,
        # Reward/formatting
        reward_func: Optional[Callable] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Callable, List[Callable]]] = None,
        # External transitions (multi-turn)
        external_transition: Optional[Callable] = None,
        # Logging/eval
        wandb_config: Optional[Dict[str, Any]] = None,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
        # Training args
        args: Optional[HAVPPOConfig] = None,
        # LoRA configuration
        use_lora: bool = False,
        # Value network (optional, will be created if not provided)
        value_network: Optional[CausalLMWithValueHead] = None,
    ):
        # Check for GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU not found. HAVPPOTrainer requires GPU for training."
            )

        if model is None and agents is None:
            raise ValueError("Either model or agents must be provided")
        if model is not None and agents is not None:
            raise ValueError("Cannot provide both model and agents parameters")

        # Training arguments
        self.args = args if args is not None else HAVPPOConfig()
        self.env_step = 0
        self._last_train_log_step = -1

        # Reward and formatting
        self._setup_formatters(formatters, num_agents)
        self._setup_reward_function(reward_func, reward_processor)

        self.model_config = model_config if model_config else {}

        if agents is not None:
            self.agents = agents
            self.num_agents = len(agents)
            if (
                hasattr(agents[0], "base_model")
                and hasattr(agents[0].base_model, "config")
                and hasattr(agents[0].base_model.config, "model_type")
            ):
                self.model_name = agents[0].base_model.config.model_type
            elif hasattr(agents[0], "config") and hasattr(
                agents[0].config, "_name_or_path"
            ):
                self.model_name = agents[0].config._name_or_path
            else:
                self.model_name = agents[0].__class__.__name__
        else:
            self.num_agents = num_agents
            if isinstance(model, str):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_load_kwargs = dict(self.model_config.get("model_kwargs", {}))
                if "attn_implementation" not in model_load_kwargs:
                    model_load_kwargs["attn_implementation"] = "flash_attention_2"

                self.agents = [
                    AutoModelForCausalLM.from_pretrained(model, **model_load_kwargs)
                    for _ in range(num_agents)
                ]
                self.model_name = model

                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model, **self.model_config.get("tokenizer_kwargs", {})
                    )
                    special_tokens = self.model_config.get("special_tokens", {})
                    if special_tokens:
                        self.tokenizer.add_special_tokens(special_tokens)
            else:
                raise ValueError(
                    "Model should be a string to create homogeneous agents"
                )

        # Validation
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.args.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 (group baseline requires multiple samples)."
            )
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("HAVPPO requires per_device_train_batch_size to be 1.")
        if self.args.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")

        self.rollout_buffers: List[List[NodeSample]] = [
            [] for _ in range(self.num_agents)
        ]

        # Check for external_transition requirement in multi-turn training
        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError(
                "Multi-turn training requires an external_transition function."
            )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator
        self.external_transition = external_transition

        # Store LoRA configuration
        self.use_lora = use_lora

        # Device
        self.device = torch.device("cuda")

        # Setup value network (centralized critic)
        self._setup_value_network(value_network)

        # Create optimizers for each agent
        # When using LoRA, only optimize adapter parameters (requires_grad=True)
        if self.use_lora:
            self.optimizers = []
            for agent_idx, agent in enumerate(self.agents):
                # Get only trainable parameters (LoRA adapter parameters)
                trainable_params = [p for p in agent.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
                self.optimizers.append(optimizer)
        else:
            # Full model training - optimize all parameters
            self.optimizers = [
                torch.optim.AdamW(
                    agent.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
                for agent in self.agents
            ]

        # Value head optimizer (only value head parameters)
        self.value_optimizer = torch.optim.AdamW(
            self.value_network.value_head.parameters(),
            lr=self.args.value_learning_rate,
            weight_decay=self.args.weight_decay,
        )

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

        # Dataset type
        self.dataset_type = dataset_type or None
        if self.dataset_type is None:
            try:
                if isinstance(self.wandb_config, dict):
                    sections = self.wandb_config.get("config_sections", {})
                    if isinstance(sections, dict):
                        ds = sections.get("dataset", {})
                        if isinstance(ds, dict):
                            self.dataset_type = ds.get("type")
            except Exception:
                self.dataset_type = None

        # Verbosity
        self.verbose = True
        try:
            if isinstance(self.wandb_config, dict):
                sections = self.wandb_config.get("config_sections", {})
                if isinstance(sections, dict):
                    out = sections.get("output", {})
                    if isinstance(out, dict) and "verbose" in out:
                        self.verbose = bool(out.get("verbose"))
        except Exception:
            pass

    def _setup_value_network(self, value_network: Optional[CausalLMWithValueHead]):
        """Setup the centralized value network."""
        if value_network is not None:
            self.value_network = value_network
        else:
            # Create a new value network with frozen backbone
            model_load_kwargs = dict(self.model_config.get("model_kwargs", {}))
            if "attn_implementation" not in model_load_kwargs:
                model_load_kwargs["attn_implementation"] = "flash_attention_2"

            value_base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_load_kwargs
            )

            # Freeze all base model parameters
            for param in value_base_model.parameters():
                param.requires_grad = False

            self.value_network = CausalLMWithValueHead(
                base_model=value_base_model,
                attach_value_head=True,
                value_head_hidden_dim=self.args.value_head_hidden_dim,
            )

        self.value_network.to(self.device)

    def _setup_formatters(self, formatters, num_agents):
        """Set up format functions for each agent."""
        default_format_func = lambda x, external_prompts=None: x.get("prompt", "")

        if formatters is None:
            self.formatters = [default_format_func] * num_agents
        elif callable(formatters) and not isinstance(formatters, list):
            original_formatter = formatters
            sig = inspect.signature(original_formatter)
            if "external_prompts" in sig.parameters:
                wrapped_formatter = lambda x, external_prompts=None: (
                    original_formatter(x, external_prompts=external_prompts)
                    if external_prompts is not None
                    else original_formatter(x)
                )
            else:
                wrapped_formatter = lambda x, external_prompts=None: original_formatter(x)
            self.formatters = [wrapped_formatter] * num_agents
        elif isinstance(formatters, list):
            if len(formatters) != num_agents:
                raise ValueError(
                    f"Number of formatters ({len(formatters)}) must match "
                    f"number of agents ({num_agents})"
                )
            wrapped_formatters = []
            for formatter in formatters:
                sig = inspect.signature(formatter)
                if "external_prompts" in sig.parameters:
                    def make_wrapper(f):
                        def wrapped(x, external_prompts=None):
                            return f(x, external_prompts=external_prompts)
                        return wrapped
                    wrapped_formatters.append(make_wrapper(formatter))
                else:
                    wrapped = lambda x, external_prompts=None, f=formatter: f(x)
                    wrapped_formatters.append(wrapped)
            self.formatters = wrapped_formatters
        else:
            raise ValueError(
                f"formatters must be a callable, a list of callables, or None. "
                f"Got {type(formatters)}"
            )

    def _setup_reward_function(self, reward_func, reward_processor=None):
        """Set up a single reward function with an optional processor."""
        if reward_func is None or not callable(reward_func):
            raise ValueError(
                "reward_func must be a callable that returns a list of floats"
            )
        self.reward_func = reward_func
        self.reward_processor = (
            reward_processor if reward_processor is not None else (lambda x: x)
        )

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "HumanEval")
            wandb_entity = self.wandb_config.get("entity", "contrl")

            if self.args.num_turns == 1:
                wandb_name = self.wandb_config.get("name", "test-havppo")
            else:
                wandb_name = self.wandb_config.get("name", "test-mt-havppo")

            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "num_turns": self.args.num_turns,
                "algorithm": "HAVPPO",
                "ppo_clip_eps": self.args.ppo_clip_eps,
                "m_clip_min": self.args.m_clip_min,
                "m_clip_max": self.args.m_clip_max,
                "shuffle_agent_order": self.args.shuffle_agent_order,
                "use_ppo_clip": self.args.use_ppo_clip,
                "learning_rate": self.args.learning_rate,
                "value_learning_rate": self.args.value_learning_rate,
                "value_head_hidden_dim": self.args.value_head_hidden_dim,
                "value_loss_coef": self.args.value_loss_coef,
                "advantage_normalization": self.args.advantage_normalization,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
                "use_lora": self.use_lora,
            }

            sections = (
                self.wandb_config.get("config_sections")
                if isinstance(self.wandb_config, dict)
                else None
            )
            if isinstance(sections, dict):
                dataset_section = sections.get("dataset") or {}
                model_section = sections.get("model") or {}
                output_section = sections.get("output") or {}
                external_section = sections.get("external") or {}
                trainer_section = sections.get("trainer") or {}

                config_dict.update(
                    {
                        "dataset": dataset_section,
                        "model": model_section,
                        "output": output_section,
                        "external": external_section,
                        "trainer": trainer_section,
                    }
                )

                dataset_name = (
                    dataset_section.get("name")
                    if isinstance(dataset_section, dict)
                    else None
                )
                dataset_type = (
                    dataset_section.get("type")
                    if isinstance(dataset_section, dict)
                    else None
                )
                if dataset_name:
                    config_dict["dataset_name"] = dataset_name
                if dataset_type:
                    config_dict["dataset_type"] = dataset_type

                ext_mode = (
                    external_section.get("mode")
                    if isinstance(external_section, dict)
                    else None
                )
                if ext_mode:
                    config_dict["external_mode"] = ext_mode

            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }

            if wandb_dir is not None:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir

            tags = (
                self.wandb_config.get("tags")
                if isinstance(self.wandb_config, dict)
                else None
            )
            if isinstance(tags, list):
                init_kwargs["tags"] = tags

            wandb.init(**init_kwargs)
            self.wandb_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Returns the evaluation DataLoader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def _build_joint_state(
        self,
        prompts: List[str],
        completions: Optional[List[str]] = None,
    ) -> str:
        """
        Build joint state input for centralized value network.
        
        Args:
            prompts: List of prompts per agent
            completions: Optional list of completions per agent (for Q(s,a) style)
            
        Returns:
            Joint state string for value network input
        """
        pieces = [f"[Agent {idx}] {p}" for idx, p in enumerate(prompts)]
        joint_prompt = "\n\n".join(pieces)

        if completions is not None:
            action_parts = [
                f"[Agent {idx} action]\n{c}" for idx, c in enumerate(completions)
            ]
            joint_prompt += "\n\n[Joint Action]\n" + "\n\n".join(action_parts)

        return joint_prompt

    def _estimate_value(self, joint_state: str) -> torch.Tensor:
        """
        Get V(s) from the centralized value network.
        
        Args:
            joint_state: Joint state string from _build_joint_state
            
        Returns:
            Value estimate tensor
        """
        encoded = self.tokenizer(
            joint_state,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.value_network(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_values=True,
            )

        # Return value at last token position
        return outputs.values[:, -1]

    def _estimate_value_with_grad(self, joint_state: str) -> torch.Tensor:
        """
        Get V(s) from the centralized value network with gradient.
        
        Args:
            joint_state: Joint state string from _build_joint_state
            
        Returns:
            Value estimate tensor with gradient
        """
        encoded = self.tokenizer(
            joint_state,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.value_network(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_values=True,
        )

        # Return value at last token position
        return outputs.values[:, -1]

    def evaluate(self, num_eval_samples: int = 4) -> Dict[str, float]:
        """Unified evaluation supporting both single-turn and multi-turn."""
        if self.eval_dataset is None:
            return {}

        all_agent_completions_turns = [[] for _ in range(self.num_agents)]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []
        eval_turn_rewards: List[List[float]] = [[] for _ in range(self.args.num_turns)]

        eval_dataloader = self.get_eval_dataloader()

        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break

                for batch_item in batch:
                    self._evaluate_sample(
                        batch_item,
                        all_agent_completions_turns,
                        all_test_cases,
                        all_entry_points,
                        all_prompts,
                        eval_turn_rewards,
                    )

        extra_eval_metrics: Dict[str, Any] = {}

        n_turns = self.args.num_turns
        if n_turns > 0 and eval_turn_rewards and eval_turn_rewards[0]:
            n_samp = len(eval_turn_rewards[0])
            gamma = float(getattr(self.args, "discount", 0.9))
            sum_returns = [0.0] * n_turns
            for s in range(n_samp):
                rs = [
                    eval_turn_rewards[t][s] if s < len(eval_turn_rewards[t]) else 0.0
                    for t in range(n_turns)
                ]
                ret = [0.0] * n_turns
                ret[-1] = rs[-1]
                for t in range(n_turns - 2, -1, -1):
                    ret[t] = rs[t] + gamma * ret[t + 1]
                for t in range(n_turns):
                    sum_returns[t] += ret[t]
            for t in range(n_turns):
                extra_eval_metrics[f"eval/turn_{t+1}/mean_reward"] = float(
                    np.mean(eval_turn_rewards[t]) if eval_turn_rewards[t] else 0.0
                )
                extra_eval_metrics[f"eval/turn_{t+1}/mean_return"] = float(
                    sum_returns[t] / n_samp if n_samp > 0 else 0.0
                )

        eval_metrics = self._log_eval_metrics(
            all_agent_completions_turns,
            all_test_cases,
            all_entry_points,
            all_prompts,
            extra_metrics=extra_eval_metrics,
        )
        return eval_metrics

    def _evaluate_sample(
        self,
        batch_item,
        all_agent_completions_turns,
        all_test_cases,
        all_entry_points,
        all_prompts,
        eval_turn_rewards,
    ):
        """Evaluate a single sample for any number of turns."""
        agent_sample_completions = [[] for _ in range(self.num_agents)]

        all_test_cases.append(batch_item.get("test", ""))
        all_entry_points.append(batch_item.get("entry_point", ""))
        all_prompts.append(batch_item.get("prompt", ""))

        previous_turn_completions = [None] * self.num_agents
        eval_prompt_history = [[] for _ in range(self.num_agents)]
        eval_response_history = [[] for _ in range(self.num_agents)]

        for turn_idx in range(self.args.num_turns):
            agent_external_prompts = [None] * self.num_agents

            if turn_idx > 0 and all(c is not None for c in previous_turn_completions):
                selected_prev = list(previous_turn_completions)
                if self.external_transition is not None:
                    transition_result = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=selected_prev,
                        num_agents=self.num_agents,
                        prompt_history_per_agent=eval_prompt_history,
                        response_history_per_agent=[
                            list(eval_response_history[i])
                            for i in range(self.num_agents)
                        ],
                    )

                    if isinstance(transition_result, (list, tuple)):
                        if len(transition_result) != self.num_agents:
                            raise ValueError(
                                f"External transition returned {len(transition_result)} values "
                                f"but expected {self.num_agents}"
                            )
                        agent_external_prompts = list(transition_result)
                    else:
                        raise ValueError(
                            "External transition must return a list or tuple of prompts"
                        )

            for agent_idx in range(self.num_agents):
                agent_completions = self._generate_completions_with_external_prompts(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=1,
                    max_new_tokens=self.args.max_new_tokens,
                    external_prompts=agent_external_prompts[agent_idx],
                    do_sample=True,
                )
                completion = agent_completions["completions"][0][0]
                used_prompt = agent_completions["prompts"][0]
                eval_prompt_history[agent_idx].append(used_prompt)
                agent_sample_completions[agent_idx].append(completion)

            agent_completions_for_reward = [
                [agent_sample_completions[i][-1]] for i in range(self.num_agents)
            ]
            prompt = self.formatters[0](batch_item)
            rewards = self._compute_rewards(
                [prompt], agent_completions_for_reward, batch_items=[batch_item]
            )
            if rewards:
                eval_turn_rewards[turn_idx].append(float(rewards[0]))
                for agent_idx in range(self.num_agents):
                    chosen = agent_sample_completions[agent_idx][-1]
                    previous_turn_completions[agent_idx] = chosen
                    eval_response_history[agent_idx].append(chosen)

        for agent_idx in range(self.num_agents):
            all_agent_completions_turns[agent_idx].append(
                agent_sample_completions[agent_idx]
            )

    def _log_eval_metrics(
        self,
        all_agent_completions_turns,
        all_test_cases,
        all_entry_points,
        all_prompts,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Log evaluation metrics."""
        eval_metrics = {}

        if (
            self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions_turns
            and all(agent_comps for agent_comps in all_agent_completions_turns)
        ):
            detailed_metrics = self.eval_logger(
                agent_completions_turns=all_agent_completions_turns,
                test_cases=all_test_cases,
                entry_points=all_entry_points,
                prompts=all_prompts,
            )

            aggregated_detailed_metrics = self.eval_aggregator(
                detailed_metrics, num_turns=self.args.num_turns
            )
            for key, value in aggregated_detailed_metrics.items():
                eval_metrics[f"eval/{key}"] = value

        if isinstance(extra_metrics, dict) and extra_metrics:
            eval_metrics.update(extra_metrics)

        if self.wandb_initialized and wandb.run is not None:
            wandb.log(eval_metrics, step=self.env_step)

        return eval_metrics

    def train(self, **kwargs):
        """
        Main training loop implementing HAVPPO with sequential agent updates.
        """
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        for agent in self.agents:
            agent.to(self.device)
            agent.train()

        self.value_network.train()

        for epoch in range(0, int(self.args.num_train_epochs)):
            epoch_turn_rewards = [[] for _ in range(self.args.num_turns)]
            epoch_turn_returns = [[] for _ in range(self.args.num_turns)]

            dl = self.get_train_dataloader()
            if not getattr(self, "verbose", True):
                it = enumerate(
                    tqdm(
                        dl,
                        total=len(dl),
                        desc=f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}",
                    )
                )
            else:
                it = enumerate(dl)

            for batch_idx, batch in it:
                # Periodic evaluation
                if int(self.args.eval_interval) > 0 and (
                    batch_idx % int(self.args.eval_interval) == 0
                ):
                    _ = self.evaluate(num_eval_samples=int(self.args.eval_num_samples))

                batch_item = batch[0]
                # HAVPPO training step with sequential updates
                batch_loss, _batch_stats = self._train_step_havppo(
                    batch_item,
                    epoch_turn_rewards,
                    epoch_turn_returns,
                    **kwargs,
                )

            # Process remaining buffered samples
            for agent_idx, buffer in enumerate(self.rollout_buffers):
                if buffer:
                    self._process_buffer(agent_idx, buffer)

            # Log epoch metrics
            if self.wandb_initialized and wandb.run is not None:
                epoch_log: Dict[str, Any] = {}
                n_turns = max(1, int(self.args.num_turns))
                for turn_idx in range(n_turns):
                    if epoch_turn_rewards and epoch_turn_rewards[turn_idx]:
                        epoch_log[f"turn_{turn_idx + 1}/epoch_reward_mean"] = float(
                            np.mean(epoch_turn_rewards[turn_idx])
                        )
                    if epoch_turn_returns and epoch_turn_returns[turn_idx]:
                        epoch_log[f"turn_{turn_idx + 1}/epoch_avg_return"] = float(
                            np.mean(epoch_turn_returns[turn_idx])
                        )
                if epoch_log:
                    wandb.log(epoch_log, step=self.env_step)

    def _train_step_havppo(
        self,
        batch_item,
        epoch_turn_rewards,
        epoch_turn_returns,
        **kwargs,
    ):
        """
        HAVPPO training step with sequential agent updates.
        
        Key differences from HAGRPO:
        1. Uses centralized value network for advantage estimation
        2. Value head is updated after policy updates
        """
        num_turns = int(self.args.num_turns)
        num_gens = int(self.args.num_generations)
        gamma = float(getattr(self.args, "discount", 0.9))

        turn_reward_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        turn_return_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        turn_node_counts: List[int] = [0 for _ in range(num_turns)]

        def build_node(
            turn_idx: int,
            prompts_per_agent=None,
            prompt_history_per_agent: Optional[List[List[str]]] = None,
            response_history_per_agent: Optional[List[List[str]]] = None,
        ):
            comps_per_agent = []
            log_probs_per_agent = []

            for agent_idx in range(self.num_agents):
                # Generate completions and compute old log probs
                comps = self._generate_completions_with_log_probs(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=num_gens,
                    max_new_tokens=self.args.max_new_tokens,
                    external_prompts=(
                        prompts_per_agent[agent_idx] if prompts_per_agent else None
                    ),
                    **kwargs,
                )
                comps_per_agent.append(comps)
                # Store old log probs for importance ratio calculation
                log_probs_per_agent.append(comps.get("sequence_log_probs", []))

            agent_completions_list = [
                comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
            ]
            prompts_used_this_turn = [
                comps_per_agent[i]["prompts"][0] for i in range(self.num_agents)
            ]
            formatted_prompt = comps_per_agent[0]["prompts"][0]

            if prompt_history_per_agent is None:
                prompt_history_per_agent = [[] for _ in range(self.num_agents)]
            if response_history_per_agent is None:
                response_history_per_agent = [[] for _ in range(self.num_agents)]

            next_prompt_history = [
                list(prompt_history_per_agent[i]) + [prompts_used_this_turn[i]]
                for i in range(self.num_agents)
            ]

            # Compute rewards
            joint_mode = str(getattr(self.args, "joint_mode", "aligned")).lower()
            rewards_vec: List[float] = []
            combo_indices: List[Tuple[int, ...]] = []

            if joint_mode in ["cross", "crossed"] and self.num_agents > 1:
                per_agent_ranges = [
                    range(len(agent_completions_list[i]))
                    for i in range(self.num_agents)
                ]
                for idx_tuple in itertools.product(*per_agent_ranges):
                    completion_args = [
                        [agent_completions_list[a][idx_tuple[a]]]
                        for a in range(self.num_agents)
                    ]
                    try:
                        sig = inspect.signature(self.reward_func)
                        if "batch_items" in sig.parameters:
                            rlist = self.reward_func(
                                *completion_args, batch_items=[batch_item]
                            )
                        else:
                            rlist = self.reward_func(*completion_args)
                    except TypeError:
                        rlist = self.reward_func(
                            [
                                agent_completions_list[a][idx_tuple[a]]
                                for a in range(self.num_agents)
                            ]
                        )
                    processed = [self.reward_processor(r) for r in rlist]
                    rewards_vec.append(float(processed[0] if processed else 0.0))
                    combo_indices.append(tuple(idx_tuple))
            elif joint_mode in ["align", "aligned"] and self.num_agents > 1:
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j] * self.num_agents) for j in range(k)]
            elif self.num_agents == 1:
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j]) for j in range(k)]
            else:
                raise ValueError(f"Unsupported joint_mode: {joint_mode}")

            if 0 <= turn_idx < len(epoch_turn_rewards):
                epoch_turn_rewards[turn_idx].append(
                    np.mean(rewards_vec) if rewards_vec else 0.0
                )

            self.env_step += len(rewards_vec)
            node_env_step = int(self.env_step)
            node_mean_reward = float(np.mean(rewards_vec)) if rewards_vec else 0.0
            turn_reward_node_means[turn_idx].append(node_mean_reward)
            turn_node_counts[turn_idx] += 1

            # Compute value estimates for each joint action
            values_vec = []
            for j, idx_tuple in enumerate(combo_indices):
                joint_completions = [
                    agent_completions_list[i][idx_tuple[i]]
                    for i in range(self.num_agents)
                ]
                joint_state = self._build_joint_state(
                    prompts_used_this_turn, joint_completions
                )
                value = self._estimate_value(joint_state)
                values_vec.append(value.item())

            term_threshold = getattr(self.args, "termination_threshold", None)
            terminate_here = False
            if term_threshold is not None and rewards_vec:
                try:
                    terminate_here = float(np.mean(rewards_vec)) > float(term_threshold)
                except Exception:
                    terminate_here = False

            node = {
                "turn": turn_idx,
                "completions": comps_per_agent,
                "log_probs": log_probs_per_agent,
                "rewards": rewards_vec,
                "values": values_vec,
                "children": [],
                "returns": None,
                "combo_indices": combo_indices,
                "env_step": node_env_step,
                "prompts": prompts_used_this_turn,
            }

            if turn_idx < num_turns - 1 and not terminate_here:
                for j in range(len(rewards_vec)):
                    idx_tuple = combo_indices[j]
                    parent_joint = [
                        agent_completions_list[i][idx_tuple[i]]
                        for i in range(self.num_agents)
                    ]
                    next_response_history = [
                        list(response_history_per_agent[i])
                        + [agent_completions_list[i][idx_tuple[i]]]
                        for i in range(self.num_agents)
                    ]

                    child_prompts = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=parent_joint,
                        num_agents=self.num_agents,
                        prompt_history_per_agent=next_prompt_history,
                        response_history_per_agent=next_response_history,
                    )
                    if (
                        not isinstance(child_prompts, (list, tuple))
                        or len(child_prompts) != self.num_agents
                    ):
                        raise ValueError(
                            "External transition must return per-agent prompts"
                        )
                    child = build_node(
                        turn_idx + 1,
                        prompts_per_agent=list(child_prompts),
                        prompt_history_per_agent=next_prompt_history,
                        response_history_per_agent=next_response_history,
                    )
                    node["children"].append(child)
            return node

        root = build_node(
            0,
            prompts_per_agent=None,
            prompt_history_per_agent=[[] for _ in range(self.num_agents)],
            response_history_per_agent=[[] for _ in range(self.num_agents)],
        )

        def compute_returns(node):
            if not node["children"]:
                node["returns"] = list(node["rewards"]) if node["rewards"] else []
                return node["returns"]
            parent_returns = []
            for j, rj in enumerate(node["rewards"] or []):
                child_node = node["children"][j]
                child_returns = compute_returns(child_node)
                mean_child = float(np.mean(child_returns)) if child_returns else 0.0
                parent_returns.append(rj + gamma * mean_child)
            node["returns"] = parent_returns
            return parent_returns

        compute_returns(root)

        def record_turn_returns(node):
            t = node["turn"]
            if 0 <= t < len(epoch_turn_returns):
                vals = node.get("returns") or []
                if vals:
                    mean_ret = float(np.mean(vals))
                    epoch_turn_returns[t].append(mean_ret)
                    turn_return_node_means[t].append(mean_ret)
            for ch in node["children"]:
                record_turn_returns(ch)

        record_turn_returns(root)

        # Collect all nodes for sequential update
        all_nodes = []

        def collect_nodes(node):
            all_nodes.append(node)
            for child in node["children"]:
                collect_nodes(child)

        collect_nodes(root)

        # HAVPPO: Sequential update with value-based advantage
        self._sequential_update_havppo(all_nodes)

        batch_loss = float(np.mean(np.abs(root.get("returns") or [0.0])))
        batch_stats: Dict[int, Dict[str, Any]] = {}
        for t in range(num_turns):
            stats: Dict[str, Any] = {}
            if turn_reward_node_means[t]:
                stats["batch_mean_reward"] = float(np.mean(turn_reward_node_means[t]))
            if turn_return_node_means[t]:
                stats["batch_expected_return"] = float(
                    np.mean(turn_return_node_means[t])
                )
            batch_stats[t] = stats

        return batch_loss, batch_stats

    def _sequential_update_havppo(self, all_nodes: List[Dict[str, Any]]):
        """
        Perform sequential agent updates following HAPPO scheme with value-based advantage.
        
        For each node:
        1. Compute value-based advantages (A = R - V)
        2. Draw random permutation of agents
        3. Set M = advantages for first agent
        4. For each agent in sequence:
           - Update with PPO-clip objective using M
           - Compute new M = old_M * clipped_ratio
        5. Update value head once after all policy updates
        """
        for node in all_nodes:
            returns_vec = node.get("returns") or []
            values_vec = node.get("values") or []
            if not returns_vec:
                continue

            comps_per_agent = node["completions"]
            log_probs_per_agent = node.get("log_probs", [])
            combo_indices = node.get("combo_indices") or []
            prompts = node.get("prompts") or []

            # Compute value-based advantages
            returns_tensor = torch.tensor(returns_vec, dtype=torch.float, device=self.device)
            values_tensor = torch.tensor(values_vec, dtype=torch.float, device=self.device)
            advantages = returns_tensor - values_tensor

            # Normalize advantages if enabled
            if self.args.advantage_normalization and len(advantages) > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std().clamp(min=1e-8)
                advantages = (advantages - adv_mean) / adv_std

            # Determine agent update order
            if self.args.shuffle_agent_order:
                agent_order = list(range(self.num_agents))
                random.shuffle(agent_order)
            elif self.args.reverse_agent_order:
                # Reverse order: main (last agent) first, helper (first agent) last
                agent_order = list(range(self.num_agents - 1, -1, -1))
            else:
                # Default order: helper (first agent) first, main (last agent) last
                agent_order = list(range(self.num_agents))

            # Initialize M factor with advantages
            # M shape: [num_joint_actions]
            M = advantages.clone()

            # Sequential update for each agent
            for agent_idx in agent_order:
                if agent_idx >= len(comps_per_agent):
                    continue

                agent = self.agents[agent_idx]
                optimizer = self.optimizers[agent_idx]
                comps_data = comps_per_agent[agent_idx]

                # Get old log probs for this agent
                old_log_probs = (
                    log_probs_per_agent[agent_idx]
                    if agent_idx < len(log_probs_per_agent)
                    else []
                )

                # Map joint actions to per-agent completion indices
                joint_mode = str(getattr(self.args, "joint_mode", "aligned")).lower()

                if joint_mode == "cross" and combo_indices:
                    # For cross mode: group M values by completion index
                    k = len(comps_data["completions"][0]) if comps_data["completions"] else 0
                    per_completion_M = [[] for _ in range(k)]
                    per_completion_indices = [[] for _ in range(k)]

                    for j, idx_tuple in enumerate(combo_indices):
                        comp_idx = idx_tuple[agent_idx]
                        per_completion_M[comp_idx].append(M[j].item())
                        per_completion_indices[comp_idx].append(j)

                    # Average M for each completion
                    completion_M = torch.tensor(
                        [np.mean(m) if m else 0.0 for m in per_completion_M],
                        dtype=torch.float,
                        device=self.device,
                    )
                else:
                    # Aligned mode: M directly corresponds to completion indices
                    completion_M = M.clone()

                # Update agent with PPO-clip objective
                new_log_probs, loss = self._update_agent_ppo_clip(
                    agent,
                    optimizer,
                    comps_data,
                    completion_M,
                    old_log_probs,
                )

                # Compute new M for next agent
                if agent_idx != agent_order[-1]:  # Not the last agent
                    M = self._update_M_factor(
                        M,
                        old_log_probs,
                        new_log_probs,
                        combo_indices,
                        agent_idx,
                        joint_mode,
                        self.device,
                    )

            # Update value head once after all policy updates
            self._update_value_head(node, returns_tensor)

    def _update_agent_ppo_clip(
        self,
        agent,
        optimizer,
        comps_data: Dict[str, Any],
        M: torch.Tensor,
        old_log_probs: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Update a single agent using either PPO-clip or simple policy gradient with M factor.
        
        If use_ppo_clip=True (default):
            loss = -min(ratio * M, clip(ratio, 1-eps, 1+eps) * M)
        If use_ppo_clip=False (MAGRPO-style):
            loss = -log_prob * M
        
        Returns:
            new_log_probs: List of new log probabilities for each completion
            loss: The computed loss value
        """
        device = self.device
        eps = self.args.ppo_clip_eps
        use_ppo_clip = self.args.use_ppo_clip

        agent.train()
        optimizer.zero_grad()

        prompt_input_ids = comps_data["prompt_input_ids"].to(device)
        completion_input_ids = comps_data["completion_input_ids"]
        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids = [[t.to(device) for t in completion_input_ids]]

        prompt_ids = prompt_input_ids[0]
        new_log_probs = []
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            if seq_idx >= len(M):
                break

            m_value = M[seq_idx]

            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                target_ids = completion_tokens
                attention_mask = torch.ones(len(input_ids), device=device)

                outputs = agent(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )

                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                # Calculate new log probabilities
                log_probs = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(0):
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                        log_probs.append(token_log_prob)

                if log_probs:
                    new_seq_log_prob = torch.stack(log_probs).sum()
                    # Store detached log prob for M factor computation
                    # (gradient should only flow through current agent's update)
                    new_log_probs.append(new_seq_log_prob.detach())

                    if use_ppo_clip:
                        # PPO-clip objective with M factor
                        # Get old log prob
                        if seq_idx < len(old_log_probs) and old_log_probs[seq_idx] is not None:
                            old_log_prob = old_log_probs[seq_idx]
                            if isinstance(old_log_prob, torch.Tensor):
                                old_log_prob = old_log_prob.to(device)
                            else:
                                old_log_prob = torch.tensor(old_log_prob, device=device)
                        else:
                            old_log_prob = new_seq_log_prob.detach()

                        # Compute importance ratio
                        ratio = torch.exp(new_seq_log_prob - old_log_prob)

                        # L = min(ratio * M, clip(ratio, 1-eps, 1+eps) * M)
                        clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
                        surr1 = ratio * m_value
                        surr2 = clipped_ratio * m_value

                        # For maximization, take minimum (pessimistic bound)
                        # Then negate for gradient descent
                        loss = -torch.min(surr1, surr2)
                    else:
                        # Simple policy gradient with M factor (MAGRPO-style)
                        # loss = -log_prob * M
                        loss = -new_seq_log_prob * m_value

                    total_loss = total_loss + loss
                    num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)

        total_loss.backward()
        optimizer.step()

        return new_log_probs, total_loss.detach()

    def _update_M_factor(
        self,
        M: torch.Tensor,
        old_log_probs: List[torch.Tensor],
        new_log_probs: List[torch.Tensor],
        combo_indices: List[Tuple[int, ...]],
        agent_idx: int,
        joint_mode: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Update M factor by multiplying with clipped importance ratios.
        
        M^{i_{1:m+1}} = clip(ratio) * M^{i_{1:m}}
        """
        m_clip_min = self.args.m_clip_min
        m_clip_max = self.args.m_clip_max

        new_M = M.clone()

        if joint_mode == "cross" and combo_indices:
            # For cross mode: update M for each joint action
            for j, idx_tuple in enumerate(combo_indices):
                comp_idx = idx_tuple[agent_idx]
                if comp_idx < len(new_log_probs) and comp_idx < len(old_log_probs):
                    new_lp = new_log_probs[comp_idx]
                    old_lp = old_log_probs[comp_idx]

                    if isinstance(old_lp, torch.Tensor):
                        old_lp = old_lp.to(device)
                    else:
                        old_lp = torch.tensor(old_lp, device=device)

                    ratio = torch.exp(new_lp - old_lp)
                    # Clip ratio to prevent M from exploding
                    clipped_ratio = torch.clamp(ratio, m_clip_min, m_clip_max)
                    new_M[j] = new_M[j] * clipped_ratio.item()
        else:
            # Aligned mode: direct correspondence
            for j in range(min(len(M), len(new_log_probs), len(old_log_probs))):
                new_lp = new_log_probs[j]
                old_lp = old_log_probs[j]

                if isinstance(old_lp, torch.Tensor):
                    old_lp = old_lp.to(device)
                else:
                    old_lp = torch.tensor(old_lp, device=device)

                ratio = torch.exp(new_lp - old_lp)
                clipped_ratio = torch.clamp(ratio, m_clip_min, m_clip_max)
                new_M[j] = new_M[j] * clipped_ratio.item()

        # Additional clipping on M itself to ensure stability
        new_M = torch.clamp(new_M, -m_clip_max, m_clip_max)

        return new_M

    def _update_value_head(
        self,
        node: Dict[str, Any],
        returns: torch.Tensor,
    ):
        """
        Update the value head using MSE loss.
        
        Args:
            node: Node containing prompts and completions
            returns: Target returns tensor
        """
        combo_indices = node.get("combo_indices") or []
        prompts = node.get("prompts") or []
        comps_per_agent = node["completions"]

        if not combo_indices or not prompts:
            return

        self.value_optimizer.zero_grad()

        total_value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_samples = 0

        for j, idx_tuple in enumerate(combo_indices):
            if j >= len(returns):
                break

            # Get completions for this joint action
            agent_completions_list = [
                comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
            ]
            joint_completions = [
                agent_completions_list[i][idx_tuple[i]]
                for i in range(self.num_agents)
            ]

            # Build joint state and get value prediction with gradient
            joint_state = self._build_joint_state(prompts, joint_completions)
            value_pred = self._estimate_value_with_grad(joint_state)

            # MSE loss
            target = returns[j]
            value_loss = (value_pred.squeeze() - target) ** 2

            total_value_loss = total_value_loss + value_loss
            num_samples += 1

        if num_samples > 0:
            total_value_loss = total_value_loss / num_samples
            total_value_loss = total_value_loss * self.args.value_loss_coef

            if not torch.isnan(total_value_loss) and not torch.isinf(total_value_loss):
                total_value_loss.backward()
                self.value_optimizer.step()

    def _generate_completions_with_log_probs(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        external_prompts=None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completions and compute sequence log probabilities.
        Extended from _generate_completions to also return log probs.
        """
        # First generate completions using the standard method
        if self.args.num_turns == 1 or external_prompts is None:
            comps_data = self._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )
        else:
            prompts = [external_prompts for _ in batch_items]
            if getattr(self.args, "external_prompt_passthrough", False):
                comps_data = self._generate_completions(
                    agent,
                    batch_items,
                    agent_idx=agent_idx,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    prompts_override=prompts,
                    do_sample=do_sample,
                    **kwargs,
                )
            else:
                modified_items = []
                for item, prompt in zip(batch_items, prompts):
                    modified_item = item.copy() if hasattr(item, "copy") else dict(item)
                    modified_item["_original_prompt"] = modified_item.get("prompt", "")
                    modified_item["prompt"] = prompt
                    modified_items.append(modified_item)

                comps_data = self._generate_completions(
                    agent,
                    modified_items,
                    agent_idx=agent_idx,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    **kwargs,
                )

                for i, item in enumerate(comps_data["batch_items"]):
                    if "_original_prompt" in item:
                        item["prompt"] = item["_original_prompt"]
                        del item["_original_prompt"]

                comps_data["prompts"] = prompts

        # Now compute log probabilities for each completion
        device = self.device
        prompt_input_ids = comps_data["prompt_input_ids"].to(device)
        completion_input_ids = comps_data["completion_input_ids"]

        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids_list = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids_list = [[t.to(device) for t in completion_input_ids]]

        prompt_ids = prompt_input_ids[0]
        sequence_log_probs = []

        # Set to eval mode temporarily for log prob computation
        training_mode = agent.training
        agent.eval()

        with torch.no_grad():
            for completion_tokens in completion_input_ids_list[0]:
                if len(completion_tokens) > 0:
                    input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                    target_ids = completion_tokens
                    attention_mask = torch.ones(len(input_ids), device=device)

                    outputs = agent(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                    )

                    completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                    log_probs = []
                    for i, token_id in enumerate(target_ids):
                        if i < completion_logits.size(0):
                            token_logits = completion_logits[i]
                            token_log_prob = torch.log_softmax(token_logits, dim=-1)[
                                token_id
                            ]
                            log_probs.append(token_log_prob)

                    if log_probs:
                        seq_log_prob = torch.stack(log_probs).sum()
                        sequence_log_probs.append(seq_log_prob.detach())
                    else:
                        sequence_log_probs.append(torch.tensor(0.0, device=device))
                else:
                    sequence_log_probs.append(torch.tensor(0.0, device=device))

        agent.train(training_mode)
        comps_data["sequence_log_probs"] = sequence_log_probs

        return comps_data

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        prompts_override: Optional[List[str]] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ):
        """Generate completions from an agent given prompts."""
        device = self.device

        if prompts_override is not None:
            if len(prompts_override) != len(batch_items):
                raise ValueError(
                    "prompts_override must have the same length as batch_items"
                )
            prompts = prompts_override
        else:
            format_func = self.formatters[agent_idx]
            prompts = [format_func(item) for item in batch_items]

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_encodings = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        training_mode = agent.training
        original_requires_grad = {}

        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        agent.eval()

        generation_output = None
        try:
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            top_k = getattr(self.args, "top_k", None)
            if do_sample is None and num_return_sequences > 1:
                generation_update = {
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                }
                if top_k is not None:
                    generation_update["top_k"] = top_k
                generation_kwargs.update(generation_update)
            elif do_sample is not None:
                generation_kwargs.update(
                    {
                        "do_sample": bool(do_sample),
                        "num_beams": 1,
                        "num_return_sequences": num_return_sequences,
                    }
                )
                if do_sample:
                    generation_kwargs.update(
                        {
                            "temperature": self.args.temperature,
                            "top_p": self.args.top_p,
                        }
                    )
                    if top_k is not None:
                        generation_kwargs["top_k"] = top_k

            if (
                "pad_token_id" not in generation_kwargs
                or generation_kwargs["pad_token_id"] is None
            ):
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            generation_kwargs.update(kwargs)
            generation_output = agent.generate(**generation_kwargs)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        completion_input_ids = generation_output.sequences

        prompt_len = prompt_input_ids[0].shape[0]
        pad_positions = (prompt_input_ids[0] == self.tokenizer.pad_token_id).nonzero()
        if pad_positions.shape[0] > 0:
            prompt_len = pad_positions[0].item()

        completions = []
        completion_tokens_list = []

        total_sequences = completion_input_ids.shape[0]

        batch_completions = []
        batch_completion_tokens = []

        end_idx = min(num_return_sequences, total_sequences)

        for s in range(end_idx):
            completion_tokens = completion_input_ids[s, prompt_len:]
            batch_completion_tokens.append(completion_tokens)

            completion_text = self.tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            batch_completions.append(completion_text)

        completions.append(batch_completions)
        completion_tokens_list.append(batch_completion_tokens)

        completion_attention_masks = []
        batch_masks = []
        for tokens in completion_tokens_list[0]:
            mask = torch.ones(len(tokens), device=device)
            batch_masks.append(mask)
        completion_attention_masks.append(batch_masks)

        logits = (
            generation_output.scores if hasattr(generation_output, "scores") else []
        )

        return {
            "prompts": prompts,
            "batch_items": batch_items,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": completions,
            "completion_input_ids": completion_tokens_list,
            "completion_attention_mask": completion_attention_masks,
            "logits": logits,
        }

    def _generate_completions_with_external_prompts(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        external_prompts=None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ):
        """Generate completions with optional external prompts."""
        if self.args.num_turns == 1 or external_prompts is None:
            return self._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )

        prompts = [external_prompts for _ in batch_items]
        if getattr(self.args, "external_prompt_passthrough", False):
            return self._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                prompts_override=prompts,
                do_sample=do_sample,
                **kwargs,
            )

        modified_items = []
        for item, prompt in zip(batch_items, prompts):
            modified_item = item.copy() if hasattr(item, "copy") else dict(item)
            modified_item["_original_prompt"] = modified_item.get("prompt", "")
            modified_item["prompt"] = prompt
            modified_items.append(modified_item)

        completions_data = self._generate_completions(
            agent,
            modified_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        for i, item in enumerate(completions_data["batch_items"]):
            if "_original_prompt" in item:
                item["prompt"] = item["_original_prompt"]
                del item["_original_prompt"]

        completions_data["prompts"] = prompts

        return completions_data

    def _compute_rewards(
        self, prompts, completions_list, batch_items=None
    ) -> List[float]:
        """Compute rewards using a single reward function and optional processor."""
        all_rewards = []

        for i in range(self.num_agents):
            if not isinstance(completions_list[i], list):
                completions_list[i] = (
                    [completions_list[i]]
                    if not isinstance(completions_list[i], list)
                    else completions_list[i]
                )

        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))

        for completion_idx in range(min_completions):
            agent_completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]

            try:
                completion_args = [[comp] for comp in agent_completions]
                sig = inspect.signature(self.reward_func)
                if "batch_items" in sig.parameters:
                    func_rewards = self.reward_func(
                        *completion_args, batch_items=batch_items
                    )
                else:
                    func_rewards = self.reward_func(*completion_args)
            except TypeError:
                func_rewards = self.reward_func(agent_completions)

            processed_rewards = [self.reward_processor(r) for r in func_rewards]
            all_rewards.append(processed_rewards[0])

        return all_rewards

    def _pack_completions_for_buffer(
        self, completions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt_ids = completions_data["prompt_input_ids"].detach().cpu()
        completion_ids = completions_data["completion_input_ids"]
        if completion_ids and isinstance(completion_ids[0], list):
            packed_completion_ids = [[t.detach().cpu() for t in completion_ids[0]]]
        else:
            packed_completion_ids = [[t.detach().cpu() for t in completion_ids]]
        return {
            "prompt_input_ids": prompt_ids,
            "completion_input_ids": packed_completion_ids,
        }

    def _append_to_buffer(self, agent_idx: int, sample: NodeSample) -> None:
        buffer = self.rollout_buffers[agent_idx]
        buffer.append(sample)
        if len(buffer) >= int(self.args.rollout_buffer_size):
            self._process_buffer(agent_idx, buffer)

    def _should_log_train(self, step: int) -> bool:
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_train_log_step = step
            return True
        if (
            self._last_train_log_step < 0
            or (step - self._last_train_log_step) >= interval
        ):
            self._last_train_log_step = step
            return True
        return False

    def _process_buffer(self, agent_idx: int, buffer: List[NodeSample]) -> None:
        """Process buffered samples - for HAVPPO this is simplified since updates happen in _sequential_update_havppo."""
        if not buffer:
            return
        # Log metrics from buffer
        if self.wandb_initialized and wandb.run is not None and buffer:
            turn_groups: Dict[int, List[NodeSample]] = {}
            for sample in buffer:
                t_idx = int(sample.turn_idx)
                turn_groups.setdefault(t_idx, []).append(sample)

            for t_idx in sorted(turn_groups.keys()):
                samples = turn_groups[t_idx]
                batch_log: Dict[str, Any] = {}
                prefix = f"turn_{t_idx + 1}/"
                batch_log[prefix + "reward_mean"] = float(
                    np.mean([s.node_mean_reward for s in samples])
                )
                batch_log[prefix + "expected_return"] = float(
                    np.mean([s.node_mean_return for s in samples])
                )
                step = max(s.node_env_step for s in samples)
                if self._should_log_train(step):
                    wandb.log(batch_log, step=step)

        buffer.clear()

    def save_model(self, output_dir):
        """Save the final trained models or LoRA adapters and value head."""
        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            if self.use_lora:
                # Save only LoRA adapter weights
                agent_dir = f"{output_dir}/agent_{agent_idx}_lora"
            else:
                # Save full model
                agent_dir = f"{output_dir}/agent_{agent_idx}"

            os.makedirs(agent_dir, exist_ok=True)

            # For PEFT models, save_pretrained saves only the adapter weights
            # For regular models, it saves the full model
            agent.save_pretrained(agent_dir)

            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

        # Save value head
        value_head_dir = f"{output_dir}/value_head"
        os.makedirs(value_head_dir, exist_ok=True)
        torch.save(
            self.value_network.value_head.state_dict(),
            f"{value_head_dir}/value_head.pt",
        )

        if self.wandb_initialized and wandb.run is not None:
            save_type = "lora_adapters" if self.use_lora else "full_models"
            wandb.log({
                "final_model_saved": output_dir,
                "save_type": save_type,
                "value_head_saved": True,
            })
            wandb.finish()
