"""
Agent-Chained Policy Optimization (ACPPO) Trainer.

ACPPO combines:
- Simultaneous model updates (like MAPPO/MAGRPO)
- Per-agent value networks (decentralized, not centralized like HAVPPO)
- Agent chaining from MAGRPO
- KL-based similarity reward for auxiliary function prediction
- TD-based refined advantage calculation

Key Design Principle:
- Advantage calculation: Uses TD residuals (zeta) with gamma' and lambda'
- Critic update: MSE loss against actual returns (cumulative rewards), NOT TD targets
"""

from __future__ import annotations

import inspect
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments

from comlrl.models.actor_critic import ValueHead


RewardFunc = Callable[..., Sequence[float]]
Formatter = Callable[[Dict[str, Any]], str]
MetricsCallback = Callable[[List["NodeSample"]], Dict[str, float]]


@dataclass
class ACPPOConfig(TrainingArguments):
    """
    Configuration for ACPPO training.
    
    ACPPO: Agent-Chained Policy Optimization with:
    - Simultaneous model updates
    - Per-agent value networks
    - TD-based refined advantages
    - KL similarity reward for agent chaining
    """

    # Core setup
    num_train_epochs: float = field(
        default=20,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Per-device batch size (must be 1 for ACPPO)."},
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
        metadata={"help": "Number of agents."},
    )

    # Sampling/generation
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations to sample per prompt for each agent."},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate."},
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
        metadata={"help": "Top-k for sampling."},
    )

    # Multi-turn / tree rollout
    num_turns: Optional[int] = field(
        default=1,
        metadata={"help": "Number of turns per episode."},
    )
    discount: float = field(
        default=0.9,
        metadata={"help": "Discount factor (gamma) for returns."},
    )
    joint_mode: str = field(
        default="aligned",
        metadata={"help": "Joint action composition: 'cross' or 'aligned'."},
    )
    termination_threshold: Optional[float] = field(
        default=-0.2,
        metadata={"help": "Early stop threshold."},
    )
    external_prompt_passthrough: bool = field(
        default=False,
        metadata={"help": "Use external prompts directly in multi-turn."},
    )

    # Evaluation
    eval_interval: int = field(
        default=16,
        metadata={"help": "Run evaluation every N batches."},
    )
    eval_num_samples: int = field(
        default=4,
        metadata={"help": "Number of samples per evaluation."},
    )
    rollout_buffer_size: int = field(
        default=2,
        metadata={"help": "Number of node samples to buffer before update."},
    )

    # Per-agent Value Network parameters (ACPPO-specific)
    value_head_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for 2-layer MLP value head."},
    )
    value_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for value head optimizer."},
    )
    value_loss_coef: float = field(
        default=0.5,
        metadata={"help": "Coefficient for value loss."},
    )

    # TD-based advantage parameters (ACPPO-specific)
    gamma_prime: float = field(
        default=0.99,
        metadata={"help": "Gamma' for TD residuals in advantage calculation."},
    )
    lambda_prime: float = field(
        default=0.95,
        metadata={"help": "Lambda' for GAE-like weighting in advantage calculation."},
    )
    advantage_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages."},
    )

    # Agent Chaining
    agent_chaining: bool = field(
        default=True,
        metadata={"help": "Enable agent chaining mode."},
    )
    max_new_tokens_per_agent: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of max_new_tokens for each agent."},
    )

    # PPO Clip parameters
    use_ppo_clip: bool = field(
        default=True,
        metadata={
            "help": "Whether to use PPO-clip objective. "
            "If False, uses simple policy gradient: loss = -log_prob * advantage."
        },
    )
    ppo_clip_eps: float = field(
        default=0.2,
        metadata={"help": "PPO clipping epsilon for importance ratio."},
    )

    # Chat Template
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Use chat template for Instruct models."},
    )
    chat_template_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt for chat template."},
    )


@dataclass
class NodeSample:
    """Sample from a rollout node for buffer storage."""
    agent_idx: int
    turn_idx: int
    completions_data: Dict[str, Any]
    returns: List[float]
    advantages: List[float]  # Refined advantages from TD residuals
    node_mean_reward: float
    node_mean_return: float
    node_env_step: int


class ACPPOTrainer:
    """
    Agent-Chained Policy Optimization Trainer.
    
    Features:
    - Simultaneous model updates (all agents update independently)
    - Per-agent value networks (2-layer MLP value head on frozen backbone)
    - TD-based refined advantage calculation
    - KL similarity reward for agent chaining
    - Agent chaining: Agent 2 predicts Agent 1's aux before generating main
    """

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[List[PreTrainedModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_func: Optional[RewardFunc] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Formatter, Sequence[Formatter]]] = None,
        args: Optional[ACPPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        metrics_callback: Optional[MetricsCallback] = None,
        external_transition: Optional[Callable] = None,
        use_lora: bool = False,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
    ) -> None:
        """
        Initialize ACPPO Trainer.
        
        Args:
            model: Base model name/path for homogeneous agents
            agents: Pre-initialized list of agent models (alternative to model)
            tokenizer: Tokenizer for the models
            reward_func: Reward function for evaluating joint actions
            reward_processor: Optional reward scaling/shifting processor
            formatters: Prompt formatters for each agent
            args: ACPPO configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            model_config: Additional model configuration
            wandb_config: W&B logging configuration
            metrics_callback: Callback for custom metrics
            external_transition: External transition function for multi-turn
            use_lora: Whether models use LoRA adapters
            eval_logger: Logger function for detailed evaluation metrics (e.g., fully_passed_rate)
            eval_aggregator: Aggregator function for eval_logger metrics
        """
        if reward_func is None or not callable(reward_func):
            raise ValueError("A callable reward_func must be provided.")
        
        self.args = args if args is not None else ACPPOConfig(output_dir="./acppo_output")
        self.reward_func = reward_func
        self.reward_processor = reward_processor or (lambda x: x)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_callback = metrics_callback
        self.model_config = model_config or {}
        self.use_lora = use_lora
        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Training will run on CPU.")

        # Setup tokenizer
        self.tokenizer = self._ensure_tokenizer(model, tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Number of agents
        self.num_agents = self.args.num_agents

        # Setup agents
        if agents is not None:
            self.agents = agents
            self.model_name = getattr(agents[0], "name_or_path", "custom_model")
        elif model is not None:
            self.model_name = model if isinstance(model, str) else getattr(model, "name_or_path", "model")
            self.agents = self._load_agents(model)
        else:
            raise ValueError("Either model or agents must be provided.")

        # Move agents to device
        for agent in self.agents:
            agent.to(self.device)

        # Setup per-agent value heads (ACPPO-specific)
        self._setup_value_heads()

        # Setup formatters
        self.formatters = self._setup_formatters(formatters)

        # Setup optimizers for policy models
        self.optimizers: List[torch.optim.Optimizer] = []
        for agent in self.agents:
            trainable_params = [p for p in agent.parameters() if p.requires_grad]
            if trainable_params:
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.args.learning_rate,
                    weight_decay=getattr(self.args, "weight_decay", 0.0),
                )
                self.optimizers.append(optimizer)
            else:
                # Dummy optimizer if no trainable params
                self.optimizers.append(None)

        # Rollout buffers per agent
        self.rollout_buffers: List[List[NodeSample]] = [
            [] for _ in range(self.num_agents)
        ]

        # External transition for multi-turn
        self.external_transition = external_transition
        if self.args.num_turns > 1 and external_transition is None:
            print("Warning: Multi-turn training requires external_transition.")

        # Agent chaining setup
        self.agent_chaining = self.args.agent_chaining

        # Chat template setup
        self.use_chat_template = self.args.use_chat_template

        # Reward function signature inference
        self._reward_signature = self._infer_reward_signature(reward_func)

        # W&B setup
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        self.env_step = 0
        self._last_train_log_step = -1
        if wandb_config is not None:
            self._init_wandb()

        # Verbosity
        self.verbose = True
        if wandb_config and isinstance(wandb_config, dict):
            sections = wandb_config.get("config_sections", {})
            if isinstance(sections, dict):
                out = sections.get("output", {})
                if isinstance(out, dict):
                    self.verbose = out.get("verbose", True)

    def _setup_value_heads(self) -> None:
        """
        Setup per-agent value heads.
        
        Each agent gets its own value head (2-layer MLP) attached to the
        frozen backbone. Only the value head is trainable for value updates.
        All value heads use bfloat16 for consistency with model.
        """
        self.value_heads: List[ValueHead] = []
        self.value_optimizers: List[torch.optim.Optimizer] = []
        
        # Use bfloat16 for all computations (consistent with model dtype)
        self.compute_dtype = torch.bfloat16

        for agent_idx, agent in enumerate(self.agents):
            # Get hidden size from agent config
            config = getattr(agent, "config", None)
            if config is None:
                # Try to get from base_model for PEFT models
                base_model = getattr(agent, "base_model", None)
                if base_model is not None:
                    model_obj = getattr(base_model, "model", base_model)
                    config = getattr(model_obj, "config", None)
            
            if config is None:
                raise ValueError(f"Cannot determine hidden size for agent {agent_idx}")

            hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
            if hidden_size is None:
                raise ValueError(f"Agent {agent_idx} config does not have hidden_size")

            # Create value head
            value_head = ValueHead(
                input_dim=hidden_size,
                hidden_dim=self.args.value_head_hidden_dim,
            )
            # Move to device and convert to bfloat16
            value_head.to(device=self.device, dtype=self.compute_dtype)
            self.value_heads.append(value_head)

            # Create optimizer for value head only
            value_optimizer = torch.optim.AdamW(
                value_head.parameters(),
                lr=self.args.value_learning_rate,
            )
            self.value_optimizers.append(value_optimizer)

    def _ensure_tokenizer(
        self, 
        model: Optional[Union[str, PreTrainedModel]], 
        tokenizer: Optional[PreTrainedTokenizerBase]
    ) -> PreTrainedTokenizerBase:
        """Ensure we have a valid tokenizer."""
        if tokenizer is not None:
            return tokenizer
        
        if isinstance(model, str):
            from transformers import AutoTokenizer
            tok_kwargs = self.model_config.get("tokenizer_kwargs", {})
            return AutoTokenizer.from_pretrained(model, **tok_kwargs)
        
        raise ValueError("Tokenizer must be provided if model is not a string path.")

    def _load_agents(self, model: Union[str, PreTrainedModel]) -> List[PreTrainedModel]:
        """Load agent models."""
        from transformers import AutoModelForCausalLM

        agents = []
        model_kwargs = self.model_config.get("model_kwargs", {})

        for _ in range(self.num_agents):
            if isinstance(model, str):
                agent = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
            else:
                # Clone the model for each agent
                agent = type(model).from_pretrained(
                    model.name_or_path, **model_kwargs
                )
            agents.append(agent)

        return agents

    def _setup_formatters(
        self, 
        formatters: Optional[Union[Formatter, Sequence[Formatter]]]
    ) -> List[Formatter]:
        """Setup prompt formatters for each agent."""
        if formatters is None:
            # Default identity formatter
            return [lambda x: x.get("prompt", str(x)) for _ in range(self.num_agents)]
        
        if callable(formatters) and not isinstance(formatters, (list, tuple)):
            return [formatters for _ in range(self.num_agents)]
        
        formatters_list = list(formatters)
        if len(formatters_list) < self.num_agents:
            # Extend with last formatter
            formatters_list.extend(
                [formatters_list[-1]] * (self.num_agents - len(formatters_list))
            )
        
        return formatters_list[:self.num_agents]

    def _infer_reward_signature(self, reward_func: RewardFunc) -> List[str]:
        """Infer reward function parameter names."""
        try:
            sig = inspect.signature(reward_func)
            return list(sig.parameters.keys())
        except (ValueError, TypeError):
            return []

    def _get_max_tokens_for_agent(self, agent_idx: int) -> int:
        """Get max_new_tokens for a specific agent."""
        if self.args.max_new_tokens_per_agent is not None:
            if agent_idx < len(self.args.max_new_tokens_per_agent):
                return self.args.max_new_tokens_per_agent[agent_idx]
        return self.args.max_new_tokens

    # =========================================================================
    # Value Estimation Methods
    # =========================================================================

    def _estimate_value_for_agent(
        self,
        agent_idx: int,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """
        Estimate V^{(i)}([s_t, b_t^{(i)}]) for agent i without gradients.
        
        Uses the frozen backbone to get hidden states, then the value head
        to predict the value.
        """
        agent = self.agents[agent_idx]
        value_head = self.value_heads[agent_idx]

        # Tokenize prompt + completion
        text = prompt + completion
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Get hidden states from backbone (no grad)
        agent.eval()
        with torch.no_grad():
            outputs = agent(
                **inputs,
                output_hidden_states=True,
            )
            # Get last layer hidden states
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1][:, -1, :]  # Last token
            else:
                # Fallback for models that don't return hidden states by default
                last_hidden = self._get_hidden_states(agent, inputs)

        # Value head prediction (no grad for inference)
        with torch.no_grad():
            value = value_head(last_hidden)

        return value.squeeze()

    def _estimate_value_with_grad_for_agent(
        self,
        agent_idx: int,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """
        Estimate value with gradients for value head training.
        
        Backbone is frozen (no grad), only value head has gradients.
        """
        agent = self.agents[agent_idx]
        value_head = self.value_heads[agent_idx]

        # Tokenize
        text = prompt + completion
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Get hidden states from frozen backbone (no grad)
        agent.eval()
        with torch.no_grad():
            outputs = agent(
                **inputs,
                output_hidden_states=True,
            )
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1][:, -1, :]
            else:
                last_hidden = self._get_hidden_states(agent, inputs)

        # Detach hidden states to ensure no gradients flow to backbone
        last_hidden = last_hidden.detach()

        # Value head prediction with gradient
        value_head.train()
        value = value_head(last_hidden)

        return value.squeeze()

    def _get_hidden_states(
        self, 
        agent: PreTrainedModel, 
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fallback method to get hidden states."""
        # Try to get the base model for PEFT models
        base_model = getattr(agent, "base_model", None)
        if base_model is not None:
            model_obj = getattr(base_model, "model", base_model)
        else:
            model_obj = getattr(agent, "model", agent)

        # Try to get the transformer/core model
        if hasattr(model_obj, "transformer"):
            core = model_obj.transformer
        elif hasattr(model_obj, "model"):
            core = model_obj.model
        else:
            core = model_obj

        # Forward through core to get hidden states
        outputs = core(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )

        if hasattr(outputs, "hidden_states"):
            return outputs.hidden_states[-1][:, -1, :]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, -1, :]
        else:
            raise ValueError("Cannot extract hidden states from model outputs")

    # =========================================================================
    # TD Residual and Advantage Calculation
    # =========================================================================

    def _compute_td_residuals(
        self,
        agent_values: List[torch.Tensor],
        reward: float,
        next_turn_first_agent_value: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Compute TD residuals for all agents.
        
        Following the Bellman operators:
        - For i < N: zeta^{(i)} = gamma' * V^{(i+1)} - V^{(i)}
        - For i = N: zeta^{(N)} = r_t + gamma' * V^{(1)}(s_{t+1}) - V^{(N)}
        
        Note: Intermediate agents (i < N) don't receive reward during micro-steps.
        """
        N = self.num_agents
        gamma_prime = self.args.gamma_prime
        td_residuals = []

        for i in range(N):
            if i < N - 1:
                # Intermediate agents: no reward, bootstrap from next agent
                zeta = gamma_prime * agent_values[i + 1] - agent_values[i]
            else:
                # Last agent (N): receives reward
                r_t = torch.tensor(reward, dtype=self.compute_dtype, device=self.device)
                if next_turn_first_agent_value is not None:
                    # Multi-turn: bootstrap from first agent's value at next turn
                    zeta = r_t + gamma_prime * next_turn_first_agent_value - agent_values[i]
                else:
                    # Single-turn or terminal: no bootstrap
                    zeta = r_t - agent_values[i]
            td_residuals.append(zeta)

        return td_residuals

    def _compute_refined_advantage(
        self,
        agent_idx: int,
        td_residuals_per_turn: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute refined advantage A_t^{(i)} using TD residuals.
        
        A_t^{(i)} = sum_{j=i}^N (gamma' * lambda')^{j-i} * zeta^{(j)}_t
                 + sum_{k=1}^inf sum_{j=1}^N (gamma' * lambda')^{kN+j-i} * zeta^{(j)}_{t+k}
        
        The second term accounts for future turns in multi-turn settings.
        """
        gamma_prime = self.args.gamma_prime
        lambda_prime = self.args.lambda_prime
        N = self.num_agents

        advantage = torch.tensor(0.0, dtype=self.compute_dtype, device=self.device)

        # Current turn: j from agent_idx to N-1 (0-indexed)
        if td_residuals_per_turn:
            current_residuals = td_residuals_per_turn[0]
            for j in range(agent_idx, N):
                exponent = j - agent_idx
                coef = (gamma_prime * lambda_prime) ** exponent
                advantage = advantage + coef * current_residuals[j]

        # Future turns (for multi-turn)
        for k, future_residuals in enumerate(td_residuals_per_turn[1:], start=1):
            for j in range(N):
                exponent = k * N + j - agent_idx
                coef = (gamma_prime * lambda_prime) ** exponent
                advantage = advantage + coef * future_residuals[j]

        return advantage

    # =========================================================================
    # Agent Chaining Helper Methods
    # =========================================================================

    def _extract_main_from_chaining_output(self, completion: str, entry_point: str = None) -> str:
        """
        Extract ONLY the main function from Agent 2's chaining output.
        
        IMPORTANT: Returns only the main function (entry_point), NOT the predicted aux.
        
        Args:
            completion: The full completion text
            entry_point: Optional entry point function name to look for
        """
        if not completion:
            return ""
        
        def _extract_single_function(text: str, start_pos: int) -> str:
            """Extract a single function definition starting at start_pos."""
            func_text = text[start_pos:]
            # Find the next function definition (if any) to cut
            next_func = re.search(r'\n(?=def\s+\w+\s*\()', func_text[1:])  # Skip first char to avoid matching itself
            if next_func:
                return func_text[:next_func.start() + 1].strip()
            return func_text.strip()
        
        # Method 1: Try XML tags (legacy format)
        main_match = re.search(
            r'<main_function>(.*?)</main_function>',
            completion,
            flags=re.DOTALL
        )
        if main_match:
            main_content = main_match.group(1).strip()
            if main_content:
                return main_content
        
        # Method 2: If entry_point is provided, find that function directly (PRIORITY)
        if entry_point:
            match = re.search(
                rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:',
                completion
            )
            if match:
                return _extract_single_function(completion, match.start())
        
        # Method 3: Look for "# Main function" comment and extract the function after it
        main_comment_match = re.search(
            r'#\s*Main\s+function[^\n]*\n\s*(def\s+\w+\s*\([^)]*\)\s*:)',
            completion,
            flags=re.IGNORECASE
        )
        if main_comment_match:
            func_start = main_comment_match.start(1)
            return _extract_single_function(completion, func_start)
        
        # Method 4: Find any function definition that's not 'aux'
        matches = list(re.finditer(r'def\s+(\w+)\s*\([^)]*\)\s*:', completion))
        for match in matches:
            func_name = match.group(1)
            if func_name != 'aux':
                return _extract_single_function(completion, match.start())
        
        # Method 5: If no main function found, return empty (not the whole completion)
        return ""

    # =========================================================================
    # Generation Methods
    # =========================================================================

    def _generate_completions(
        self,
        agent: PreTrainedModel,
        batch_items: List[Dict],
        agent_idx: int = 0,
        num_return_sequences: int = 1,
        max_new_tokens: int = 256,
        prompts_override: Optional[List[str]] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate completions from an agent.
        
        Returns dict with:
        - prompts: List of prompts
        - completions: List of completion lists
        - prompt_input_ids: Tokenized prompts
        - completion_input_ids: Tokenized completions
        - batch_items: Original batch items
        """
        device = self.device

        # Apply formatter to create prompts
        if prompts_override is not None:
            prompts = prompts_override
        else:
            format_func = self.formatters[agent_idx]
            prompts = [format_func(item) for item in batch_items]

        # Apply chat template if enabled
        if self.use_chat_template:
            prompts = self._apply_chat_template(prompts)

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original training state
        training_mode = agent.training
        original_requires_grad = {}
        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        agent.eval()

        # Generation kwargs
        generation_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": max_new_tokens,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,  # Explicitly set to suppress warning
        }

        # Sampling settings - always use sampling with temperature/top_p
        # (greedy decoding causes models to copy prompt patterns)
        if do_sample is False:
            # Explicitly disabled - use greedy
            generation_kwargs["do_sample"] = False
            generation_kwargs["num_return_sequences"] = 1
        else:
            # Default: always use sampling (even with num_return_sequences=1)
            generation_kwargs.update({
                "do_sample": True,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "num_beams": 1,
                "num_return_sequences": num_return_sequences,
            })
            if self.args.top_k is not None:
                generation_kwargs["top_k"] = self.args.top_k

        # Generate
        with torch.no_grad():
            generation_output = agent.generate(**generation_kwargs)

        # Extract completions
        generated_ids = generation_output.sequences
        batch_size = prompt_input_ids.size(0)
        # Get padded prompt length (may include padding tokens)
        padded_prompt_length = prompt_input_ids.size(1)

        completions_per_prompt = []
        completion_ids_per_prompt = []

        for batch_idx in range(batch_size):
            start_idx = batch_idx * num_return_sequences
            end_idx = start_idx + num_return_sequences
            batch_sequences = generated_ids[start_idx:end_idx]

            # Calculate actual prompt length for this sample (excluding padding)
            # This handles both left and right padding correctly
            actual_prompt_length = prompt_attention_mask[batch_idx].sum().item()
            
            # For left padding, we need to use padded_prompt_length as the split point
            # because generate() preserves the full padded sequence
            # For right padding, same logic applies
            prompt_length_for_split = padded_prompt_length

            completions = []
            completion_ids = []

            for seq in batch_sequences:
                completion_tokens = seq[prompt_length_for_split:]
                
                # Filter out padding tokens from completion if any
                if self.tokenizer.pad_token_id is not None:
                    completion_tokens = completion_tokens[
                        completion_tokens != self.tokenizer.pad_token_id
                    ]
                
                completion_text = self.tokenizer.decode(
                    completion_tokens,
                    skip_special_tokens=True,
                )
                raw_completion_text = completion_text  # Save before cleaning
                completion_text = self._clean_completion(completion_text)
                
                # Debug: log when clean_completion significantly changed the output
                if len(raw_completion_text) > 50 and len(completion_text) < len(raw_completion_text) * 0.3:
                    print(f"\n[DEBUG] _clean_completion significantly truncated output for agent_idx={agent_idx}")
                    print(f"  Before clean length: {len(raw_completion_text)}")
                    print(f"  After clean length: {len(completion_text)}")
                    print(f"  Before clean (first 500 chars): {repr(raw_completion_text[:500])}")
                    print(f"  After clean: {repr(completion_text[:300])}")
                
                # Debug: log when NO tokens were generated at all
                if len(completion_tokens) == 0:
                    print(f"\n[DEBUG] NO completion tokens generated for agent_idx={agent_idx}, batch_idx={batch_idx}")
                    print(f"  Original sequence length: {len(seq)}")
                    print(f"  Prompt split point: {prompt_length_for_split}")
                    print(f"  Full sequence tokens (first 50): {seq.tolist()[:50]}")
                    # Check if model just output EOS immediately
                    if len(seq) > prompt_length_for_split:
                        post_prompt = seq[prompt_length_for_split:]
                        print(f"  Post-prompt tokens: {post_prompt.tolist()}")
                    print(f"  Prompt text: {repr(prompts[batch_idx][:200] if batch_idx < len(prompts) else 'N/A')}")
                
                # Debug: log when tokens exist but text is empty
                elif len(completion_text.strip()) == 0:
                    print(f"\n[DEBUG] Empty completion text but {len(completion_tokens)} tokens generated")
                    print(f"  Token IDs (first 30): {completion_tokens.tolist()[:30]}")
                    raw_decoded = self.tokenizer.decode(completion_tokens, skip_special_tokens=False)
                    print(f"  Raw decoded (with special tokens): {repr(raw_decoded[:200])}")
                    # Check for special tokens
                    eos_count = (completion_tokens == self.tokenizer.eos_token_id).sum().item()
                    pad_count = (completion_tokens == self.tokenizer.pad_token_id).sum().item() if self.tokenizer.pad_token_id else 0
                    print(f"  EOS tokens: {eos_count}, PAD tokens: {pad_count}")
                
                completions.append(completion_text)
                completion_ids.append(completion_tokens)

            completions_per_prompt.append(completions)
            completion_ids_per_prompt.append(completion_ids)

        # Restore model state
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]
        if training_mode:
            agent.train()

        return {
            "prompts": prompts,
            "completions": completions_per_prompt,
            "prompt_input_ids": prompt_input_ids,
            "completion_input_ids": completion_ids_per_prompt,
            "batch_items": batch_items,
        }

    def _clean_completion(self, completion: str) -> str:
        """Clean up generated completion.
        
        For agent chaining mode, we preserve both aux and main functions.
        The model outputs in natural comment format:
        - # Predicted helper function
        - # Main function
        """
        # Strip leading newlines/whitespace first
        completion = completion.lstrip('\n')
        
        # Check if this looks like agent chaining output (has both aux and main function patterns)
        has_aux_comment = "# Predicted helper function" in completion or "# predicted helper" in completion.lower()
        has_main_comment = "# Main function" in completion or "# main function" in completion.lower()
        has_aux_def = "def aux(" in completion
        has_main_def = re.search(r'def\s+(?!aux\b)\w+\s*\(', completion) is not None
        
        # Agent chaining mode: preserve both functions
        if (has_aux_comment or has_aux_def) and (has_main_comment or has_main_def):
            # Find all function definitions
            func_matches = list(re.finditer(r'\ndef\s+\w+\s*\([^)]*\)\s*:', completion))
            if not func_matches:
                func_matches = list(re.finditer(r'^def\s+\w+\s*\([^)]*\)\s*:', completion, re.MULTILINE))
            
            if len(func_matches) >= 2:
                # Keep from the first function definition to end of second function
                first_func_start = func_matches[0].start()
                
                # Find end of second function (next function or end of completion)
                if len(func_matches) > 2:
                    # Cut before third function
                    second_func_end = func_matches[2].start()
                    completion = completion[first_func_start:second_func_end].rstrip()
                else:
                    # Only two functions, keep both
                    completion = completion[first_func_start:].rstrip()
                
                # Apply stop tokens only after function extraction
                stop_tokens = ["\n\n\n", "```", "\nclass ", "\nif __name__"]
                for token in stop_tokens:
                    if token in completion:
                        completion = completion.split(token)[0]
                
                return completion.strip()
        
        # Check for XML tags (legacy format)
        has_predicted_aux = "<predicted_aux>" in completion and "</predicted_aux>" in completion
        has_main_function = "<main_function>" in completion and "</main_function>" in completion
        
        if has_predicted_aux or has_main_function:
            # Agent chaining mode with tags
            if has_main_function:
                end_tag = "</main_function>"
                end_pos = completion.find(end_tag)
                if end_pos != -1:
                    completion = completion[:end_pos + len(end_tag)]
            elif has_predicted_aux:
                tag_end = completion.find("</predicted_aux>")
                after_tag = completion[tag_end:]
                parts = after_tag.split("\n\ndef ")
                if len(parts) > 2:
                    after_tag_cleaned = parts[0] + "\n\ndef " + parts[1]
                    completion = completion[:tag_end] + after_tag_cleaned
            return completion.strip()
        
        # Standard mode (single function): apply stop tokens first
        stop_tokens = ["\n\n\n", "```", "\nclass ", "\nif __name__"]
        for token in stop_tokens:
            if token in completion:
                completion = completion.split(token)[0]
        
        completion = completion.lstrip('\n')
        
        # Keep only the first function definition
        if "\n\ndef " in completion:
            parts = completion.split("\n\ndef ")
            first_part = parts[0].strip()
            if first_part:
                completion = first_part
            elif len(parts) > 1:
                completion = "def " + parts[1]
        
        return completion.strip()

    def _apply_chat_template(self, prompts: List[str]) -> List[str]:
        """Apply chat template for instruct models."""
        templated = []
        for prompt in prompts:
            messages = []
            if self.args.chat_template_system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.args.chat_template_system_prompt
                })
            messages.append({"role": "user", "content": prompt})

            try:
                templated_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                templated.append(templated_prompt)
            except Exception:
                templated.append(prompt)

        return templated

    # =========================================================================
    # Policy Update Methods
    # =========================================================================

    def _update_policy_for_agent(
        self,
        agent_idx: int,
        completions_data: Dict[str, Any],
        advantages: List[float],
        old_log_probs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[float, List[torch.Tensor]]:
        """
        Update policy for a single agent using refined advantages.
        
        If use_ppo_clip=True and old_log_probs provided:
            loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        Otherwise:
            loss = -log_prob * advantage (simple policy gradient)
        
        Returns:
            loss: The computed loss value
            new_log_probs: List of new log probabilities for each completion
        """
        agent = self.agents[agent_idx]
        optimizer = self.optimizers[agent_idx]
        new_log_probs_out = []

        if optimizer is None:
            return 0.0, new_log_probs_out

        device = self.device
        eps = self.args.ppo_clip_eps
        use_ppo_clip = self.args.use_ppo_clip

        # Debug: check input data
        if self.verbose:
            print(f"\n[DEBUG] _update_policy_for_agent: agent_idx={agent_idx}")
            print(f"  advantages: {advantages}")
            print(f"  use_ppo_clip: {use_ppo_clip}, eps: {eps}")
            print(f"  old_log_probs provided: {old_log_probs is not None}")

        if not advantages:
            if self.verbose:
                print(f"  [WARN] No advantages provided, skipping update")
            return 0.0, new_log_probs_out

        advantages_tensor = torch.tensor(advantages, dtype=self.compute_dtype, device=device)

        # Normalize advantages if enabled (only if multiple samples)
        if self.args.advantage_normalization and len(advantages_tensor) > 1:
            adv_mean = advantages_tensor.mean()
            adv_std = advantages_tensor.std().clamp(min=1e-8)
            advantages_tensor = (advantages_tensor - adv_mean) / adv_std
            if self.verbose:
                print(f"  Normalized advantages: mean={adv_mean.item():.4f}, std={adv_std.item():.4f}")

        agent.train()
        optimizer.zero_grad()

        # Handle prompt_input_ids
        prompt_input_ids = completions_data.get("prompt_input_ids")
        if prompt_input_ids is None:
            if self.verbose:
                print(f"  [ERROR] prompt_input_ids is None")
            return 0.0, new_log_probs_out
        prompt_input_ids = prompt_input_ids.to(device)

        # Handle completion_input_ids - this has complex nested structure
        completion_input_ids = completions_data.get("completion_input_ids")
        if completion_input_ids is None:
            if self.verbose:
                print(f"  [ERROR] completion_input_ids is None")
            return 0.0, new_log_probs_out

        # Debug: print structure
        if self.verbose:
            print(f"  completion_input_ids type: {type(completion_input_ids)}")
            if isinstance(completion_input_ids, list) and completion_input_ids:
                print(f"  completion_input_ids[0] type: {type(completion_input_ids[0])}")
                if isinstance(completion_input_ids[0], list) and completion_input_ids[0]:
                    print(f"  completion_input_ids[0][0] type: {type(completion_input_ids[0][0])}")
                    if hasattr(completion_input_ids[0][0], 'shape'):
                        print(f"  completion_input_ids[0][0] shape: {completion_input_ids[0][0].shape}")

        # Flatten nested structure properly
        flat_completion_ids = []
        if isinstance(completion_input_ids, list):
            for item in completion_input_ids:
                if isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, torch.Tensor):
                            flat_completion_ids.append(sub_item.to(device))
                elif isinstance(item, torch.Tensor):
                    flat_completion_ids.append(item.to(device))

        if not flat_completion_ids:
            if self.verbose:
                print(f"  [ERROR] No completion tokens found after flattening")
            return 0.0, new_log_probs_out

        if self.verbose:
            print(f"  Flattened completion_ids count: {len(flat_completion_ids)}")

        prompt_ids = prompt_input_ids[0]
        total_loss = torch.tensor(0.0, dtype=self.compute_dtype, device=device, requires_grad=True)
        num_samples = 0

        for seq_idx, completion_tokens in enumerate(flat_completion_ids):
            if seq_idx >= len(advantages_tensor):
                break

            advantage = advantages_tensor[seq_idx]

            if len(completion_tokens) > 0:
                # Build input: prompt + completion[:-1] to predict completion
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                target_ids = completion_tokens
                attention_mask = torch.ones(len(input_ids), device=device)

                outputs = agent(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )

                # Get logits for completion tokens
                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1: -1, :]

                log_probs = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(0):
                        token_log_prob = F.log_softmax(completion_logits[i], dim=-1)[token_id]
                        log_probs.append(token_log_prob)

                if log_probs:
                    new_seq_log_prob = torch.stack(log_probs).sum()
                    new_log_probs_out.append(new_seq_log_prob.detach())

                    if use_ppo_clip and old_log_probs is not None and seq_idx < len(old_log_probs):
                        # PPO-clip objective
                        old_log_prob = old_log_probs[seq_idx]
                        if isinstance(old_log_prob, torch.Tensor):
                            old_log_prob = old_log_prob.to(device)
                        else:
                            old_log_prob = torch.tensor(old_log_prob, dtype=self.compute_dtype, device=device)

                        # Importance ratio
                        ratio = torch.exp(new_seq_log_prob - old_log_prob)
                        
                        # Clipped surrogate objective
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage
                        loss = -torch.min(surr1, surr2)

                        if self.verbose and seq_idx == 0:
                            print(f"  [PPO] ratio={ratio.item():.4f}, surr1={surr1.item():.4f}, surr2={surr2.item():.4f}")
                    else:
                        # Simple policy gradient
                        loss = -new_seq_log_prob * advantage

                    total_loss = total_loss + loss
                    num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

            if self.verbose:
                print(f"  total_loss: {total_loss.item():.6f}, num_samples: {num_samples}")

            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                optimizer.step()
            else:
                if self.verbose:
                    print(f"  [WARN] Loss is nan/inf, skipping update")
        else:
            if self.verbose:
                print(f"  [WARN] num_samples=0, no update performed")

        return total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss, new_log_probs_out

    def _update_value_head_for_agent(
        self,
        agent_idx: int,
        prompts: List[str],
        completions: List[str],
        returns: List[float],
    ) -> float:
        """
        Update value head using MSE loss against actual returns.
        
        Loss = (1/K) * sum_k (V^{(i)}(s, b^{(i)}_k) - G_k)^2
        
        Where G_k is the actual return (cumulative reward), NOT a TD target.
        """
        if not returns or not completions:
            return 0.0

        value_head = self.value_heads[agent_idx]
        optimizer = self.value_optimizers[agent_idx]

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, dtype=self.compute_dtype, device=self.device, requires_grad=True)
        num_samples = 0

        for prompt, completion, ret in zip(prompts, completions, returns):
            # Get value prediction with gradient
            value_pred = self._estimate_value_with_grad_for_agent(
                agent_idx, prompt, completion
            )

            # MSE loss against actual return (NOT TD target)
            target = torch.tensor(ret, dtype=self.compute_dtype, device=self.device)
            loss = (value_pred - target) ** 2
            total_loss = total_loss + loss
            num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples
            total_loss = total_loss * self.args.value_loss_coef

            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                optimizer.step()

        return total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

    # =========================================================================
    # Simultaneous Update
    # =========================================================================

    def _simultaneous_update(self, all_nodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Update all agents simultaneously.
        
        Each agent is updated independently based on their refined advantages.
        This differs from HAPPO which updates agents sequentially with M factor.
        
        Process:
        1. Compute value estimates for all agents
        2. Compute TD residuals
        3. Compute refined advantages for each agent
        4. Update each agent's policy using their advantages
        5. Update each agent's value head using MSE against returns
        """
        update_stats = {}

        for node in all_nodes:
            returns_vec = node.get("returns") or []
            rewards_vec = node.get("rewards") or []
            comps_per_agent = node.get("completions") or []
            prompts = node.get("prompts") or []

            if not returns_vec or not comps_per_agent:
                continue

            # Process each joint action
            num_joint_actions = len(returns_vec)

            for joint_idx in range(num_joint_actions):
                # Get completions for this joint action
                joint_completions = []
                joint_prompts = []

                for agent_idx in range(self.num_agents):
                    if agent_idx < len(comps_per_agent):
                        agent_comps = comps_per_agent[agent_idx].get("completions", [[]])[0]
                        if joint_idx < len(agent_comps):
                            joint_completions.append(agent_comps[joint_idx])
                        else:
                            joint_completions.append("")
                    else:
                        joint_completions.append("")

                    # Get prompt for this agent
                    if agent_idx < len(prompts) and prompts:
                        if isinstance(prompts[0], list):
                            joint_prompts.append(prompts[agent_idx][0] if prompts[agent_idx] else "")
                        else:
                            joint_prompts.append(prompts[0] if prompts else "")
                    else:
                        joint_prompts.append("")

                # Compute value estimates for all agents
                agent_values = []
                for agent_idx in range(self.num_agents):
                    value = self._estimate_value_for_agent(
                        agent_idx,
                        joint_prompts[agent_idx] if agent_idx < len(joint_prompts) else "",
                        joint_completions[agent_idx] if agent_idx < len(joint_completions) else "",
                    )
                    agent_values.append(value)

                # Compute TD residuals
                reward = rewards_vec[joint_idx] if joint_idx < len(rewards_vec) else 0.0
                td_residuals = self._compute_td_residuals(agent_values, reward)

                # Compute refined advantages for each agent
                td_residuals_per_turn = [td_residuals]  # Single turn for now

                for agent_idx in range(self.num_agents):
                    advantage = self._compute_refined_advantage(
                        agent_idx, td_residuals_per_turn
                    )

                    # Store advantage in node for later use
                    if "advantages" not in node:
                        node["advantages"] = [[] for _ in range(self.num_agents)]
                    node["advantages"][agent_idx].append(advantage.item())

            # Now update each agent
            for agent_idx in range(self.num_agents):
                if agent_idx >= len(comps_per_agent):
                    continue

                agent_completions = comps_per_agent[agent_idx].get("completions", [[]])[0]
                agent_advantages = node.get("advantages", [[] for _ in range(self.num_agents)])[agent_idx]
                agent_returns = list(returns_vec)

                # Update policy
                if agent_advantages and agent_idx < len(comps_per_agent):
                    # Get old_log_probs if available (for PPO clip)
                    old_log_probs = node.get("old_log_probs_per_agent", [None] * self.num_agents)
                    agent_old_log_probs = old_log_probs[agent_idx] if agent_idx < len(old_log_probs) else None
                    
                    policy_loss, new_log_probs = self._update_policy_for_agent(
                        agent_idx,
                        comps_per_agent[agent_idx],
                        agent_advantages,
                        old_log_probs=agent_old_log_probs,
                    )
                    update_stats[f"agent_{agent_idx}_policy_loss"] = policy_loss

                # Update value head
                if prompts and agent_completions:
                    agent_prompts = []
                    for _ in range(len(agent_completions)):
                        if isinstance(prompts[0], list) and agent_idx < len(prompts):
                            agent_prompts.append(prompts[agent_idx][0] if prompts[agent_idx] else "")
                        elif prompts:
                            agent_prompts.append(prompts[0])
                        else:
                            agent_prompts.append("")

                    value_loss = self._update_value_head_for_agent(
                        agent_idx,
                        agent_prompts,
                        agent_completions,
                        agent_returns,
                    )
                    update_stats[f"agent_{agent_idx}_value_loss"] = value_loss

        return update_stats

    # =========================================================================
    # Reward Computation
    # =========================================================================

    def _compute_rewards(
        self,
        batch_items: List[Dict],
        completions_per_agent: List[Dict[str, Any]],
        combo_indices: List[Tuple[int, ...]],
    ) -> List[float]:
        """Compute rewards for joint actions."""
        rewards = []
        
        # Get entry_point from batch_items for main function extraction
        entry_point = None
        if batch_items and len(batch_items) > 0:
            entry_point = batch_items[0].get("entry_point")

        for idx_tuple in combo_indices:
            # Get completions for this combination
            agent_completions = []
            for agent_idx, comp_idx in enumerate(idx_tuple):
                if agent_idx < len(completions_per_agent):
                    comps = completions_per_agent[agent_idx].get("completions", [[]])[0]
                    if comp_idx < len(comps):
                        completion = comps[comp_idx]
                        # For agent chaining, extract main function from agent 2
                        if self.agent_chaining and agent_idx == self.num_agents - 1:
                            original_completion = completion
                            completion = self._extract_main_from_chaining_output(
                                completion, entry_point=entry_point
                            )
                            # Debug: log when extraction results in empty string
                            if not completion.strip() and original_completion.strip():
                                print(f"\n[DEBUG] _extract_main_from_chaining_output returned EMPTY")
                                print(f"  entry_point: {entry_point}")
                                print(f"  Original completion length: {len(original_completion)}")
                                print(f"  Original completion (first 500 chars):")
                                print(f"  {repr(original_completion[:500])}")
                        agent_completions.append(completion)
                    else:
                        agent_completions.append("")
                else:
                    agent_completions.append("")

            # Call reward function
            # Build completion_args: each agent's completion wrapped in a list
            # This matches the expected signature: reward_func([comp1], [comp2], ..., batch_items=...)
            completion_args = [[comp] for comp in agent_completions]

            try:
                # Check if reward function accepts batch_items parameter
                sig = inspect.signature(self.reward_func)
                if "batch_items" in sig.parameters:
                    # Call with batch_items keyword argument
                    reward_result = self.reward_func(
                        *completion_args, batch_items=batch_items
                    )
                else:
                    # Fallback: try without batch_items
                    reward_result = self.reward_func(*completion_args)

                if isinstance(reward_result, (list, tuple)):
                    reward = float(reward_result[0]) if reward_result else 0.0
                else:
                    reward = float(reward_result)

                reward = self.reward_processor(reward)
                rewards.append(reward)

            except TypeError:
                # Fallback for different signature patterns
                try:
                    reward_result = self.reward_func(agent_completions)
                    if isinstance(reward_result, (list, tuple)):
                        reward = float(reward_result[0]) if reward_result else 0.0
                    else:
                        reward = float(reward_result)
                    reward = self.reward_processor(reward)
                    rewards.append(reward)
                except Exception as e:
                    if self.verbose:
                        print(f"Reward computation error (fallback): {e}")
                    rewards.append(0.0)

            except Exception as e:
                if self.verbose:
                    print(f"Reward computation error: {e}")
                rewards.append(0.0)

        return rewards

    # =========================================================================
    # Training Step
    # =========================================================================

    def _train_step(
        self,
        batch_items: List[Dict],
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Execute a single training step.
        
        1. Generate completions for all agents
        2. Compute rewards for joint actions
        3. Compute KL similarity rewards (if agent chaining)
        4. Compute returns
        5. Update all agents simultaneously
        """
        num_generations = self.args.num_generations
        joint_mode = str(self.args.joint_mode).lower()

        # Generate completions for each agent
        completions_per_agent = []
        for agent_idx in range(self.num_agents):
            max_tokens = self._get_max_tokens_for_agent(agent_idx)
            comps_data = self._generate_completions(
                self.agents[agent_idx],
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_generations,
                max_new_tokens=max_tokens,
            )
            completions_per_agent.append(comps_data)

        # Build joint action combinations
        if joint_mode == "cross":
            # Cartesian product
            indices_per_agent = [list(range(num_generations)) for _ in range(self.num_agents)]
            import itertools
            combo_indices = list(itertools.product(*indices_per_agent))
        else:
            # Aligned: index-matched
            combo_indices = [(i,) * self.num_agents for i in range(num_generations)]

        # Compute rewards
        rewards = self._compute_rewards(batch_items, completions_per_agent, combo_indices)

        # Compute returns (single turn for now)
        returns = list(rewards)

        # Build node
        node = {
            "completions": completions_per_agent,
            "prompts": [cd.get("prompts", []) for cd in completions_per_agent],
            "rewards": rewards,
            "returns": returns,
            "combo_indices": combo_indices,
            "env_step": self.env_step,
        }

        # Simultaneous update
        update_stats = self._simultaneous_update([node])

        self.env_step += 1

        # Compute batch stats
        batch_loss = float(np.mean(np.abs(rewards))) if rewards else 0.0
        batch_stats = {
            "batch_mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "batch_mean_return": float(np.mean(returns)) if returns else 0.0,
        }
        batch_stats.update(update_stats)

        return batch_loss, batch_stats

    # =========================================================================
    # Training Loop
    # =========================================================================

    def train(self) -> None:
        """Main training loop."""
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided for training.")

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

        num_epochs = int(self.args.num_train_epochs)
        total_steps = num_epochs * len(train_dataloader)

        if self.verbose:
            print(f"Starting ACPPO training for {num_epochs} epochs ({total_steps} steps)")
            print(f"  - Num agents: {self.num_agents}")
            print(f"  - Agent chaining: {self.agent_chaining}")
            print(f"  - Gamma': {self.args.gamma_prime}, Lambda': {self.args.lambda_prime}")

        global_step = 0

        for epoch in range(num_epochs):
            epoch_rewards = []
            epoch_returns = []

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=False,  # Always show tqdm regardless of verbose setting
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Convert batch to list of dicts
                if isinstance(batch, dict):
                    batch_items = [{k: v[i] for k, v in batch.items()} for i in range(len(next(iter(batch.values()))))]
                else:
                    batch_items = [batch]

                # Training step
                batch_loss, batch_stats = self._train_step(batch_items)

                epoch_rewards.append(batch_stats.get("batch_mean_reward", 0.0))
                epoch_returns.append(batch_stats.get("batch_mean_return", 0.0))

                # Update progress bar
                progress_bar.set_postfix({
                    "reward": f"{batch_stats.get('batch_mean_reward', 0):.3f}",
                })

                # Logging
                if (
                    self.wandb_initialized
                    and wandb.run is not None
                    and global_step % self.args.logging_steps == 0
                ):
                    log_dict = {
                        "train/reward": batch_stats.get("batch_mean_reward", 0),
                        "train/return": batch_stats.get("batch_mean_return", 0),
                        "train/epoch": epoch + 1,
                    }
                    for key, val in batch_stats.items():
                        if "loss" in key:
                            log_dict[f"train/{key}"] = val
                    wandb.log(log_dict, step=global_step)

                # Evaluation
                if (
                    self.eval_dataset is not None
                    and global_step > 0
                    and global_step % self.args.eval_interval == 0
                ):
                    eval_stats = self._evaluate()
                    if self.wandb_initialized and wandb.run is not None:
                        wandb.log(
                            {f"eval/{k}": v for k, v in eval_stats.items()},
                            step=global_step,
                        )

                global_step += 1

            # Epoch summary
            if self.verbose:
                print(
                    f"Epoch {epoch + 1} - Mean reward: {np.mean(epoch_rewards):.4f}, "
                    f"Mean return: {np.mean(epoch_returns):.4f}"
                )

        if self.verbose:
            print("Training completed!")

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation with detailed logging support."""
        if self.eval_dataset is None:
            return {}

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
        )

        eval_rewards = []
        num_samples = min(self.args.eval_num_samples, len(self.eval_dataset))

        # For detailed logging
        all_agent_completions_turns: List[List[List[str]]] = [
            [] for _ in range(self.num_agents)
        ]
        all_test_cases: List[str] = []
        all_entry_points: List[str] = []
        all_prompts: List[str] = []
        all_code_prompts: List[str] = []  # For BigCodeBench

        for agent in self.agents:
            agent.eval()

        with torch.no_grad():
            for idx, batch in enumerate(eval_dataloader):
                if idx >= num_samples:
                    break

                if isinstance(batch, dict):
                    batch_items = [{k: v[i] for k, v in batch.items()} for i in range(len(next(iter(batch.values()))))]
                else:
                    batch_items = [batch]

                # Collect metadata for logging
                for item in batch_items:
                    all_test_cases.append(item.get("test", ""))
                    all_entry_points.append(item.get("entry_point", ""))
                    all_prompts.append(item.get("prompt", ""))
                    all_code_prompts.append(item.get("code_prompt", ""))  # For BigCodeBench

                # Generate single completion per agent (greedy)
                completions_per_agent = []
                for agent_idx in range(self.num_agents):
                    comps_data = self._generate_completions(
                        self.agents[agent_idx],
                        batch_items,
                        agent_idx=agent_idx,
                        num_return_sequences=1,
                        max_new_tokens=self._get_max_tokens_for_agent(agent_idx),
                        do_sample=False,
                    )
                    completions_per_agent.append(comps_data)

                    # Store completions for logging (per agent, per sample, per turn)
                    # Single turn for now
                    agent_comps = comps_data.get("completions", [[]])[0]
                    completion_text = agent_comps[0] if agent_comps else ""
                    
                    # For agent chaining, extract main function from last agent
                    if self.agent_chaining and agent_idx == self.num_agents - 1:
                        entry_point = batch_items[0].get("entry_point") if batch_items else None
                        completion_text = self._extract_main_from_chaining_output(
                            completion_text, entry_point=entry_point
                        )
                    
                    all_agent_completions_turns[agent_idx].append([completion_text])

                # Compute reward
                combo_indices = [(0,) * self.num_agents]
                rewards = self._compute_rewards(batch_items, completions_per_agent, combo_indices)
                if rewards:
                    eval_rewards.append(rewards[0])

        for agent in self.agents:
            agent.train()

        # Basic metrics
        eval_metrics = {
            "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "num_samples": len(eval_rewards),
        }

        # Detailed logging with eval_logger and eval_aggregator (fully_passed_rate, etc.)
        if (
            self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions_turns
            and all(agent_comps for agent_comps in all_agent_completions_turns)
        ):
            try:
                detailed_metrics = self.eval_logger(
                    agent_completions_turns=all_agent_completions_turns,
                    test_cases=all_test_cases,
                    entry_points=all_entry_points,
                    prompts=all_prompts,
                    code_prompts=all_code_prompts,  # For BigCodeBench
                )

                # Aggregate metrics for logging
                aggregated_detailed_metrics = self.eval_aggregator(
                    detailed_metrics, num_turns=self.args.num_turns
                )
                for key, value in aggregated_detailed_metrics.items():
                    eval_metrics[key] = value
            except Exception as e:
                if self.verbose:
                    print(f"Warning: eval_logger/eval_aggregator failed: {e}")

        return eval_metrics

    # =========================================================================
    # W&B and Utility Methods
    # =========================================================================

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases."""
        if self.wandb_initialized:
            return

        if self.wandb_config is None:
            self.wandb_config = {}

        wandb_project = self.wandb_config.get("project", "acppo")
        wandb_entity = self.wandb_config.get("entity", None)
        wandb_name = self.wandb_config.get("name", "acppo_run")
        wandb_dir = self.wandb_config.get("dir", None)
        wandb_tags = self.wandb_config.get("tags", ["acppo"])

        config_dict = {
            "model_name": self.model_name,
            "num_agents": self.num_agents,
            "agent_chaining": self.agent_chaining,
            "learning_rate": self.args.learning_rate,
            "value_learning_rate": self.args.value_learning_rate,
            "gamma_prime": self.args.gamma_prime,
            "lambda_prime": self.args.lambda_prime,
            "num_train_epochs": self.args.num_train_epochs,
            "num_generations": self.args.num_generations,
        }

        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                dir=wandb_dir,
                tags=wandb_tags,
                config=config_dict,
            )
            self.wandb_initialized = True
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            self.wandb_initialized = False

    def save_model(self, output_dir: str) -> None:
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = os.path.join(output_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)

            if self.use_lora:
                agent.save_pretrained(agent_dir)
            else:
                agent.save_pretrained(agent_dir)

            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

            # Save value head
            value_head_path = os.path.join(agent_dir, "value_head.pt")
            torch.save(self.value_heads[agent_idx].state_dict(), value_head_path)

        if self.verbose:
            print(f"Models saved to {output_dir}")
