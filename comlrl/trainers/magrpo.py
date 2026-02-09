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
from tqdm import tqdm  # type: ignore
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments


@dataclass
class MAGRPOConfig(TrainingArguments):
    """
    Configuration for MAGRPO training, inheriting from TrainingArguments.
    Supports both single-turn and multi-turn training modes.
    """

    # Core setup
    num_train_epochs: float = field(
        default=20,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Per-device batch size (must be 1 for MAGRPO)."},
    )
    learning_rate: float = field(
        default=5.0e-6,
        metadata={"help": "Learning rate for optimizer."},
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
        metadata={"help": "Number of agents; set to 1 for single-agent GRPO."},
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
        metadata={
            "help": "Temperature for sampling (present for completeness; generation uses model_config if provided)."
        },
    )
    top_p: float = field(
        default=0.6,
        metadata={
            "help": "Top-p for sampling (present for completeness; generation uses model_config if provided)."
        },
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

    # Agent Chaining
    agent_chaining: bool = field(
        default=False,
        metadata={
            "help": "Enable agent chaining mode where Agent 2 first reasons about Agent 1's output then generates main."
        },
    )
    max_new_tokens_per_agent: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of max_new_tokens for each agent. If None, uses max_new_tokens for all. "
                    "Agent 2 typically needs more tokens to include both reasoning and main function."
        },
    )
    format_reward_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for format reward (reasoning accuracy) in agent chaining mode."
        },
    )

    # Chat Template (for Instruct models)
    use_chat_template: bool = field(
        default=False,
        metadata={
            "help": "Use chat template for Instruct models. Applies tokenizer.apply_chat_template() to prompts."
        },
    )
    chat_template_system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional system prompt to include in chat template. If None, no system prompt is added."
        },
    )


@dataclass
class NodeSample:
    agent_idx: int
    turn_idx: int
    completions_data: Dict[str, Any]
    returns: List[float]
    node_mean_reward: float
    node_mean_return: float
    node_env_step: int
    node_mean_format_reward: Optional[float] = None  # Agent Chaining format reward


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization Trainer (MAGRPO).
    Supports both single-turn and multi-turn training with external transitions.

    When num_turns=1, this trainer behaves as a standard MAGRPO trainer.
    When num_turns>1, it adds multi-turn capabilities with external transitions between turns.

    Args:
        model: The model to be trained for homogeneous agents
        agents: List of agent models (alternative to model)
        num_agents: The number of agents
        reward_func: Single reward function callable
        reward_processor: Optional processor to apply to the reward (e.g., scaling)
        formatters: Formatters to apply to dataset items for each agent
        args: The training arguments
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        tokenizer: The tokenizer
        wandb_config: Configuration for Weights & Biases logging
        model_config: Model configuration dict
        eval_logger: Evaluation logger function
        eval_aggregator: Evaluation aggregator function
        external_transition: Function that provides external transitions between turns
        dataset_type: Optional explicit dataset type (e.g., "humaneval")
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
        args: Optional[MAGRPOConfig] = None,
        # LoRA configuration
        use_lora: bool = False,
    ):
        # Check for GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU not found. MAGRPOTrainer requires GPU for training."
            )

        if model is None and agents is None:
            raise ValueError("Either model or agents must be provided")
        if model is not None and agents is not None:
            raise ValueError("Cannot provide both model and agents parameters")

        # Training arguments
        self.args = args if args is not None else MAGRPOConfig()
        self.env_step = 0
        self._last_train_log_step = -1

        # Reward and formatting
        self._setup_formatters(formatters, num_agents)
        self._setup_reward_function(reward_func, reward_processor)

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

            self.model_config = model_config if model_config else {}
        else:
            self.model_config = model_config if model_config else {}
            self.num_agents = num_agents
            if isinstance(model, str):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model_load_kwargs = dict(self.model_config.get("model_kwargs", {}))
                if "attn_implementation" not in model_load_kwargs:
                    model_load_kwargs["attn_implementation"] = "flash_attention_2"

                # self.agents = [
                #     AutoModelForCausalLM.from_pretrained(
                #         model, **self.model_config.get("model_kwargs", {})
                #     )
                #     for _ in range(num_agents)
                # ]
                self.agents = [
                    AutoModelForCausalLM.from_pretrained(
                        model, **model_load_kwargs
                    )
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

        # Allow single-agent as a special case (GRPO)
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.args.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 (group baseline requires multiple samples)."
            )
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("MAGRPO requires per_device_train_batch_size to be 1. ")
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

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

        # Dataset type: prefer explicit parameter, fallback to config sections
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

        # Agent Chaining setup
        self.agent_chaining = getattr(self.args, "agent_chaining", False)
        self.format_reward_func = None  # Will be set externally if needed
        
        # Setup per-agent max_new_tokens
        max_tokens_per_agent = getattr(self.args, "max_new_tokens_per_agent", None)
        if max_tokens_per_agent is not None:
            if len(max_tokens_per_agent) != self.num_agents:
                raise ValueError(
                    f"max_new_tokens_per_agent length ({len(max_tokens_per_agent)}) "
                    f"must match num_agents ({self.num_agents})"
                )
            self.max_new_tokens_per_agent = list(max_tokens_per_agent)
        else:
            # Use default max_new_tokens for all agents
            self.max_new_tokens_per_agent = [self.args.max_new_tokens] * self.num_agents

        # Chat template setup (for Instruct models)
        self.use_chat_template = getattr(self.args, "use_chat_template", False)
        self.chat_template_system_prompt = getattr(self.args, "chat_template_system_prompt", None)

        # Verbosity from config (default True)
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

    def _setup_formatters(self, formatters, num_agents):
        """Set up format functions for each agent that can handle external transitions."""
        # Use multi-turn compatible default formatter that accepts external prompts
        default_format_func = lambda x, external_prompts=None: x.get("prompt", "")

        if formatters is None:
            # Just use the default formatter for all agents
            self.formatters = [default_format_func] * num_agents
        elif callable(formatters) and not isinstance(formatters, list):
            # We have a single formatter and we should apply it to all agents
            # Wrap the formatter to accept external_prompts parameter
            original_formatter = formatters
            sig = inspect.signature(original_formatter)
            if "external_prompts" in sig.parameters:
                wrapped_formatter = lambda x, external_prompts=None: (
                    original_formatter(x, external_prompts=external_prompts)
                    if external_prompts is not None
                    else original_formatter(x)
                )
            else:
                wrapped_formatter = lambda x, external_prompts=None: original_formatter(
                    x
                )
            self.formatters = [wrapped_formatter] * num_agents
        elif isinstance(formatters, list):
            # We have a list of formatters and we should apply them to all agents
            if len(formatters) != num_agents:
                raise ValueError(
                    f"Number of formatters ({len(formatters)}) must match "
                    f"number of agents ({num_agents})"
                )
            # Ensure all formatters can accept external_prompts
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
                    # Wrap to accept but ignore parameter
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

    def set_format_reward_func(self, func: Callable):
        """Set the format reward function for agent chaining.
        
        This function evaluates how well Agent 2's reasoning matches Agent 1's actual output.
        Signature: func(reasoning: str, actual_aux: str, batch_item: Dict) -> float
        """
        self.format_reward_func = func

    def _get_max_tokens_for_agent(self, agent_idx: int) -> int:
        """Get the max_new_tokens for a specific agent."""
        if hasattr(self, "max_new_tokens_per_agent") and self.max_new_tokens_per_agent:
            return self.max_new_tokens_per_agent[agent_idx]
        return self.args.max_new_tokens

    def _apply_chat_template(self, prompts: List[str]) -> List[str]:
        """Apply chat template to prompts for Instruct models.
        
        Converts raw prompts into chat format expected by Instruct models.
        Uses tokenizer.apply_chat_template() if available.
        
        Args:
            prompts: List of raw prompt strings
            
        Returns:
            List of prompts with chat template applied
        """
        if not self.use_chat_template:
            return prompts
        
        templated_prompts = []
        for prompt in prompts:
            # Build messages list
            messages = []
            
            # Add system prompt if configured
            if self.chat_template_system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.chat_template_system_prompt
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Apply chat template using tokenizer
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    templated = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True  # Add assistant prompt start
                    )
                    templated_prompts.append(templated)
                except Exception as e:
                    # Fallback: return original prompt if template fails
                    if self.verbose:
                        print(f"Warning: Chat template application failed: {e}")
                    templated_prompts.append(prompt)
            else:
                # Tokenizer doesn't support chat template
                if self.verbose:
                    print("Warning: Tokenizer does not support apply_chat_template")
                templated_prompts.append(prompt)
        
        return templated_prompts

    def _extract_assistant_response(self, completion_text: str) -> str:
        """Extract the assistant's response from a completion that may contain chat template tokens.
        
        For Instruct models, the completion might include special tokens or formatting.
        This method cleans up the response to extract just the content.
        
        Args:
            completion_text: Raw completion text from the model
            
        Returns:
            Cleaned assistant response
        """
        if not self.use_chat_template:
            return completion_text
        
        # Common patterns to remove from assistant responses
        # These vary by model family, so we handle common cases
        response = completion_text
        
        # Remove common end-of-turn tokens
        end_tokens = [
            "<|im_end|>",  # Qwen, ChatML
            "<|eot_id|>",  # Llama 3
            "</s>",        # Various models
            "<|end|>",     # Some models
            "[/INST]",     # Llama 2
            "### Response:", # Alpaca-style
        ]
        
        for token in end_tokens:
            if token in response:
                response = response.split(token)[0]
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        return response

    def _extract_main_from_chaining_output(self, completion: str) -> str:
        """Extract only the main function from Agent 2's chaining output.
        
        In agent chaining mode, Agent 2's output contains:
        1. <predicted_aux>...</predicted_aux> block with predicted aux function
        2. Main function implementation
        
        This extracts only the main function part for reward calculation.
        
        Args:
            completion: Agent 2's full completion including predicted_aux block
            
        Returns:
            Only the main function code (predicted_aux block removed)
        """
        import re
        
        if not completion:
            return ""
        
        # Remove <predicted_aux>...</predicted_aux> block
        cleaned = re.sub(
            r'<predicted_aux>.*?</predicted_aux>',
            '',
            completion,
            flags=re.DOTALL
        )
        
        # Clean up extra whitespace
        cleaned = cleaned.strip()
        
        return cleaned

    def _compute_format_rewards(
        self,
        agent2_completions: List[str],
        actual_aux_completions: List[str],
        batch_items: List[Dict],
    ) -> List[float]:
        """Compute format rewards for Agent 2's reasoning accuracy.
        
        In agent chaining mode, Agent 2's completion includes both:
        1. Reasoning about Agent 1's expected aux function
        2. The main function implementation
        
        This evaluates how well the reasoning part predicted Agent 1's actual aux output.
        The format_reward_func should parse the reasoning from agent2's completion.
        """
        if self.format_reward_func is None:
            # Default: return 0 if no format reward function is set
            return [0.0] * len(agent2_completions)

        format_rewards = []
        for agent2_comp, actual_aux, batch_item in zip(
            agent2_completions, actual_aux_completions, batch_items
        ):
            try:
                reward = self.format_reward_func(agent2_comp, actual_aux, batch_item)
                format_rewards.append(float(reward))
            except Exception:
                format_rewards.append(0.0)
        
        return format_rewards

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking with multi-turn config."""
        if not self.wandb_initialized:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "mlrl")
            wandb_entity = self.wandb_config.get("entity", "OpenMLRL")

            # Use different default names based on num_turns
            if self.args.num_turns == 1:
                wandb_name = self.wandb_config.get("name", "test-magrpo")
            else:
                wandb_name = self.wandb_config.get("name", "test-mt-magrpo")

            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "num_turns": self.args.num_turns,
                # single reward function; keep legacy fields out
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
                "use_lora": self.use_lora,
                # Agent Chaining settings
                "agent_chaining": getattr(self.args, "agent_chaining", False),
                "max_new_tokens_per_agent": getattr(self, "max_new_tokens_per_agent", None),
                "format_reward_weight": getattr(self.args, "format_reward_weight", 0.1),
                # Chat Template settings
                "use_chat_template": getattr(self, "use_chat_template", False),
                "chat_template_system_prompt": getattr(self, "chat_template_system_prompt", None),
            }

            # No per-turn weighting or early termination config

            # Incorporate full config sections and derived fields for searchability
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

                # Attach full sections
                config_dict.update(
                    {
                        "dataset": dataset_section,
                        "model": model_section,
                        "output": output_section,
                        "external": external_section,
                        "trainer": trainer_section,
                    }
                )

                # Derived convenience keys
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

                # External mode-specific fields
                ext_mode = (
                    external_section.get("mode")
                    if isinstance(external_section, dict)
                    else None
                )
                if ext_mode:
                    config_dict["external_mode"] = ext_mode
                    if "original_prompt" in external_section:
                        config_dict["original_prompt"] = external_section.get(
                            "original_prompt"
                        )
                    if "previous_response" in external_section:
                        config_dict["previous_response"] = external_section.get(
                            "previous_response"
                        )

            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }

            if wandb_dir is not None:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir

            # Optionally support tags if provided by caller
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

    def evaluate(self, num_eval_samples: int = 4) -> Dict[str, float]:
        """
        Unified evaluation that supports both single-turn and multi-turn.

        Args:
            num_eval_samples: Number of samples to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        # Storage for completions across turns for all agents
        all_agent_completions_turns = [[] for _ in range(self.num_agents)]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []
        # Collect per-turn immediate rewards across evaluated samples
        eval_turn_rewards: List[List[float]] = [[] for _ in range(self.args.num_turns)]
        # No per-function tracking; single reward function handles composition

        # Get evaluation dataloader
        eval_dataloader = self.get_eval_dataloader()

        # Evaluate on specified number of samples
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break

                # Process each batch item
                for batch_item in batch:
                    self._evaluate_sample(
                        batch_item,
                        all_agent_completions_turns,
                        all_test_cases,
                        all_entry_points,
                        all_prompts,
                        eval_turn_rewards,
                    )

        # Prepare extra metrics to pass into logging after computing returns/components
        extra_eval_metrics: Dict[str, Any] = {}

        # Compute eval returns per turn and add to extra metrics
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

        # No per-reward-function logging when using a single reward function

        # Calculate and log metrics (including extra_eval_metrics)
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
        # no per-function component tracking
    ):
        """Evaluate a single sample for any number of turns."""
        # Storage for each agent's completions across turns
        agent_sample_completions = [[] for _ in range(self.num_agents)]

        # Store sample information
        all_test_cases.append(batch_item.get("test", ""))
        all_entry_points.append(batch_item.get("entry_point", ""))
        all_prompts.append(batch_item.get("prompt", ""))

        # Track the selected completions from the previous turn (evaluation traces a single path)
        previous_turn_completions = [None] * self.num_agents
        # Track full history per agent for evaluation path
        eval_prompt_history = [[] for _ in range(self.num_agents)]
        eval_response_history = [[] for _ in range(self.num_agents)]

        # Run episode with configured number of turns
        for turn_idx in range(self.args.num_turns):
            # Prepare external prompts for turns after the first
            agent_external_prompts = [None] * self.num_agents

            if turn_idx > 0 and all(c is not None for c in previous_turn_completions):
                # Use previously selected completions to form next-turn prompts (single eval path)
                selected_prev = list(previous_turn_completions)
                # Get external transitions based on selected prior completions
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

                    # External transition should return prompts for each agent
                    if isinstance(transition_result, (list, tuple)):
                        if len(transition_result) != self.num_agents:
                            raise ValueError(
                                f"External transition returned {len(transition_result)} values but expected {self.num_agents}"
                            )
                        agent_external_prompts = list(transition_result)
                    else:
                        raise ValueError(
                            "External transition must return a list or tuple of external prompts for each agent"
                        )

            # Generate and extract one completion from each agent for evaluation
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
                # Extract the completion directly
                completion = agent_completions["completions"][0][0]
                # Record prompt used this turn
                used_prompt = agent_completions["prompts"][0]
                eval_prompt_history[agent_idx].append(used_prompt)
                agent_sample_completions[agent_idx].append(completion)

            # Compute immediate reward at this turn (single joint sample)
            agent_completions_for_reward = [
                [agent_sample_completions[i][-1]] for i in range(self.num_agents)
            ]
            prompt = self.formatters[0](batch_item)
            rewards = self._compute_rewards(
                [prompt], agent_completions_for_reward, batch_items=[batch_item]
            )
            if rewards:
                # Track per-turn reward across samples
                eval_turn_rewards[turn_idx].append(float(rewards[0]))
                # Update selected previous-turn completions for next-turn prompts
                for agent_idx in range(self.num_agents):
                    chosen = agent_sample_completions[agent_idx][-1]
                    previous_turn_completions[agent_idx] = chosen
                    eval_response_history[agent_idx].append(chosen)

        # Store completions for all agents
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
        """Log evaluation metrics for any number of turns."""
        eval_metrics = {}

        # Detailed logging (if logger is provided), standardized to modern interface
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

            # Aggregate metrics for logging
            # Aggregate strictly per-turn; aggregator already returns turn_k/* keys only
            aggregated_detailed_metrics = self.eval_aggregator(
                detailed_metrics, num_turns=self.args.num_turns
            )
            for key, value in aggregated_detailed_metrics.items():
                eval_metrics[f"eval/{key}"] = value

        # Merge any extra metrics (already with full key prefixes like 'eval/...')
        if isinstance(extra_metrics, dict) and extra_metrics:
            eval_metrics.update(extra_metrics)

        # Log evaluation metrics
        if self.wandb_initialized and wandb.run is not None:
            wandb.log(eval_metrics, step=self.env_step)

        return eval_metrics

    def train(self, **kwargs):
        """
        Unified train method that supports both single-turn and multi-turn training.
        """
        # Initialize wandb if not already done
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        # Setup devices for training (GPU is required)
        device = torch.device("cuda")
        for agent in self.agents:
            agent.to(device)
            agent.train()

        # Create the data pipeline for generating examples
        for epoch in range(0, int(self.args.num_train_epochs)):
            # No per-agent reward tracking in single reward mode

            # Turn tracking for all cases (including single-turn)
            epoch_turn_rewards = [
                [] for _ in range(self.args.num_turns)
            ]  # immediate rewards
            epoch_turn_returns = [[] for _ in range(self.args.num_turns)]  # returns
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
                # Periodic evaluation based on configuration
                if int(self.args.eval_interval) > 0 and (
                    batch_idx % int(self.args.eval_interval) == 0
                ):
                    # evaluate() already logs its metrics; avoid duplicate logging here
                    _ = self.evaluate(num_eval_samples=int(self.args.eval_num_samples))

                # Process single batch item (batch_size=1 enforced)
                batch_item = batch[0]
                # Unified training step (returns-based, backward updates)
                batch_loss, _batch_stats = self._train_step_returns(
                    batch_item,
                    epoch_turn_rewards,
                    epoch_turn_returns,
                    **kwargs,
                )

            for agent_idx, buffer in enumerate(self.rollout_buffers):
                if buffer:
                    self._process_buffer(agent_idx, buffer)

            # Log per-turn epoch averages inline (avoid custom system/* metrics)
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

    def _train_step_returns(
        self,
        batch_item,
        epoch_turn_rewards,
        epoch_turn_returns,
        **kwargs,
    ):
        """Branching rollout with returns; updates backward from last turn to first.

        Returns an additional per-turn batch summary for logging:
        - batch_mean_reward (immediate reward mean averaged across nodes at the turn)
        - batch_expected_return (expected return averaged across nodes at the turn)
        - no per-function breakdown (single reward function)
        - levels (code-only: mean of level_1/2/3 and bonus across nodes)
        """
        num_turns = int(self.args.num_turns)
        num_gens = int(self.args.num_generations)
        gamma = float(getattr(self.args, "discount", 0.9))

        # Per-turn accumulators for batch-level summaries
        turn_reward_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        turn_return_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        # No per-function accumulation in single reward mode
        turn_node_counts: List[int] = [0 for _ in range(num_turns)]

        def build_node(
            turn_idx: int,
            prompts_per_agent=None,
            prompt_history_per_agent: Optional[List[List[str]]] = None,
            response_history_per_agent: Optional[List[List[str]]] = None,
        ):
            comps_per_agent = []
            chaining_reasoning_completions = None  # Store Agent 2's reasoning
            
            # Generate completions for all agents
            # In Agent Chaining mode, Agent 2's formatter already includes instruction
            # to reason about Agent 1's aux first, then generate main function
            for agent_idx in range(self.num_agents):
                comps = self._generate_completions_with_external_prompts(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=num_gens,
                    max_new_tokens=None,  # Will use agent-specific max_new_tokens
                    external_prompts=(
                        prompts_per_agent[agent_idx] if prompts_per_agent else None
                    ),
                    **kwargs,
                )
                comps_per_agent.append(comps)
            
            # In Agent Chaining mode, extract reasoning from Agent 2's completions
            # Agent 2's output contains both reasoning about aux and main function
            agent2_raw_completions = None  # Store Agent 2's raw completions (with predicted_aux)
            if self.agent_chaining and self.num_agents >= 2:
                agent2_raw_completions = comps_per_agent[self.num_agents - 1]["completions"][0]
                chaining_reasoning_completions = agent2_raw_completions  # For format reward calculation

            # Build agent_completions_list
            # In Agent Chaining mode, extract only main function from Agent 2's output
            agent_completions_list = []
            for i in range(self.num_agents):
                completions = comps_per_agent[i]["completions"][0]
                if self.agent_chaining and i == self.num_agents - 1:
                    # Agent 2: extract main function only (remove <predicted_aux> block)
                    completions = [
                        self._extract_main_from_chaining_output(comp)
                        for comp in completions
                    ]
                agent_completions_list.append(completions)
            # Prompts actually used this turn, per agent (may differ across agents)
            prompts_used_this_turn = [
                comps_per_agent[i]["prompts"][0] for i in range(self.num_agents)
            ]
            formatted_prompt = comps_per_agent[0]["prompts"][0]

            # Initialize history containers if not provided
            if prompt_history_per_agent is None:
                prompt_history_per_agent = [[] for _ in range(self.num_agents)]
            if response_history_per_agent is None:
                response_history_per_agent = [[] for _ in range(self.num_agents)]

            # Extend prompt history with this turn's prompts
            next_prompt_history = [
                list(prompt_history_per_agent[i]) + [prompts_used_this_turn[i]]
                for i in range(self.num_agents)
            ]
            # Compute rewards per joint action depending on joint_mode
            joint_mode = str(getattr(self.args, "joint_mode", "aligned")).lower()
            rewards_vec: List[float] = []
            combo_indices: List[Tuple[int, ...]] = []
            if joint_mode in ["cross", "crossed"] and self.num_agents > 1:
                # Cartesian product of per-agent completion indices
                per_agent_ranges = [
                    range(len(agent_completions_list[i]))
                    for i in range(self.num_agents)
                ]
                for idx_tuple in itertools.product(*per_agent_ranges):
                    # Build per-agent single-element lists
                    completion_args = [
                        [agent_completions_list[a][idx_tuple[a]]]
                        for a in range(self.num_agents)
                    ]
                    # Call reward function for this joint action
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
                    # Apply processor
                    processed = [self.reward_processor(r) for r in rlist]
                    rewards_vec.append(float(processed[0] if processed else 0.0))
                    combo_indices.append(tuple(idx_tuple))
            elif joint_mode in ["align", "aligned"] and self.num_agents > 1:
                # Aligned by index
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                # combo indices: align j with (j,j,...)
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j] * self.num_agents) for j in range(k)]
            elif self.num_agents == 1:
                # Single-agent mode (GRPO)
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                # combo indices: single agent, each completion gets its own index
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j]) for j in range(k)]
            else:
                raise ValueError(f"Unsupported joint_mode: {joint_mode}")

            # Agent Chaining: Compute format rewards for Agent 2 (applied only to Agent 2's returns later)
            format_rewards_vec = None
            if (
                self.agent_chaining
                and chaining_reasoning_completions is not None
                and self.format_reward_func is not None
                and self.num_agents >= 2
            ):
                # Compute format rewards for each completion pair
                format_rewards_vec = []
                
                # For aligned mode: compute format reward for each aligned pair
                agent1_completions = agent_completions_list[0]  # Agent 1's aux completions
                
                for j in range(len(rewards_vec)):
                    if joint_mode in ["align", "aligned"]:
                        # Aligned: reasoning[j] should predict aux[j]
                        reasoning_idx = j if j < len(chaining_reasoning_completions) else 0
                        aux_idx = j if j < len(agent1_completions) else 0
                    else:
                        # Cross mode: use combo_indices to get the right aux completion
                        idx_tuple = combo_indices[j]
                        reasoning_idx = idx_tuple[self.num_agents - 1] if len(idx_tuple) > 1 else 0
                        aux_idx = idx_tuple[0]
                    
                    reasoning = chaining_reasoning_completions[reasoning_idx] if reasoning_idx < len(chaining_reasoning_completions) else ""
                    actual_aux = agent1_completions[aux_idx] if aux_idx < len(agent1_completions) else ""
                    
                    try:
                        format_reward = self.format_reward_func(
                            reasoning, actual_aux, batch_item
                        )
                        format_rewards_vec.append(float(format_reward))
                    except Exception:
                        format_rewards_vec.append(0.0)
                
                # NOTE: format_rewards_vec is NOT added to rewards_vec here
                # It will be applied only to Agent 2's returns in post_order_update

            if 0 <= turn_idx < len(epoch_turn_rewards):
                epoch_turn_rewards[turn_idx].append(
                    np.mean(rewards_vec) if rewards_vec else 0.0
                )

            # Per-node means for batch-level summaries
            self.env_step += len(rewards_vec)
            node_env_step = int(self.env_step)
            node_mean_reward = float(np.mean(rewards_vec)) if rewards_vec else 0.0
            turn_reward_node_means[turn_idx].append(node_mean_reward)

            turn_node_counts[turn_idx] += 1

            # Early termination: stop expanding this branch if mean reward exceeds threshold
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
                "rewards": rewards_vec,
                "children": [],
                "returns": None,
                "combo_indices": combo_indices,
                "env_step": node_env_step,
                "format_rewards": format_rewards_vec,  # Agent Chaining format rewards
                "chaining_reasoning": chaining_reasoning_completions,  # Agent 2's reasoning
            }

            if turn_idx < num_turns - 1 and not terminate_here:
                for j in range(len(rewards_vec)):
                    # Map j to per-agent indices
                    idx_tuple = combo_indices[j]
                    parent_joint = [
                        agent_completions_list[i][idx_tuple[i]]
                        for i in range(self.num_agents)
                    ]
                    # Extend response history with selected completions on this branch
                    next_response_history = [
                        list(response_history_per_agent[i])
                        + [agent_completions_list[i][idx_tuple[i]]]
                        for i in range(self.num_agents)
                    ]

                    child_prompts = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=parent_joint,
                        num_agents=self.num_agents,
                        # Full history along this branch up to (and including) this turn
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

        # After returns computed, record per-turn mean returns
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

        pending_samples: List[List[NodeSample]] = [[] for _ in range(self.num_agents)]

        def post_order_update(node):
            for child in node["children"]:
                post_order_update(child)
            returns_vec = node.get("returns") or []
            comps_per_agent = node["completions"]
            if not returns_vec:
                return
            # If cross mode, build per-agent joint reward sums (accumulate joint returns
            # for each completion across all joint actions it participates in)
            joint_mode_local = str(getattr(self.args, "joint_mode", "aligned")).lower()
            combo_idx_list = node.get("combo_indices") or []
            per_agent_joint_sums: List[List[float]] = []
            if joint_mode_local == "cross" and combo_idx_list:
                # Determine K per agent
                k = len(comps_per_agent[0]["completions"][0]) if comps_per_agent else 0
                for a in range(self.num_agents):
                    sums = [0.0] * k
                    counts = [0] * k
                    for j, ret in enumerate(returns_vec):
                        idx_a = combo_idx_list[j][a]
                        sums[idx_a] += float(ret)
                        counts[idx_a] += 1
                    # Use joint reward sum per completion (no averaging)
                    per_agent_joint_sums.append(sums)
            else:
                # Aligned: returns already length K
                k = len(returns_vec)
                per_agent_joint_sums = [
                    list(map(float, returns_vec)) for _ in range(self.num_agents)
                ]
            node_env_step = int(node.get("env_step", self.env_step))
            # Get format rewards if available (Agent Chaining mode)
            format_rewards = node.get("format_rewards") or []
            node_mean_format_reward = (
                float(np.mean(format_rewards)) if format_rewards else None
            )
            format_reward_weight = getattr(self.args, "format_reward_weight", 0.1)
            
            for agent_idx in range(self.num_agents):
                node_rewards = node.get("rewards") or []
                node_mean_reward = float(np.mean(node_rewards)) if node_rewards else 0.0
                node_mean_return = float(np.mean(returns_vec)) if returns_vec else 0.0
                
                # Compute returns for this agent
                agent_returns = [float(r) for r in per_agent_joint_sums[agent_idx]]
                
                # Agent Chaining: Add format reward ONLY to Agent 2's returns
                if (
                    self.agent_chaining
                    and agent_idx == self.num_agents - 1  # Agent 2 (last agent)
                    and format_rewards
                ):
                    # Add weighted format reward to Agent 2's returns
                    for j in range(len(agent_returns)):
                        if j < len(format_rewards):
                            agent_returns[j] += format_reward_weight * format_rewards[j]
                
                sample = NodeSample(
                    agent_idx=agent_idx,
                    turn_idx=int(node.get("turn", 0)),
                    completions_data=self._pack_completions_for_buffer(
                        comps_per_agent[agent_idx]
                    ),
                    returns=agent_returns,
                    node_mean_reward=node_mean_reward,
                    node_mean_return=node_mean_return,
                    node_env_step=node_env_step,
                    node_mean_format_reward=node_mean_format_reward if agent_idx == self.num_agents - 1 else None,
                )
                pending_samples[agent_idx].append(sample)

        post_order_update(root)

        for agent_idx, samples in enumerate(pending_samples):
            samples.sort(key=lambda s: s.node_env_step)
            for sample in samples:
                self._append_to_buffer(agent_idx, sample)

        # Build per-turn batch summary
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
            # No per-reward-function means; use a single reward function
            batch_stats[t] = stats

        return batch_loss, batch_stats

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
        """
        Generate completions from an agent given prompts, preserving model state.

        Args:
            agent: The agent model to generate completions
            batch_items: List of data items (dictionaries from dataset)
            agent_idx: Index of the agent (used to select the appropriate formatter)
            num_return_sequences: Number of completions to generate per prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional arguments to pass to the model during generation

        Returns:
            Dict: A dictionary containing generated completions and associated data
        """
        device = agent.device

        # Apply the appropriate formatter to create prompts from batch items
        if prompts_override is not None:
            if len(prompts_override) != len(batch_items):
                raise ValueError(
                    "prompts_override must have the same length as batch_items"
                )
            prompts = prompts_override
        else:
            format_func = self.formatters[agent_idx]
            prompts = [format_func(item) for item in batch_items]
        # batch_size is always 1 due to enforced constraint
        
        # Store original prompts for reference (before chat template)
        original_prompts = list(prompts)

        # Ensure tokenizer exists
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply chat template if enabled (for Instruct models)
        if self.use_chat_template:
            prompts = self._apply_chat_template(prompts)

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original model state and gradient settings
        training_mode = agent.training
        original_requires_grad = {}

        # Save original requires_grad states
        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False  # Temporarily disable gradients for generation

        agent.eval()  # Set to eval mode for generation

        # Generate completions without gradients
        generation_output = None
        try:
            # Use max_new_tokens instead of max_length
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,  # Changed from max_length
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            # If requesting multiple sequences, use sampling for diversity
            top_k = getattr(self.args, "top_k", None)
            if do_sample is None and num_return_sequences > 1:
                # Use generation parameters from config
                generation_update = {
                    "do_sample": True,  # Enable sampling for randomness
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,  # Disable beam search when sampling
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

            # Set pad_token_id from tokenizer if not set
            if (
                "pad_token_id" not in generation_kwargs
                or generation_kwargs["pad_token_id"] is None
            ):
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            # Add any additional user-provided kwargs (these override model defaults)
            generation_kwargs.update(kwargs)
            generation_output = agent.generate(**generation_kwargs)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore original model state and gradients
        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Extract completion tokens (excluding prompt tokens)
        completion_input_ids = generation_output.sequences

        # For single prompt, find its actual length in tokens
        # to properly extract just the completion part
        prompt_len = prompt_input_ids[0].shape[0]
        # Find where padding token starts if any
        pad_positions = (prompt_input_ids[0] == self.tokenizer.pad_token_id).nonzero()
        if pad_positions.shape[0] > 0:
            prompt_len = pad_positions[
                0
            ].item()  # prompt ends at index prompt_len, this is the index of the first pad token

        # Extract completion text for single prompt
        completions = []
        completion_tokens_list = []

        # Calculate total sequence count
        total_sequences = completion_input_ids.shape[0]

        # Process single prompt and its multiple completions
        batch_completions = []
        batch_completion_tokens = []

        # Get all sequences for this prompt (start_idx=0, end_idx=num_return_sequences)
        end_idx = min(num_return_sequences, total_sequences)

        for s in range(end_idx):
            # Get only the completion part (exclude the prompt tokens)
            completion_tokens = completion_input_ids[s, prompt_len:]
            batch_completion_tokens.append(completion_tokens)

            # Decode to text
            completion_text = self.tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            
            # Extract assistant response if using chat template
            if self.use_chat_template:
                completion_text = self._extract_assistant_response(completion_text)
            
            batch_completions.append(completion_text)

        completions.append(batch_completions)
        completion_tokens_list.append(batch_completion_tokens)

        # Create attention masks for completions (single batch)
        completion_attention_masks = []
        batch_masks = []
        for tokens in completion_tokens_list[0]:  # Only one batch
            mask = torch.ones(len(tokens), device=device)
            batch_masks.append(mask)
        completion_attention_masks.append(batch_masks)

        # Extract logit for computing loss
        logits = (
            generation_output.scores if hasattr(generation_output, "scores") else []
        )

        return {
            "prompts": prompts,
            "batch_items": batch_items,  # Store original batch items for reference
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
        max_new_tokens=None,
        external_prompts=None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ):
        """
        Generate completions with optional external prompts.
        This wraps the _generate_completions method to handle external transitions.

        When num_turns=1 or external_prompts is None, behaves like _generate_completions.
        
        If max_new_tokens is None, uses the agent-specific max_new_tokens from config.
        """
        # Use agent-specific max_new_tokens if not explicitly provided
        if max_new_tokens is None:
            max_new_tokens = self._get_max_tokens_for_agent(agent_idx)
        
        # If single-turn or no external prompts, use standard method directly
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

        # Multi-turn with external prompts: external modes return next-turn prompts.

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

        # Temporarily replace prompts in batch_items
        modified_items = []
        for item, prompt in zip(batch_items, prompts):
            modified_item = item.copy() if hasattr(item, "copy") else dict(item)
            modified_item["_original_prompt"] = modified_item.get("prompt", "")
            modified_item["prompt"] = prompt
            modified_items.append(modified_item)

        # Use _generate_completions with modified items
        completions_data = self._generate_completions(
            agent,
            modified_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        # Restore original prompts in batch_items
        for i, item in enumerate(completions_data["batch_items"]):
            if "_original_prompt" in item:
                item["prompt"] = item["_original_prompt"]
                del item["_original_prompt"]

        # Update prompts in completions_data to reflect the formatted prompts
        completions_data["prompts"] = prompts

        return completions_data

    def _compute_rewards(
        self, prompts, completions_list, batch_items=None
    ) -> List[float]:
        """
        Compute rewards using a single reward function and optional processor.

        Args:
            prompts: List of prompts (unused by default, passed via batch_items to reward_fn)
            completions_list: List of completions from each agent

        Returns:
            List of final processed rewards
        """
        # Initialize list to store rewards
        all_rewards = []

        # Single prompt case (batch_size=1 enforced)
        # Ensure correct structure for all agents
        for i in range(self.num_agents):
            if not isinstance(completions_list[i], list):
                completions_list[i] = (
                    [completions_list[i]]
                    if not isinstance(completions_list[i], list)
                    else completions_list[i]
                )

        # Find minimum number of completions across all agents
        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))

        for completion_idx in range(min_completions):
            # Extract one completion from each agent
            agent_completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]

            # Call the single reward function
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

            # Apply processor to rewards (single processor)
            processed_rewards = [self.reward_processor(r) for r in func_rewards]

            # Take the processed reward for the chosen completion
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
        if not buffer:
            return
        turn_groups: Dict[int, List[NodeSample]] = {}
        for sample in buffer:
            t_idx = int(sample.turn_idx)
            turn_groups.setdefault(t_idx, []).append(sample)
        buffer.clear()
        for t_idx in sorted(turn_groups.keys()):
            samples = turn_groups[t_idx]
            self._update_from_samples(agent_idx, samples)
            if self.wandb_initialized and wandb.run is not None and samples:
                batch_log: Dict[str, Any] = {}
                prefix = f"turn_{t_idx + 1}/"
                batch_log[prefix + "reward_mean"] = float(
                    np.mean([s.node_mean_reward for s in samples])
                )
                batch_log[prefix + "expected_return"] = float(
                    np.mean([s.node_mean_return for s in samples])
                )
                
                # Log format reward if available (Agent Chaining mode)
                format_rewards = [
                    s.node_mean_format_reward for s in samples
                    if s.node_mean_format_reward is not None
                ]
                if format_rewards:
                    batch_log[prefix + "format_reward_mean"] = float(np.mean(format_rewards))
                
                step = max(s.node_env_step for s in samples)
                if self._should_log_train(step):
                    wandb.log(batch_log, step=step)

    def _update_from_samples(self, agent_idx: int, samples: List[NodeSample]) -> None:
        if not samples:
            return
        random.shuffle(samples)
        self.optimizers[agent_idx].zero_grad()
        scale = 1.0 / len(samples)
        for sample in samples:
            loss = self._compute_loss_with_gradients(
                self.agents[agent_idx],
                sample.completions_data,
                sample.returns,
            )
            (loss * scale).backward()
        self.optimizers[agent_idx].step()

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        """
        Compute loss with proper gradient tracking by performing a new forward pass.

        Args:
            agent: The agent model
            completions_data: The completions data from _generate_completions
            returns: The returns for each completion (not immediate rewards)

        Returns:
            torch.Tensor: The computed loss with gradients attached
        """
        device = agent.device

        # Make sure we have the correct number of rewards
        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert returns to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)

        # Group-relative advantage based on returns (mean baseline, no z-score normalization)
        mean_ret = returns_tensor.mean()
        advantages = returns_tensor - mean_ret

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"].to(device)
        completion_input_ids = completions_data["completion_input_ids"]
        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids = [[t.to(device) for t in completion_input_ids]]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Process single prompt (batch_size=1)
        prompt_ids = prompt_input_ids[0]

        # Token-based loss: concatenate prompt + completion
        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            # Break if we've processed enough completions for the available rewards
            if seq_idx >= len(advantages):
                break

            advantage = advantages[seq_idx]

            # Create input sequence by concatenating prompt with all but last token of completion
            # (we'll predict the next token at each step)
            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])

                # Target is the completion tokens
                target_ids = completion_tokens

                # Create attention mask for the full sequence
                attention_mask = torch.ones(len(input_ids), device=device)

                # Forward pass with gradients enabled
                outputs = agent(
                    input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                    attention_mask=attention_mask.unsqueeze(0),  # Add batch dimension
                )

                # Get logits for the completion part (excluding prompt)
                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                # Calculate log probabilities
                log_probs = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(
                        0
                    ):  # Check if we have logits for this position
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[
                            token_id
                        ]
                        log_probs.append(token_log_prob)

                if log_probs:
                    sequence_log_prob = torch.stack(log_probs).sum()
                    # Policy gradient loss: -log_prob * advantage
                    loss = -sequence_log_prob * advantage
                    total_loss = total_loss + loss
                    num_samples += 1

        # Average the loss over all processed samples
        if num_samples > 0:
            total_loss = total_loss / num_samples

        # Safety check for invalid loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def save_model(self, output_dir):
        """
        Save the final trained models or LoRA adapters.

        Args:
            output_dir: Directory to save the models to
        """
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

        # Log final model saving to wandb
        if self.wandb_initialized and wandb.run is not None:
            save_type = "lora_adapters" if self.use_lora else "full_models"
            wandb.log({"final_model_saved": output_dir, "save_type": save_type})
            wandb.finish()
