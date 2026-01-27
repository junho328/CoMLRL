import inspect
import itertools
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Optional PEFT import for LoRA support
try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# Module-level function for ProcessPoolExecutor (instance methods can't be pickled)
def _compute_reward_worker(
    reward_func: Callable,
    reward_processor: Callable,
    agent_completions: List[str],
    batch_item: Dict,
) -> float:
    """Worker function for parallel reward computation."""
    try:
        completion_args = [[comp] for comp in agent_completions]
        sig = inspect.signature(reward_func)
        if "batch_items" in sig.parameters:
            func_rewards = reward_func(*completion_args, batch_items=[batch_item])
        else:
            func_rewards = reward_func(*completion_args)
    except TypeError:
        func_rewards = reward_func(agent_completions)

    processed_rewards = [reward_processor(r) for r in func_rewards]
    return float(processed_rewards[0] if processed_rewards else 0.0)


@dataclass
class MAGRPOConfig(TrainingArguments):
    """
    Configuration for Multi-Agent GRPO training (single-turn).
    Supports multiple agents with batch generation, LoRA, and multi-GPU.
    """

    # Core setup
    num_train_epochs: float = field(
        default=3,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Per-device batch size"},
    )
    learning_rate: float = field(
        default=5.0e-6,
        metadata={"help": "Learning rate for optimizer."},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every N steps."},
    )
    save_steps: int = field(
        default=100,
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
        default=512,
        metadata={"help": "Maximum number of new tokens to generate after the prompt."},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for sampling."},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k for sampling (set to None to disable)."},
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Repetition penalty for generation (set to None to disable, >1.0 to penalize)."},
    )

    # Joint mode for multi-agent reward computation
    joint_mode: str = field(
        default="aligned",
        metadata={
            "help": "Joint action mode: 'aligned' (same idx) or 'cross' (all combinations)."
        },
    )

    # Evaluation
    eval_interval: int = field(
        default=20,
        metadata={"help": "Run evaluation every N training batches."},
    )
    eval_num_samples: int = field(
        default=8,
        metadata={"help": "Number of samples to evaluate per evaluation run."},
    )
    rollout_buffer_size: int = field(
        default=32,
        metadata={"help": "Number of samples to buffer before an update."},
    )

    # Parallelism for reward computation
    parallel_reward: bool = field(
        default=True,
        metadata={"help": "Enable parallel reward computation."},
    )
    max_reward_workers: int = field(
        default=8,
        metadata={"help": "Maximum number of workers for parallel reward computation."},
    )
    reward_parallel_backend: str = field(
        default="process",
        metadata={"help": "Backend for parallel reward: 'process' (true parallelism) or 'thread' (GIL-limited)."},
    )

    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA adapters."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability."},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. None for auto-detect."},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained LoRA adapter to load."},
    )

    # Multi-GPU configuration
    use_distributed: bool = field(
        default=False,
        metadata={"help": "Enable distributed training across multiple GPUs."},
    )

    # Separate adapters for multi-agent (mental simulation mode)
    separate_agent_adapters: bool = field(
        default=False,
        metadata={"help": "Use separate LoRA adapters for each agent."},
    )

    # Mental simulation configuration for Agent 2
    enable_mental_simulation: bool = field(
        default=False,
        metadata={"help": "Enable two-stage generation for Agent 2 with mental simulation."},
    )
    inference_max_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens for Agent 2's inference (mental simulation)."},
    )

    # Similarity reward configuration
    similarity_weight_schedule: str = field(
        default="linear",
        metadata={"help": "Schedule for similarity weight: 'linear', 'cosine', or 'step'."},
    )
    similarity_weight_start: float = field(
        default=0.0,
        metadata={"help": "Starting value for similarity weight lambda(t)."},
    )
    similarity_weight_end: float = field(
        default=0.5,
        metadata={"help": "Ending value for similarity weight lambda(t)."},
    )
    similarity_weight_warmup_steps: int = field(
        default=20,
        metadata={"help": "Number of warmup steps for similarity weight schedule."},
    )
    code_reward_threshold: float = field(
        default=1.5,
        metadata={"help": "Minimum code reward threshold to enable similarity reward (gating)."},
    )


@dataclass
class RolloutSample:
    """Single rollout sample for buffer-based updates."""
    agent_idx: int
    completions_data: Dict[str, Any]
    returns: List[float]
    combo_indices: List[Tuple[int, ...]]  # For cross mode
    mean_reward: float
    mean_return: float
    env_step: int
    # Differential reward fields for mental simulation mode
    inference_returns: Optional[List[float]] = None  # Returns for inference tokens
    main_returns: Optional[List[float]] = None  # Returns for main function tokens
    inference_token_boundaries: Optional[List[int]] = None  # Token index where inference ends


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization Trainer (Single-Turn).

    Features:
    - Batch generation for efficiency
    - LoRA adapter support via PEFT
    - Multi-GPU training via DDP
    - Joint mode: aligned (same idx) or cross (all combinations)
    """

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[List[PreTrainedModel]] = None,
        num_agents: int = 2,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_config: Optional[Dict[str, Any]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        dataset_type: Optional[str] = None,
        reward_func: Optional[Callable] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Callable, List[Callable]]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
        args: Optional[MAGRPOConfig] = None,
    ):
        self.args = args if args is not None else MAGRPOConfig()

        # Initialize distributed training if enabled
        self._setup_distributed()

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not found. MAGRPOTrainer requires GPU for training.")

        if model is None and agents is None:
            raise ValueError("Either model or agents must be provided")
        if model is not None and agents is not None:
            raise ValueError("Cannot provide both model and agents parameters")

        self.env_step = 0
        self._last_train_log_step = -1
        self.model_config = model_config if model_config else {}

        self._setup_formatters(formatters, num_agents)
        self._setup_reward_function(reward_func, reward_processor)

        # Setup model with optional LoRA
        # For separate_agent_adapters mode: create separate models for each agent
        # Otherwise: use a single shared model
        self.agent_models = None  # Will be set if using separate models
        
        if agents is not None:
            self.num_agents = len(agents)
            self.shared_model = agents[0]
            self.model_name = self._extract_model_name(agents[0])
            
            if self.args.separate_agent_adapters and self.args.use_lora:
                # Create separate models with their own LoRA adapters
                self.agent_models = self._setup_separate_agent_models(agents)
                # shared_model points to agent 0's model for compatibility
                self.shared_model = self.agent_models[0]
            elif self.args.use_lora:
                self.shared_model = self._apply_lora(self.shared_model)
            
            self.agents = [self.shared_model for _ in range(self.num_agents)]
        else:
            self.num_agents = num_agents
            if isinstance(model, str):
                self.model_name = model

                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model, **self.model_config.get("tokenizer_kwargs", {})
                    )
                    special_tokens = self.model_config.get("special_tokens", {})
                    if special_tokens:
                        self.tokenizer.add_special_tokens(special_tokens)

                if self.args.separate_agent_adapters and self.args.use_lora:
                    # Load separate pretrained models for each agent
                    self.agent_models = self._load_separate_agent_models(model)
                    self.shared_model = self.agent_models[0]
                else:
                    self.shared_model = self._load_model(model)

                self.agents = [self.shared_model for _ in range(num_agents)]
            else:
                raise ValueError("Model should be a string to create homogeneous agents")

        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.args.num_generations < 2:
            raise ValueError("num_generations must be >= 2 for GRPO baseline.")
        if self.args.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")

        self.rollout_buffers: List[List[RolloutSample]] = [
            [] for _ in range(self.num_agents)
        ]

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator

        # Setup optimizer(s)
        if self.agent_models is not None:
            # Separate optimizers for each agent's model
            self.agent_optimizers = []
            for i, agent_model in enumerate(self.agent_models):
                optimizer_i = torch.optim.AdamW(
                    [p for p in agent_model.parameters() if p.requires_grad],
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
                self.agent_optimizers.append(optimizer_i)
            # Keep reference to first optimizer for compatibility
            self.optimizer = self.agent_optimizers[0]
        else:
            # Single optimizer for shared model
            self.optimizer = torch.optim.AdamW(
                self.shared_model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            self.agent_optimizers = None

        # Initialize similarity reward function (will be set from train script if mental simulation enabled)
        self.similarity_reward_func = None

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None and self.is_main_process:
            self._init_wandb()

        self.dataset_type = dataset_type
        if self.dataset_type is None:
            self.dataset_type = self._extract_dataset_type()

        self.verbose = self._extract_verbose()

    def _setup_distributed(self):
        """Setup distributed training environment."""
        # Check if already running in distributed mode (e.g., launched via torchrun)
        # even if use_distributed is False in config
        already_distributed = dist.is_initialized() or "LOCAL_RANK" in os.environ
        
        self.distributed = self.args.use_distributed and torch.cuda.device_count() > 1

        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif already_distributed:
            # Script launched with torchrun but use_distributed=False
            # Still need to properly identify main process to avoid duplicate logging
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

        self.is_main_process = self.global_rank == 0

    def _load_model(self, model_path: str) -> PreTrainedModel:
        """Load model with optional LoRA configuration."""
        model_load_kwargs = dict(self.model_config.get("model_kwargs", {}))
        if "attn_implementation" not in model_load_kwargs:
            model_load_kwargs["attn_implementation"] = "flash_attention_2"

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)

        # Apply LoRA if enabled
        if self.args.use_lora:
            if self.args.separate_agent_adapters:
                model = self._setup_separate_adapters(model)
            else:
                model = self._apply_lora(model)

        return model

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA adapter to model."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for LoRA support. Install with: pip install peft"
            )

        # Load existing LoRA adapter if path provided
        if self.args.lora_path is not None:
            if self.is_main_process:
                print(f"Loading LoRA adapter from {self.args.lora_path}")
            model = PeftModel.from_pretrained(model, self.args.lora_path)
            return model

        # Create new LoRA adapter
        target_modules = self.args.lora_target_modules
        if target_modules is None:
            # Auto-detect target modules based on model architecture
            target_modules = self._get_default_lora_targets(model)

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.is_main_process:
            print(f"Applying LoRA with r={self.args.lora_r}, alpha={self.args.lora_alpha}")
            print(f"Target modules: {target_modules}")

        model = get_peft_model(model, lora_config)

        if self.is_main_process:
            model.print_trainable_parameters()

        return model

    def _load_separate_agent_models(self, model_path: str) -> List[PreTrainedModel]:
        """
        Load separate pretrained models for each agent, each with its own LoRA adapter.
        
        This creates completely independent models (not adapter switching) so each agent
        has its own model instance and trainable parameters.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for separate agent models. Install with: pip install peft"
            )

        model_load_kwargs = dict(self.model_config.get("model_kwargs", {}))
        if "attn_implementation" not in model_load_kwargs:
            model_load_kwargs["attn_implementation"] = "flash_attention_2"

        target_modules = self.args.lora_target_modules

        agent_models = []
        
        if self.is_main_process:
            print(f"Loading {self.num_agents} separate models for mental simulation")
            print(f"LoRA config: r={self.args.lora_r}, alpha={self.args.lora_alpha}")

        for i in range(self.num_agents):
            if self.is_main_process:
                print(f"Loading model for Agent {i}...")
            
            # Load a fresh pretrained model for each agent
            base_model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)
            
            # Auto-detect target modules if not specified
            if target_modules is None:
                target_modules = self._get_default_lora_targets(base_model)
            
            # Create LoRA config for this agent
            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA adapter
            model_with_lora = get_peft_model(base_model, lora_config)
            
            if self.is_main_process:
                print(f"Agent {i} model ready")
                model_with_lora.print_trainable_parameters()
            
            agent_models.append(model_with_lora)

        if self.is_main_process:
            print(f"All {self.num_agents} agent models loaded successfully")

        return agent_models

    def _setup_separate_agent_models(self, agents: List[PreTrainedModel]) -> List[PreTrainedModel]:
        """
        Setup separate LoRA adapters on provided agent models.
        
        Each agent gets its own LoRA adapter applied to its model instance.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library is required for separate agent models. Install with: pip install peft"
            )

        target_modules = self.args.lora_target_modules
        
        if self.is_main_process:
            print(f"Setting up {len(agents)} separate agent models with LoRA")
            print(f"LoRA config: r={self.args.lora_r}, alpha={self.args.lora_alpha}")

        agent_models = []
        
        for i, agent_model in enumerate(agents):
            # Auto-detect target modules if not specified
            if target_modules is None:
                target_modules = self._get_default_lora_targets(agent_model)
            
            # Create LoRA config for this agent
            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA adapter
            model_with_lora = get_peft_model(agent_model, lora_config)
            
            if self.is_main_process:
                print(f"Agent {i} model ready")
                model_with_lora.print_trainable_parameters()
            
            agent_models.append(model_with_lora)

        return agent_models

    def _get_agent_model(self, agent_idx: int) -> PreTrainedModel:
        """Get the model for a specific agent."""
        if self.agent_models is not None:
            return self.agent_models[agent_idx]
        return self.shared_model

    def _get_similarity_weight(self, step: int) -> float:
        """Get the current similarity weight lambda(t) based on schedule."""
        schedule = self.args.similarity_weight_schedule
        start = self.args.similarity_weight_start
        end = self.args.similarity_weight_end
        warmup_steps = self.args.similarity_weight_warmup_steps

        if warmup_steps <= 0:
            return end

        progress = min(step / warmup_steps, 1.0)

        if schedule == "linear":
            return start + (end - start) * progress
        elif schedule == "cosine":
            # Cosine annealing from start to end
            return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2
        elif schedule == "step":
            # Step function: jump to end after warmup
            return end if progress >= 1.0 else start
        else:
            # Default to linear
            return start + (end - start) * progress

    def set_similarity_reward_func(self, func: Callable):
        """Set the similarity reward function for mental simulation mode."""
        self.similarity_reward_func = func

    def _get_default_lora_targets(self, model: PreTrainedModel) -> List[str]:
        """Get default LoRA target modules based on model architecture."""
        model_type = getattr(model.config, "model_type", "").lower()

        # Common target modules for different architectures
        targets_map = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj", "c_fc"],
            "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        }

        for key, targets in targets_map.items():
            if key in model_type:
                return targets

        # Default fallback
        return ["q_proj", "v_proj"]

    def _extract_model_name(self, agent) -> str:
        """Extract model name from agent."""
        if hasattr(agent, "base_model") and hasattr(agent.base_model, "config"):
            if hasattr(agent.base_model.config, "model_type"):
                return agent.base_model.config.model_type
        if hasattr(agent, "config") and hasattr(agent.config, "_name_or_path"):
            return agent.config._name_or_path
        return agent.__class__.__name__

    def _extract_dataset_type(self) -> Optional[str]:
        """Extract dataset type from wandb config."""
        try:
            if isinstance(self.wandb_config, dict):
                sections = self.wandb_config.get("config_sections", {})
                if isinstance(sections, dict):
                    ds = sections.get("dataset", {})
                    if isinstance(ds, dict):
                        return ds.get("type")
        except Exception:
            pass
        return None

    def _extract_verbose(self) -> bool:
        """Extract verbose setting from wandb config."""
        try:
            if isinstance(self.wandb_config, dict):
                sections = self.wandb_config.get("config_sections", {})
                if isinstance(sections, dict):
                    out = sections.get("output", {})
                    if isinstance(out, dict) and "verbose" in out:
                        return bool(out.get("verbose"))
        except Exception:
            pass
        return True

    def _setup_formatters(self, formatters, num_agents):
        """Set up format functions for each agent."""
        default_format_func = lambda x: x.get("prompt", "")

        if formatters is None:
            self.formatters = [default_format_func] * num_agents
        elif callable(formatters) and not isinstance(formatters, list):
            self.formatters = [formatters] * num_agents
        elif isinstance(formatters, list):
            if len(formatters) != num_agents:
                raise ValueError(
                    f"Number of formatters ({len(formatters)}) must match "
                    f"number of agents ({num_agents})"
                )
            self.formatters = formatters
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
        if not self.wandb_initialized and self.is_main_process:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "bigcodebench")
            wandb_entity = self.wandb_config.get("entity", "contrl")
            wandb_name = self.wandb_config.get("name", "bcb-magrpo")
            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
                "joint_mode": self.args.joint_mode,
                "use_lora": self.args.use_lora,
                "distributed": self.distributed,
                "world_size": self.world_size,
            }

            if self.args.use_lora:
                config_dict.update({
                    "lora_r": self.args.lora_r,
                    "lora_alpha": self.args.lora_alpha,
                    "lora_dropout": self.args.lora_dropout,
                })

            sections = self.wandb_config.get("config_sections") if isinstance(self.wandb_config, dict) else None
            if isinstance(sections, dict):
                for key in ["dataset", "model", "output", "trainer"]:
                    if key in sections:
                        config_dict[key] = sections[key]

                dataset_section = sections.get("dataset", {})
                if isinstance(dataset_section, dict):
                    if dataset_section.get("name"):
                        config_dict["dataset_name"] = dataset_section["name"]
                    if dataset_section.get("type"):
                        config_dict["dataset_type"] = dataset_section["type"]

            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }

            if wandb_dir is not None:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir

            tags = self.wandb_config.get("tags") if isinstance(self.wandb_config, dict) else None
            if isinstance(tags, list):
                init_kwargs["tags"] = tags

            wandb.init(**init_kwargs)
            self.wandb_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader with distributed sampler if needed."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sampler = None
        shuffle = False
        if self.distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
        else:
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Returns the evaluation DataLoader with distributed sampler if needed."""
        if self.eval_dataset is None:
            return None

        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                self.eval_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,  # No shuffling for eval
            )

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def _batch_generate_all_agents(
        self,
        batch_item: Dict,
        num_return_sequences: int,
        max_new_tokens: int,
        do_sample: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for all agents using batch generation.
        """
        # Get model for generation (unwrap DDP if needed)
        model = self.shared_model
        if isinstance(model, DDP):
            model = model.module

        device = self.device

        # Prepare prompts for all agents
        all_prompts = []
        for agent_idx in range(self.num_agents):
            prompt = self.formatters[agent_idx](batch_item)
            all_prompts.append(prompt)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize all prompts together
        prompt_encodings = self.tokenizer(
            all_prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original state
        training_mode = model.training
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        model.eval()

        try:
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            top_k = getattr(self.args, "top_k", None)
            repetition_penalty = getattr(self.args, "repetition_penalty", None)
            if do_sample and num_return_sequences > 1:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                })
                if top_k is not None:
                    generation_kwargs["top_k"] = top_k
                if repetition_penalty is not None:
                    generation_kwargs["repetition_penalty"] = repetition_penalty
            else:
                generation_kwargs.update({
                    "do_sample": do_sample,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                })

            if generation_kwargs.get("pad_token_id") is None:
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            generation_output = model.generate(**generation_kwargs)

        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore model state
        model.train(training_mode)
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Parse output
        all_sequences = generation_output.sequences

        # Split results per agent
        results = []
        for agent_idx in range(self.num_agents):
            start_idx = agent_idx * num_return_sequences
            end_idx = start_idx + num_return_sequences

            agent_sequences = all_sequences[start_idx:end_idx]
            agent_prompt_ids = prompt_input_ids[agent_idx]

            # prompt_len is the total input length (including any padding)
            # model.generate() appends new tokens AFTER the full input sequence
            # So we need to skip the entire input to get just the generated completion
            prompt_len = agent_prompt_ids.shape[0]

            # Extract completions
            completions = []
            completion_tokens_list = []

            for seq in agent_sequences:
                completion_tokens = seq[prompt_len:]
                completion_tokens_list.append(completion_tokens)
                completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                completions.append(completion_text)

            completion_masks = [torch.ones(len(t), device=device) for t in completion_tokens_list]

            results.append({
                "prompts": [all_prompts[agent_idx]],
                "batch_items": [batch_item],
                "prompt_input_ids": agent_prompt_ids.unsqueeze(0),
                "prompt_attention_mask": prompt_attention_mask[agent_idx].unsqueeze(0),
                "completions": [completions],
                "completion_input_ids": [completion_tokens_list],
                "completion_attention_mask": [completion_masks],
            })

        return results

    def _generate_single_agent(
        self,
        prompt: str,
        num_return_sequences: int,
        max_new_tokens: int,
        do_sample: bool = True,
        agent_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate completions for a single agent with a given prompt.
        Used for sequential generation in mental simulation mode.
        """
        # Get the appropriate model for this agent
        model = self._get_agent_model(agent_idx)
        if isinstance(model, DDP):
            model = model.module

        device = self.device

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encoding.input_ids
        prompt_attention_mask = prompt_encoding.attention_mask

        # Store original state
        training_mode = model.training
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        model.eval()

        try:
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            top_k = getattr(self.args, "top_k", None)
            repetition_penalty = getattr(self.args, "repetition_penalty", None)
            if do_sample and num_return_sequences > 1:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                })
                if top_k is not None:
                    generation_kwargs["top_k"] = top_k
                if repetition_penalty is not None:
                    generation_kwargs["repetition_penalty"] = repetition_penalty
            else:
                generation_kwargs.update({
                    "do_sample": do_sample,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                })

            if generation_kwargs.get("pad_token_id") is None:
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            generation_output = model.generate(**generation_kwargs)

        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore model state
        model.train(training_mode)
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Parse output
        all_sequences = generation_output.sequences
        prompt_len = prompt_input_ids.shape[1]

        completions = []
        completion_tokens_list = []

        for seq in all_sequences:
            completion_tokens = seq[prompt_len:]
            completion_tokens_list.append(completion_tokens)
            completion_text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
            completions.append(completion_text)

        completion_masks = [torch.ones(len(t), device=device) for t in completion_tokens_list]

        return {
            "prompts": [prompt],
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": [completions],
            "completion_input_ids": [completion_tokens_list],
            "completion_attention_mask": [completion_masks],
        }

    def _generate_with_mental_simulation(
        self,
        batch_item: Dict,
        num_return_sequences: int,
        max_new_tokens: int,
        do_sample: bool = True,
        inference_formatter: Optional[Callable] = None,
        conditioned_formatter: Optional[Callable] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[int]]:
        """
        Generate completions with mental simulation for Agent 2.

        Flow:
        1. Agent 1 generates helper function
        2. Agent 2 Step A: Generate inference (mental simulation of Agent 1's code)
        3. Agent 2 Step B: Generate main function conditioned on inference

        Returns:
            results: List of completion data per agent
            agent2_inferences: List of inference texts (for similarity computation)
            inference_boundaries: List of token indices where inference ends (for each generation)
        """
        device = self.device
        results = []
        agent2_inferences = []
        inference_boundaries = []

        # ====================================================================
        # Step 1: Agent 1 generates helper function
        # ====================================================================
        agent1_prompt = self.formatters[0](batch_item)
        agent1_result = self._generate_single_agent(
            prompt=agent1_prompt,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            agent_idx=0,
        )
        agent1_result["batch_items"] = [batch_item]
        results.append(agent1_result)

        # ====================================================================
        # Step 2: Agent 2 Step A - Mental Simulation (generate inference)
        # ====================================================================
        if inference_formatter is not None:
            inference_prompt = inference_formatter(batch_item)
        else:
            # Default inference prompt
            inference_prompt = self._create_inference_prompt(batch_item)

        inference_result = self._generate_single_agent(
            prompt=inference_prompt,
            num_return_sequences=num_return_sequences,
            max_new_tokens=self.args.inference_max_tokens,
            do_sample=do_sample,
            agent_idx=1,  # Agent 2's adapter
        )

        # Store inference completions
        inference_completions = inference_result["completions"][0]  # List of inference texts
        agent2_inferences = inference_completions

        # Track inference token lengths for each generation
        inference_token_lengths = [len(t) for t in inference_result["completion_input_ids"][0]]

        # ====================================================================
        # Step 3: Agent 2 Step B - Generate main function conditioned on inference
        # ====================================================================
        agent2_combined_completions = []
        agent2_combined_tokens = []
        agent2_prompts = []

        for gen_idx in range(num_return_sequences):
            inference_text = inference_completions[gen_idx]

            # Create conditioned prompt
            if conditioned_formatter is not None:
                conditioned_prompt = conditioned_formatter(batch_item, inference_text)
            else:
                conditioned_prompt = self._create_conditioned_prompt(batch_item, inference_text)

            # Generate main function for this specific inference
            # Note: For efficiency, we generate 1 main per inference
            main_result = self._generate_single_agent(
                prompt=conditioned_prompt,
                num_return_sequences=1,  # 1 main per inference
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                agent_idx=1,  # Still Agent 2's adapter
            )

            main_completion = main_result["completions"][0][0]
            main_tokens = main_result["completion_input_ids"][0][0]

            # Combine inference + main for the full Agent 2 completion
            full_completion = inference_text + "\n\n" + main_completion
            agent2_combined_completions.append(full_completion)

            # Store the main completion separately (this is what gets tested)
            # For training, we track both inference and main tokens
            inference_tokens = inference_result["completion_input_ids"][0][gen_idx]
            combined_tokens = torch.cat([inference_tokens, main_tokens])
            agent2_combined_tokens.append(combined_tokens)

            # Track boundary where inference ends
            inference_boundaries.append(len(inference_tokens))

            agent2_prompts.append(conditioned_prompt)

        # Build Agent 2 result structure
        # For reward computation, we use just the main completion
        agent2_result = {
            "prompts": agent2_prompts,
            "batch_items": [batch_item],
            "prompt_input_ids": inference_result["prompt_input_ids"],
            "prompt_attention_mask": inference_result["prompt_attention_mask"],
            "completions": [agent2_combined_completions],
            "completion_input_ids": [agent2_combined_tokens],
            "completion_attention_mask": [[torch.ones(len(t), device=device) for t in agent2_combined_tokens]],
            # Additional fields for mental simulation
            "inference_completions": inference_completions,
            "inference_token_ids": inference_result["completion_input_ids"][0],
            "main_completions": [main_result["completions"][0][0] for _ in range(num_return_sequences)],
            "inference_boundaries": inference_boundaries,
        }
        results.append(agent2_result)

        return results, agent2_inferences, inference_boundaries

    def _create_inference_prompt(self, batch_item: Dict) -> str:
        """Create the default inference prompt for mental simulation."""
        instruct_prompt = batch_item.get("instruct_prompt", "")
        entry_point = batch_item.get("entry_point", "")

        if not instruct_prompt:
            instruct_prompt = batch_item.get("prompt", "")

        return f"""Before writing the main function '{entry_point}', predict how your partner will implement the helper function 'aux'.

Problem: {instruct_prompt}

Write your prediction of how the 'aux' helper function will be implemented. Output ONLY the predicted function code starting with 'def aux(':

"""

    def _create_conditioned_prompt(self, batch_item: Dict, inference_text: str) -> str:
        """Create the conditioned prompt for main function generation."""
        # Get the original main formatter prompt
        main_prompt = self.formatters[1](batch_item) if len(self.formatters) > 1 else ""

        # Prepend the inference conditioning
        conditioned_prompt = f"""Your partner will write the helper function like this:

{inference_text}

Considering this helper function implementation, {main_prompt}"""

        return conditioned_prompt

    def _compute_single_reward(
        self,
        agent_completions: List[str],
        batch_item: Dict,
    ) -> float:
        """Compute reward for a single joint action."""
        try:
            completion_args = [[comp] for comp in agent_completions]
            sig = inspect.signature(self.reward_func)
            if "batch_items" in sig.parameters:
                func_rewards = self.reward_func(*completion_args, batch_items=[batch_item])
            else:
                func_rewards = self.reward_func(*completion_args)
        except TypeError:
            func_rewards = self.reward_func(agent_completions)

        processed_rewards = [self.reward_processor(r) for r in func_rewards]
        return float(processed_rewards[0] if processed_rewards else 0.0)

    def _compute_rewards_with_joint_mode(
        self,
        batch_item: Dict,
        agent_completions_list: List[List[str]],
    ) -> Tuple[List[float], List[Tuple[int, ...]]]:
        """
        Compute rewards based on joint_mode setting.
        
        Returns:
            rewards: List of reward values
            combo_indices: List of (agent_0_idx, agent_1_idx, ...) tuples indicating
                          which completion from each agent was used
        """
        joint_mode = self.args.joint_mode.lower()

        if joint_mode == "cross" and self.num_agents > 1:
            # Cross mode: compute rewards for all combinations
            return self._compute_rewards_cross(batch_item, agent_completions_list)
        else:
            # Aligned mode: compute rewards for same-index completions
            return self._compute_rewards_aligned(batch_item, agent_completions_list)

    def _compute_rewards_aligned(
        self,
        batch_item: Dict,
        agent_completions_list: List[List[str]],
    ) -> Tuple[List[float], List[Tuple[int, ...]]]:
        """Compute rewards for aligned (same index) completions."""
        num_completions = min(len(comps) for comps in agent_completions_list)

        if not self.args.parallel_reward or num_completions <= 1:
            rewards = []
            combo_indices = []
            for idx in range(num_completions):
                agent_completions = [
                    agent_completions_list[agent_idx][idx]
                    for agent_idx in range(self.num_agents)
                ]
                reward = self._compute_single_reward(agent_completions, batch_item)
                rewards.append(reward)
                combo_indices.append(tuple([idx] * self.num_agents))
            return rewards, combo_indices

        # Parallel computation
        rewards = [0.0] * num_completions
        combo_indices = [tuple([idx] * self.num_agents) for idx in range(num_completions)]
        max_workers = min(self.args.max_reward_workers, num_completions)

        # Choose executor based on backend config
        use_process = self.args.reward_parallel_backend.lower() == "process"
        ExecutorClass = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        with ExecutorClass(max_workers=max_workers) as executor:
            futures = {}
            for idx in range(num_completions):
                agent_completions = [
                    agent_completions_list[agent_idx][idx]
                    for agent_idx in range(self.num_agents)
                ]
                if use_process:
                    # Use module-level function for ProcessPoolExecutor (picklable)
                    future = executor.submit(
                        _compute_reward_worker,
                        self.reward_func,
                        self.reward_processor,
                        agent_completions,
                        batch_item,
                    )
                else:
                    # Use instance method for ThreadPoolExecutor
                    future = executor.submit(
                        self._compute_single_reward, agent_completions, batch_item
                    )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    # Fallback to 0.0 on error, log if verbose
                    if self.verbose:
                        print(f"Reward computation error: {e}")
                    rewards[idx] = 0.0

        return rewards, combo_indices

    def _compute_rewards_cross(
        self,
        batch_item: Dict,
        agent_completions_list: List[List[str]],
    ) -> Tuple[List[float], List[Tuple[int, ...]]]:
        """Compute rewards for all cross-product combinations."""
        # Generate all combinations
        per_agent_ranges = [range(len(comps)) for comps in agent_completions_list]
        all_combos = list(itertools.product(*per_agent_ranges))
        num_combos = len(all_combos)

        if not self.args.parallel_reward or num_combos <= 1:
            rewards = []
            for combo in all_combos:
                agent_completions = [
                    agent_completions_list[agent_idx][combo[agent_idx]]
                    for agent_idx in range(self.num_agents)
                ]
                reward = self._compute_single_reward(agent_completions, batch_item)
                rewards.append(reward)
            return rewards, all_combos

        # Parallel computation
        rewards = [0.0] * num_combos
        max_workers = min(self.args.max_reward_workers, num_combos)

        # Choose executor based on backend config
        use_process = self.args.reward_parallel_backend.lower() == "process"
        ExecutorClass = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        with ExecutorClass(max_workers=max_workers) as executor:
            futures = {}
            for combo_idx, combo in enumerate(all_combos):
                agent_completions = [
                    agent_completions_list[agent_idx][combo[agent_idx]]
                    for agent_idx in range(self.num_agents)
                ]
                if use_process:
                    # Use module-level function for ProcessPoolExecutor (picklable)
                    future = executor.submit(
                        _compute_reward_worker,
                        self.reward_func,
                        self.reward_processor,
                        agent_completions,
                        batch_item,
                    )
                else:
                    # Use instance method for ThreadPoolExecutor
                    future = executor.submit(
                        self._compute_single_reward, agent_completions, batch_item
                    )
                futures[future] = combo_idx

            for future in as_completed(futures):
                combo_idx = futures[future]
                try:
                    rewards[combo_idx] = future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"Reward computation error: {e}")
                    rewards[combo_idx] = 0.0

        return rewards, all_combos

    def train(self, **kwargs):
        """Main training loop for single-turn GRPO with multi-GPU support."""
        if self.wandb_config is not None and not self.wandb_initialized and self.is_main_process:
            self._init_wandb()

        # Move model to device
        self.shared_model.to(self.device)

        # Wrap model with DDP for distributed training
        if self.distributed:
            self.shared_model = DDP(
                self.shared_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )

        self.shared_model.train()

        for epoch in range(int(self.args.num_train_epochs)):
            epoch_rewards = []
            epoch_returns = []

            dl = self.get_train_dataloader()

            # Set epoch for distributed sampler
            if self.distributed and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)

            iterator = enumerate(dl)
            if not self.verbose and self.is_main_process:
                iterator = enumerate(
                    tqdm(dl, total=len(dl), desc=f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}")
                )

            for batch_idx, batch in iterator:
                # Periodic evaluation (all processes to avoid DDP sync issues)
                if self.args.eval_interval > 0 and batch_idx % self.args.eval_interval == 0:
                    # Synchronize before eval to ensure all processes are at the same point
                    if self.distributed:
                        dist.barrier()
                    
                    # All processes run eval (logging only on main)
                    self.evaluate(num_eval_samples=self.args.eval_num_samples)
                    
                    # Synchronize after eval before resuming training
                    if self.distributed:
                        dist.barrier()

                # Process single batch item
                batch_item = batch[0]
                batch_loss, batch_rewards, batch_returns = self._train_step(
                    batch_item, **kwargs
                )

                epoch_rewards.extend(batch_rewards)
                epoch_returns.extend(batch_returns)

            # Process remaining samples in buffers
            for agent_idx, buffer in enumerate(self.rollout_buffers):
                if buffer:
                    self._process_buffer(agent_idx, buffer)
                    buffer.clear()

            # Log epoch metrics (main process only)
            if self.wandb_initialized and wandb.run is not None and self.is_main_process:
                epoch_log = {}
                if epoch_rewards:
                    epoch_log["epoch_reward_mean"] = float(np.mean(epoch_rewards))
                if epoch_returns:
                    epoch_log["epoch_return_mean"] = float(np.mean(epoch_returns))
                if epoch_log:
                    wandb.log(epoch_log, step=self.env_step)

        # Cleanup distributed training
        if self.distributed:
            dist.barrier()

    def _train_step(
        self,
        batch_item: Dict,
        **kwargs,
    ) -> tuple:
        """Single training step with joint_mode support and optional mental simulation."""
        num_gens = self.args.num_generations

        # Check if mental simulation is enabled
        if self.args.enable_mental_simulation and self.num_agents >= 2:
            return self._train_step_with_mental_simulation(batch_item, num_gens, **kwargs)

        # Standard generation for all agents (original behavior)
        comps_per_agent = self._batch_generate_all_agents(
            batch_item,
            num_return_sequences=num_gens,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
        )

        # Extract completions list per agent
        agent_completions_list = [
            comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
        ]

        # Compute rewards with joint_mode
        rewards_vec, combo_indices = self._compute_rewards_with_joint_mode(
            batch_item, agent_completions_list
        )

        self.env_step += len(rewards_vec)
        mean_reward = float(np.mean(rewards_vec)) if rewards_vec else 0.0
        returns_vec = list(rewards_vec)
        mean_return = float(np.mean(returns_vec)) if returns_vec else 0.0

        # Create samples for each agent
        for agent_idx in range(self.num_agents):
            # For cross mode, compute per-agent returns
            if self.args.joint_mode.lower() == "cross" and self.num_agents > 1:
                agent_returns = self._compute_per_agent_returns(
                    agent_idx, rewards_vec, combo_indices, len(agent_completions_list[agent_idx])
                )
            else:
                agent_returns = returns_vec

            sample = RolloutSample(
                agent_idx=agent_idx,
                completions_data=self._pack_completions_for_buffer(comps_per_agent[agent_idx]),
                returns=agent_returns,
                combo_indices=combo_indices,
                mean_reward=mean_reward,
                mean_return=mean_return,
                env_step=self.env_step,
            )
            self._append_to_buffer(agent_idx, sample)

        return mean_return, rewards_vec, returns_vec

    def _train_step_with_mental_simulation(
        self,
        batch_item: Dict,
        num_gens: int,
        **kwargs,
    ) -> tuple:
        """
        Training step with mental simulation for Agent 2.
        
        Agent 2 first generates an inference about Agent 1's code,
        then generates the main function conditioned on that inference.
        Rewards are assigned differentially to inference vs main tokens.
        """
        # Generate with mental simulation
        comps_per_agent, agent2_inferences, inference_boundaries = self._generate_with_mental_simulation(
            batch_item,
            num_return_sequences=num_gens,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
        )

        # For reward computation, we use Agent 1's actual code and Agent 2's main function
        # Extract main completions for Agent 2 (excluding inference)
        agent1_completions = comps_per_agent[0]["completions"][0]
        
        # For reward, we need the main function part of Agent 2's completion
        # Get main completions from the mental simulation result
        if "main_completions" in comps_per_agent[1]:
            agent2_main_completions = comps_per_agent[1]["main_completions"]
        else:
            # Fallback: use full completion
            agent2_main_completions = comps_per_agent[1]["completions"][0]

        agent_completions_list = [agent1_completions, agent2_main_completions]

        # Compute base code rewards (R_corr)
        code_rewards_vec, combo_indices = self._compute_rewards_with_joint_mode(
            batch_item, agent_completions_list
        )

        # Compute similarity rewards if similarity function is set
        similarity_rewards = [0.0] * len(code_rewards_vec)
        lambda_t = self._get_similarity_weight(self.env_step)
        threshold = self.args.code_reward_threshold

        if self.similarity_reward_func is not None:
            for idx in range(len(code_rewards_vec)):
                # Gate similarity reward based on code reward threshold
                if code_rewards_vec[idx] >= threshold:
                    # Compute similarity between inference and Agent 1's actual code
                    inference_code = agent2_inferences[idx] if idx < len(agent2_inferences) else ""
                    actual_code = agent1_completions[idx] if idx < len(agent1_completions) else ""
                    
                    try:
                        sim_reward = self.similarity_reward_func(inference_code, actual_code)
                        similarity_rewards[idx] = sim_reward
                    except Exception:
                        similarity_rewards[idx] = 0.0
                else:
                    # Gated: similarity reward is 0 if code reward below threshold
                    similarity_rewards[idx] = 0.0

        # Compute differential returns for Agent 2
        # Inference tokens: R_corr + lambda_t * R_sim
        # Main tokens: R_corr only
        inference_returns = [
            code_rewards_vec[i] + lambda_t * similarity_rewards[i]
            for i in range(len(code_rewards_vec))
        ]
        main_returns = list(code_rewards_vec)

        # Agent 1 gets standard returns (code reward)
        agent1_returns = list(code_rewards_vec)

        self.env_step += len(code_rewards_vec)
        mean_reward = float(np.mean(code_rewards_vec)) if code_rewards_vec else 0.0
        mean_return = float(np.mean(code_rewards_vec)) if code_rewards_vec else 0.0

        # Create sample for Agent 1 (standard)
        sample_agent1 = RolloutSample(
            agent_idx=0,
            completions_data=self._pack_completions_for_buffer(comps_per_agent[0]),
            returns=agent1_returns,
            combo_indices=combo_indices,
            mean_reward=mean_reward,
            mean_return=mean_return,
            env_step=self.env_step,
        )
        self._append_to_buffer(0, sample_agent1)

        # Create sample for Agent 2 (with differential rewards)
        agent2_completions_data = self._pack_completions_for_buffer_with_boundaries(
            comps_per_agent[1], inference_boundaries
        )
        sample_agent2 = RolloutSample(
            agent_idx=1,
            completions_data=agent2_completions_data,
            returns=main_returns,  # Main function gets code reward
            combo_indices=combo_indices,
            mean_reward=mean_reward,
            mean_return=mean_return,
            env_step=self.env_step,
            # Differential reward fields
            inference_returns=inference_returns,
            main_returns=main_returns,
            inference_token_boundaries=inference_boundaries,
        )
        self._append_to_buffer(1, sample_agent2)

        # Log mental simulation metrics
        if self.wandb_initialized and wandb.run is not None and self.is_main_process:
            ms_log = {
                "train/lambda_t": lambda_t,
                "train/similarity_reward_mean": float(np.mean(similarity_rewards)) if similarity_rewards else 0.0,
                "train/inference_return_mean": float(np.mean(inference_returns)) if inference_returns else 0.0,
            }
            if self._should_log_train(self.env_step):
                wandb.log(ms_log, step=self.env_step)

        return mean_return, code_rewards_vec, list(code_rewards_vec)

    def _pack_completions_for_buffer_with_boundaries(
        self,
        completions_data: Dict[str, Any],
        inference_boundaries: List[int],
    ) -> Dict[str, Any]:
        """Pack completions data for buffer with token boundaries for mental simulation."""
        base_packed = self._pack_completions_for_buffer(completions_data)
        base_packed["inference_boundaries"] = inference_boundaries
        return base_packed

    def _compute_per_agent_returns(
        self,
        agent_idx: int,
        rewards: List[float],
        combo_indices: List[Tuple[int, ...]],
        num_completions: int,
    ) -> List[float]:
        """
        For cross mode, compute average reward for each completion of this agent
        by averaging over all combinations that include that completion.
        """
        agent_returns = [0.0] * num_completions
        agent_counts = [0] * num_completions

        for reward, combo in zip(rewards, combo_indices):
            comp_idx = combo[agent_idx]
            agent_returns[comp_idx] += reward
            agent_counts[comp_idx] += 1

        # Average
        for i in range(num_completions):
            if agent_counts[i] > 0:
                agent_returns[i] /= agent_counts[i]

        return agent_returns

    def _pack_completions_for_buffer(self, completions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pack completions data for buffer storage."""
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

    def _append_to_buffer(self, agent_idx: int, sample: RolloutSample) -> None:
        """Append sample to buffer and process if full."""
        buffer = self.rollout_buffers[agent_idx]
        buffer.append(sample)
        if len(buffer) >= self.args.rollout_buffer_size:
            self._process_buffer(agent_idx, buffer)
            buffer.clear()

    def _should_log_train(self, step: int) -> bool:
        """Check if we should log training metrics at this step."""
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_train_log_step = step
            return True
        if self._last_train_log_step < 0 or (step - self._last_train_log_step) >= interval:
            self._last_train_log_step = step
            return True
        return False

    def _process_buffer(self, agent_idx: int, buffer: List[RolloutSample]) -> None:
        """Process buffered samples and perform gradient update."""
        if not buffer:
            return

        self._update_from_samples(buffer)

        if self.wandb_initialized and wandb.run is not None and self.is_main_process:
            batch_log = {
                "train/reward_mean": float(np.mean([s.mean_reward for s in buffer])),
                "train/return_mean": float(np.mean([s.mean_return for s in buffer])),
            }
            step = max(s.env_step for s in buffer)
            if self._should_log_train(step):
                wandb.log(batch_log, step=step)

    def _update_from_samples(self, samples: List[RolloutSample]) -> None:
        """Perform gradient update from buffered samples."""
        if not samples:
            return

        random.shuffle(samples)

        # Get the appropriate optimizer and model for this agent
        agent_idx = samples[0].agent_idx if samples else 0
        if self.agent_optimizers is not None and agent_idx < len(self.agent_optimizers):
            optimizer = self.agent_optimizers[agent_idx]
        else:
            optimizer = self.optimizer

        optimizer.zero_grad()
        scale = 1.0 / len(samples)

        for sample in samples:
            # Check if this sample has differential rewards (mental simulation mode)
            if (sample.inference_returns is not None and 
                sample.main_returns is not None and
                sample.inference_token_boundaries is not None):
                # Use differential loss computation
                loss = self._compute_loss_with_differential_rewards(
                    sample.completions_data,
                    sample.inference_returns,
                    sample.main_returns,
                    sample.inference_token_boundaries,
                    agent_idx=agent_idx,
                )
            else:
                # Standard loss computation
                loss = self._compute_loss_with_gradients(
                    sample.completions_data,
                    sample.returns,
                    agent_idx=agent_idx,
                )
            (loss * scale).backward()

        # Gradient synchronization happens automatically with DDP
        optimizer.step()

    def _compute_loss_with_gradients(
        self,
        completions_data: Dict[str, Any],
        returns: List[float],
        agent_idx: int = 0,
    ) -> torch.Tensor:
        """Compute GRPO loss with proper gradient tracking."""
        device = self.device

        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)

        # Group-relative advantage (mean baseline) - within this prompt's group only
        mean_ret = returns_tensor.mean()
        advantages = returns_tensor - mean_ret

        # Get model for forward pass (use agent-specific model if available)
        model = self._get_agent_model(agent_idx)
        if isinstance(model, DDP):
            model = model.module

        model.train()

        prompt_input_ids = completions_data["prompt_input_ids"].to(device)
        completion_input_ids = completions_data["completion_input_ids"]
        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids = [[t.to(device) for t in completion_input_ids]]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        prompt_ids = prompt_input_ids[0]

        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            if seq_idx >= len(advantages):
                break

            advantage = advantages[seq_idx]

            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                target_ids = completion_tokens
                attention_mask = torch.ones(len(input_ids), device=device)

                # Forward pass through the agent's model
                agent_model = self._get_agent_model(agent_idx)
                outputs = agent_model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )

                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                log_probs = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(0):
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                        log_probs.append(token_log_prob)

                if log_probs:
                    sequence_log_prob = torch.stack(log_probs).sum()
                    loss = -sequence_log_prob * advantage
                    total_loss = total_loss + loss
                    num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def _compute_loss_with_differential_rewards(
        self,
        completions_data: Dict[str, Any],
        inference_returns: List[float],
        main_returns: List[float],
        inference_boundaries: List[int],
        normalize_std: bool = True,
        agent_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute GRPO loss with differential rewards for inference vs main tokens.
        
        This method applies different advantages to different parts of the completion:
        - Tokens before inference_boundary: Use inference_returns (includes R_sim)
        - Tokens after inference_boundary: Use main_returns (R_corr only)
        
        Advantage normalization is performed within each reward group:
        - Inference advantages: normalized from inference_returns within the group
        - Main advantages: normalized from main_returns within the group
        
        Args:
            completions_data: Dictionary containing completion tokens and prompts.
            inference_returns: Returns for inference tokens (R_corr + lambda*R_sim).
            main_returns: Returns for main function tokens (R_corr only).
            inference_boundaries: List of token indices where inference ends for each sample.
            normalize_std: Whether to also divide by std for advantage normalization.
            agent_idx: Index of the agent whose model to use.
        
        Returns:
            Total loss tensor with gradients.
        """
        device = self.device

        if len(inference_returns) == 0 or len(main_returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert to tensors
        inference_returns_tensor = torch.tensor(inference_returns, dtype=torch.float, device=device)
        main_returns_tensor = torch.tensor(main_returns, dtype=torch.float, device=device)

        # Group-relative advantage normalization
        # For inference tokens: advantage = (R_infer - mean(R_infer)) / std(R_infer)
        # For main tokens: advantage = (R_main - mean(R_main)) / std(R_main)
        
        inference_mean = inference_returns_tensor.mean()
        main_mean = main_returns_tensor.mean()

        inference_advantages = inference_returns_tensor - inference_mean
        main_advantages = main_returns_tensor - main_mean

        # Optional: normalize by standard deviation for more stable gradients
        if normalize_std:
            inference_std = inference_returns_tensor.std()
            main_std = main_returns_tensor.std()
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            if inference_std > eps:
                inference_advantages = inference_advantages / (inference_std + eps)
            if main_std > eps:
                main_advantages = main_advantages / (main_std + eps)

        # Get model for forward pass (use agent-specific model if available)
        model = self._get_agent_model(agent_idx)
        if isinstance(model, DDP):
            model = model.module

        model.train()

        prompt_input_ids = completions_data["prompt_input_ids"].to(device)
        completion_input_ids = completions_data["completion_input_ids"]
        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids = [[t.to(device) for t in completion_input_ids]]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        prompt_ids = prompt_input_ids[0]

        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            if seq_idx >= len(inference_advantages) or seq_idx >= len(main_advantages):
                break

            # Get the boundary for this sample
            boundary = inference_boundaries[seq_idx] if seq_idx < len(inference_boundaries) else 0

            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                target_ids = completion_tokens
                attention_mask = torch.ones(len(input_ids), device=device)

                # Forward pass through the agent's model
                agent_model = self._get_agent_model(agent_idx)
                outputs = agent_model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )

                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                # Compute log probs with differential advantages
                inference_log_probs = []
                main_log_probs = []

                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(0):
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]

                        if i < boundary:
                            # This is an inference token
                            inference_log_probs.append(token_log_prob)
                        else:
                            # This is a main function token
                            main_log_probs.append(token_log_prob)

                # Compute loss for inference tokens
                if inference_log_probs:
                    inference_log_prob_sum = torch.stack(inference_log_probs).sum()
                    inference_loss = -inference_log_prob_sum * inference_advantages[seq_idx]
                    total_loss = total_loss + inference_loss

                # Compute loss for main tokens
                if main_log_probs:
                    main_log_prob_sum = torch.stack(main_log_probs).sum()
                    main_loss = -main_log_prob_sum * main_advantages[seq_idx]
                    total_loss = total_loss + main_loss

                num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def evaluate(self, num_eval_samples: int = 8) -> Dict[str, float]:
        """Run evaluation on the eval dataset with distributed support.
        
        In distributed mode, each GPU processes different eval samples and
        results are gathered on the main process for logging.
        """
        if self.eval_dataset is None:
            return {}

        # Local results for this process
        local_agent_completions = [[] for _ in range(self.num_agents)]
        local_test_cases = []
        local_entry_points = []
        local_prompts = []
        local_rewards = []

        eval_dataloader = self.get_eval_dataloader()
        
        # In distributed mode, calculate samples per GPU
        # Each GPU processes num_eval_samples // world_size samples
        if self.distributed:
            samples_per_gpu = max(1, num_eval_samples // self.world_size)
        else:
            samples_per_gpu = num_eval_samples

        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= samples_per_gpu:
                    break

                for batch_item in batch:
                    sample_rewards = self._evaluate_sample(
                        batch_item,
                        local_agent_completions,
                        local_test_cases,
                        local_entry_points,
                        local_prompts,
                    )
                    local_rewards.extend(sample_rewards)

        # Gather results from all processes in distributed mode
        if self.distributed:
            all_agent_completions, all_test_cases, all_entry_points, all_prompts, all_rewards = \
                self._gather_eval_results(
                    local_agent_completions,
                    local_test_cases,
                    local_entry_points,
                    local_prompts,
                    local_rewards,
                )
        else:
            all_agent_completions = local_agent_completions
            all_test_cases = local_test_cases
            all_entry_points = local_entry_points
            all_prompts = local_prompts
            all_rewards = local_rewards

        eval_metrics = self._log_eval_metrics(
            all_agent_completions,
            all_test_cases,
            all_entry_points,
            all_prompts,
            all_rewards,
        )
        return eval_metrics

    def _gather_eval_results(
        self,
        local_agent_completions: List[List],
        local_test_cases: List,
        local_entry_points: List,
        local_prompts: List,
        local_rewards: List[float],
    ) -> Tuple[List[List], List, List, List, List[float]]:
        """Gather evaluation results from all processes in distributed mode.
        
        Returns aggregated results from all GPUs.
        """
        # Prepare local data as a single object for gathering
        local_data = {
            "agent_completions": local_agent_completions,
            "test_cases": local_test_cases,
            "entry_points": local_entry_points,
            "prompts": local_prompts,
            "rewards": local_rewards,
        }
        
        # Gather from all processes
        gathered_data = [None] * self.world_size
        dist.all_gather_object(gathered_data, local_data)
        
        # Aggregate results from all processes
        all_agent_completions = [[] for _ in range(self.num_agents)]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []
        all_rewards = []
        
        for data in gathered_data:
            if data is None:
                continue
            # Merge agent completions
            for agent_idx in range(self.num_agents):
                if agent_idx < len(data["agent_completions"]):
                    all_agent_completions[agent_idx].extend(data["agent_completions"][agent_idx])
            all_test_cases.extend(data["test_cases"])
            all_entry_points.extend(data["entry_points"])
            all_prompts.extend(data["prompts"])
            all_rewards.extend(data["rewards"])
        
        return all_agent_completions, all_test_cases, all_entry_points, all_prompts, all_rewards

    def _evaluate_sample(
        self,
        batch_item: Dict,
        all_agent_completions: List[List],
        all_test_cases: List,
        all_entry_points: List,
        all_prompts: List,
    ) -> List[float]:
        """Evaluate a single sample."""
        all_test_cases.append(batch_item.get("test", ""))
        all_entry_points.append(batch_item.get("entry_point", ""))
        # For BigCodeBench, use code_prompt instead of prompt (contains imports)
        if self.dataset_type and self.dataset_type.lower() in ["bigcodebench", "bcb"]:
            all_prompts.append(batch_item.get("code_prompt", ""))
        else:
            all_prompts.append(batch_item.get("prompt", ""))

        comps_per_agent = self._batch_generate_all_agents(
            batch_item,
            num_return_sequences=self.args.num_generations,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
        )

        for agent_idx in range(self.num_agents):
            completion = comps_per_agent[agent_idx]["completions"][0][0]
            all_agent_completions[agent_idx].append(completion)

        agent_completions_list = [
            comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
        ]
        rewards, _ = self._compute_rewards_with_joint_mode(batch_item, agent_completions_list)

        return rewards

    def _log_eval_metrics(
        self,
        all_agent_completions: List[List],
        all_test_cases: List,
        all_entry_points: List,
        all_prompts: List,
        all_rewards: List[float],
    ) -> Dict[str, float]:
        """Log evaluation metrics. Only main process logs to wandb."""
        eval_metrics = {}

        if all_rewards:
            eval_metrics["eval/reward_mean"] = float(np.mean(all_rewards))
            eval_metrics["eval/reward_std"] = float(np.std(all_rewards))

        # Only run eval_logger on main process to avoid duplicate verbose output
        if (
            self.is_main_process
            and self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions
            and all(agent_comps for agent_comps in all_agent_completions)
        ):
            detailed_metrics = self.eval_logger(
                agent_completions=all_agent_completions,
                test_cases=all_test_cases,
                entry_points=all_entry_points,
                prompts=all_prompts,
            )

            aggregated_metrics = self.eval_aggregator(detailed_metrics)
            for key, value in aggregated_metrics.items():
                eval_metrics[f"eval/{key}"] = value

        if self.wandb_initialized and wandb.run is not None and self.is_main_process:
            wandb.log(eval_metrics, step=self.env_step)

        return eval_metrics

    def save_model(self, output_dir: str):
        """Save the trained model (and LoRA adapter if used)."""
        if not self.is_main_process:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Get the underlying model (unwrap DDP if needed)
        model = self.shared_model
        if isinstance(model, DDP):
            model = model.module

        model_dir = f"{output_dir}/model"
        os.makedirs(model_dir, exist_ok=True)

        # Save LoRA adapter separately if using PEFT
        if self.args.use_lora and PEFT_AVAILABLE and hasattr(model, "save_pretrained"):
            # Save only the LoRA adapter
            lora_dir = f"{output_dir}/lora_adapter"
            os.makedirs(lora_dir, exist_ok=True)
            model.save_pretrained(lora_dir)
            print(f"LoRA adapter saved to {lora_dir}")

            # Optionally save merged model
            if hasattr(model, "merge_and_unload"):
                merged_dir = f"{output_dir}/merged_model"
                os.makedirs(merged_dir, exist_ok=True)
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(merged_dir)
                print(f"Merged model saved to {merged_dir}")
        else:
            # Save full model
            model.save_pretrained(model_dir)

        if self.tokenizer:
            self.tokenizer.save_pretrained(model_dir)

        if self.wandb_initialized and wandb.run is not None:
            wandb.log({"final_model_saved": output_dir})
            wandb.finish()

    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
