"""
PuB-MDP (Public Belief MDP) Trainer for Collaborative Code Generation.

This trainer implements a centralized planning / decentralized execution paradigm:
- Public Agent: Learns to generate prescriptions (instructions) for worker agents
- Worker Agents: Frozen LLMs that execute code generation based on prescriptions

The Public Agent generates:
- z1: Prescription for Agent 1 (auxiliary function specification)
- z2: Prescription for Agent 2 (main function specification), conditioned on z1

Training uses GRPO (Group Relative Policy Optimization) to update only the Public Agent.

Features:
- Multi-GPU data parallelism support
- Flash Attention 2 support
- Parallel generation and reward computation
- Separate model configurations for Public Agent and Worker Agents
"""

import inspect
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments


@dataclass
class PuBMDPConfig(TrainingArguments):
    """
    Configuration for PuB-MDP training.
    
    The Public Agent learns a meta-policy that generates prescriptions for worker agents.
    Worker agents can be frozen or trained based on `train_workers` option.
    """
    
    # Core setup
    num_train_epochs: float = field(
        default=4,
        metadata={"help": "Number of training epochs."},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Per-device batch size."},
    )
    learning_rate: float = field(
        default=5.0e-6,
        metadata={"help": "Learning rate for Public Agent optimizer."},
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "Log every N steps."},
    )
    save_steps: int = field(
        default=200,
        metadata={"help": "Save every N steps."},
    )
    
    # Agent configuration
    num_workers: int = field(
        default=2,
        metadata={"help": "Number of worker agents (Agent 1 for aux, Agent 2 for main)."},
    )
    
    # Worker training options
    train_workers: bool = field(
        default=False,
        metadata={"help": "Whether to train worker agents. If False, workers are frozen."},
    )
    worker_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for worker agents. If None, uses same as public agent."},
    )
    
    # Sampling/generation for Public Agent (prescriptions)
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of prescription groups to sample per task."},
    )
    max_prescription_tokens: int = field(
        default=256,
        metadata={"help": "Maximum tokens for prescription generation."},
    )
    
    # Sampling/generation for Worker Agents (code)
    max_code_tokens: int = field(
        default=512,
        metadata={"help": "Maximum tokens for code generation by worker agents."},
    )
    
    # Generation parameters
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for sampling."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p for nucleus sampling."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "Top-k for sampling (set to None to disable)."},
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
    
    # Buffer for gradient accumulation
    rollout_buffer_size: int = field(
        default=64,
        metadata={"help": "Number of samples to buffer before an update."},
    )
    
    # Parallelization
    num_reward_workers: int = field(
        default=4,
        metadata={"help": "Number of parallel workers for reward computation."},
    )


@dataclass
class PrescriptionSample:
    """Stores a prescription sample for buffer-based training."""
    prescription_data: Dict[str, Any]  # z1, z2 generation data
    returns: List[float]  # Returns for each prescription group
    mean_reward: float
    mean_return: float
    env_step: int
    prompt_id: int  # ID to ensure group normalization within same prompt
    # Worker data (only used when train_workers=True)
    worker_data: Optional[Dict[str, Any]] = None  # aux/main prompt_ids and completion_ids


class PuBMDPTrainer:
    """
    PuB-MDP Trainer for Collaborative Code Generation with Multi-GPU support.
    
    Implements centralized planning (Public Agent) with decentralized execution (Worker Agents).
    
    Features:
    - Multi-GPU data parallelism (different samples on different GPUs)
    - Flash Attention 2 support
    - Parallel generation and reward computation
    - Separate model configurations for Public Agent and Worker Agents
    
    System Flow:
    1. Public Agent receives task T
    2. Public Agent generates z1 (prescription for Agent 1)
    3. Public Agent generates z2 (prescription for Agent 2), conditioned on T and z1
    4. Worker Agent 1 (frozen): Takes z1 + aux_instruction -> generates aux code
    5. Worker Agent 2 (frozen): Takes z2 + main_instruction -> generates main code
    6. Reward computed from combined code execution
    7. Only Public Agent is updated using GRPO
    """
    
    def __init__(
        self,
        # Models
        public_agent: Optional[PreTrainedModel] = None,
        worker_agents: Optional[List[PreTrainedModel]] = None,
        # Tokenizers
        public_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        worker_tokenizers: Optional[List[PreTrainedTokenizerBase]] = None,
        # Data
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        dataset_type: Optional[str] = None,
        # Reward
        reward_func: Optional[Callable] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        # Formatters and templates
        task_formatter: Optional[Callable] = None,
        prescription_prompt_template: Optional[str] = None,
        aux_instruction_template: Optional[Callable] = None,
        main_instruction_template: Optional[Callable] = None,
        # Logging
        wandb_config: Optional[Dict[str, Any]] = None,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
        # Config
        args: Optional[PuBMDPConfig] = None,
        # Distributed training
        local_rank: int = -1,
    ):
        # Validate GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not found. PuBMDPTrainer requires GPU for training.")
        
        if public_agent is None:
            raise ValueError("public_agent must be provided")
        if worker_agents is None or len(worker_agents) != 2:
            raise ValueError("worker_agents must be a list of 2 models [Agent1, Agent2]")
        
        self.args = args if args is not None else PuBMDPConfig()
        self.env_step = 0
        self._last_train_log_step = -1
        self._prompt_counter = 0  # For tracking prompt IDs
        
        # Distributed training setup
        self.local_rank = local_rank
        self.is_distributed = local_rank != -1
        self.world_size = 1
        self.rank = 0
        
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda")
        
        # Public Agent (trainable) - store the raw model, DDP wrapping done in train()
        self._public_agent_raw = public_agent
        self.public_agent = public_agent  # Will be replaced with DDP in train()
        self.public_tokenizer = public_tokenizer
        
        # Worker Agents - can be frozen or trainable based on config
        self._worker_agents_raw = worker_agents  # Store raw models for DDP wrapping
        self.worker_agents = worker_agents  # Will be replaced with DDP in train() if training
        self.num_workers = len(worker_agents)
        self.train_workers = self.args.train_workers
        
        # Use same tokenizer for workers if not specified
        if worker_tokenizers is None:
            self.worker_tokenizers = [public_tokenizer] * self.num_workers
        else:
            self.worker_tokenizers = worker_tokenizers
        
        # Freeze or enable training for worker agents based on config
        if not self.train_workers:
            # Freeze worker agents - ensure they never get gradients
            for worker in self.worker_agents:
                worker.eval()
                for param in worker.parameters():
                    param.requires_grad = False
        else:
            # Enable training for worker agents
            for worker in self.worker_agents:
                worker.train()
                for param in worker.parameters():
                    param.requires_grad = True
        
        # Data
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset_type = dataset_type
        
        # Reward
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable")
        self.reward_func = reward_func
        self.reward_processor = reward_processor if reward_processor else (lambda x: x)
        
        # Setup formatters and templates
        self._setup_formatters(
            task_formatter,
            prescription_prompt_template,
            aux_instruction_template,
            main_instruction_template,
        )
        
        # Logging
        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        
        # Optimizers will be created in train() after DDP wrapping
        self.optimizer = None  # Public Agent optimizer
        self.worker_optimizers = None  # Worker Agent optimizers (only if train_workers=True)
        
        # Rollout buffer for prescription samples
        self.rollout_buffer: List[PrescriptionSample] = []
        
        # Thread pool for parallel reward computation
        self.reward_executor = ThreadPoolExecutor(
            max_workers=self.args.num_reward_workers
        )
        
        # Initialize W&B only on rank 0
        if self.wandb_config is not None and self.rank == 0:
            self._init_wandb()
        
        # Verbosity
        self.verbose = True
    
    def _get_base_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Get the underlying model from DDP wrapper if applicable."""
        if hasattr(model, 'module'):
            return model.module
        return model
    
    def _setup_formatters(
        self,
        task_formatter,
        prescription_prompt_template,
        aux_instruction_template,
        main_instruction_template,
    ):
        """Setup formatters and instruction templates."""
        
        # Default task formatter
        if task_formatter is None:
            self.task_formatter = lambda x: x.get("prompt", "")
        else:
            self.task_formatter = task_formatter
        
        # Default prescription prompt template for Public Agent
        # Note: Uses only {task} placeholder, entry_point is passed separately
        if prescription_prompt_template is None:
            self.prescription_prompt_template = """You are a coding team leader (Public Agent). Your job is to coordinate two developers working on a coding task.

Task:
{task}

You need to provide clear specifications (prescriptions) for each developer:

1. **Developer 1 (Auxiliary Function)**: Will create a helper function named 'aux' that assists with the main problem.
2. **Developer 2 (Main Function)**: Will implement the main solution using the aux function.

Important: Developer 2 cannot see Developer 1's code. They can only see your specifications.

First, write a detailed specification for Developer 1 describing:
- What the aux function should do
- What parameters it should accept
- What it should return
- Key implementation details

[PRESCRIPTION_1]
"""
        else:
            self.prescription_prompt_template = prescription_prompt_template
        
        # Default aux instruction template
        if aux_instruction_template is None:
            def default_aux_template(task: str, prescription: str, entry_point: str) -> str:
                return f"""Create a helper function based on the following specification.

Problem Context:
{task}

Specification from Team Leader:
{prescription}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Create a helper function named 'aux'
- Follow the specification exactly

Your output should follow this format:

def aux(...):\n    # your function code here\n    return result
"""
            self.aux_instruction_template = default_aux_template
        else:
            self.aux_instruction_template = aux_instruction_template
        
        # Default main instruction template
        if main_instruction_template is None:
            def default_main_template(task: str, prescription: str, entry_point: str) -> str:
                return f"""Implement the main function based on the following specification.

Problem:
{task}

Specification from Team Leader:
{prescription}

You have access to a helper function: aux(...)
The aux function will be provided by your teammate based on the team leader's coordination.

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function
- Implement ONLY the '{entry_point}' function as specified
- You can call aux() within your function if helpful

Your output should follow this format:

def {entry_point}(...):\n    # your function code here\n    return result
"""
            self.main_instruction_template = default_main_template
        else:
            self.main_instruction_template = main_instruction_template
    
    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized and self.wandb_config:
            wandb_project = self.wandb_config.get("project", "pubmdp-coding")
            wandb_entity = self.wandb_config.get("entity", "contrl")
            wandb_name = self.wandb_config.get("name", "pubmdp-training")
            wandb_dir = self.wandb_config.get("dir", None)
            
            config_dict = {
                "num_workers": self.num_workers,
                "learning_rate": self.args.learning_rate,
                "num_generations": self.args.num_generations,
                "max_prescription_tokens": self.args.max_prescription_tokens,
                "max_code_tokens": self.args.max_code_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "world_size": self.world_size,
            }
            
            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }
            
            if wandb_dir:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir
            
            tags = self.wandb_config.get("tags", ["pubmdp", "code-generation"])
            if isinstance(tags, list):
                init_kwargs["tags"] = tags
            
            wandb.init(**init_kwargs)
            self.wandb_initialized = True
    
    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader with distributed sampler if needed."""
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=(sampler is None),
            sampler=sampler,
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
    
    def _generate_prescriptions_batch(
        self,
        batch_items: List[Dict[str, Any]],
        num_generations: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate prescription pairs (z1, z2) for a batch of items in PARALLEL.
        
        All z1 prescriptions are generated first in a single batch,
        then all z2 prescriptions are generated in a single batch.
        
        Args:
            batch_items: List of dataset items
            num_generations: Number of prescription pairs per item
        
        Returns:
            List of prescription data dictionaries, one per batch item
        """
        # Prepare all z1 prompts
        z1_prompts = []
        entry_points = []
        for item in batch_items:
            task = self.task_formatter(item)
            entry_point = item.get("entry_point", "solution")
            entry_points.append(entry_point)
            # Use only {task} in format - template only has {task} placeholder
            z1_prompt = self.prescription_prompt_template.format(task=task)
            z1_prompts.append(z1_prompt)
        
        # Batch generate ALL z1 prescriptions at once
        z1_batch_results = self._generate_text_batch(
            model=self.public_agent,
            tokenizer=self.public_tokenizer,
            prompts=z1_prompts,
            num_return_sequences=num_generations,
            max_new_tokens=self.args.max_prescription_tokens,
            do_sample=True,
        )
        
        # Prepare ALL z2 prompts at once
        all_z2_prompts = []
        z2_prompt_to_item_idx = []  # Track which item each z2 prompt belongs to
        z2_prompt_to_gen_idx = []   # Track which generation within item
        
        for batch_idx, (item, z1_result, entry_point) in enumerate(
            zip(batch_items, z1_batch_results, entry_points)
        ):
            task = self.task_formatter(item)
            prescriptions_z1 = z1_result["completions"]
            z1_base_prompt = self.prescription_prompt_template.format(task=task)
            
            for gen_idx, z1 in enumerate(prescriptions_z1):
                z2_prompt = f"""{z1_base_prompt}

{z1}

---

Now, based on the aux function specification above, write a detailed specification for Developer 2's main function '{entry_point}'.

Note: Developer 2 will see BOTH:
1. The aux function specification above (so they know how to use aux())
2. Your specification below (for implementing the main function)

Write a specification for the main function that:
- Describes what '{entry_point}' should accomplish
- Explains HOW to use the aux() function (based on the specification above)
- Specifies the parameters it should accept
- Specifies what it should return
- Includes key implementation logic

[SPECIFICATION FOR MAIN FUNCTION]
"""
                all_z2_prompts.append(z2_prompt)
                z2_prompt_to_item_idx.append(batch_idx)
                z2_prompt_to_gen_idx.append(gen_idx)
        
        # Batch generate ALL z2 prescriptions at once
        z2_all_results = self._generate_text_batch(
            model=self.public_agent,
            tokenizer=self.public_tokenizer,
            prompts=all_z2_prompts,
            num_return_sequences=1,
            max_new_tokens=self.args.max_prescription_tokens,
            do_sample=True,
        )
        
        # Organize results by batch item
        results = []
        for batch_idx, (item, z1_result, entry_point) in enumerate(
            zip(batch_items, z1_batch_results, entry_points)
        ):
            task = self.task_formatter(item)
            prescriptions_z1 = z1_result["completions"]
            z1_input_ids = z1_result["prompt_input_ids"]
            z1_completion_ids = z1_result["completion_input_ids"]
            z1_base_prompt = self.prescription_prompt_template.format(task=task)
            
            # Collect z2 results for this item
            prescriptions_z2 = []
            z2_completion_ids = []
            z2_input_ids_list = []
            z2_prompts_for_item = []
            
            for i, (item_idx, gen_idx) in enumerate(zip(z2_prompt_to_item_idx, z2_prompt_to_gen_idx)):
                if item_idx == batch_idx:
                    z2_result = z2_all_results[i]
                    prescriptions_z2.append(z2_result["completions"][0])
                    z2_completion_ids.append(z2_result["completion_input_ids"][0])
                    z2_input_ids_list.append(z2_result["prompt_input_ids"])
                    z2_prompts_for_item.append(all_z2_prompts[i])
            
            results.append({
                "prescriptions_z1": prescriptions_z1,
                "prescriptions_z2": prescriptions_z2,
                "prompt_for_z1": z1_base_prompt,
                "prompts_for_z2": z2_prompts_for_item,
                "z1_input_ids": z1_input_ids,
                "z2_input_ids": z2_input_ids_list,
                "z1_completion_ids": z1_completion_ids,
                "z2_completion_ids": z2_completion_ids,
                "batch_item": item,
            })
        
        return results
    
    def _execute_worker_agents_batch(
        self,
        prescription_results: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[List[str], List[str]]], Optional[Dict[str, Any]]]:
        """
        Execute worker agents for ALL prescriptions in PARALLEL.
        
        All aux prompts are generated in a single batch, then all main prompts.
        
        Args:
            prescription_results: List of prescription data from _generate_prescriptions_batch
        
        Returns:
            Tuple of:
            - List of (aux_codes, main_codes) tuples
            - Worker data dict (only if train_workers=True, else None)
        """
        all_aux_prompts = []
        all_main_prompts = []
        prompt_indices = []  # Track which item each prompt belongs to
        
        # Collect all prompts
        for idx, result in enumerate(prescription_results):
            item = result["batch_item"]
            task = self.task_formatter(item)
            entry_point = item.get("entry_point", "solution")
            
            for z1, z2 in zip(result["prescriptions_z1"], result["prescriptions_z2"]):
                # Agent 1: receives z1 (aux specification)
                aux_prompt = self.aux_instruction_template(task, z1, entry_point)
                
                # Agent 2: receives z2 (main specification) AND z1 (aux specification)
                # Check if main_instruction_template accepts aux_prescription parameter
                import inspect
                sig = inspect.signature(self.main_instruction_template)
                if 'aux_prescription' in sig.parameters:
                    main_prompt = self.main_instruction_template(task, z2, entry_point, aux_prescription=z1)
                else:
                    # Fallback for backward compatibility
                    main_prompt = self.main_instruction_template(task, z2, entry_point)
                
                all_aux_prompts.append(aux_prompt)
                all_main_prompts.append(main_prompt)
                prompt_indices.append(idx)
        
        # Use sampling when training workers, greedy otherwise
        do_sample = self.train_workers
        
        # Batch generate ALL auxiliary functions at once
        aux_results = self._generate_text_batch(
            model=self.worker_agents[0],
            tokenizer=self.worker_tokenizers[0],
            prompts=all_aux_prompts,
            num_return_sequences=1,
            max_new_tokens=self.args.max_code_tokens,
            do_sample=do_sample,
        )
        
        # Batch generate ALL main functions at once
        main_results = self._generate_text_batch(
            model=self.worker_agents[1],
            tokenizer=self.worker_tokenizers[1],
            prompts=all_main_prompts,
            num_return_sequences=1,
            max_new_tokens=self.args.max_code_tokens,
            do_sample=do_sample,
        )
        
        # Organize results by batch item
        outputs = [[] for _ in prescription_results]
        for i, (aux_r, main_r, idx) in enumerate(zip(aux_results, main_results, prompt_indices)):
            outputs[idx].append((aux_r["completions"][0], main_r["completions"][0]))
        
        # Convert to (aux_codes, main_codes) format
        final_results = []
        for item_outputs in outputs:
            aux_codes = [o[0] for o in item_outputs]
            main_codes = [o[1] for o in item_outputs]
            final_results.append((aux_codes, main_codes))
        
        # Prepare worker data for training (if enabled)
        worker_data = None
        if self.train_workers:
            # Organize by batch item for gradient computation
            worker_data_per_item = []
            result_idx = 0
            for idx, result in enumerate(prescription_results):
                num_gens = len(result["prescriptions_z1"])
                item_aux_results = aux_results[result_idx:result_idx + num_gens]
                item_main_results = main_results[result_idx:result_idx + num_gens]
                result_idx += num_gens
                
                worker_data_per_item.append({
                    "aux_prompt_ids": [r["prompt_input_ids"] for r in item_aux_results],
                    "aux_completion_ids": [r["completion_input_ids"][0] for r in item_aux_results],
                    "main_prompt_ids": [r["prompt_input_ids"] for r in item_main_results],
                    "main_completion_ids": [r["completion_input_ids"][0] for r in item_main_results],
                })
            
            worker_data = {"per_item": worker_data_per_item}
        
        return final_results, worker_data
    
    def _generate_text_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: List[str],
        num_return_sequences: int = 1,
        max_new_tokens: int = 256,
        do_sample: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate text from a model for a batch of prompts.
        
        Handles DDP-wrapped models correctly.
        
        Returns:
            List of dicts with completions, prompt_input_ids, completion_input_ids
        """
        if not prompts:
            return []
        
        # Get the base model for generation (unwrap DDP if necessary)
        base_model = self._get_base_model(model)
        device = next(base_model.parameters()).device
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        results = []
        
        # Process prompts in batches to avoid OOM
        batch_size = min(len(prompts), 8)  # Adjust based on memory
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            encoding = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            prompt_input_ids = encoding.input_ids
            prompt_attention_mask = encoding.attention_mask
            
            # Generation kwargs
            gen_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "num_return_sequences": num_return_sequences,
            }
            
            if do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,
                })
                if self.args.top_k is not None:
                    gen_kwargs["top_k"] = self.args.top_k
            else:
                gen_kwargs["do_sample"] = False
            
            # Store training state
            was_training = base_model.training
            base_model.eval()
            
            with torch.no_grad():
                # Use base model for generation (not DDP wrapper)
                output = base_model.generate(**gen_kwargs)
            
            if was_training:
                base_model.train()
            
            # Process outputs
            total_seqs = output.shape[0]
            seqs_per_prompt = num_return_sequences
            
            for prompt_idx in range(len(batch_prompts)):
                # Get actual prompt length (excluding padding)
                prompt_mask = prompt_attention_mask[prompt_idx]
                actual_prompt_len = (prompt_mask == 1).sum().item()
                
                # Extract only non-padded prompt tokens (remove left padding)
                # Left padding: [PAD, PAD, ..., token1, token2, ...]
                # We want: [token1, token2, ...]
                prompt_ids_no_pad = prompt_input_ids[prompt_idx][prompt_mask == 1].detach().cpu()
                
                completions = []
                completion_ids = []
                
                for seq_idx in range(seqs_per_prompt):
                    global_idx = prompt_idx * seqs_per_prompt + seq_idx
                    if global_idx >= total_seqs:
                        break
                    
                    # Extract completion (everything after actual prompt, not padding)
                    completion_tokens = output[global_idx, actual_prompt_len:]
                    
                    # Remove trailing pad tokens from completion
                    if tokenizer.pad_token_id is not None:
                        non_pad_mask = completion_tokens != tokenizer.pad_token_id
                        if non_pad_mask.any():
                            # Find last non-pad token
                            last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                            completion_tokens = completion_tokens[:last_non_pad]
                    
                    completion_ids.append(completion_tokens.detach().cpu())
                    completion_text = tokenizer.decode(
                        completion_tokens, skip_special_tokens=True
                    )
                    completions.append(completion_text)
                
                results.append({
                    "completions": completions,
                    "prompt_input_ids": prompt_ids_no_pad.unsqueeze(0),  # [1, actual_seq_len] without padding
                    "completion_input_ids": completion_ids,
                })
        
        return results
    
    def _compute_rewards_batch(
        self,
        batch_items: List[Dict[str, Any]],
        all_aux_codes: List[List[str]],
        all_main_codes: List[List[str]],
    ) -> List[List[float]]:
        """
        Compute rewards for a batch of items in PARALLEL using ThreadPoolExecutor.
        
        Args:
            batch_items: List of dataset items
            all_aux_codes: List of lists of aux codes (one list per item)
            all_main_codes: List of lists of main codes (one list per item)
        
        Returns:
            List of reward lists (one per item, maintaining prompt grouping)
        """
        all_rewards = []
        futures = []
        
        # Submit all reward computations in parallel
        for item, aux_codes, main_codes in zip(batch_items, all_aux_codes, all_main_codes):
            item_futures = []
            for aux_code, main_code in zip(aux_codes, main_codes):
                future = self.reward_executor.submit(
                    self._compute_single_reward,
                    item, aux_code, main_code
                )
                item_futures.append(future)
            futures.append(item_futures)
        
        # Collect results maintaining prompt grouping
        for item_futures in futures:
            item_rewards = []
            for future in item_futures:
                reward = future.result()
                item_rewards.append(reward)
            all_rewards.append(item_rewards)
        
        return all_rewards
    
    def _compute_single_reward(
        self,
        batch_item: Dict[str, Any],
        aux_code: str,
        main_code: str,
    ) -> float:
        """Compute reward for a single (aux_code, main_code) pair."""
        try:
            sig = inspect.signature(self.reward_func)
            if "batch_items" in sig.parameters:
                reward_list = self.reward_func(
                    [aux_code], [main_code], batch_items=[batch_item]
                )
            else:
                reward_list = self.reward_func([aux_code], [main_code])
            
            reward = self.reward_processor(reward_list[0])
        except Exception as e:
            if self.verbose:
                print(f"Reward computation error: {e}")
            reward = 0.0
        
        return float(reward)
    
    def _compute_grpo_loss_batch(
        self,
        samples: List[PrescriptionSample],
        accumulate_gradients: bool = True,
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a batch of samples with gradient accumulation.
        
        Uses gradient accumulation pattern to avoid OOM by performing backward()
        after each sample instead of accumulating all losses in one computation graph.
        
        IMPORTANT: Group normalization is done per-prompt to avoid mixing
        samples from different prompts.
        
        Args:
            samples: List of PrescriptionSample objects
            accumulate_gradients: If True, perform backward() after each sample
                                  to accumulate gradients (memory efficient).
                                  If False, return total loss (original behavior).
        
        Returns:
            Total loss tensor (for logging purposes)
        """
        device = self.device
        
        if not samples:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Group samples by prompt_id to ensure proper group normalization
        prompt_groups: Dict[int, List[PrescriptionSample]] = {}
        for sample in samples:
            pid = sample.prompt_id
            if pid not in prompt_groups:
                prompt_groups[pid] = []
            prompt_groups[pid].append(sample)
        
        # Get base model for forward pass
        base_model = self._get_base_model(self.public_agent)
        base_model.train()
        
        # First pass: count total samples for proper normalization
        total_samples = 0
        for prompt_id, group_samples in prompt_groups.items():
            for sample in group_samples:
                for g in range(len(sample.returns)):
                    prescription_data = sample.prescription_data
                    z1_completion_ids = prescription_data["z1_completion_ids"]
                    z2_completion_ids = prescription_data["z2_completion_ids"]
                    if len(z1_completion_ids[g]) > 0 and len(z2_completion_ids[g]) > 0:
                        total_samples += 1
        
        if total_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        accumulated_loss = 0.0  # For logging only
        
        # Collect all (loss_data) tuples first, then process with gradient accumulation
        loss_items = []
        
        for prompt_id, group_samples in prompt_groups.items():
            # Compute group-relative advantages WITHIN this prompt group only
            all_returns = []
            for s in group_samples:
                all_returns.extend(s.returns)
            
            if not all_returns:
                continue
            
            returns_tensor = torch.tensor(all_returns, dtype=torch.float, device=device)
            mean_return = returns_tensor.mean()
            
            # Process each sample in the group
            return_idx = 0
            for sample in group_samples:
                prescription_data = sample.prescription_data
                z1_input_ids = prescription_data["z1_input_ids"]
                z1_completion_ids = prescription_data["z1_completion_ids"]
                z2_input_ids_list = prescription_data["z2_input_ids"]
                z2_completion_ids = prescription_data["z2_completion_ids"]
                
                # Get pad_token_id from public tokenizer
                pad_token_id = self.public_tokenizer.pad_token_id if self.public_tokenizer else None
                
                for g in range(len(sample.returns)):
                    if return_idx >= len(returns_tensor):
                        break
                    
                    advantage = returns_tensor[return_idx] - mean_return
                    return_idx += 1
                    
                    z1_comp_ids = z1_completion_ids[g]
                    z2_comp_ids = z2_completion_ids[g]
                    
                    if len(z1_comp_ids) > 0 and len(z2_comp_ids) > 0:
                        loss_items.append({
                            'z1_input_ids': z1_input_ids,
                            'z1_comp_ids': z1_comp_ids,
                            'z2_prompt_ids': z2_input_ids_list[g],
                            'z2_comp_ids': z2_comp_ids,
                            'advantage': advantage,
                            'pad_token_id': pad_token_id,
                        })
        
        if not loss_items:
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        num_items = len(loss_items)
        
        # Process with gradient accumulation
        # Use DDP no_sync for all but the last backward to avoid redundant gradient sync
        for idx, item in enumerate(loss_items):
            is_last = (idx == num_items - 1)
            
            # Use no_sync context for all but the last backward (DDP optimization)
            if self.is_distributed and hasattr(self.public_agent, 'no_sync') and not is_last:
                context_manager = self.public_agent.no_sync()
            else:
                from contextlib import nullcontext
                context_manager = nullcontext()
            
            with context_manager:
                # Move tensors to device
                z1_input_ids = item['z1_input_ids'].to(device)
                z1_comp_ids = item['z1_comp_ids'].to(device)
                z2_prompt_ids = item['z2_prompt_ids'].to(device)
                z2_comp_ids = item['z2_comp_ids'].to(device)
                advantage = item['advantage']
                pad_token_id = item['pad_token_id']
                
                # Compute log prob for z1
                z1_log_prob = self._compute_sequence_log_prob(
                    z1_input_ids[0], z1_comp_ids, base_model, pad_token_id
                )
                
                # Compute log prob for z2
                z2_log_prob = self._compute_sequence_log_prob(
                    z2_prompt_ids[0], z2_comp_ids, base_model, pad_token_id
                )
                
                # Combined log prob
                combined_log_prob = z1_log_prob + z2_log_prob
                
                # Policy gradient loss (normalized by total samples)
                loss = (-combined_log_prob * advantage) / num_items
                
                if accumulate_gradients:
                    # Perform backward immediately to free computation graph
                    loss.backward()
                    accumulated_loss += loss.item()
                else:
                    accumulated_loss += loss.item()
        
        # Return accumulated loss as tensor for logging
        return torch.tensor(accumulated_loss, device=device, requires_grad=False)
    
    def _compute_sequence_log_prob(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        model: PreTrainedModel,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of a completion sequence.
        
        Args:
            prompt_ids: Prompt token IDs [seq_len] or [1, seq_len] (should NOT contain padding)
            completion_ids: Completion token IDs [seq_len] (should NOT contain padding)
            model: The model to use for forward pass (should be base model)
            pad_token_id: Optional pad token ID to filter out (safety check)
        
        Returns:
            Sum of log probabilities for the completion (scalar tensor with grad)
        """
        device = self.device
        
        if len(completion_ids) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure tensors are on correct device and 1D
        prompt_ids = prompt_ids.to(device)
        completion_ids = completion_ids.to(device)
        
        if prompt_ids.dim() > 1:
            prompt_ids = prompt_ids.squeeze(0)
        if completion_ids.dim() > 1:
            completion_ids = completion_ids.squeeze(0)
        
        # Safety check: Remove any remaining pad tokens from prompt
        # (This should not happen if _generate_text_batch works correctly, but added for safety)
        if pad_token_id is not None:
            non_pad_mask = prompt_ids != pad_token_id
            if non_pad_mask.any():
                prompt_ids = prompt_ids[non_pad_mask]
            
            # Also remove pad tokens from completion
            non_pad_mask_comp = completion_ids != pad_token_id
            if non_pad_mask_comp.any():
                completion_ids = completion_ids[non_pad_mask_comp]
        
        # Check again after filtering
        if len(prompt_ids) == 0 or len(completion_ids) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # input_ids = [prompt] + [completion[:-1]]
        # We predict completion tokens given prompt + previous completion tokens
        input_ids = torch.cat([prompt_ids, completion_ids[:-1]])
        attention_mask = torch.ones(len(input_ids), device=device, dtype=torch.long)
        
        # Forward pass through model
        outputs = model(
            input_ids=input_ids.unsqueeze(0),  # [1, seq_len]
            attention_mask=attention_mask.unsqueeze(0),  # [1, seq_len]
        )
        
        # logits shape: [1, seq_len, vocab_size]
        # We want logits for predicting completion tokens
        # Position prompt_len-1 predicts first completion token
        prompt_len = len(prompt_ids)
        completion_logits = outputs.logits[0, prompt_len - 1:-1, :]  # [completion_len, vocab_size]
        
        # Compute log probs for each completion token
        log_probs = []
        for i, token_id in enumerate(completion_ids):
            if i < completion_logits.size(0):
                token_logits = completion_logits[i]  # [vocab_size]
                token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                log_probs.append(token_log_prob)
        
        if log_probs:
            return torch.stack(log_probs).sum()
        
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_worker_loss_batch(
        self,
        samples: List[PrescriptionSample],
        worker_idx: int,
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a worker agent.
        
        Args:
            samples: List of PrescriptionSample objects with worker_data
            worker_idx: Index of worker agent (0 for aux, 1 for main)
        
        Returns:
            Loss tensor with gradients for the specified worker
        """
        device = self.device
        
        if not samples or not self.train_workers:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Group samples by prompt_id for proper group normalization
        prompt_groups: Dict[int, List[PrescriptionSample]] = {}
        for sample in samples:
            if sample.worker_data is None:
                continue
            pid = sample.prompt_id
            if pid not in prompt_groups:
                prompt_groups[pid] = []
            prompt_groups[pid].append(sample)
        
        # Get base model for forward pass
        base_model = self._get_base_model(self.worker_agents[worker_idx])
        base_model.train()
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_samples = 0
        
        # Select data keys based on worker index
        prompt_key = "aux_prompt_ids" if worker_idx == 0 else "main_prompt_ids"
        completion_key = "aux_completion_ids" if worker_idx == 0 else "main_completion_ids"
        
        for prompt_id, group_samples in prompt_groups.items():
            # Compute group-relative advantages WITHIN this prompt group only
            all_returns = []
            for s in group_samples:
                all_returns.extend(s.returns)
            
            if not all_returns:
                continue
            
            returns_tensor = torch.tensor(all_returns, dtype=torch.float, device=device)
            mean_return = returns_tensor.mean()
            
            # Process each sample in the group
            return_idx = 0
            # Get pad_token_id from worker tokenizer
            worker_tokenizer = self.worker_tokenizers[worker_idx] if self.worker_tokenizers else None
            pad_token_id = worker_tokenizer.pad_token_id if worker_tokenizer else None
            
            for sample in group_samples:
                worker_data = sample.worker_data
                prompt_ids_list = worker_data[prompt_key]
                completion_ids_list = worker_data[completion_key]
                
                for g in range(len(sample.returns)):
                    if return_idx >= len(returns_tensor):
                        break
                    
                    advantage = returns_tensor[return_idx] - mean_return
                    return_idx += 1
                    
                    prompt_ids = prompt_ids_list[g].to(device)
                    completion_ids = completion_ids_list[g].to(device)
                    
                    if len(completion_ids) > 0:
                        log_prob = self._compute_sequence_log_prob(
                            prompt_ids[0], completion_ids, base_model, pad_token_id
                        )
                        
                        # Policy gradient loss
                        loss = -log_prob * advantage
                        total_loss = total_loss + loss
                        total_samples += 1
        
        if total_samples > 0:
            total_loss = total_loss / total_samples
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)
        
        return total_loss
    
    def _train_step_batch(
        self,
        batch_items: List[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform training step for a batch of items with full parallelization.
        
        Args:
            batch_items: List of dataset items
        
        Returns:
            Tuple of (mean_return, metrics_dict)
        """
        num_gens = self.args.num_generations
        
        # Step 1: Generate prescriptions for ALL items in PARALLEL
        prescription_results = self._generate_prescriptions_batch(
            batch_items, num_generations=num_gens
        )
        
        # Step 2: Execute worker agents in PARALLEL
        worker_results, worker_data = self._execute_worker_agents_batch(prescription_results)
        
        # Step 3: Compute rewards in PARALLEL
        all_aux_codes = [r[0] for r in worker_results]
        all_main_codes = [r[1] for r in worker_results]
        all_rewards = self._compute_rewards_batch(batch_items, all_aux_codes, all_main_codes)
        
        # Step 4: Create samples with prompt IDs for proper grouping
        all_mean_rewards = []
        all_mean_returns = []
        
        for idx, (pres_result, rewards) in enumerate(zip(prescription_results, all_rewards)):
            self._prompt_counter += 1
            prompt_id = self._prompt_counter
            
            returns = rewards  # Single-step, returns = rewards
            
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            mean_return = float(np.mean(returns)) if returns else 0.0
            
            all_mean_rewards.append(mean_reward)
            all_mean_returns.append(mean_return)
            
            self.env_step += len(rewards)
            
            # Include worker data if training workers
            sample_worker_data = None
            if self.train_workers and worker_data is not None:
                sample_worker_data = worker_data["per_item"][idx]
            
            sample = PrescriptionSample(
                prescription_data={
                    "z1_input_ids": pres_result["z1_input_ids"],
                    "z1_completion_ids": pres_result["z1_completion_ids"],
                    "z2_input_ids": pres_result["z2_input_ids"],
                    "z2_completion_ids": pres_result["z2_completion_ids"],
                },
                returns=returns,
                mean_reward=mean_reward,
                mean_return=mean_return,
                env_step=self.env_step,
                prompt_id=prompt_id,
                worker_data=sample_worker_data,
            )
            
            self.rollout_buffer.append(sample)
        
        # Process buffer if full
        if len(self.rollout_buffer) >= self.args.rollout_buffer_size:
            self._process_buffer()
        
        batch_mean_reward = float(np.mean(all_mean_rewards))
        batch_mean_return = float(np.mean(all_mean_returns))
        
        metrics = {
            "mean_reward": batch_mean_reward,
            "mean_return": batch_mean_return,
        }
        
        return batch_mean_return, metrics
    
    def _process_buffer(self):
        """
        Process rollout buffer and update agents.
        
        Uses gradient accumulation to avoid OOM by performing backward()
        after each sample instead of accumulating all losses.
        
        Updates:
        - Public Agent: Always updated
        - Worker Agents: Updated only if train_workers=True
        """
        if not self.rollout_buffer:
            return
        
        # === Update Public Agent ===
        self.optimizer.zero_grad()
        
        # Compute loss with gradient accumulation
        # backward() is called inside _compute_grpo_loss_batch for each sample
        public_loss = self._compute_grpo_loss_batch(
            self.rollout_buffer, 
            accumulate_gradients=True
        )
        
        # Gradient synchronization for DDP happens during backward() calls
        # No need for explicit backward() here as it's done inside the function
        self.optimizer.step()
        
        # === Update Worker Agents (if enabled) ===
        worker_losses = []
        if self.train_workers and self.worker_optimizers is not None:
            for worker_idx in range(self.num_workers):
                self.worker_optimizers[worker_idx].zero_grad()
                
                worker_loss = self._compute_worker_loss_batch(
                    self.rollout_buffer, worker_idx
                )
                worker_loss.backward()
                
                # When using DDP, backward() already synchronizes gradients
                self.worker_optimizers[worker_idx].step()
                
                worker_losses.append(float(worker_loss.item()))
        
        # Log metrics (only on rank 0)
        if self.rank == 0 and self.wandb_initialized and wandb.run is not None:
            mean_reward = float(np.mean([s.mean_reward for s in self.rollout_buffer]))
            mean_return = float(np.mean([s.mean_return for s in self.rollout_buffer]))
            step = max(s.env_step for s in self.rollout_buffer)
            
            if self._should_log_train(step):
                log_dict = {
                    "train/mean_reward": mean_reward,
                    "train/mean_return": mean_return,
                    "train/public_loss": float(public_loss.item()),
                }
                
                # Log worker losses if training workers
                if worker_losses:
                    log_dict["train/worker_aux_loss"] = worker_losses[0]
                    if len(worker_losses) > 1:
                        log_dict["train/worker_main_loss"] = worker_losses[1]
                
                wandb.log(log_dict, step=step)
        
        self.rollout_buffer.clear()
    
    def _should_log_train(self, step: int) -> bool:
        """Check if we should log at this step."""
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_train_log_step = step
            return True
        if self._last_train_log_step < 0 or (step - self._last_train_log_step) >= interval:
            self._last_train_log_step = step
            return True
        return False
    
    def evaluate(self, num_eval_samples: int = 4) -> Dict[str, float]:
        """Evaluate the Public Agent."""
        if self.eval_dataset is None:
            return {}
        
        eval_rewards = []
        all_aux_codes = []
        all_main_codes = []
        all_test_cases = []
        all_entry_points = []
        all_prompts = []
        
        eval_dataloader = self.get_eval_dataloader()
        
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break
                
                # Batch evaluation
                prescription_results = self._generate_prescriptions_batch(
                    batch, num_generations=1
                )
                # Correctly unpack tuple: (List[Tuple[aux_codes, main_codes]], Optional[Dict])
                worker_results_list, _ = self._execute_worker_agents_batch(prescription_results)
                
                aux_codes_batch = [r[0] for r in worker_results_list]
                main_codes_batch = [r[1] for r in worker_results_list]
                rewards_batch = self._compute_rewards_batch(batch, aux_codes_batch, main_codes_batch)
                
                for item, aux_codes, main_codes, rewards in zip(
                    batch, aux_codes_batch, main_codes_batch, rewards_batch
                ):
                    eval_rewards.extend(rewards)
                    all_aux_codes.append(aux_codes)
                    all_main_codes.append(main_codes)
                    all_test_cases.append(item.get("test", ""))
                    all_entry_points.append(item.get("entry_point", ""))
                    all_prompts.append(item.get("prompt", ""))
        
        eval_metrics = {
            "eval/mean_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        }
        
        # Use custom logger if provided
        if self.eval_logger is not None and self.eval_aggregator is not None:
            agent_completions_turns = [
                [[aux[0]] for aux in all_aux_codes],
                [[main[0]] for main in all_main_codes],
            ]
            
            detailed_metrics = self.eval_logger(
                agent_completions_turns=agent_completions_turns,
                test_cases=all_test_cases,
                entry_points=all_entry_points,
                prompts=all_prompts,
            )
            
            aggregated = self.eval_aggregator(detailed_metrics, num_turns=1)
            for key, value in aggregated.items():
                eval_metrics[f"eval/{key}"] = value
        
        # Log to W&B (only on rank 0)
        if self.rank == 0 and self.wandb_initialized and wandb.run is not None:
            wandb.log(eval_metrics, step=self.env_step)
        
        return eval_metrics
    
    def train(self):
        """
        Main training loop with multi-GPU support.
        
        Each GPU has:
        - One Public Agent (trainable, wrapped in DDP)
        - Two Worker Agents (trainable or frozen based on train_workers option)
        
        Different GPUs process different data samples.
        Gradients are synchronized across GPUs via DDP.
        """
        if self.rank == 0 and self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()
        
        # Move models to device
        self._public_agent_raw.to(self.device)
        self._public_agent_raw.train()
        
        for worker in self._worker_agents_raw:
            worker.to(self.device)
            if self.train_workers:
                worker.train()
            else:
                worker.eval()
        
        # Wrap public agent with DDP for distributed training
        # DDP handles gradient synchronization automatically during backward()
        if self.is_distributed:
            self.public_agent = DDP(
                self._public_agent_raw,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.public_agent = self._public_agent_raw
        
        # Wrap worker agents with DDP if training them
        if self.train_workers and self.is_distributed:
            self.worker_agents = [
                DDP(
                    worker,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
                for worker in self._worker_agents_raw
            ]
        else:
            self.worker_agents = self._worker_agents_raw
        
        # Create optimizer AFTER DDP wrapping so it has correct parameter references
        # Only optimizes Public Agent parameters
        self.optimizer = torch.optim.AdamW(
            self._public_agent_raw.parameters(),  # Use raw model's parameters
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # Create worker optimizers if training workers
        if self.train_workers:
            worker_lr = self.args.worker_learning_rate or self.args.learning_rate
            self.worker_optimizers = [
                torch.optim.AdamW(
                    worker.parameters(),  # Use raw model's parameters
                    lr=worker_lr,
                    weight_decay=self.args.weight_decay,
                )
                for worker in self._worker_agents_raw
            ]
            if self.rank == 0 and self.verbose:
                print(f"Training Worker Agents with learning rate: {worker_lr}")
        
        for epoch in range(int(self.args.num_train_epochs)):
            epoch_rewards = []
            epoch_returns = []
            
            dl = self.get_train_dataloader()
            
            # Set epoch for distributed sampler to ensure different data per GPU
            if self.is_distributed and hasattr(dl.sampler, 'set_epoch'):
                dl.sampler.set_epoch(epoch)
            
            if self.rank == 0 and not self.verbose:
                it = enumerate(tqdm(
                    dl,
                    total=len(dl),
                    desc=f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}",
                ))
            else:
                it = enumerate(dl)
            
            for batch_idx, batch in it:
                # Periodic evaluation (only on rank 0)
                if self.rank == 0 and self.args.eval_interval > 0 and batch_idx % self.args.eval_interval == 0:
                    self.evaluate(num_eval_samples=self.args.eval_num_samples)
                
                # Synchronize all ranks after evaluation to prevent NCCL timeout
                # (Rank 0 may take longer due to evaluation while other ranks wait)
                if self.is_distributed and self.args.eval_interval > 0 and batch_idx % self.args.eval_interval == 0:
                    dist.barrier()
                
                # Train step with batch
                loss, metrics = self._train_step_batch(batch)
                
                epoch_rewards.append(metrics["mean_reward"])
                epoch_returns.append(metrics["mean_return"])
            
            # Process remaining buffer
            if self.rollout_buffer:
                self._process_buffer()
            
            # Synchronize at epoch end
            if self.is_distributed:
                dist.barrier()
            
            # Log epoch metrics (only on rank 0)
            if self.rank == 0 and self.wandb_initialized and wandb.run is not None:
                wandb.log({
                    "epoch/mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                    "epoch/mean_return": float(np.mean(epoch_returns)) if epoch_returns else 0.0,
                    "epoch": epoch + 1,
                }, step=self.env_step)
    
    def save_model(self, output_dir: str):
        """
        Save trained models (only on rank 0).
        
        Saves:
        - Public Agent: Always saved
        - Worker Agents: Saved only if train_workers=True
        """
        if self.rank != 0:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Public Agent
        public_dir = os.path.join(output_dir, "public_agent")
        os.makedirs(public_dir, exist_ok=True)
        
        model = self._get_base_model(self.public_agent)
        model.save_pretrained(public_dir)
        
        if self.public_tokenizer:
            self.public_tokenizer.save_pretrained(public_dir)
        
        # Save Worker Agents if they were trained
        if self.train_workers:
            for idx, worker in enumerate(self._worker_agents_raw):
                worker_dir = os.path.join(output_dir, f"worker_agent_{idx}")
                os.makedirs(worker_dir, exist_ok=True)
                
                worker_model = self._get_base_model(worker)
                worker_model.save_pretrained(worker_dir)
                
                if idx < len(self.worker_tokenizers):
                    self.worker_tokenizers[idx].save_pretrained(worker_dir)
            
            if self.verbose:
                print(f"Saved {len(self._worker_agents_raw)} worker agents")
        
        if self.wandb_initialized and wandb.run is not None:
            wandb.log({"model_saved": output_dir})
            wandb.finish()
    
    def cleanup(self):
        """Cleanup resources."""
        self.reward_executor.shutdown(wait=True)
        if self.is_distributed:
            dist.destroy_process_group()
