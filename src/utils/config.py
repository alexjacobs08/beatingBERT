"""Configuration management for experiments."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from typing import Any


# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


# Task configurations
TASK_CONFIGS = {
    "sst2": {
        "name": "sst2",
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "text_keys": ["sentence"],
        "metric": "accuracy",
    },
    "mnli": {
        "name": "mnli",
        "num_labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "text_keys": ["premise", "hypothesis"],
        "metric": "accuracy",
    },
    "mnli_matched": {
        "name": "mnli_matched",
        "num_labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "text_keys": ["premise", "hypothesis"],
        "metric": "accuracy",
    },
    "mnli_mismatched": {
        "name": "mnli_mismatched",
        "num_labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "text_keys": ["premise", "hypothesis"],
        "metric": "accuracy",
    },
    "rte": {
        "name": "rte",
        "num_labels": 2,
        "label_names": ["entailment", "not_entailment"],
        "text_keys": ["sentence1", "sentence2"],
        "metric": "accuracy",
    },
    "qqp": {
        "name": "qqp",
        "num_labels": 2,
        "label_names": ["not_duplicate", "duplicate"],
        "text_keys": ["question1", "question2"],
        "metric": "accuracy",
    },
    "mrpc": {
        "name": "mrpc",
        "num_labels": 2,
        "label_names": ["not_equivalent", "equivalent"],
        "text_keys": ["sentence1", "sentence2"],
        "metric": "accuracy",
    },
}


@dataclass
class BERTConfig:
    """Configuration for BERT/DeBERTa fine-tuning."""
    
    model_name: str = "bert-base-uncased"
    task: str = "sst2"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 128
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_samples: int | None = None  # For sample efficiency experiments
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class GemmaConfig:
    """Configuration for Gemma inference and LoRA."""
    
    model_name: str = "google/gemma-3-1b-it"
    task: str = "sst2"
    temperature: float = 0.0
    max_new_tokens: int = 3  # Changed from 10 to 3 for single-token answers
    batch_size: int = 8
    use_fp16: bool = True
    
    # Few-shot config
    num_few_shot: int = 0  # 0 for zero-shot, 5 or 10 for few-shot
    
    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Training config (for LoRA)
    learning_rate: float = 1e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    max_samples: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_task_config(task: str) -> dict[str, Any]:
    """
    Get configuration for a specific task.
    
    Args:
        task: Task name (e.g., 'sst2', 'mnli', 'rte')
        
    Returns:
        Task configuration dictionary
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task]


def ensure_dirs() -> None:
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PROMPTS_DIR.mkdir(exist_ok=True)

