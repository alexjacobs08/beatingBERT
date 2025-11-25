"""Data preprocessing utilities for BERT and Gemma models."""

from typing import Any
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

from src.utils.config import TASK_CONFIGS
from src.prompts.templates import get_zero_shot_prompt, create_few_shot_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTPreprocessor:
    """Preprocessor for BERT/DeBERTa models."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Initialize BERT preprocessor.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def preprocess_function(self, examples: dict[str, list], task: str) -> dict[str, Any]:
        """
        Preprocess examples for BERT-style models.
        
        Args:
            examples: Batch of examples
            task: Task name
            
        Returns:
            Dictionary with tokenized inputs
        """
        task_config = TASK_CONFIGS[task]
        text_keys = task_config["text_keys"]
        
        # Handle single or paired inputs
        if len(text_keys) == 1:
            # Single sentence tasks (SST-2)
            texts = examples[text_keys[0]]
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,  # Return lists for datasets library
            )
        else:
            # Sentence pair tasks (MNLI, RTE, etc.)
            texts_a = examples[text_keys[0]]
            texts_b = examples[text_keys[1]]
            tokenized = self.tokenizer(
                texts_a,
                texts_b,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
        
        return tokenized
    
    def preprocess_dataset(self, dataset: Dataset, task: str) -> Dataset:
        """
        Preprocess entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            task: Task name
            
        Returns:
            Preprocessed dataset
        """
        logger.info(f"Preprocessing dataset for task: {task}")
        
        # Apply preprocessing
        processed = dataset.map(
            lambda examples: self.preprocess_function(examples, task),
            batched=True,
            desc="Tokenizing",
        )
        
        # Rename 'label' to 'labels' for consistency with transformers
        if "label" in processed.column_names:
            processed = processed.rename_column("label", "labels")
        
        # Set format for PyTorch
        processed.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return processed


class GemmaPreprocessor:
    """Preprocessor for Gemma models (prompt-based)."""
    
    def __init__(self, task: str, num_few_shot: int = 0, few_shot_examples: list[dict] | None = None):
        """
        Initialize Gemma preprocessor.
        
        Args:
            task: Task name
            num_few_shot: Number of few-shot examples (0 for zero-shot)
            few_shot_examples: List of examples to use for few-shot learning
        """
        self.task = task
        self.num_few_shot = num_few_shot
        self.few_shot_examples = few_shot_examples or []
        
        if num_few_shot > 0 and len(self.few_shot_examples) != num_few_shot:
            logger.warning(
                f"num_few_shot={num_few_shot} but got {len(self.few_shot_examples)} examples"
            )
    
    def create_prompt(self, example: dict[str, Any]) -> str:
        """
        Create prompt for an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Formatted prompt
        """
        if self.num_few_shot == 0:
            # Zero-shot
            return get_zero_shot_prompt(self.task, example)
        else:
            # Few-shot
            return create_few_shot_prompt(
                self.task,
                self.few_shot_examples,
                example
            )
    
    def preprocess_function(self, examples: dict[str, list]) -> dict[str, list]:
        """
        Create prompts for a batch of examples.
        
        Args:
            examples: Batch of examples
            
        Returns:
            Dictionary with prompts
        """
        # Get number of examples in batch
        num_examples = len(examples["label"])
        
        # Create prompt for each example
        prompts = []
        for i in range(num_examples):
            # Extract single example
            example = {key: examples[key][i] for key in examples.keys()}
            # Create prompt
            prompt = self.create_prompt(example)
            prompts.append(prompt)
        
        return {"prompt": prompts}
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Add prompts to dataset.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Dataset with prompts
        """
        logger.info(
            f"Creating prompts for task: {self.task} "
            f"(num_few_shot={self.num_few_shot})"
        )
        
        # Add prompts
        processed = dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Creating prompts",
        )
        
        return processed


def collate_fn_bert(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for BERT models.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched tensors
    """
    import torch
    
    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "attention_mask": torch.stack([example["attention_mask"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]),
    }


def get_text_from_example(example: dict[str, Any], task: str) -> str:
    """
    Extract readable text from an example.
    
    Args:
        example: Example dictionary
        task: Task name
        
    Returns:
        Human-readable text representation
    """
    task_config = TASK_CONFIGS[task]
    text_keys = task_config["text_keys"]
    
    texts = [example[key] for key in text_keys]
    return " [SEP] ".join(texts)

