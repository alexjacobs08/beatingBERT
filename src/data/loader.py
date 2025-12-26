"""Dataset loading utilities for GLUE and reasoning benchmarks."""

from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
from typing import Tuple
import logging

from src.utils.config import DATA_DIR, TASK_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Non-GLUE datasets that need special loading
SPECIAL_DATASETS = {
    "anli_r1": ("anli", None),
    "anli_r2": ("anli", None),
    "anli_r3": ("anli", None),
    "hellaswag": ("Rowan/hellaswag", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "boolq": ("google/boolq", None),
}


def load_glue_dataset(
    task: str,
    cache_dir: Path | None = None,
    max_samples: int | None = None
) -> DatasetDict:
    """
    Load a dataset from HuggingFace (GLUE or reasoning benchmarks).
    
    Args:
        task: Task name (e.g., 'sst2', 'mnli', 'rte', 'anli_r1', 'hellaswag', etc.)
        cache_dir: Directory to cache datasets (default: DATA_DIR)
        max_samples: Maximum number of samples to load from train split (for testing/debugging)
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    if cache_dir is None:
        cache_dir = DATA_DIR
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate task
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    
    logger.info(f"Loading dataset: {task}")
    
    # Check if this is a special (non-GLUE) dataset
    if task in SPECIAL_DATASETS:
        dataset = _load_special_dataset(task, cache_dir)
    # Handle MNLI special cases
    elif task in ["mnli_matched", "mnli_mismatched"]:
        # Load full MNLI dataset
        dataset = load_dataset("glue", "mnli", cache_dir=str(cache_dir))
        
        # MNLI has different validation splits
        if task == "mnli_matched":
            dataset["validation"] = dataset["validation_matched"]
        else:  # mnli_mismatched
            dataset["validation"] = dataset["validation_mismatched"]
        
        # Remove the extra validation splits to avoid confusion
        if "validation_matched" in dataset:
            del dataset["validation_matched"]
        if "validation_mismatched" in dataset:
            del dataset["validation_mismatched"]
    else:
        # Load standard GLUE task
        logger.info(f"Loading GLUE dataset: {task}")
        dataset = load_dataset("glue", task, cache_dir=str(cache_dir))
    
    # Limit train samples if specified (for sample efficiency experiments)
    if max_samples is not None and max_samples > 0 and "train" in dataset:
        logger.info(f"Limiting train split to {max_samples} samples")
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
    
    # Log dataset sizes
    logger.info(f"Dataset sizes:")
    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} examples")
    
    return dataset


def _load_special_dataset(task: str, cache_dir: Path) -> DatasetDict:
    """Load non-GLUE datasets with special handling."""
    
    dataset_name, subset = SPECIAL_DATASETS[task]
    
    # ANLI - Adversarial NLI
    if task.startswith("anli_"):
        round_num = task[-1]  # r1, r2, or r3
        dataset = load_dataset("anli", cache_dir=str(cache_dir))
        
        # ANLI has separate splits per round
        result = DatasetDict({
            "train": dataset[f"train_r{round_num}"],
            "validation": dataset[f"dev_r{round_num}"],
            "test": dataset[f"test_r{round_num}"],
        })
        return result
    
    # HellaSwag - Commonsense completion
    elif task == "hellaswag":
        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
        
        # HellaSwag has train/validation/test
        # Test split has empty labels, so we filter those out
        def process_hellaswag(example):
            # Handle empty labels (test set)
            if example["label"] == "" or example["label"] is None:
                example["label"] = -1  # Mark as unlabeled
            else:
                example["label"] = int(example["label"])
            return example
        
        dataset = dataset.map(process_hellaswag)
        return dataset
    
    # WinoGrande - Pronoun resolution
    elif task == "winogrande":
        dataset = load_dataset(dataset_name, subset, cache_dir=str(cache_dir))
        
        # WinoGrande labels are "1" or "2" as strings, convert to 0/1
        # Test split has empty labels
        def process_winogrande(example):
            if example["answer"] == "" or example["answer"] is None:
                example["label"] = -1  # Mark as unlabeled
            else:
                example["label"] = int(example["answer"]) - 1  # "1"->0, "2"->1
            return example
        
        dataset = dataset.map(process_winogrande)
        return dataset
    
    # ARC-Challenge - Science reasoning
    elif task == "arc_challenge":
        dataset = load_dataset(dataset_name, subset, cache_dir=str(cache_dir))
        
        # ARC has answerKey as A/B/C/D, convert to 0/1/2/3
        def process_arc(example):
            answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            answer = example.get("answerKey", "")
            if answer == "" or answer is None:
                example["label"] = -1  # Mark as unlabeled
            else:
                example["label"] = answer_map.get(answer, -1)
            return example
        
        dataset = dataset.map(process_arc)
        return dataset
    
    # BoolQ - Boolean QA
    elif task == "boolq":
        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
        
        # BoolQ labels are already boolean, convert to int
        def process_boolq(example):
            answer = example.get("answer")
            if answer is None:
                example["label"] = -1  # Mark as unlabeled
            else:
                example["label"] = int(answer)  # True->1, False->0
            return example
        
        dataset = dataset.map(process_boolq)
        return dataset
    
    else:
        raise ValueError(f"Unknown special dataset: {task}")


def get_few_shot_examples(
    dataset: Dataset,
    num_examples: int = 5,
    seed: int = 99
) -> list[dict]:
    """
    Get a balanced set of examples for few-shot learning.
    
    Args:
        dataset: Training dataset
        num_examples: Total number of examples to return
        seed: Random seed for reproducibility
        
    Returns:
        List of example dictionaries
    """
    # Get unique labels
    labels = dataset.unique("label")
    num_labels = len([l for l in labels if l != -1])  # Exclude -1 (unlabeled)
    
    # Calculate examples per label
    examples_per_label = num_examples // num_labels
    remainder = num_examples % num_labels
    
    # Sample examples for each label
    examples = []
    valid_label_idx = 0  # Separate counter for valid labels only
    for label in sorted(labels):
        if label == -1:  # Skip unlabeled data
            continue

        # Get all examples with this label
        label_examples = dataset.filter(lambda x, l=label: x["label"] == l)

        # Determine how many to sample for this label
        # Use valid_label_idx (not enumerate index) for fair remainder distribution
        n_to_sample = examples_per_label + (1 if valid_label_idx < remainder else 0)
        n_to_sample = min(n_to_sample, len(label_examples))

        # Sample examples
        sampled = label_examples.shuffle(seed=seed).select(range(n_to_sample))
        examples.extend(sampled)

        valid_label_idx += 1
    
    return examples


def prepare_data_splits(
    task: str,
    max_samples: int | None = None,
    cache_dir: Path | None = None
) -> Tuple[DatasetDict, dict]:
    """
    Prepare data splits with metadata.
    
    Args:
        task: Task name
        max_samples: Maximum training samples
        cache_dir: Cache directory
        
    Returns:
        Tuple of (dataset_dict, metadata_dict)
    """
    # Load dataset
    dataset = load_glue_dataset(task, cache_dir=cache_dir, max_samples=max_samples)
    
    # Get task config
    task_config = TASK_CONFIGS[task]
    
    # Create metadata
    metadata = {
        "task": task,
        "num_labels": task_config["num_labels"],
        "label_names": task_config["label_names"],
        "text_keys": task_config["text_keys"],
        "train_size": len(dataset["train"]),
        "val_size": len(dataset["validation"]),
        "test_size": len(dataset.get("test", [])) if "test" in dataset else 0,
    }
    
    return dataset, metadata


def get_dataset_statistics(dataset: DatasetDict, task: str) -> dict:
    """
    Compute statistics about the dataset.
    
    Args:
        dataset: Dataset dictionary
        task: Task name
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    for split_name, split_data in dataset.items():
        # Label distribution
        labels = split_data["label"]
        label_counts = {}
        for label in labels:
            if label != -1:  # Skip unlabeled
                label_counts[label] = label_counts.get(label, 0) + 1
        
        stats[split_name] = {
            "num_examples": len(split_data),
            "label_distribution": label_counts,
        }
        
        # Text length statistics (for first text field)
        task_config = TASK_CONFIGS[task]
        first_text_key = task_config["text_keys"][0]
        if first_text_key in split_data.column_names:
            text_lengths = [len(text.split()) for text in split_data[first_text_key]]
            stats[split_name]["avg_text_length"] = sum(text_lengths) / len(text_lengths)
            stats[split_name]["max_text_length"] = max(text_lengths)
            stats[split_name]["min_text_length"] = min(text_lengths)
    
    return stats





