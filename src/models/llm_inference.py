"""LLM inference for zero-shot and few-shot classification."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
import json
import logging
from typing import Any

from src.data.preprocessing import GemmaPreprocessor
from src.prompts.templates import parse_model_output, label_to_id, id_to_label
from src.evaluation.metrics import MetricsTracker, get_memory_usage
from src.utils.reproducibility import set_seed, get_device
from src.utils.config import GemmaConfig, TASK_CONFIGS
from src.data.loader import get_few_shot_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInference:
    """Inference engine for LLMs on classification tasks."""
    
    def __init__(
        self,
        config: GemmaConfig,
        few_shot_examples: list[dict] | None = None
    ):
        """
        Initialize LLM inference.
        
        Args:
            config: Inference configuration
            few_shot_examples: Examples for few-shot learning (if num_few_shot > 0)
        """
        self.config = config
        self.few_shot_examples = few_shot_examples or []
        
        # Get task info
        self.task_config = TASK_CONFIGS[config.task]
        self.num_labels = self.task_config["num_labels"]
        self.label_names = self.task_config["label_names"]
        
        # Setup
        set_seed()
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        model_display_name = config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name
        logger.info(f"Loading {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Load model with appropriate precision
        if config.use_fp16 and self.device.type != "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.model.to(self.device)
        
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = GemmaPreprocessor(
            task=config.task,
            num_few_shot=config.num_few_shot,
            few_shot_examples=self.few_shot_examples
        )
        
        model_display_name = config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name
        logger.info(f"Initialized {model_display_name} for {config.task}")
        logger.info(f"Mode: {'Zero-shot' if config.num_few_shot == 0 else f'{config.num_few_shot}-shot'}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode (only new tokens)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def predict_single(self, example: dict[str, Any]) -> tuple[int | None, str]:
        """
        Predict label for a single example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Tuple of (predicted_label_id, raw_output)
        """
        # Create prompt
        prompt = self.preprocessor.create_prompt(example)
        
        # Generate
        output = self.generate(prompt)
        
        # Parse output
        parsed_label = parse_model_output(output, self.config.task)
        
        if parsed_label is None:
            return None, output
        
        # Convert to ID
        try:
            label_id = label_to_id(parsed_label, self.config.task)
            return label_id, output
        except (KeyError, ValueError):
            return None, output
    
    def predict(self, dataset, return_raw: bool = False) -> tuple[list[int], dict[str, Any], list[str] | None]:
        """
        Predict labels for entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            return_raw: Whether to return raw model outputs
            
        Returns:
            Tuple of (predictions, metrics, raw_outputs)
        """
        logger.info(f"Running inference on {len(dataset)} examples...")
        
        predictions = []
        raw_outputs = [] if return_raw else None
        failed_parses = 0
        
        tracker = MetricsTracker()
        tracker.start_timer()
        
        for example in tqdm(dataset, desc="Predicting"):
            import time
            start = time.time()
            
            pred_id, raw_output = self.predict_single(example)
            
            latency = time.time() - start
            
            if pred_id is None:
                # Failed to parse - use random/default
                failed_parses += 1
                pred_id = 0  # Default to first label
            
            predictions.append(pred_id)
            
            if return_raw:
                raw_outputs.append(raw_output)
            
            # Track metrics (if we have labels)
            if "label" in example:
                tracker.add_batch(
                    [pred_id],
                    [example["label"]],
                    latency=latency,
                    memory=get_memory_usage()
                )
        
        tracker.end_timer()
        
        # Compute metrics
        metrics = tracker.compute_metrics()
        metrics["failed_parses"] = failed_parses
        metrics["parse_success_rate"] = 1 - (failed_parses / len(dataset))
        
        logger.info(f"Failed parses: {failed_parses}/{len(dataset)} ({failed_parses/len(dataset)*100:.1f}%)")
        
        return predictions, metrics, raw_outputs
    
    def predict_batch(
        self,
        examples: list[dict[str, Any]],
        batch_size: int | None = None
    ) -> list[tuple[int | None, str]]:
        """
        Predict labels for a batch of examples.
        
        Args:
            examples: List of examples
            batch_size: Batch size (if None, use config.batch_size)
            
        Returns:
            List of (predicted_label_id, raw_output) tuples
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        for i in tqdm(range(0, len(examples), batch_size), desc="Batch prediction"):
            batch = examples[i:i + batch_size]
            
            for example in batch:
                pred_id, raw_output = self.predict_single(example)
                results.append((pred_id, raw_output))
        
        return results
    
    def evaluate(self, dataset) -> dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Args:
            dataset: Dataset with labels
            
        Returns:
            Evaluation metrics
        """
        predictions, metrics, _ = self.predict(dataset, return_raw=False)
        return metrics
    
    def save_config(self, path: Path) -> None:
        """
        Save configuration.
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.config.to_dict()
        config_dict["few_shot_examples"] = self.few_shot_examples
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def create_llm_inference(
    task: str,
    model_name: str = "google/gemma-3-1b-it",
    num_few_shot: int = 0,
    train_dataset: Any | None = None,
    temperature: float = 0.0,
    use_fp16: bool = True
) -> LLMInference:
    """
    Create an LLM inference engine with proper setup.
    
    Args:
        task: Task name
        model_name: HuggingFace model name
        num_few_shot: Number of few-shot examples
        train_dataset: Training dataset (required if num_few_shot > 0)
        temperature: Sampling temperature
        use_fp16: Whether to use FP16
        
    Returns:
        Configured GemmaInference instance
    """
    # Get few-shot examples if needed
    few_shot_examples = []
    if num_few_shot > 0:
        if train_dataset is None:
            raise ValueError("train_dataset required for few-shot learning")
        few_shot_examples = get_few_shot_examples(
            train_dataset,
            num_examples=num_few_shot
        )
        logger.info(f"Selected {len(few_shot_examples)} few-shot examples")
    
    # Create config
    config = GemmaConfig(
        model_name=model_name,
        task=task,
        temperature=temperature,
        num_few_shot=num_few_shot,
        use_fp16=use_fp16
    )
    
    # Create inference engine
    return LLMInference(config, few_shot_examples)

