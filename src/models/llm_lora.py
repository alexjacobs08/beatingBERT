"""LLM LoRA fine-tuning for classification tasks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from pathlib import Path
from tqdm import tqdm
import json
import logging
from typing import Any

from src.prompts.templates import parse_model_output, label_to_id, id_to_label
from src.evaluation.metrics import MetricsTracker, get_memory_usage
from src.utils.reproducibility import set_seed, get_device
from src.utils.config import GemmaConfig, TASK_CONFIGS
from src.data.preprocessing import GemmaPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMDataset(Dataset):
    """Dataset wrapper for LLM training."""
    
    def __init__(self, dataset, preprocessor: GemmaPreprocessor, tokenizer, max_length: int = 256):
        """
        Initialize dataset.
        
        Args:
            dataset: HuggingFace dataset
            preprocessor: Gemma preprocessor
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        # Create prompt
        prompt = self.preprocessor.create_prompt(example)
        
        # Get target label text
        label_text = id_to_label(example["label"], self.preprocessor.task)
        
        # Combine prompt and target
        full_text = f"{prompt} {label_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask prompt tokens, only train on target)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]
        
        labels = encoding["input_ids"].clone()
        labels[:, :prompt_length] = -100  # Mask prompt tokens
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


class LLMLoRATrainer:
    """Trainer for LLMs with LoRA fine-tuning."""
    
    def __init__(
        self,
        config: GemmaConfig,
        output_dir: Path | None = None,
        few_shot_examples: list[dict] | None = None
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save model and results
            few_shot_examples: Few-shot examples (if using few-shot)
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.few_shot_examples = few_shot_examples or []
        
        # Get task info
        self.task_config = TASK_CONFIGS[config.task]
        self.num_labels = self.task_config["num_labels"]
        self.label_names = self.task_config["label_names"]
        
        # Setup
        set_seed()
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {config.model_name}")
        if config.use_fp16 and self.device.type != "cpu":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.base_model.to(self.device)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        
        # Preprocessor
        self.preprocessor = GemmaPreprocessor(
            task=config.task,
            num_few_shot=config.num_few_shot,
            few_shot_examples=self.few_shot_examples
        )
        
        model_display_name = config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name
        logger.info(f"Initialized {model_display_name} LoRA for {config.task}")
        logger.info(f"LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    
    def prepare_data(self, dataset) -> LLMDataset:
        """
        Prepare dataset for training.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            LLMDataset
        """
        return LLMDataset(
            dataset,
            self.preprocessor,
            self.tokenizer,
            max_length=self.config.max_length
        )
    
    def train(self, train_dataset, val_dataset) -> dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training history
        """
        # Prepare data
        logger.info("Preparing datasets...")
        train_data = self.prepare_data(train_dataset)
        val_data = self.prepare_data(val_dataset)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = (len(train_loader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_acc = -1.0  # Start at -1 so even 0.0 accuracy triggers a save
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.4f}")
            
            # Save best model (always save first epoch at minimum)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.output_dir:
                    self.save_model(self.output_dir / "best_adapter")
                    logger.info(f"Saved best model (acc: {best_val_acc:.4f})")
        
        # Save final model
        if self.output_dir:
            self.save_model(self.output_dir / "final_adapter")
        
        return history
    
    def _train_epoch(self, train_loader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader) -> tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions (argmax over vocabulary for the last non-masked token)
                logits = outputs.logits
                labels = batch["labels"]
                
                # Find last non-masked position for each example
                for i in range(labels.shape[0]):
                    non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(non_masked) > 0:
                        # Use the last non-masked position
                        last_pos = non_masked[-1]
                        pred_token = torch.argmax(logits[i, last_pos])
                        true_token = labels[i, last_pos]
                        
                        if pred_token == true_token:
                            correct += 1
                        total += 1
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def predict(self, dataset) -> tuple[list[int], dict[str, Any]]:
        """
        Make predictions using the fine-tuned model.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model.eval()
        
        predictions = []
        true_labels = []
        tracker = MetricsTracker()
        tracker.start_timer()
        
        for example in tqdm(dataset, desc="Predicting"):
            import time
            start = time.time()
            
            # Create prompt
            prompt = self.preprocessor.create_prompt(example)
            
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
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse
            parsed_label = parse_model_output(generated_text, self.config.task)
            
            if parsed_label is None:
                pred_id = 0  # Default
            else:
                try:
                    pred_id = label_to_id(parsed_label, self.config.task)
                except (KeyError, ValueError):
                    pred_id = 0
            
            predictions.append(pred_id)
            
            latency = time.time() - start
            
            if "label" in example:
                true_labels.append(example["label"])
                tracker.add_batch(
                    [pred_id],
                    [example["label"]],
                    latency=latency,
                    memory=get_memory_usage()
                )
        
        tracker.end_timer()
        metrics = tracker.compute_metrics()
        
        return predictions, metrics
    
    def save_model(self, path: Path) -> None:
        """
        Save LoRA adapter and config.
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_path = path / "training_config.json"
        config_dict = self.config.to_dict()
        config_dict["few_shot_examples"] = self.few_shot_examples
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved LoRA adapter to {path}")
    
    def load_adapter(self, path: Path) -> None:
        """
        Load LoRA adapter.
        
        Args:
            path: Load path
        """
        path = Path(path)
        
        # Load adapter onto base model (convert Path to string for peft)
        self.model = PeftModel.from_pretrained(self.base_model, str(path))
        self.model.to(self.device)
        
        logger.info(f"Loaded LoRA adapter from {path}")

