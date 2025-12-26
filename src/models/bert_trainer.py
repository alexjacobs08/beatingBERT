"""BERT/DeBERTa trainer for GLUE tasks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import json
import logging
from typing import Any

from src.data.preprocessing import BERTPreprocessor
from src.evaluation.metrics import MetricsTracker, get_memory_usage
from src.utils.reproducibility import set_seed, get_device
from src.utils.config import BERTConfig, TASK_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTTrainer:
    """Trainer for BERT/DeBERTa models on classification tasks."""
    
    def __init__(
        self,
        config: BERTConfig,
        output_dir: Path | None = None
    ):
        """
        Initialize BERT trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save model and results
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get task info
        self.task_config = TASK_CONFIGS[config.task]
        self.num_labels = self.task_config["num_labels"]
        
        # Setup
        set_seed()
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Preprocessor
        self.preprocessor = BERTPreprocessor(
            self.tokenizer,
            max_length=config.max_length
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        logger.info(f"Initialized {config.model_name} for {config.task}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, dataset):
        """
        Prepare dataset for training.
        
        Args:
            dataset: HuggingFace dataset
            
        Returns:
            Processed dataset
        """
        return self.preprocessor.preprocess_dataset(dataset, self.config.task)
    
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
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }
        
        best_val_acc = 0.0
        
        self.metrics_tracker.start_timer()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history["train_loss"].append(train_loss)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            history["val_loss"].append(val_metrics.get("loss", 0.0))
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["macro_f1"])
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['macro_f1']:.4f}")
            
            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                if self.output_dir:
                    self.save_model(self.output_dir / "best_model")
                    logger.info(f"Saved best model (acc: {best_val_acc:.4f})")
        
        self.metrics_tracker.end_timer()
        
        # Save final model
        if self.output_dir:
            self.save_model(self.output_dir / "final_model")
        
        return history
    
    def _train_epoch(self, train_loader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (progress_bar.n + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Finalize any remaining gradients from incomplete accumulation
        if (progress_bar.n + 1) % self.config.gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return total_loss / len(train_loader)
    
    def evaluate(self, dataloader) -> dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        tracker = MetricsTracker()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Measure time and memory
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                import time
                cpu_start = time.time()
                
                if start_time:
                    start_time.record()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    latency = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                else:
                    latency = time.time() - cpu_start
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Track metrics
                tracker.add_batch(
                    predictions.cpu().numpy(),
                    labels.cpu().numpy(),
                    latency=latency,
                    memory=get_memory_usage()
                )
                
                total_loss += loss.item()
        
        # Compute metrics
        metrics = tracker.compute_metrics()
        metrics["loss"] = total_loss / len(dataloader)
        
        return metrics
    
    def predict(self, dataset) -> tuple[list[int], dict[str, Any]]:
        """
        Make predictions on a dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Tuple of (predictions, metrics)
        """
        # Prepare data
        processed_data = self.prepare_data(dataset)
        dataloader = DataLoader(
            processed_data,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Evaluate
        metrics = self.evaluate(dataloader)
        
        # Get predictions from metrics tracker
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy().tolist())
        
        return predictions, metrics
    
    def save_model(self, path: Path) -> None:
        """
        Save model and tokenizer.
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_path = path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def load_model(self, path: Path) -> None:
        """
        Load model and tokenizer.
        
        Args:
            path: Load path
        """
        path = Path(path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        logger.info(f"Loaded model from {path}")


def load_pretrained_bert(
    model_name: str,
    task: str,
    device: torch.device | None = None
) -> tuple[nn.Module, AutoTokenizer]:
    """
    Load a pre-trained BERT model for a specific task.
    
    Args:
        model_name: HuggingFace model name
        task: Task name
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    task_config = TASK_CONFIGS[task]
    num_labels = task_config["num_labels"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    
    return model, tokenizer

