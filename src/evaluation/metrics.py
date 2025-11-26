"""Evaluation metrics for model comparison."""

import time
import psutil
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from typing import Any
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track performance and efficiency metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.labels = []
        self.latencies = []
        self.memory_usage = []
        self.start_time = None
        self.end_time = None
    
    def add_batch(
        self,
        predictions: list[int] | np.ndarray,
        labels: list[int] | np.ndarray,
        latency: float | None = None,
        memory: float | None = None
    ) -> None:
        """
        Add a batch of predictions and labels.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            latency: Batch inference time in seconds
            memory: Memory usage in GB
        """
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        
        if latency is not None:
            self.latencies.append(latency)
        
        if memory is not None:
            self.memory_usage.append(memory)
    
    def start_timer(self) -> None:
        """Start timing."""
        self.start_time = time.time()
    
    def end_timer(self) -> None:
        """End timing."""
        self.end_time = time.time()
    
    def compute_metrics(self, average: str = "macro") -> dict[str, Any]:
        """
        Compute all metrics.
        
        Args:
            average: Averaging strategy for F1 ('macro', 'micro', 'weighted')
            
        Returns:
            Dictionary of metrics
        """
        if len(self.predictions) == 0:
            logger.warning("No predictions to evaluate")
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Performance metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        }
        
        # Matthews Correlation Coefficient (good for imbalanced datasets)
        try:
            metrics["mcc"] = matthews_corrcoef(labels, predictions)
        except Exception as e:
            logger.warning(f"Could not compute MCC: {e}")
            metrics["mcc"] = None
        
        # Efficiency metrics
        if len(self.latencies) > 0:
            total_samples = len(predictions)
            total_time = sum(self.latencies)
            metrics["avg_latency_ms"] = (total_time / len(self.latencies)) * 1000
            metrics["total_time_s"] = total_time
            metrics["samples_per_second"] = total_samples / total_time if total_time > 0 else 0
        
        if len(self.memory_usage) > 0:
            metrics["peak_memory_gb"] = max(self.memory_usage)
            metrics["avg_memory_gb"] = np.mean(self.memory_usage)
        
        # Wall clock time
        if self.start_time is not None and self.end_time is not None:
            metrics["wall_clock_time_s"] = self.end_time - self.start_time
        
        # Number of samples
        metrics["num_samples"] = len(predictions)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(self.labels, self.predictions)
    
    def get_classification_report(self, target_names: list[str] | None = None) -> str:
        """
        Get detailed classification report.
        
        Args:
            target_names: List of label names
            
        Returns:
            Classification report string
        """
        return classification_report(
            self.labels,
            self.predictions,
            target_names=target_names,
            zero_division=0
        )
    
    def get_errors(self) -> list[tuple[int, int, int]]:
        """
        Get all error cases.
        
        Returns:
            List of (index, true_label, predicted_label) tuples
        """
        errors = []
        for idx, (true_label, pred_label) in enumerate(zip(self.labels, self.predictions)):
            if true_label != pred_label:
                errors.append((idx, true_label, pred_label))
        return errors


def compute_accuracy(predictions: list[int], labels: list[int]) -> float:
    """
    Compute accuracy.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(labels, predictions)


def compute_f1(
    predictions: list[int],
    labels: list[int],
    average: str = "macro"
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        average: Averaging strategy
        
    Returns:
        F1 score
    """
    return f1_score(labels, predictions, average=average, zero_division=0)


def compute_mcc(predictions: list[int], labels: list[int]) -> float:
    """
    Compute Matthews Correlation Coefficient.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        MCC score
    """
    return matthews_corrcoef(labels, predictions)


def measure_inference_time(model_fn, *args, num_warmup: int = 3, **kwargs) -> tuple[Any, float]:
    """
    Measure inference time for a model.
    
    Args:
        model_fn: Model function to call
        *args: Arguments to pass to model
        num_warmup: Number of warmup runs
        **kwargs: Keyword arguments to pass to model
        
    Returns:
        Tuple of (result, time_in_seconds)
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = model_fn(*args, **kwargs)
    
    # Actual timed run
    start = time.time()
    result = model_fn(*args, **kwargs)
    end = time.time()
    
    return result, end - start


def get_memory_usage() -> float:
    """
    Get current memory usage in GB.
    
    Returns:
        Memory usage in GB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # Convert to GB


def get_gpu_memory_usage() -> float:
    """
    Get GPU memory usage in GB.
    
    Returns:
        GPU memory usage in GB, or 0 if no GPU
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory tracking, use system memory
        return get_memory_usage()
    return 0.0


def print_metrics(metrics: dict[str, Any], title: str | None = None) -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Metrics dictionary
        title: Optional title to print
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"{title:^60}")
        print(f"{'=' * 60}")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    if "accuracy" in metrics:
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    if "macro_f1" in metrics:
        print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    if "weighted_f1" in metrics:
        print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    if "mcc" in metrics and metrics["mcc"] is not None:
        print(f"  MCC:          {metrics['mcc']:.4f}")
    
    # Efficiency metrics
    if any(k in metrics for k in ["avg_latency_ms", "samples_per_second", "peak_memory_gb"]):
        print("\nEfficiency Metrics:")
    
    if "avg_latency_ms" in metrics:
        print(f"  Avg Latency:  {metrics['avg_latency_ms']:.2f} ms")
    if "samples_per_second" in metrics:
        print(f"  Throughput:   {metrics['samples_per_second']:.2f} samples/s")
    if "peak_memory_gb" in metrics:
        print(f"  Peak Memory:  {metrics['peak_memory_gb']:.2f} GB")
    if "wall_clock_time_s" in metrics:
        print(f"  Wall Time:    {metrics['wall_clock_time_s']:.2f} s")
    
    if "num_samples" in metrics:
        print(f"\nTotal Samples: {metrics['num_samples']}")
    
    print()





