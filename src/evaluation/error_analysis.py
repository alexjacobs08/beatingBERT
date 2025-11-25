"""Error analysis utilities for comparing models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_confusion_matrix_plot(
    true_labels: list[int],
    pred_labels: list[int],
    label_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Path | None = None
) -> None:
    """
    Create and save a confusion matrix plot.
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        label_names: Names of labels for display
        title: Plot title
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.close()


def analyze_error_overlap(
    model1_predictions: list[int],
    model2_predictions: list[int],
    true_labels: list[int],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> dict[str, Any]:
    """
    Analyze error overlap between two models.
    
    Args:
        model1_predictions: Predictions from model 1
        model2_predictions: Predictions from model 2
        true_labels: True labels
        model1_name: Name of model 1
        model2_name: Name of model 2
        
    Returns:
        Dictionary with overlap statistics
    """
    model1_errors = set()
    model2_errors = set()
    
    for idx, (pred1, pred2, true) in enumerate(zip(
        model1_predictions, model2_predictions, true_labels
    )):
        if pred1 != true:
            model1_errors.add(idx)
        if pred2 != true:
            model2_errors.add(idx)
    
    # Compute overlaps
    both_correct = len([
        i for i in range(len(true_labels))
        if i not in model1_errors and i not in model2_errors
    ])
    both_wrong = len(model1_errors & model2_errors)
    only_model1_wrong = len(model1_errors - model2_errors)
    only_model2_wrong = len(model2_errors - model1_errors)
    
    total = len(true_labels)
    
    results = {
        "total_examples": total,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_model1_wrong": only_model1_wrong,
        "only_model2_wrong": only_model2_wrong,
        f"{model1_name}_errors": len(model1_errors),
        f"{model2_name}_errors": len(model2_errors),
        f"{model1_name}_accuracy": 1 - (len(model1_errors) / total),
        f"{model2_name}_accuracy": 1 - (len(model2_errors) / total),
        "error_overlap_ratio": both_wrong / max(len(model1_errors), 1),
        "model1_error_indices": sorted(model1_errors),
        "model2_error_indices": sorted(model2_errors),
        "shared_error_indices": sorted(model1_errors & model2_errors),
    }
    
    return results


def create_error_overlap_venn(
    overlap_stats: dict[str, Any],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    save_path: Path | None = None
) -> None:
    """
    Create a Venn diagram-style visualization of error overlap.
    
    Args:
        overlap_stats: Statistics from analyze_error_overlap
        model1_name: Name of model 1
        model2_name: Name of model 2
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = [
        'Both\nCorrect',
        f'Only {model1_name}\nWrong',
        f'Only {model2_name}\nWrong',
        'Both\nWrong'
    ]
    
    counts = [
        overlap_stats['both_correct'],
        overlap_stats['only_model1_wrong'],
        overlap_stats['only_model2_wrong'],
        overlap_stats['both_wrong']
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count}\n({count/overlap_stats["total_examples"]*100:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_ylabel('Number of Examples', fontsize=12)
    ax.set_title(
        f'Error Analysis: {model1_name} vs {model2_name}\n'
        f'Total Examples: {overlap_stats["total_examples"]}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error overlap plot to {save_path}")
    
    plt.close()


def get_top_errors(
    predictions: list[int],
    true_labels: list[int],
    examples: list[dict[str, Any]],
    n: int = 10
) -> pd.DataFrame:
    """
    Get top N error cases.
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        examples: List of example dictionaries
        n: Number of errors to return
        
    Returns:
        DataFrame with error cases
    """
    errors = []
    
    for idx, (pred, true, example) in enumerate(zip(predictions, true_labels, examples)):
        if pred != true:
            error_info = {
                'index': idx,
                'true_label': true,
                'predicted_label': pred,
            }
            # Add text fields from example
            error_info.update(example)
            errors.append(error_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(errors)
    
    # Return top N
    return df.head(n) if len(df) > n else df


def create_error_distribution_plot(
    model_errors: dict[str, list[int]],
    true_labels: list[int],
    label_names: list[str],
    save_path: Path | None = None
) -> None:
    """
    Create a plot showing error distribution across labels for multiple models.
    
    Args:
        model_errors: Dictionary mapping model names to their error indices
        true_labels: True labels
        label_names: Names of labels
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate errors per label for each model
    x = np.arange(len(label_names))
    width = 0.8 / len(model_errors)
    
    for i, (model_name, error_indices) in enumerate(model_errors.items()):
        errors_per_label = []
        total_per_label = []
        
        for label_id in range(len(label_names)):
            # Count errors for this label
            label_errors = sum(
                1 for idx in error_indices
                if true_labels[idx] == label_id
            )
            # Count total examples for this label
            label_total = sum(1 for l in true_labels if l == label_id)
            
            errors_per_label.append(label_errors)
            total_per_label.append(label_total)
        
        # Calculate error rates
        error_rates = [
            (e / t * 100) if t > 0 else 0
            for e, t in zip(errors_per_label, total_per_label)
        ]
        
        offset = width * i
        ax.bar(
            x + offset,
            error_rates,
            width,
            label=model_name,
            alpha=0.8
        )
    
    ax.set_xlabel('Label', fontsize=12)
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Distribution Across Labels', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(model_errors) - 1) / 2)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")
    
    plt.close()


def print_error_overlap_summary(overlap_stats: dict[str, Any]) -> None:
    """
    Print a summary of error overlap statistics.
    
    Args:
        overlap_stats: Statistics from analyze_error_overlap
    """
    print("\n" + "=" * 60)
    print("Error Overlap Analysis")
    print("=" * 60)
    
    print(f"\nTotal Examples: {overlap_stats['total_examples']}")
    print(f"\nBoth Correct:   {overlap_stats['both_correct']} "
          f"({overlap_stats['both_correct']/overlap_stats['total_examples']*100:.1f}%)")
    print(f"Both Wrong:     {overlap_stats['both_wrong']} "
          f"({overlap_stats['both_wrong']/overlap_stats['total_examples']*100:.1f}%)")
    print(f"Only Model 1:   {overlap_stats['only_model1_wrong']} "
          f"({overlap_stats['only_model1_wrong']/overlap_stats['total_examples']*100:.1f}%)")
    print(f"Only Model 2:   {overlap_stats['only_model2_wrong']} "
          f"({overlap_stats['only_model2_wrong']/overlap_stats['total_examples']*100:.1f}%)")
    
    print(f"\nModel 1 Errors: {overlap_stats.get('Model 1_errors', 'N/A')}")
    print(f"Model 2 Errors: {overlap_stats.get('Model 2_errors', 'N/A')}")
    print(f"Error Overlap:  {overlap_stats['error_overlap_ratio']*100:.1f}%")
    print()


def save_error_analysis_report(
    predictions: list[int],
    true_labels: list[int],
    examples: list[dict[str, Any]],
    label_names: list[str],
    output_dir: Path,
    model_name: str = "model"
) -> None:
    """
    Save a comprehensive error analysis report.
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        examples: List of examples
        label_names: Label names
        output_dir: Output directory
        model_name: Model name for file naming
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top errors
    errors_df = get_top_errors(predictions, true_labels, examples, n=50)
    
    # Save to CSV
    errors_csv_path = output_dir / f"{model_name}_errors.csv"
    errors_df.to_csv(errors_csv_path, index=False)
    logger.info(f"Saved errors to {errors_csv_path}")
    
    # Create confusion matrix
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    create_confusion_matrix_plot(
        true_labels,
        predictions,
        label_names,
        title=f"Confusion Matrix - {model_name}",
        save_path=cm_path
    )




