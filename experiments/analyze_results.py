"""Analyze and compare experimental results."""

import argparse
from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import RESULTS_DIR
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """
    Load all results.json files from a directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        List of result dictionaries with metadata
    """
    results = []
    
    # Find all results.json files
    for result_file in results_dir.rglob("results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Add metadata
            data["result_path"] = str(result_file.parent)
            data["run_name"] = result_file.parent.name
            
            results.append(data)
            logger.info(f"Loaded: {result_file.parent.name}")
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
    
    return results


def extract_metrics(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Extract metrics into a DataFrame.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with organized metrics
    """
    rows = []
    
    for result in results:
        config = result.get("config", {})
        val_metrics = result.get("validation_metrics", {})
        test_metrics = result.get("test_metrics", {})
        
        # Extract model info
        model_name = config.get("model_name", "unknown")
        task = config.get("task", "unknown")
        
        # Determine method
        if "lora_rank" in config and config.get("lora_rank"):
            method = f"LoRA (r={config['lora_rank']})"
        elif config.get("num_few_shot", 0) > 0:
            method = f"Few-shot (k={config['num_few_shot']})"
        elif "gemma" in model_name.lower():
            method = "Zero-shot"
        else:
            method = "Fine-tuned"
        
        # Simplify model name
        if "bert-base" in model_name:
            model_simple = "BERT-base"
        elif "deberta-v3-base" in model_name:
            model_simple = "DeBERTa-v3"
        elif "gemma" in model_name.lower():
            model_simple = "Gemma"
        else:
            model_simple = model_name.split("/")[-1]
        
        row = {
            "Model": model_simple,
            "Method": method,
            "Task": task,
            "Run": result["run_name"],
            # Validation metrics
            "Val Accuracy": val_metrics.get("accuracy"),
            "Val Macro F1": val_metrics.get("macro_f1"),
            "Val MCC": val_metrics.get("mcc"),
            "Latency (ms)": val_metrics.get("avg_latency_ms"),
            "Peak Memory (GB)": val_metrics.get("peak_memory_gb"),
            "Throughput (samples/s)": val_metrics.get("samples_per_second"),
            # Test metrics (if available)
            "Test Accuracy": test_metrics.get("accuracy"),
            "Test Macro F1": test_metrics.get("macro_f1"),
            "Test MCC": test_metrics.get("mcc"),
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def create_performance_table(df: pd.DataFrame, output_path: Path | None = None) -> str:
    """
    Create a performance comparison table.
    
    Args:
        df: DataFrame with metrics
        output_path: Path to save table
        
    Returns:
        Markdown table string
    """
    # Group by Model and Method
    perf_cols = ["Model", "Method", "Task", "Val Accuracy", "Val Macro F1", "Val MCC"]
    perf_df = df[perf_cols].copy()
    
    # Format percentages
    for col in ["Val Accuracy", "Val Macro F1"]:
        if col in perf_df.columns:
            perf_df[col] = perf_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    
    if "Val MCC" in perf_df.columns:
        perf_df["Val MCC"] = perf_df["Val MCC"].apply(lambda x: f"{x:.4f}" if pd.notna(x) and x is not None else "-")
    
    # Convert to markdown
    md_table = perf_df.to_markdown(index=False)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("# Performance Comparison\n\n")
            f.write(md_table)
        logger.info(f"Saved performance table to {output_path}")
    
    return md_table


def create_efficiency_table(df: pd.DataFrame, output_path: Path | None = None) -> str:
    """
    Create an efficiency comparison table.
    
    Args:
        df: DataFrame with metrics
        output_path: Path to save table
        
    Returns:
        Markdown table string
    """
    eff_cols = ["Model", "Method", "Latency (ms)", "Peak Memory (GB)", "Throughput (samples/s)"]
    eff_df = df[eff_cols].copy()
    
    # Format numbers
    for col in ["Latency (ms)", "Peak Memory (GB)", "Throughput (samples/s)"]:
        if col in eff_df.columns:
            eff_df[col] = eff_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    # Convert to markdown
    md_table = eff_df.to_markdown(index=False)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("# Efficiency Comparison\n\n")
            f.write(md_table)
        logger.info(f"Saved efficiency table to {output_path}")
    
    return md_table


def create_comparison_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comparison plots.
    
    Args:
        df: DataFrame with metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy comparison
    if "Val Accuracy" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar plot
        df_plot = df[df["Val Accuracy"].notna()].copy()
        if len(df_plot) > 0:
            x_labels = [f"{row['Model']}\n{row['Method']}" for _, row in df_plot.iterrows()]
            colors = ['#3498db' if 'BERT' in model else '#e74c3c' for model in df_plot["Model"]]
            
            bars = ax.bar(range(len(df_plot)), df_plot["Val Accuracy"], color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(df_plot)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_ylabel('Validation Accuracy', fontsize=12)
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, df_plot["Val Accuracy"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved accuracy comparison to {output_dir / 'accuracy_comparison.png'}")
    
    # Plot 2: Accuracy vs Latency tradeoff
    if "Val Accuracy" in df.columns and "Latency (ms)" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df_plot = df[(df["Val Accuracy"].notna()) & (df["Latency (ms)"].notna())].copy()
        if len(df_plot) > 0:
            for model in df_plot["Model"].unique():
                model_data = df_plot[df_plot["Model"] == model]
                ax.scatter(model_data["Latency (ms)"], model_data["Val Accuracy"], 
                          label=model, s=100, alpha=0.7)
            
            ax.set_xlabel('Latency (ms)', fontsize=12)
            ax.set_ylabel('Validation Accuracy', fontsize=12)
            ax.set_title('Accuracy vs Latency Tradeoff', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_vs_latency.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved accuracy vs latency plot to {output_dir / 'accuracy_vs_latency.png'}")
    
    # Plot 3: F1 Score comparison
    if "Val Macro F1" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_plot = df[df["Val Macro F1"].notna()].copy()
        if len(df_plot) > 0:
            x_labels = [f"{row['Model']}\n{row['Method']}" for _, row in df_plot.iterrows()]
            colors = ['#2ecc71' if 'BERT' in model else '#f39c12' for model in df_plot["Model"]]
            
            bars = ax.bar(range(len(df_plot)), df_plot["Val Macro F1"], color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(df_plot)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_ylabel('Validation Macro F1', fontsize=12)
            ax.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, df_plot["Val Macro F1"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "f1_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved F1 comparison to {output_dir / 'f1_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing results (default: all results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results/analysis)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["all", "tables", "plots", "csv"],
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Set directories
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading results from: {results_dir}")
    
    # Load results
    results = load_results(results_dir)
    
    if len(results) == 0:
        logger.error("No results found!")
        return
    
    logger.info(f"Loaded {len(results)} result files")
    
    # Extract metrics
    df = extract_metrics(results)
    
    # Save raw DataFrame
    if args.format in ["all", "csv"]:
        csv_path = output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved raw results to {csv_path}")
    
    # Create performance table
    if args.format in ["all", "tables"]:
        perf_table = create_performance_table(df, output_dir / "performance_table.md")
        print("\n" + perf_table + "\n")
        
        # Create efficiency table
        eff_table = create_efficiency_table(df, output_dir / "efficiency_table.md")
        print("\n" + eff_table + "\n")
    
    # Create plots
    if args.format in ["all", "plots"]:
        create_comparison_plots(df, output_dir / "plots")
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()





