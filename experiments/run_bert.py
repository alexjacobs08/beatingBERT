"""CLI for running BERT/DeBERTa experiments."""

import argparse
from pathlib import Path
import json
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_trainer import BERTTrainer
from src.data.loader import load_glue_dataset
from src.utils.config import BERTConfig, MODELS_DIR, RESULTS_DIR, ensure_dirs
from src.evaluation.metrics import print_metrics
from src.evaluation.error_analysis import save_error_analysis_report
from src.utils.reproducibility import print_system_info
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BERT/DeBERTa on GLUE tasks")
    
    # Model and task
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "microsoft/deberta-v3-base", "textattack/bert-base-uncased-SST-2"],
        help="Model to use"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sst2", "mnli", "mnli_matched", "mnli_mismatched", "rte", "qqp", "mrpc",
                 "anli_r1", "anli_r2", "anli_r3", "hellaswag", "winogrande", "arc_challenge", "boolq"],
        help="GLUE task"
    )
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum validation/test samples (for quick testing)")
    
    # Modes
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "predict"],
        help="Mode: train, eval, or predict"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (for eval/predict)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for organizing results"
    )
    
    args = parser.parse_args()
    
    # Create directories
    ensure_dirs()
    
    # Print system info
    print_system_info()
    
    # Generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = args.model.replace("/", "_")
        run_name = args.run_name or f"{model_name_clean}_{args.task}_{timestamp}"
        args.output_dir = RESULTS_DIR / run_name
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create config
    config = BERTConfig(
        model_name=args.model,
        task=args.task,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    # Save config
    config.save(args.output_dir / "config.json")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.task}")
    dataset = load_glue_dataset(args.task, max_samples=args.max_samples)
    
    # Create trainer
    model_output_dir = args.output_dir / "model"
    trainer = BERTTrainer(config, output_dir=model_output_dir)
    
    if args.mode == "train":
        # Train
        logger.info("Starting training...")
        history = trainer.train(dataset["train"], dataset["validation"])
        
        # Save history
        history_path = args.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training complete. Saved to {model_output_dir}")
        
        # Evaluate on validation set
        logger.info("\nEvaluating on validation set...")
        trainer.load_model(model_output_dir / "best_model")
        
        val_dataset = dataset["validation"]
        if args.max_eval_samples:
            logger.info(f"Limiting validation to {args.max_eval_samples} samples for quick testing")
            val_dataset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))
        
        val_predictions, val_metrics = trainer.predict(val_dataset)
        
        print_metrics(val_metrics, "Validation Results")
        
        # Save validation results
        results = {
            "config": config.to_dict(),
            "validation_metrics": val_metrics,
            "history": history
        }
        
        results_path = args.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Error analysis
        logger.info("Performing error analysis...")
        from src.utils.config import TASK_CONFIGS
        task_config = TASK_CONFIGS[args.task]
        
        save_error_analysis_report(
            val_predictions,
            val_dataset["label"],
            [dict(val_dataset[i]) for i in range(len(val_dataset))],
            task_config["label_names"],
            args.output_dir / "error_analysis",
            model_name=args.model.replace("/", "_")
        )
        
        # Evaluate on test set if available and has valid labels
        if "test" in dataset and len(dataset["test"]) > 0:
            # Check if test has labels (GLUE test sets have -1 for unlabeled)
            if "label" in dataset["test"].column_names:
                # Check if labels are valid (not all -1)
                test_labels = dataset["test"]["label"]
                has_valid_labels = any(label != -1 for label in test_labels)
                
                if has_valid_labels:
                    logger.info("\nEvaluating on test set...")
                    test_predictions, test_metrics = trainer.predict(dataset["test"])
                    print_metrics(test_metrics, "Test Results")
                    
                    results["test_metrics"] = test_metrics
                    
                    # Save updated results
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                else:
                    logger.info("\nTest set has no valid labels (GLUE submission format). Skipping test evaluation.")
        
        logger.info(f"\nAll results saved to {args.output_dir}")
    
    elif args.mode == "eval":
        # Evaluate existing model
        if args.model_path is None:
            raise ValueError("--model_path required for eval mode")
        
        logger.info(f"Loading model from {args.model_path}")
        trainer.load_model(Path(args.model_path))
        
        logger.info("Evaluating on validation set...")
        val_predictions, val_metrics = trainer.predict(dataset["validation"])
        
        print_metrics(val_metrics, "Validation Results")
        
        # Save results
        results = {
            "config": config.to_dict(),
            "model_path": args.model_path,
            "validation_metrics": val_metrics
        }
        
        results_path = args.output_dir / "eval_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    elif args.mode == "predict":
        # Generate predictions
        if args.model_path is None:
            raise ValueError("--model_path required for predict mode")
        
        logger.info(f"Loading model from {args.model_path}")
        trainer.load_model(Path(args.model_path))
        
        # Predict on test set
        split = "test" if "test" in dataset else "validation"
        logger.info(f"Predicting on {split} set...")
        predictions, metrics = trainer.predict(dataset[split])
        
        print_metrics(metrics, f"{split.capitalize()} Results")
        
        # Save predictions
        predictions_path = args.output_dir / f"predictions_{split}.json"
        with open(predictions_path, 'w') as f:
            json.dump({
                "predictions": predictions,
                "metrics": metrics
            }, f, indent=2)
        
        logger.info(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()

