"""CLI for running LLM experiments (zero-shot, few-shot, LoRA) with any causal language model."""

import argparse
from pathlib import Path
import json
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm_inference import create_llm_inference
from src.models.llm_lora import LLMLoRATrainer
from src.models.llm_dspy import LLMDSPyBase  # Only used for DSPy optimization
from src.data.loader import load_glue_dataset, get_few_shot_examples
from src.utils.config import GemmaConfig, MODELS_DIR, RESULTS_DIR, PROMPTS_DIR, ensure_dirs
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
    parser = argparse.ArgumentParser(description="Run LLM experiments on GLUE tasks (Gemma, TinyLlama, Qwen, etc.)")
    
    # Model and task
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model to use (e.g., google/gemma-3-1b-it, TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2-0.5B-Instruct)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sst2", "mnli", "mnli_matched", "mnli_mismatched", "rte", "qqp", "mrpc",
                 "anli_r1", "anli_r2", "anli_r3", "hellaswag", "winogrande", "arc_challenge", "boolq"],
        help="GLUE task"
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["zero-shot", "few-shot", "lora", "dspy"],
        help="Inference mode"
    )
    
    # Few-shot / LoRA / DSPy parameters
    parser.add_argument(
        "--num_few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples (for few-shot, lora, and dspy modes)"
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=300,
        help="Max training examples for DSPy optimization (expensive)"
    )
    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=100,
        help="Max validation examples for DSPy optimization"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        choices=[8, 16, 32],
        help="LoRA rank (for lora mode)"
    )
    
    # Inference parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum validation/test samples (for quick testing)")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max new tokens to generate (3 for single token)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16")
    
    # LoRA training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (for LoRA)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (for LoRA)")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length (for LoRA)")
    
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
    
    # Load adapter (for LoRA inference only)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (for inference with pretrained adapter)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    ensure_dirs()
    
    # Print system info
    print_system_info()
    
    # Log model being used
    logger.info(f"Model: {args.model}")
    if args.mode == "dspy":
        logger.info(f"DSPy will optimize prompts FOR this model and USE this model")
    
    # Generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = args.model.split('/')[-1] if '/' in args.model else args.model
        model_name_clean = model_name_clean.replace("-", "_").lower()
        
        mode_suffix = f"{args.mode}_k{args.num_few_shot}"
        if args.mode == "lora":
            mode_suffix += f"_r{args.lora_rank}"
        run_name = args.run_name or f"{model_name_clean}_{mode_suffix}_{args.task}_{timestamp}"
        args.output_dir = RESULTS_DIR / run_name
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.task}")
    dataset = load_glue_dataset(args.task, max_samples=args.max_samples)
    
    if args.mode == "zero-shot":
        # Zero-shot inference with transformers
        logger.info("Running zero-shot inference...")
        
        config = GemmaConfig(
            model_name=args.model,
            task=args.task,
            temperature=args.temperature,
            num_few_shot=0,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_fp16=not args.no_fp16
        )
        
        config.save(args.output_dir / "config.json")
        
        # Create inference engine with transformers
        inference = create_llm_inference(
            task=args.task,
            model_name=args.model,
            num_few_shot=0,
            temperature=args.temperature,
            use_fp16=not args.no_fp16
        )
        
        # Save prompts for reproducibility
        from src.prompts.templates import ZERO_SHOT_PROMPTS
        prompt_file = PROMPTS_DIR / f"zero_shot_{args.task}.txt"
        prompt_file.parent.mkdir(exist_ok=True)
        with open(prompt_file, 'w') as f:
            f.write(ZERO_SHOT_PROMPTS[args.task])
        logger.info(f"Saved prompt template to {prompt_file}")
        
        # Predict on validation set
        logger.info("Evaluating on validation set...")
        val_dataset = dataset["validation"]
        if args.max_eval_samples:
            logger.info(f"Limiting validation to {args.max_eval_samples} samples for quick testing")
            val_dataset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))
        
        predictions, metrics, raw_outputs = inference.predict(
            val_dataset,
            return_raw=True
        )
        
        print_metrics(metrics, "Validation Results")
        
        # Save results
        results = {
            "config": config.to_dict(),
            "validation_metrics": metrics,
            "predictions": predictions
        }
        
        results_path = args.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save raw outputs (sample)
        raw_outputs_path = args.output_dir / "raw_outputs_sample.json"
        with open(raw_outputs_path, 'w') as f:
            json.dump(raw_outputs[:100], f, indent=2)  # Save first 100
        
        # Error analysis
        logger.info("Performing error analysis...")
        from src.utils.config import TASK_CONFIGS
        task_config = TASK_CONFIGS[args.task]
        
        model_name_clean = args.model.split('/')[-1] if '/' in args.model else args.model
        model_name_clean = model_name_clean.replace("-", "_").lower()
        
        save_error_analysis_report(
            predictions,
            val_dataset["label"],
            [dict(val_dataset[i]) for i in range(len(val_dataset))],
            task_config["label_names"],
            args.output_dir / "error_analysis",
            model_name=f"{model_name_clean}_zero_shot"
        )
    
    elif args.mode == "few-shot":
        # Few-shot inference with transformers
        logger.info(f"Running {args.num_few_shot}-shot inference...")
        
        config = GemmaConfig(
            model_name=args.model,
            task=args.task,
            temperature=args.temperature,
            num_few_shot=args.num_few_shot,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_fp16=not args.no_fp16
        )
        
        config.save(args.output_dir / "config.json")
        
        # Get few-shot examples
        few_shot_examples = get_few_shot_examples(
            dataset["train"],
            num_examples=args.num_few_shot
        )
        
        # Save few-shot examples
        few_shot_file = PROMPTS_DIR / f"few_shot_{args.task}_k{args.num_few_shot}.json"
        few_shot_file.parent.mkdir(exist_ok=True)
        with open(few_shot_file, 'w') as f:
            json.dump(few_shot_examples, f, indent=2)
        logger.info(f"Saved few-shot examples to {few_shot_file}")
        
        # Create inference engine with transformers
        inference = create_llm_inference(
            task=args.task,
            model_name=args.model,
            num_few_shot=args.num_few_shot,
            train_dataset=dataset["train"],
            temperature=args.temperature,
            use_fp16=not args.no_fp16
        )
        
        # Predict on validation set
        logger.info("Evaluating on validation set...")
        val_dataset = dataset["validation"]
        if args.max_eval_samples:
            logger.info(f"Limiting validation to {args.max_eval_samples} samples for quick testing")
            val_dataset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))
        
        predictions, metrics, raw_outputs = inference.predict(
            val_dataset,
            return_raw=True
        )
        
        print_metrics(metrics, "Validation Results")
        
        # Save results
        results = {
            "config": config.to_dict(),
            "validation_metrics": metrics,
            "predictions": predictions,
            "few_shot_examples": few_shot_examples
        }
        
        results_path = args.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save raw outputs (sample)
        raw_outputs_path = args.output_dir / "raw_outputs_sample.json"
        with open(raw_outputs_path, 'w') as f:
            json.dump(raw_outputs[:100], f, indent=2)
        
        # Error analysis
        logger.info("Performing error analysis...")
        from src.utils.config import TASK_CONFIGS
        task_config = TASK_CONFIGS[args.task]
        
        model_name_clean = args.model.split('/')[-1] if '/' in args.model else args.model
        model_name_clean = model_name_clean.replace("-", "_").lower()
        
        save_error_analysis_report(
            predictions,
            val_dataset["label"],
            [dict(val_dataset[i]) for i in range(len(val_dataset))],
            task_config["label_names"],
            args.output_dir / "error_analysis",
            model_name=f"{model_name_clean}_{args.num_few_shot}_shot"
        )
    
    elif args.mode == "lora":
        # LoRA fine-tuning
        if args.adapter_path is None:
            # Train mode
            logger.info(f"Training Gemma with LoRA (rank={args.lora_rank})...")
            
            # Get few-shot examples if specified
            few_shot_examples = []
            if args.num_few_shot > 0:
                few_shot_examples = get_few_shot_examples(
                    dataset["train"],
                    num_examples=args.num_few_shot
                )
            
            config = GemmaConfig(
                model_name=args.model,
                task=args.task,
                num_few_shot=args.num_few_shot,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                learning_rate=args.lr,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                use_fp16=not args.no_fp16
            )
            
            config.save(args.output_dir / "config.json")
            
            # Create trainer
            model_output_dir = args.output_dir / "adapter"
            trainer = LLMLoRATrainer(
                config,
                output_dir=model_output_dir,
                few_shot_examples=few_shot_examples
            )
            
            # Train
            logger.info("Starting training...")
            history = trainer.train(dataset["train"], dataset["validation"])
            
            # Save history
            history_path = args.output_dir / "history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Training complete. Adapter saved to {model_output_dir}")
            
            # Load best adapter and evaluate
            logger.info("\nEvaluating best adapter on validation set...")
            trainer.load_adapter(model_output_dir / "best_adapter")
            val_predictions, val_metrics = trainer.predict(dataset["validation"])
            
            print_metrics(val_metrics, "Validation Results")
            
            # Save results
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
            
            model_name_clean = args.model.split('/')[-1] if '/' in args.model else args.model
            model_name_clean = model_name_clean.replace("-", "_").lower()
            
            save_error_analysis_report(
                val_predictions,
                dataset["validation"]["label"],
                [dict(dataset["validation"][i]) for i in range(len(dataset["validation"]))],
                task_config["label_names"],
                args.output_dir / "error_analysis",
                model_name=f"{model_name_clean}_lora_r{args.lora_rank}"
            )
            
            logger.info(f"\nAll results saved to {args.output_dir}")
        
        else:
            # Inference mode with pretrained adapter
            logger.info(f"Loading LoRA adapter from {args.adapter_path}")
            
            config = GemmaConfig(
                model_name=args.model,
                task=args.task,
                lora_rank=args.lora_rank,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                use_fp16=not args.no_fp16
            )
            
            # Create trainer and load adapter
            trainer = LLMLoRATrainer(config)
            trainer.load_adapter(Path(args.adapter_path))
            
            # Evaluate
            logger.info("Evaluating on validation set...")
            predictions, metrics = trainer.predict(dataset["validation"])
            
            print_metrics(metrics, "Validation Results")
            
            # Save results
            results = {
                "config": config.to_dict(),
                "adapter_path": args.adapter_path,
                "validation_metrics": metrics
            }
            
            results_path = args.output_dir / "eval_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {results_path}")
    
    elif args.mode == "dspy":
        # DSPy optimization with Ollama
        logger.info("Running DSPy prompt optimization with Ollama...")
        
        config = GemmaConfig(
            model_name=args.model,
            task=args.task,
            temperature=args.temperature,
            num_few_shot=args.num_few_shot,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_fp16=not args.no_fp16
        )
        
        config.save(args.output_dir / "config.json")
        
        # Create optimizer
        optimizer_output_dir = args.output_dir / "dspy_optimized"
        optimizer = LLMDSPyBase(
            config,
            output_dir=optimizer_output_dir,
            mode="dspy"
        )
        
        # Optimize
        logger.info("Starting DSPy optimization...")
        logger.info(f"This will use up to {args.max_train_examples} train and {args.max_val_examples} val examples")
        
        history = optimizer.optimize(
            dataset["train"],
            dataset["validation"],
            max_train_examples=args.max_train_examples,
            max_val_examples=args.max_val_examples
        )
        
        # Save history
        history_path = args.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Optimization complete. Saved to {optimizer_output_dir}")
        
        # Evaluate on full validation set
        logger.info("\nEvaluating optimized classifier on validation set...")
        val_dataset = dataset["validation"]
        if args.max_eval_samples:
            logger.info(f"Limiting validation to {args.max_eval_samples} samples for quick testing")
            val_dataset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))
        
        val_predictions, val_metrics = optimizer.predict(val_dataset)
        
        print_metrics(val_metrics, "Validation Results")
        
        # Save results
        results = {
            "config": config.to_dict(),
            "validation_metrics": val_metrics,
            "history": history,
            "predictions": val_predictions
        }
        
        results_path = args.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Error analysis
        logger.info("Performing error analysis...")
        from src.utils.config import TASK_CONFIGS
        task_config = TASK_CONFIGS[args.task]
        
        model_name_clean = args.model.split('/')[-1] if '/' in args.model else args.model
        model_name_clean = model_name_clean.replace("-", "_").lower()
        
        save_error_analysis_report(
            val_predictions,
            val_dataset["label"],
            [dict(val_dataset[i]) for i in range(len(val_dataset))],
            task_config["label_names"],
            args.output_dir / "error_analysis",
            model_name=f"{model_name_clean}_dspy_k{args.num_few_shot}"
        )
        
        logger.info(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

