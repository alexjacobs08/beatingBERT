"""DSPy-based prompt optimization for LLMs."""

import dspy
from pathlib import Path
import json
import logging
from typing import Any
from tqdm import tqdm

from src.evaluation.metrics import MetricsTracker, get_memory_usage
from src.utils.reproducibility import set_seed
from src.utils.config import GemmaConfig, TASK_CONFIGS
from src.prompts.templates import label_to_id, id_to_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationSignature(dspy.Signature):
    """Signature for text classification."""
    text: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="brief reasoning about the classification")
    label: str = dspy.OutputField(desc="the classification label")


class BinaryClassificationSignature(dspy.Signature):
    """Signature for binary text classification."""
    text: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="brief reasoning")
    label: str = dspy.OutputField(desc="the label: positive, negative, entailment, not_entailment, equivalent, not_equivalent, duplicate, or not_duplicate")


class PairClassificationSignature(dspy.Signature):
    """Signature for sentence pair classification."""
    text1: str = dspy.InputField()
    text2: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="brief reasoning")
    label: str = dspy.OutputField(desc="the classification label")


class DSPyClassifier(dspy.Module):
    """DSPy module for classification."""
    
    def __init__(self, task: str, num_labels: int, label_names: list[str]):
        super().__init__()
        self.task = task
        self.num_labels = num_labels
        self.label_names = label_names
        
        # Use Predict instead of ChainOfThought for simpler parsing
        # ChainOfThought is built into the signature via reasoning field
        if num_labels == 2:
            self.predictor = dspy.Predict(BinaryClassificationSignature)
        else:
            self.predictor = dspy.Predict(ClassificationSignature)
    
    def forward(self, text: str = None, text1: str = None, text2: str = None):
        """Make a prediction."""
        if text is not None:
            return self.predictor(text=text)
        else:
            return self.predictor(text1=text1, text2=text2)


class LLMDSPyBase:
    """Base class for DSPy-based LLM inference (zero-shot, few-shot, and optimized)."""
    
    def __init__(
        self,
        config: GemmaConfig,
        output_dir: Path | None = None,
        mode: str = "dspy"  # "zero-shot", "few-shot", or "dspy"
    ):
        """
        Initialize DSPy inference.
        
        Args:
            config: Configuration for the model
            output_dir: Directory to save results
            mode: Inference mode (zero-shot, few-shot, or dspy-optimized)
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        
        # Get task info
        self.task_config = TASK_CONFIGS[config.task]
        self.num_labels = self.task_config["num_labels"]
        self.label_names = self.task_config["label_names"]
        self.text_keys = self.task_config["text_keys"]
        
        # Setup
        set_seed()
        
        model_display_name = config.model_name.split('/')[-1] if '/' in config.model_name else config.model_name
        logger.info(f"Initializing DSPy with {model_display_name}")
        logger.info(f"Mode: {mode}")
        
        # Configure DSPy LM
        # Check if it's an API model
        if any(provider in config.model_name.lower() for provider in ['gpt-', 'claude-', 'openai/', 'anthropic/']):
            # API model
            import os
            
            if 'gpt-' in config.model_name.lower() or 'openai' in config.model_name.lower():
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY required")
                
                clean_model = config.model_name.replace('openai/', '')
                self.lm = dspy.LM(model=f"openai/{clean_model}", temperature=config.temperature)
                logger.info(f"Using OpenAI API model: {clean_model}")
            
            elif 'claude-' in config.model_name.lower() or 'anthropic' in config.model_name.lower():
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY required")
                
                clean_model = config.model_name.replace('anthropic/', '')
                self.lm = dspy.LM(model=f"anthropic/{clean_model}", temperature=config.temperature, api_key=api_key)
                logger.info(f"Using Anthropic API model: {clean_model}")
            else:
                self.lm = dspy.LM(model=config.model_name, temperature=config.temperature)
                logger.info(f"Using API model: {config.model_name}")
        elif 'ollama' in config.model_name.lower():
            # Ollama model (local server)
            model_name = config.model_name.replace('ollama/', '').replace('ollama_chat/', '')
            logger.info(f"Using Ollama model: {model_name}")
            
            # Check if Ollama is running and pull model if needed
            import ollama
            import subprocess
            import time
            
            try:
                # Try to list models (will fail if Ollama not running)
                ollama.list()
                logger.info("✓ Ollama is running")
            except Exception:
                logger.info("Ollama not running, attempting to start...")
                try:
                    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(3)  # Give it time to start
                    logger.info("✓ Started Ollama server")
                except Exception as e:
                    logger.error(f"Failed to start Ollama: {e}")
                    logger.error("Please install Ollama: brew install ollama")
                    raise ValueError("Ollama is not installed or cannot be started")
            
            # Check if model is pulled, if not pull it
            try:
                models = ollama.list()
                model_exists = any(model_name in str(m) for m in models.get('models', []))
                
                if not model_exists:
                    logger.info(f"Model {model_name} not found locally, pulling...")
                    ollama.pull(model_name)
                    logger.info(f"✓ Pulled {model_name}")
                else:
                    logger.info(f"✓ Model {model_name} already available")
            except Exception as e:
                logger.warning(f"Could not verify/pull model: {e}")
            
            try:
                # Try dspy.ollama.OllamaLocal (official Ollama class)
                from dspy.ollama import OllamaLocal
                
                self.lm = OllamaLocal(
                    model=model_name,
                    max_tokens=200,
                    temperature=config.temperature,
                    port=11434,
                    cache=False  # Disable caching for fresh results each run
                )
                logger.info(f"✓ Connected to Ollama using OllamaLocal")
            except (ImportError, AttributeError) as e:
                logger.warning(f"OllamaLocal not available: {e}, trying generic dspy.LM")
                # Fallback to generic LM
                self.lm = dspy.LM(
                    model=f"ollama_chat/{model_name}",
                    api_base="http://localhost:11434",
                    api_key="",
                    max_tokens=200,
                    temperature=config.temperature,
                    cache=False  # Disable caching for fresh results each run
                )
                logger.info(f"✓ Connected to Ollama using generic dspy.LM")
        else:
            # HuggingFace model - need to use Ollama or similar backend
            logger.info(f"HuggingFace model requested: {model_display_name}")
            logger.error("\n" + "="*70)
            logger.error("DSPy 3.0+ requires local models to be served via inference servers")
            logger.error("="*70)
            logger.error("\nFor local HuggingFace models, use Ollama (recommended):")
            logger.error("\n  Step 1: Install Ollama")
            logger.error("    brew install ollama")
            logger.error("\n  Step 2: Map your model to Ollama equivalent:")
            logger.error("    Qwen/Qwen2-0.5B-Instruct  → ollama/qwen2:0.5b")
            logger.error("    Qwen/Qwen2-1.5B-Instruct  → ollama/qwen2:1.5b")
            logger.error("    google/gemma-2-2b-it      → ollama/gemma2:2b")
            logger.error("    TinyLlama/TinyLlama-1.1B  → ollama/tinyllama")
            logger.error("\n  Step 3: Run with Ollama model name:")
            logger.error(f"    --model 'ollama/qwen2:0.5b' instead of '{config.model_name}'")
            logger.error("\n  The script will auto-start Ollama and pull the model!")
            logger.error("="*70)
            
            # Try to give helpful suggestion based on model name
            suggestions = {
                'qwen2-0.5b': 'ollama/qwen2:0.5b',
                'qwen2-1.5b': 'ollama/qwen2:1.5b',
                'qwen2-7b': 'ollama/qwen2:7b',
                'gemma-2-2b': 'ollama/gemma2:2b',
                'gemma-3-1b': 'ollama/gemma2:2b',
                'tinyllama': 'ollama/tinyllama',
                'llama-3.2-1b': 'ollama/llama3.2:1b',
                'llama-3.2-3b': 'ollama/llama3.2:3b',
            }
            
            model_lower = config.model_name.lower()
            for key, ollama_name in suggestions.items():
                if key in model_lower:
                    logger.error(f"\n💡 Suggested command for your model:")
                    logger.error(f"   --model '{ollama_name}'")
                    break
            
            raise ValueError(f"Cannot load HuggingFace model directly. Use Ollama (see instructions above).")
        
        dspy.settings.configure(lm=self.lm)
        
        # Create classifier module
        self.classifier = DSPyClassifier(config.task, self.num_labels, self.label_names)
        
        # For zero-shot and few-shot, we use the classifier directly without optimization
        if mode in ["zero-shot", "few-shot"]:
            self.compiled_classifier = self.classifier
            logger.info(f"DSPy initialized for {mode} inference ({self.num_labels}-way classification)")
        else:
            logger.info(f"DSPy initialized for optimization ({self.num_labels}-way classification)")
    
    def _prepare_examples(self, dataset, max_examples: int | None = None):
        """
        Prepare examples for DSPy.
        
        Args:
            dataset: HuggingFace dataset
            max_examples: Maximum number of examples to use
            
        Returns:
            List of DSPy Example objects
        """
        examples = []
        max_ex = min(len(dataset), max_examples) if max_examples else len(dataset)
        
        for i in range(max_ex):
            example = dataset[i]
            label_text = id_to_label(example["label"], self.config.task)
            
            if len(self.text_keys) == 1:
                # Single text input
                ex = dspy.Example(
                    text=example[self.text_keys[0]],
                    label=label_text
                ).with_inputs("text")
            else:
                # Sentence pair input
                ex = dspy.Example(
                    text1=example[self.text_keys[0]],
                    text2=example[self.text_keys[1]],
                    label=label_text
                ).with_inputs("text1", "text2")
            
            examples.append(ex)
        
        return examples
    
    def setup_few_shot(self, train_dataset, num_examples: int | None = None):
        """
        Setup few-shot demos manually (without optimization).
        
        Args:
            train_dataset: Training dataset to sample examples from
            num_examples: Number of examples to use (defaults to config.num_few_shot)
        """
        if self.mode != "few-shot":
            logger.warning(f"setup_few_shot called in {self.mode} mode, ignoring")
            return
        
        num_examples = num_examples or self.config.num_few_shot
        if num_examples == 0:
            logger.warning("num_few_shot is 0, no demos will be set")
            return
        
        logger.info(f"Setting up {num_examples} few-shot examples...")
        examples = self._prepare_examples(train_dataset, num_examples)
        
        # Manually set demos on the predictor
        self.classifier.predictor.demos = examples
        logger.info(f"✓ Set {len(examples)} few-shot demos")
    
    def optimize(
        self,
        train_dataset,
        val_dataset,
        max_train_examples: int = 300,
        max_val_examples: int = 100
    ) -> dict[str, Any]:
        """
        Optimize the classifier using DSPy BootstrapFewShot.
        Only runs in 'dspy' mode.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            max_train_examples: Maximum training examples (DSPy is expensive)
            max_val_examples: Maximum validation examples
            
        Returns:
            Optimization history
        """
        if self.mode != "dspy":
            raise ValueError(f"optimize() should only be called in 'dspy' mode, not '{self.mode}'")
        
        logger.info(f"Preparing training examples (max: {max_train_examples})...")
        train_examples = self._prepare_examples(train_dataset, max_train_examples)
        
        logger.info(f"Preparing validation examples (max: {max_val_examples})...")
        val_examples = self._prepare_examples(val_dataset, max_val_examples)
        
        logger.info(f"Using {len(train_examples)} train and {len(val_examples)} val examples")
        
        # Define evaluation metric
        def metric(example, pred, trace=None):
            """Metric for DSPy optimization."""
            try:
                # Handle case where pred might not have label attribute (parsing failed)
                if not hasattr(pred, 'label'):
                    logger.debug(f"Prediction has no label attribute: {pred}")
                    return 0.0
                
                # Parse prediction - normalize both
                pred_label = str(pred.label).strip().lower()
                true_label = str(example.label).strip().lower()
                
                # Direct match first
                if pred_label == true_label:
                    return 1.0
                
                # Try to match any label name
                for label_name in self.label_names:
                    if label_name.lower() in pred_label:
                        return float(label_name.lower() == true_label)
                
                # Handle numeric format for binary tasks
                if self.num_labels == 2:
                    if '1' in pred_label or 'positive' in pred_label:
                        pred_normalized = self.label_names[1].lower()
                    elif '0' in pred_label or 'negative' in pred_label:
                        pred_normalized = self.label_names[0].lower()
                    else:
                        return 0.0
                    
                    return float(pred_normalized == true_label)
                
                return 0.0
            except AttributeError as e:
                # Pred object doesn't have expected attributes
                logger.debug(f"Attribute error in metric: {e}")
                return 0.0
            except Exception as e:
                logger.debug(f"Metric error: {e}")
                return 0.0
        
        # Use BootstrapFewShot optimizer with error handling
        logger.info("Starting DSPy optimization with BootstrapFewShot...")
        logger.info("This will take a while as DSPy generates and evaluates prompts...")
        logger.info("Note: Some parsing errors are normal during optimization as DSPy explores different formats")
        
        # Create a more lenient metric wrapper
        def safe_metric(example, pred, trace=None):
            """Wrapper with error handling."""
            try:
                return metric(example, pred, trace)
            except Exception as e:
                logger.debug(f"Metric evaluation failed: {e}")
                return 0.0
        
        optimizer = dspy.BootstrapFewShot(
            metric=safe_metric,
            max_bootstrapped_demos=self.config.num_few_shot or 4,
            max_labeled_demos=self.config.num_few_shot or 4,
            max_errors=10,  # Allow some errors during bootstrapping
        )
        
        # Compile/optimize the classifier
        try:
            logger.info("Compiling optimized classifier...")
            self.compiled_classifier = optimizer.compile(
                self.classifier,
                trainset=train_examples
            )
        except Exception as e:
            logger.error(f"DSPy optimization failed: {e}")
            logger.error("This might be due to model output format issues.")
            logger.error("Try: 1) Using a different teacher model, 2) Reducing num_few_shot, 3) Simplifying the task")
            raise
        
        logger.info("DSPy optimization complete!")
        logger.info("Optimized prompts will now be used for evaluation with the same model")

        history = {
            "num_train_examples": len(train_examples),
            "num_val_examples": len(val_examples),
            "optimization_complete": True
        }
        
        # Save optimized classifier
        if self.output_dir:
            self.save_classifier(self.output_dir / "dspy_optimized")
        
        return history
    
    def _evaluate(self, dataset) -> tuple[list[int], dict[str, Any]]:
        """
        Evaluate the compiled classifier.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Tuple of (predictions, metrics)
        """
        # Note: We don't start the overall timer here
        # We only time individual predictions for per-sample latency
        
        predictions = []
        true_labels = []
        failed_parses = 0
        latencies = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            import time
            # Time ONLY this prediction
            start = time.time()
            
            try:
                # Make prediction with error handling
                if len(self.text_keys) == 1:
                    pred = self.compiled_classifier(text=example[self.text_keys[0]])
                else:
                    pred = self.compiled_classifier(
                        text1=example[self.text_keys[0]],
                        text2=example[self.text_keys[1]]
                    )
                
                # Parse prediction
                pred_label = pred.label.strip().lower() if hasattr(pred, 'label') else str(pred).strip().lower()
                
                # Try multiple parsing strategies
                pred_id = None
                
                # Strategy 1: Direct label match
                try:
                    pred_id = label_to_id(pred_label, self.config.task)
                except (KeyError, ValueError):
                    pass
                
                # Strategy 2: Check if label name is in output
                if pred_id is None:
                    for label_name in self.label_names:
                        if label_name.lower() in pred_label:
                            pred_id = label_to_id(label_name, self.config.task)
                            break
                
                # Strategy 3: Binary numeric format
                if pred_id is None and self.num_labels == 2:
                    if '1' in pred_label or 'positive' in pred_label:
                        pred_id = 1
                    elif '0' in pred_label or 'negative' in pred_label:
                        pred_id = 0
                
                # Default fallback
                if pred_id is None:
                    failed_parses += 1
                    pred_id = 0
                
                predictions.append(pred_id)
                true_labels.append(example["label"])
                
                # Record ONLY inference latency
                latency = time.time() - start
                latencies.append(latency)
            
            except Exception as e:
                logger.debug(f"Prediction error: {e}")
                failed_parses += 1
                predictions.append(0)
                true_labels.append(example["label"])
                
                latency = time.time() - start
                latencies.append(latency)
        
        # Compute metrics manually (don't use tracker's timer)
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
        
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "macro_f1": f1_score(true_labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(true_labels, predictions, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(true_labels, predictions),
            # Efficiency metrics - ONLY inference time
            "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
            "total_time_s": sum(latencies),
            "samples_per_second": len(predictions) / sum(latencies) if sum(latencies) > 0 else 0,
            "peak_memory_gb": get_memory_usage(),
            "num_samples": len(predictions),
            "failed_parses": failed_parses,
            "parse_success_rate": 1 - (failed_parses / len(dataset))
        }
        
        logger.info(f"Failed parses: {failed_parses}/{len(dataset)} ({failed_parses/len(dataset)*100:.1f}%)")

        return predictions, metrics
    
    def predict(self, dataset) -> tuple[list[int], dict[str, Any]]:
        """
        Make predictions using the classifier.

        For zero-shot/few-shot: Uses classifier directly
        For dspy mode: Uses optimized compiled_classifier

        Args:
            dataset: Dataset to predict on

        Returns:
            Tuple of (predictions, metrics)
        """
        if self.mode == "dspy" and not hasattr(self, 'compiled_classifier'):
            raise ValueError("Must call optimize() first before predict() in dspy mode")

        return self._evaluate(dataset)
    
    def save_classifier(self, path: Path) -> None:
        """
        Save the optimized classifier.
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save DSPy program
        self.compiled_classifier.save(str(path / "dspy_program.json"))
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Saved DSPy classifier to {path}")
    
    def load_classifier(self, path: Path) -> None:
        """
        Load an optimized classifier.
        
        Args:
            path: Load path
        """
        path = Path(path)
        
        # Load DSPy program
        self.compiled_classifier = self.classifier.load(str(path / "dspy_program.json"))
        
        logger.info(f"Loaded DSPy classifier from {path}")


# Backwards compatibility alias
LLMDSPyOptimizer = LLMDSPyBase

