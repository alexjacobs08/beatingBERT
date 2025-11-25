"""Verify that the project is set up correctly."""

import sys
from pathlib import Path

print("=" * 60)
print("BeatingBERT Setup Verification")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Test imports
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")
    sys.exit(1)

try:
    import datasets
    print(f"✓ Datasets: {datasets.__version__}")
except ImportError as e:
    print(f"✗ Datasets import failed: {e}")
    sys.exit(1)

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
except ImportError as e:
    print(f"✗ PEFT import failed: {e}")
    sys.exit(1)

try:
    from src.utils.config import TASK_CONFIGS, BERTConfig, GemmaConfig
    print(f"✓ Config module loaded")
    print(f"  Available tasks: {', '.join(TASK_CONFIGS.keys())}")
except ImportError as e:
    print(f"✗ Config import failed: {e}")
    sys.exit(1)

try:
    from src.utils.reproducibility import set_seed, get_device
    print(f"✓ Reproducibility module loaded")
    device = get_device()
    print(f"  Device: {device}")
except ImportError as e:
    print(f"✗ Reproducibility import failed: {e}")
    sys.exit(1)

try:
    from src.data.loader import load_glue_dataset
    print(f"✓ Data loader module loaded")
except ImportError as e:
    print(f"✗ Data loader import failed: {e}")
    sys.exit(1)

try:
    from src.models.bert_trainer import BERTTrainer
    from src.models.gemma_inference import GemmaInference
    from src.models.gemma_lora import GemmaLoRATrainer
    print(f"✓ Model modules loaded")
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

try:
    from src.evaluation.metrics import MetricsTracker
    from src.evaluation.error_analysis import analyze_error_overlap
    print(f"✓ Evaluation modules loaded")
except ImportError as e:
    print(f"✗ Evaluation import failed: {e}")
    sys.exit(1)

try:
    from src.prompts.templates import ZERO_SHOT_PROMPTS
    print(f"✓ Prompt templates loaded")
    print(f"  Available prompts: {', '.join(ZERO_SHOT_PROMPTS.keys())}")
except ImportError as e:
    print(f"✗ Prompt templates import failed: {e}")
    sys.exit(1)

# Check directory structure
print("\n✓ Directory structure:")
for dir_name in ["src", "experiments", "data", "models", "results", "prompts"]:
    dir_path = Path(dir_name)
    status = "✓" if dir_path.exists() else "✗"
    print(f"  {status} {dir_name}/")

print("\n" + "=" * 60)
print("Setup verification complete! ✓")
print("=" * 60)
print("\nNext steps:")
print("  1. Run a quick test: uv run python experiments/run_bert.py --task sst2 --max_samples 10")
print("  2. Or start with: uv run python experiments/run_llm.py --task sst2 --mode zero-shot --max_samples 10")
print("  3. See README.md for full usage instructions")

