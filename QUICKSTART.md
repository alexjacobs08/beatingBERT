# 🚀 Quick Start Guide

## Installation

```bash
# Install dependencies
uv sync

# Verify setup (optional)
uv run python verify_setup.py
```

## Your First Experiment (2 minutes)

```bash
# Test BERT on 100 samples
uv run python experiments/run_bert.py --task sst2 --max_samples 100
```

✅ If this works, you're ready to go!

## Core Experiments

### 1. BERT Fine-Tuning (~30 min)
```bash
uv run python experiments/run_bert.py \
    --task sst2 \
    --model bert-base-uncased \
    --epochs 3
```

### 2. LLM Zero-Shot (~15 min)
```bash
# Default: Gemma 1B
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot

# Or try TinyLlama (faster!)
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Note:** First run downloads the model (~1-5GB depending on model).

### 3. LLM Few-Shot (~15 min)
```bash
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode few-shot \
    --num_few_shot 5
```

### 4. DSPy Optimization (~30 min)
```bash
# Automatically optimize prompts
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode dspy \
    --num_few_shot 5 \
    --max_train_examples 100
```

### 5. LLM LoRA Fine-Tuning (~2 hours)
```bash
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode lora \
    --lora_rank 16 \
    --epochs 3
```

## Compare Results

```bash
# Generate comparison tables and plots
uv run python experiments/analyze_results.py
```

View results in `results/analysis/`

## Available Tasks

- `sst2` - Sentiment analysis (easiest, start here)
- `rte` - Textual entailment (small dataset)
- `mnli` - Natural language inference (larger, ~2-3 hours)
- `qqp` - Question paraphrase detection
- `mrpc` - Sentence paraphrase detection

## Quick Tips

**Save time while testing:**
```bash
--max_samples 1000       # Limit training data
--max_eval_samples 100   # Limit validation data (NEW!)
--batch_size 8           # Reduce if out of memory
```

**Quick test mode (30 seconds):**
```bash
uv run python experiments/run_llm.py --task sst2 --mode zero-shot --max_eval_samples 10
```

**View results:**
```bash
ls results/                                    # List all experiments
cat results/*/results.json | python -m json.tool  # View metrics
```

**Low on memory?**
```bash
--batch_size 4        # Smaller batches
--no_fp16             # For Gemma (uses more memory but more compatible)
```

## Example Workflow

```bash
# Day 1: Quick comparison on SST-2
uv run python experiments/run_bert.py --task sst2
uv run python experiments/run_llm.py --task sst2 --mode zero-shot
uv run python experiments/run_llm.py --task sst2 --mode few-shot --num_few_shot 5
uv run python experiments/analyze_results.py

# Day 2: Try LoRA
uv run python experiments/run_llm.py --task sst2 --mode lora --lora_rank 16

# Day 3: Explore other tasks
uv run python experiments/run_bert.py --task rte
uv run python experiments/run_llm.py --task rte --mode zero-shot
```

## Output Structure

```
results/
├── bert_sst2_20241106_123000/
│   ├── results.json              # Metrics
│   ├── model/best_model/         # Saved model
│   └── error_analysis/           # Confusion matrix
├── gemma_zero_shot_sst2_*/
└── analysis/                     # Comparison tables & plots
    ├── performance_table.md
    ├── efficiency_table.md
    └── plots/
```

## Troubleshooting

**Import errors?**
```bash
uv sync  # Reinstall dependencies
```

**Out of memory?**
```bash
# Use smaller batches
--batch_size 4

# Use fewer samples for testing
--max_samples 100
```

**Model download slow?**
- First run downloads model (500MB-5GB depending on model)
- Cached after first download
- Takes 1-10 minutes on good connection

**Need help?**
- See full README.md for detailed documentation
- Check `src/prompts/templates.py` to see prompts
- Run `uv run python verify_setup.py` to check installation

---

**Next Steps:** See README.md for advanced usage, error analysis, and customization options.

