# 🧪 Experiment Scripts Guide

## Overview

Two scripts for running comprehensive benchmarks:
1. **`run_quick_test.sh`** - Quick verification (10-15 min)
2. **`run_full_benchmark.sh`** - Full benchmark (3-6 hours)

## Quick Test (10-15 minutes)

**Purpose**: Verify everything works before running the full benchmark

```bash
bash experiments/run_quick_test.sh
```

**What it tests:**
- ✅ BERT-base fine-tuning (1 epoch, 100 samples)
- ✅ Qwen 0.5B zero-shot
- ✅ Qwen 0.5B few-shot (3-shot)
- ✅ Qwen 0.5B LoRA (1 epoch, 100 samples)

**Dataset**: SST-2 only  
**Samples**: 100 train, 50 eval  
**Runtime**: ~10-15 minutes

**Use this to:**
- Verify your setup works
- Test after code changes
- Debug issues quickly

## Full Benchmark (3-6 hours)

**Purpose**: Comprehensive comparison of BERT vs Small LLMs

```bash
bash experiments/run_full_benchmark.sh
```

### What it runs:

#### Part 1: BERT Fine-tuning (~1-2 hours)
- **Models**: BERT-base, DeBERTa-base
- **Tasks**: SST-2, RTE, MNLI
- **Config**: 3 epochs, batch 32, up to 10K samples
- **Total**: 6 experiments

#### Part 2: LLM Zero-shot (~30-45 min)
- **Models**: Qwen 0.5B, Qwen 1.5B, Gemma 2B
- **Tasks**: SST-2, RTE, MNLI
- **Config**: Full datasets, temperature 0
- **Total**: 9 experiments

#### Part 3: LLM Few-shot (~30-45 min)
- **Models**: Qwen 0.5B, Qwen 1.5B, Gemma 2B
- **Tasks**: SST-2, RTE, MNLI
- **Config**: 5-shot, full datasets
- **Total**: 9 experiments

#### Part 4: LLM LoRA Fine-tuning (~1-2 hours)
- **Models**: Qwen 0.5B, Qwen 1.5B
- **Tasks**: SST-2, RTE, MNLI
- **Config**: rank 16, 3 epochs, up to 5K samples
- **Total**: 6 experiments

### Total Experiments: ~30

## System Requirements

### Minimum
- **RAM**: 16GB
- **Disk**: 20GB free
- **Time**: 3-6 hours
- **Power**: Keep laptop plugged in!

### Recommended
- **RAM**: 32GB
- **GPU**: M1/M2 Mac or CUDA GPU
- **Disk**: 50GB free (for multiple experiments)

## Results

All results saved to `results/` with timestamps:

```
results/
├── bert_sst2_20251108_100000/
│   ├── best_model/
│   ├── results.json
│   ├── history.json
│   └── error_analysis/
├── qwen2_0.5b_zero_shot_sst2_20251108_100530/
│   ├── results.json
│   └── error_analysis/
├── qwen2_1.5b_few_shot_sst2_20251108_101200/
│   └── ...
└── benchmark_20251108_100000.log  # Full log
```

## Analysis

After benchmark completes:

```bash
# Generate comparison tables
uv run python experiments/analyze_results.py

# Or specify directory
uv run python experiments/analyze_results.py \
    --results_dir results/ \
    --output_dir analysis/
```

This creates:
- Comparison tables (accuracy, F1, latency)
- Performance plots
- Summary statistics

## Monitoring Progress

### Watch the log:
```bash
tail -f results/benchmark_*.log
```

### Check resource usage:
```bash
# CPU/Memory
top -pid $(pgrep -f run_llm.py)

# Disk space
df -h
```

### See latest results:
```bash
ls -lt results/ | head -20
```

## Customization

Edit the scripts to customize:

### Change tasks:
```bash
TASKS=("sst2" "rte" "mnli" "qqp" "mrpc")
```

### Change models:
```bash
BERT_MODELS=("bert-base-uncased" "bert-large-uncased")
LLM_MODELS=("Qwen/Qwen2-0.5B-Instruct" "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Adjust training:
```bash
--epochs 5          # More training
--batch_size 16     # Smaller batches (less memory)
--max_samples 1000  # Quick test
--lr 3e-5           # Different learning rate
```

## Troubleshooting

### Out of memory?
Reduce batch sizes in the scripts:
- BERT: Change `--batch_size 32` → `16`
- LoRA: Change `--batch_size 8` → `4`

### Too slow?
Run fewer experiments:
```bash
# Edit script to use only one task
TASKS=("sst2")

# Or use smaller models only
LLM_MODELS=("Qwen/Qwen2-0.5B-Instruct")
```

### Script crashes?
The script logs errors but continues. Check:
```bash
grep "⚠️" results/benchmark_*.log
```

### Want to resume?
Scripts don't skip completed experiments. To avoid re-running:
1. Comment out completed sections in the script
2. Or move `results/` to backup first

## Tips

### Before running full benchmark:
1. ✅ Run quick test first
2. ✅ Ensure stable power
3. ✅ Close other apps
4. ✅ Check disk space: `df -h`
5. ✅ Estimate time: Start → Add 4 hours → When will it finish?

### During benchmark:
- 📊 Monitor with `tail -f results/benchmark_*.log`
- 💾 Don't touch the laptop (avoid interruptions)
- 🔌 Keep plugged in
- ❄️ Ensure good ventilation

### After benchmark:
- 📈 Run `analyze_results.py`
- 📊 Compare models
- 🔍 Check error analysis
- 📝 Note any patterns

## Example Workflow

```bash
# Day 1: Quick test
bash experiments/run_quick_test.sh
# → Verify everything works (15 min)

# Day 1: Start full benchmark before bed
bash experiments/run_full_benchmark.sh
# → Let it run overnight (~6 hours)

# Day 2: Analyze results
uv run python experiments/analyze_results.py
# → Generate comparison tables

# Day 2: Review findings
cd results/
ls -lt | head
# → Check individual experiment results
```

## Expected Results

Typical performance hierarchy:
1. **BERT Fine-tuned**: 88-92% (SST-2)
2. **LLM LoRA**: 86-90%
3. **LLM Few-shot**: 84-88%
4. **LLM Zero-shot**: 82-86%

Small LLMs are competitive but BERT still has an edge on short-text classification!

## Questions?

- See main `README.md` for individual experiment commands
- See `QUICKSTART.md` for setup
- See `ARCHITECTURE.md` for technical details




