# 📊 Full Benchmark Summary

## What Gets Tested

### Models (5 total)

| Model | Size | Type | Use Cases |
|-------|------|------|-----------|
| BERT-base | 110M | Encoder | Baseline fine-tuning |
| DeBERTa-base | 184M | Encoder | Better encoder |
| Qwen 0.5B | 500M | Decoder | Fastest LLM |
| Qwen 1.5B | 1.5B | Decoder | Balanced LLM |
| Gemma 2B | 2B | Decoder | Best quality LLM |

### Tasks (3 total)

| Task | Type | Size | Difficulty |
|------|------|------|------------|
| **SST-2** | Sentiment (binary) | 67K train | Easy |
| **RTE** | Entailment (binary) | 2.5K train | Hard |
| **MNLI** | NLI (3-way) | 393K train | Medium |

### Modes (4 total)

| Mode | Description | Training Time |
|------|-------------|---------------|
| **Fine-tune** | Full fine-tuning | Minutes-hours |
| **Zero-shot** | No training | Instant |
| **Few-shot** | 5 examples | Instant |
| **LoRA** | Parameter-efficient | Minutes |

## Experiment Matrix

### BERT Models (6 experiments)
```
┌─────────────┬──────┬─────┬──────┐
│ Model       │ SST2 │ RTE │ MNLI │
├─────────────┼──────┼─────┼──────┤
│ BERT-base   │  ✓   │  ✓  │  ✓   │
│ DeBERTa     │  ✓   │  ✓  │  ✓   │
└─────────────┴──────┴─────┴──────┘
```

### LLM Zero-shot (9 experiments)
```
┌─────────────┬──────┬─────┬──────┐
│ Model       │ SST2 │ RTE │ MNLI │
├─────────────┼──────┼─────┼──────┤
│ Qwen 0.5B   │  ✓   │  ✓  │  ✓   │
│ Qwen 1.5B   │  ✓   │  ✓  │  ✓   │
│ Gemma 2B    │  ✓   │  ✓  │  ✓   │
└─────────────┴──────┴─────┴──────┘
```

### LLM Few-shot (9 experiments)
```
┌─────────────┬──────┬─────┬──────┐
│ Model       │ SST2 │ RTE │ MNLI │
├─────────────┼──────┼─────┼──────┤
│ Qwen 0.5B   │  ✓   │  ✓  │  ✓   │
│ Qwen 1.5B   │  ✓   │  ✓  │  ✓   │
│ Gemma 2B    │  ✓   │  ✓  │  ✓   │
└─────────────┴──────┴─────┴──────┘
```

### LLM LoRA (6 experiments)
```
┌─────────────┬──────┬─────┬──────┐
│ Model       │ SST2 │ RTE │ MNLI │
├─────────────┼──────┼─────┼──────┤
│ Qwen 0.5B   │  ✓   │  ✓  │  ✓   │
│ Qwen 1.5B   │  ✓   │  ✓  │  ✓   │
└─────────────┴──────┴─────┴──────┘
```

## Total: 30 Experiments

## Estimated Timeline

### Quick Test (~15 min)
```
00:00 - Start
00:03 - BERT test complete
00:06 - Zero-shot complete
00:09 - Few-shot complete
00:15 - LoRA test complete
```

### Full Benchmark (~4-6 hours)

```
Hour 0:00 - Start
Hour 0:00 - BERT fine-tuning begins
├─ 00:30 - BERT-base on SST-2 ✓
├─ 00:45 - BERT-base on RTE ✓
├─ 01:15 - BERT-base on MNLI ✓
├─ 01:45 - DeBERTa on SST-2 ✓
├─ 02:00 - DeBERTa on RTE ✓
└─ 02:30 - DeBERTa on MNLI ✓

Hour 2:30 - LLM Zero-shot begins
├─ 02:40 - Qwen 0.5B complete ✓
├─ 02:55 - Qwen 1.5B complete ✓
└─ 03:15 - Gemma 2B complete ✓

Hour 3:15 - LLM Few-shot begins
├─ 03:25 - Qwen 0.5B complete ✓
├─ 03:40 - Qwen 1.5B complete ✓
└─ 04:00 - Gemma 2B complete ✓

Hour 4:00 - LLM LoRA begins
├─ 04:30 - Qwen 0.5B complete ✓
├─ 05:30 - Qwen 1.5B complete ✓
└─ 06:00 - All done! ✓
```

**Note**: Times vary based on:
- CPU/GPU speed
- Available RAM
- Dataset size
- Model size

## Expected Outputs

### Results Directory
```
results/
├── benchmark_20251108_100000.log (full log)
│
├── bert_base_uncased_sst2_20251108_100000/
│   ├── best_model/
│   ├── results.json              ← Accuracy, F1, MCC
│   ├── history.json              ← Training curves
│   └── error_analysis/
│       ├── bert_confusion_matrix.png
│       └── bert_errors.csv
│
├── qwen2_0.5b_zero_shot_sst2_20251108_103000/
│   ├── results.json              ← Performance metrics
│   └── error_analysis/
│
└── ... (27 more experiment directories)
```

### Analysis Output
After running `analyze_results.py`:
```
analysis/
├── comparison_table.csv
├── accuracy_comparison.png
├── latency_comparison.png
└── summary_stats.txt
```

## Research Questions Answered

1. **Can small LLMs match BERT on short-text classification?**
   - Compare: BERT fine-tuned vs LLM LoRA

2. **Is few-shot prompting competitive with fine-tuning?**
   - Compare: BERT fine-tuned vs LLM few-shot

3. **How much does model size matter?**
   - Compare: Qwen 0.5B vs 1.5B vs Gemma 2B

4. **Which approach is most efficient?**
   - Compare: Training time, inference latency, memory usage

5. **Where do LLMs struggle vs BERT?**
   - Analyze: Error patterns, task-specific performance

## Success Criteria

### ✅ Benchmark is successful if:
- All 30 experiments complete
- Results saved for each experiment
- No crashes or errors
- Can generate comparison tables

### 📊 Interesting findings if:
- LLM LoRA within 2% of BERT
- Few-shot competitive with fine-tuning
- Small models (0.5B) surprisingly good
- Clear efficiency trade-offs identified

## Next Steps After Benchmark

1. **Immediate** (5 min)
   ```bash
   uv run python experiments/analyze_results.py
   ```

2. **Deep Dive** (30 min)
   - Open best/worst performing experiments
   - Read error_analysis reports
   - Identify patterns

3. **Report** (1 hour)
   - Create summary table
   - Plot key comparisons
   - Write findings

4. **Iterate** (ongoing)
   - Try different hyperparameters
   - Test more models
   - Explore failure cases

## Quick Commands

```bash
# Run quick test (verify setup)
bash experiments/run_quick_test.sh

# Run full benchmark (3-6 hours)
bash experiments/run_full_benchmark.sh

# Monitor progress
tail -f results/benchmark_*.log

# Analyze when done
uv run python experiments/analyze_results.py

# Check disk space
df -h

# See latest results
ls -lt results/ | head -20
```

---

**Ready to start?** → `bash experiments/run_quick_test.sh`




