# 🧠 Beating BERT: Small LLMs vs BERT on GLUE Benchmarks

This project benchmarks small instruction-tuned LLMs (Gemma 2B) against classic BERT-family encoders on standard NLP classification tasks from the GLUE benchmark.

**Research Question**: Can a small decoder-only LLM with prompt-based learning match or exceed BERT's performance on short-text understanding tasks?

## 📋 Project Overview

- **BERT/DeBERTa**: Fine-tuned encoder models (110M-184M parameters)
- **Small LLMs**: Instruction-tuned decoder models (Gemma 2B, Qwen 0.5B/1.5B) with:
  - Zero-shot prompting
  - Few-shot prompting (k=5)
  - LoRA fine-tuning (experimental, not in main results)
  - DSPy prompt optimization (experimental, not in main results)

> **Note**: The [accompanying blog post](https://alex-jacobs.com/posts/beatingbert/) focuses on zero-shot and few-shot results. LoRA and DSPy code is available in the repo but results were inconsistent and not included in the main comparison.

### Tasks (Main Results)

- **SST-2**: Binary sentiment classification
- **RTE**: Binary textual entailment
- **BoolQ**: Yes/No question answering
- **ANLI (R1)**: Adversarial natural language inference

## 🚀 Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Clone the repository
git clone <repo-url>
cd beatingBERT

# Install dependencies with uv
uv sync

# Install Ollama (required for LLM experiments)
brew install ollama

# Or with pip
pip install -e .
```

### System Requirements

- **Python**: 3.10+
- **RAM**: 16GB+ recommended
- **GPU/MPS**: Optional but recommended for faster training
- **Disk**: ~10GB for datasets and models
- **Ollama**: Required only for DSPy prompt optimization mode

**Note**: This project uses PyTorch with standard `peft` for LoRA (MLX not required). Works on macOS (including M1/M2), Linux, and Windows. Zero-shot and few-shot use transformers directly. DSPy mode uses Ollama for prompt optimization.

## 🤖 Automated Benchmarks

### Quick Test (10-15 minutes)

Verify your setup works before running full experiments:

```bash
bash experiments/run_quick_test.sh
```

Tests all models and modes with small datasets to catch errors quickly.

### Full Benchmark (3-6 hours)

Run comprehensive comparison of BERT vs Small LLMs:

```bash
bash experiments/run_full_benchmark.sh
```

**What it runs:**
- 2 BERT models × 3 tasks = 6 experiments
- 3 LLM models × 3 tasks × 3 modes = 27 experiments
- **Total: 30+ experiments**

**Models tested:**
- BERT-base, DeBERTa-base
- Qwen 0.5B, Qwen 1.5B, Gemma 2B

**Modes tested:**
- BERT fine-tuning
- LLM zero-shot, few-shot, LoRA

See [`experiments/EXPERIMENT_GUIDE.md`](experiments/EXPERIMENT_GUIDE.md) for detailed instructions and [`experiments/BENCHMARK_SUMMARY.md`](experiments/BENCHMARK_SUMMARY.md) for what gets tested.

## 📦 Project Structure

```
beatingBERT/
├── src/
│   ├── data/              # Dataset loading and preprocessing
│   ├── models/            # Model trainers and inference
│   ├── evaluation/        # Metrics and error analysis
│   ├── prompts/           # Prompt templates
│   └── utils/             # Config and reproducibility
├── experiments/           # CLI scripts for running experiments
│   ├── run_bert.py       # BERT/DeBERTa experiments
│   ├── run_llm.py        # LLM experiments (Gemma, TinyLlama, Qwen, etc.)
│   └── analyze_results.py # Generate comparison tables
├── data/                  # Downloaded datasets (gitignored)
├── models/                # Saved models (gitignored)
├── results/               # Experiment outputs (gitignored)
└── prompts/               # Saved prompt templates
```

## 🔬 Running Experiments

### 1. BERT Fine-Tuning

```bash
# Basic BERT training on SST-2
uv run python experiments/run_bert.py --task sst2 --model bert-base-uncased

# DeBERTa on MNLI with custom hyperparameters
uv run python experiments/run_bert.py \
    --task mnli \
    --model microsoft/deberta-v3-base \
    --lr 3e-5 \
    --batch_size 32 \
    --epochs 5

# Quick test with limited samples
uv run python experiments/run_bert.py \
    --task rte \
    --model bert-base-uncased \
    --max_samples 1000
```

**Output**: Saves to `results/{model}_{task}_{timestamp}/`
- `best_model/` and `final_model/` - Model checkpoints
- `results.json` - Performance metrics
- `history.json` - Training history
- `error_analysis/` - Confusion matrices and error reports

### 2. LLM Zero-Shot (Transformers)

```bash
# Zero-shot inference on SST-2 with Qwen 0.5B (fastest)
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot \
    --model "Qwen/Qwen2-0.5B-Instruct"

# Try Gemma 1B (better quality, default)
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot \
    --model "google/gemma-3-1b-it"

# Try TinyLlama (faster)
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Output**: Saves to `results/{model}_zero_shot_{task}_{timestamp}/`
- `results.json` - Performance metrics
- `error_analysis/` - Confusion matrices and error reports

### 3. LLM Few-Shot (Transformers)

```bash
# 5-shot inference on SST-2
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode few-shot \
    --num_few_shot 5 \
    --model "Qwen/Qwen2-0.5B-Instruct"

# 10-shot on RTE with temperature sampling
uv run python experiments/run_llm.py \
    --task rte \
    --mode few-shot \
    --num_few_shot 10 \
    --model "google/gemma-3-1b-it" \
    --temperature 0.3
```

### 4. DSPy Prompt Optimization (Ollama)

```bash
# Automatically optimize prompts and few-shot examples
# Note: Requires Ollama - see OLLAMA_QUICKSTART.md for setup
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode dspy \
    --num_few_shot 5 \
    --model "ollama/qwen2:0.5b" \
    --max_train_examples 300 \
    --max_val_examples 100

# Quick test (much faster)
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode dspy \
    --num_few_shot 3 \
    --model "ollama/qwen2:0.5b" \
    --max_train_examples 50 \
    --max_val_examples 50 \
    --max_eval_samples 100
```

**Note**: DSPy optimization makes multiple LLM calls and requires Ollama. Start small!

**Output**: Saves to `results/{model}_dspy_k{num_shot}_{task}_{timestamp}/`
- `dspy_optimized/` - Optimized prompts and examples
- `results.json` - Performance metrics
- `history.json` - Optimization details

### 5. LLM LoRA Fine-Tuning

```bash
# LoRA fine-tuning with rank 16
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode lora \
    --lora_rank 16 \
    --lr 1e-4 \
    --epochs 3 \
    --batch_size 8

# Compare different ranks
for rank in 8 16 32; do
    uv run python experiments/run_llm.py \
        --task rte \
        --mode lora \
        --lora_rank $rank
done
```

**Output**: Saves to `results/gemma_lora_r{rank}_{task}_{timestamp}/`
- `adapter/best_adapter/` and `final_adapter/` - LoRA adapters
- `results.json` - Performance metrics
- `history.json` - Training curves

### Evaluation Only

```bash
# Evaluate pre-trained BERT model
uv run python experiments/run_bert.py \
    --task sst2 \
    --mode eval \
    --model_path results/bert_sst2_xxx/model/best_model

# Evaluate LoRA adapter
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode lora \
    --adapter_path results/gemma_lora_r16_sst2_xxx/adapter/best_adapter
```

## 📊 Analyzing Results

After running multiple experiments:

```bash
# Generate comparison tables and plots
uv run python experiments/analyze_results.py

# Analyze specific results directory
uv run python experiments/analyze_results.py \
    --results_dir results/my_experiments \
    --output_dir results/my_analysis

# Generate only tables (no plots)
uv run python experiments/analyze_results.py --format tables
```

**Output**: Saves to `results/analysis/`
- `performance_table.md` - Accuracy, F1, MCC comparison
- `efficiency_table.md` - Latency, memory, throughput
- `all_results.csv` - Raw data for further analysis
- `plots/` - Accuracy comparison, accuracy vs latency, F1 scores

## 📈 Evaluation Metrics

### Performance Metrics
- **Accuracy**: Primary metric for all tasks
- **Macro F1**: Balanced performance across classes
- **MCC (Matthews Correlation Coefficient)**: Robustness measure (especially for small datasets like RTE)
- **MNLI Matched/Mismatched**: Domain generalization

### Efficiency Metrics
- **Latency**: Average inference time per sample (ms)
- **Peak Memory**: Maximum memory usage during inference (GB)
- **Throughput**: Samples processed per second
- **Training Time**: Total wall-clock time for training

### Error Analysis
- **Confusion Matrices**: Per-model error patterns
- **Error Overlap**: Compare which examples each model gets wrong
- **Top-N Errors**: Qualitative analysis of failure cases

## 🛠️ Advanced Usage

### Quick Testing with Limited Samples

Speed up testing by limiting both training and evaluation samples:

```bash
# Quick test: 100 train samples, 100 eval samples
uv run python experiments/run_llm.py \
    --task sst2 \
    --mode zero-shot \
    --max_eval_samples 100

# Test BERT training quickly (1000 train, 100 eval)
uv run python experiments/run_bert.py \
    --task sst2 \
    --max_samples 1000 \
    --max_eval_samples 100
```

**Key Flags:**
- `--max_samples N` - Limit training data
- `--max_eval_samples N` - Limit validation/test data (for quick testing)

### Custom Configurations

Edit configs in code or use CLI arguments:

```python
from src.utils.config import BERTConfig, GemmaConfig

# Custom BERT config
config = BERTConfig(
    model_name="bert-base-uncased",
    task="sst2",
    learning_rate=3e-5,
    batch_size=32,
    num_epochs=5,
    max_length=128
)

# Custom Gemma config
config = GemmaConfig(
    model_name="google/gemma-3-1b-it",
    task="mnli",
    temperature=0.0,
    num_few_shot=5,
    lora_rank=16,
    learning_rate=1e-4
)
```

### Sample Efficiency Experiments

Test learning curves with limited data:

```bash
# Train with different dataset sizes
for n in 100 500 1000; do
    uv run python experiments/run_bert.py \
        --task sst2 \
        --model bert-base-uncased \
        --max_samples $n \
        --run_name "bert_sst2_n${n}"
done

# Analyze sample efficiency
uv run python experiments/analyze_results.py \
    --results_dir results/sample_efficiency
```

### Prompt Engineering

Customize prompts in `src/prompts/templates.py`:

```python
ZERO_SHOT_PROMPTS["sst2"] = """Your custom prompt here...

Review: {sentence}

Sentiment:"""
```

All prompts are automatically saved to `prompts/` for reproducibility.

## 🔍 Reproduci bility

- **Fixed seed**: 99 (set in `src/utils/reproducibility.py`)
- **Deterministic operations**: Enabled where possible
- **Saved configs**: Every experiment saves full configuration
- **Prompt tracking**: All prompts saved for reproducibility

## 📝 Key Implementation Details

### BERT Pipeline
1. Load pre-trained BERT/DeBERTa
2. Add classification head (2 or 3 classes)
3. Fine-tune with AdamW (lr=2e-5, 3-5 epochs)
4. Standard CLS token pooling

### Gemma Pipeline

**Zero-Shot/Few-Shot**:
1. Create task-specific prompt
2. Generate with greedy decoding (temp=0.0)
3. Parse output to extract label
4. Handle malformed outputs gracefully

**LoRA Fine-Tuning**:
1. Load base Gemma model (FP16 on GPU/MPS)
2. Apply LoRA to q/k/v/o projection matrices
3. Train on prompt-completion pairs
4. Save lightweight adapters (~50-200MB vs ~5GB full model)

## 🎯 Expected Results

Based on the project plan, approximate performance targets:

| Task | BERT-base | DeBERTa-v3 | Gemma Zero-Shot | Gemma LoRA |
|------|-----------|------------|-----------------|------------|
| SST-2 | ~93% | ~95% | ~85-88% | ~90-92% |
| MNLI | ~84% | ~88% | ~75-80% | ~82-85% |
| RTE | ~70% | ~75% | ~60-65% | ~68-72% |

**Efficiency**: Gemma has higher latency but better few-shot adaptability.

## 📚 References

- [GLUE Benchmark](https://gluebenchmark.com/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654)
- [Gemma Technical Report](https://arxiv.org/abs/2403.08295)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

This is a research project. To reproduce or extend:

1. Fork the repository
2. Run experiments following the guide above
3. Share your results!

## 📄 License

MIT License - see LICENSE file for details.

## 🐛 Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 4 or 8)
- Use `--no_fp16` flag for Gemma
- Limit samples with `--max_samples` for testing

### Slow Training
- Ensure GPU/MPS is being used (check system info in logs)
- Increase `--batch_size` if you have memory headroom
- Use gradient accumulation for effective larger batches

### Import Errors
- Make sure you ran `uv sync` or `pip install -e .`
- Check Python version (3.10+)

### Dataset Download Issues
- Datasets are cached to `data/` directory
- If download fails, check internet connection
- HuggingFace datasets requires network access

## 💡 Tips

1. **Start small**: Test with `--max_samples 100` first
2. **Monitor resources**: Watch memory usage, especially for Gemma
3. **Save prompts**: Check `prompts/` directory to see what was sent to models
4. **Compare carefully**: Use `analyze_results.py` for fair comparisons
5. **Error analysis**: Look at confusion matrices to understand failure modes

---

**Built with**: PyTorch • Transformers • PEFT • HuggingFace Datasets • uv

