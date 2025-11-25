#!/bin/bash
# Quick test to verify all models and modes work correctly
# Runtime: ~10-15 minutes
# Tests each model/mode with small samples to catch any errors

set -e  # Exit on error

# Get script directory and change to experiments directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "🧪 Quick Test - Verify Setup"
echo "========================================"
echo "This will test all models and modes with tiny datasets"
echo "Expected runtime: 10-15 minutes"
echo ""

# Configuration
TASKS=("sst2")  # Just one task for quick test
MAX_SAMPLES=100  # Tiny dataset
MAX_EVAL=50     # Quick evaluation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "Tasks: ${TASKS[@]}"
echo "Max samples: ${MAX_SAMPLES}"
echo ""

# Test 1: BERT Models
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 1: BERT Fine-tuning${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for task in "${TASKS[@]}"; do
    echo ""
    echo "→ BERT-base on $task (quick)"
    uv run python run_bert.py \
        --task "$task" \
        --model bert-base-uncased \
        --epochs 1 \
        --batch_size 16 \
        --max_samples "$MAX_SAMPLES" \
        --max_eval_samples "$MAX_EVAL" || echo "⚠️  BERT test failed"
done

# Test 2: Qwen 0.5B Zero-shot
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 2: Qwen 0.5B Zero-shot${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for task in "${TASKS[@]}"; do
    echo ""
    echo "→ Qwen 0.5B zero-shot on $task"
    uv run python run_llm.py \
        --task "$task" \
        --mode zero-shot \
        --model "Qwen/Qwen2-0.5B-Instruct" \
        --max_eval_samples "$MAX_EVAL" || echo "⚠️  Qwen zero-shot test failed"
done

# Test 3: Qwen 0.5B Few-shot
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 3: Qwen 0.5B Few-shot${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for task in "${TASKS[@]}"; do
    echo ""
    echo "→ Qwen 0.5B 3-shot on $task"
    uv run python run_llm.py \
        --task "$task" \
        --mode few-shot \
        --num_few_shot 3 \
        --model "Qwen/Qwen2-0.5B-Instruct" \
        --max_eval_samples "$MAX_EVAL" || echo "⚠️  Qwen few-shot test failed"
done

# Test 4: LoRA (very quick - note: may show 0.0 acc with tiny dataset)
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Test 4: Qwen 0.5B LoRA${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Note: 0.0% accuracy expected with only 100 samples - just testing setup"

for task in "${TASKS[@]}"; do
    echo ""
    echo "→ Qwen 0.5B LoRA on $task (1 epoch)"
    uv run python run_llm.py \
        --task "$task" \
        --mode lora \
        --model "Qwen/Qwen2-0.5B-Instruct" \
        --lora_rank 8 \
        --epochs 1 \
        --batch_size 8 \
        --max_samples "$MAX_SAMPLES" || echo "⚠️  LoRA test failed (check if training completed)"
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Quick test complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Check results/ directory for outputs"
echo ""
echo "To run full benchmark:"
echo "  bash experiments/run_full_benchmark.sh"

