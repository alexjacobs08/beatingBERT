#!/bin/bash
# Full benchmark: BERT vs Small LLMs on GLUE + Reasoning tasks
# Runtime: ~6-10 hours depending on hardware
# Tests multiple models, modes, and tasks

set -e  # Exit on error

# Get script directory and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "========================================"
echo "🚀 Full Benchmark - BERT vs Small LLMs"
echo "========================================"
echo "This will run comprehensive experiments"
echo "Expected runtime: 6-10 hours"
echo ""
echo "⚠️  Make sure you have:"
echo "  - 16GB+ RAM"
echo "  - 30GB+ free disk space"
echo "  - Stable power (plug in laptop!)"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Configuration
# GLUE tasks (traditional NLU)
GLUE_TASKS=("sst2" "rte" "mnli")

# Reasoning tasks (LLMs should excel here)
# - anli_r1: Adversarial NLI (designed to fool BERT)
# - hellaswag: Commonsense completion
# - winogrande: Pronoun resolution
# - arc_challenge: Science reasoning
# - boolq: Boolean QA
REASONING_TASKS=("anli_r1" "hellaswag" "winogrande" "arc_challenge" "boolq")

# Combined for LLMs
ALL_TASKS=("${GLUE_TASKS[@]}" "${REASONING_TASKS[@]}")

# BERT-compatible tasks (standard classification format)
# Note: hellaswag, arc_challenge, winogrande need special multi-choice handling
BERT_TASKS=("${GLUE_TASKS[@]}" "anli_r1" "boolq")

BERT_MODELS=("bert-base-uncased" "microsoft/deberta-v3-base")
LLM_MODELS=(
    "Qwen/Qwen2-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "google/gemma-2-2b-it"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Create experiment log
LOG_FILE="results/benchmark_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo ""

#############################################
# Part 1: BERT Models (Fine-tuning)
#############################################
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Part 1/5: BERT Fine-tuning${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: ${BERT_MODELS[@]}"
echo "Tasks: ${BERT_TASKS[@]}"
echo ""

for model in "${BERT_MODELS[@]}"; do
    model_name=$(basename "$model")
    echo ""
    echo -e "${YELLOW}→ Training $model_name${NC}"
    
    for task in "${BERT_TASKS[@]}"; do
        echo "  → Task: $task"
        
        uv run python experiments/run_bert.py \
            --task "$task" \
            --model "$model" \
            --epochs 3 \
            --batch_size 32 \
            --lr 2e-5 \
            --max_samples 10000 || {
                echo "  ⚠️  Failed: $model on $task"
                continue
            }
        
        echo "  ✓ Complete"
    done
done

#############################################
# Part 2: LLM Zero-shot
#############################################
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Part 2/5: LLM Zero-shot${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: ${LLM_MODELS[@]}"
echo "Tasks: ${ALL_TASKS[@]}"
echo ""

for model in "${LLM_MODELS[@]}"; do
    model_name=$(basename "$model")
    echo ""
    echo -e "${YELLOW}→ Testing $model_name (zero-shot)${NC}"
    
    for task in "${ALL_TASKS[@]}"; do
        echo "  → Task: $task"
        
        uv run python experiments/run_llm.py \
            --task "$task" \
            --mode zero-shot \
            --model "$model" \
            --temperature 0.0 || {
                echo "  ⚠️  Failed: $model zero-shot on $task"
                continue
            }
        
        echo "  ✓ Complete"
    done
done

#############################################
# Part 3: LLM Few-shot
#############################################
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Part 3/5: LLM Few-shot${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: ${LLM_MODELS[@]}"
echo "Tasks: ${ALL_TASKS[@]}"
echo "Shots: 5"
echo ""

for model in "${LLM_MODELS[@]}"; do
    model_name=$(basename "$model")
    echo ""
    echo -e "${YELLOW}→ Testing $model_name (5-shot)${NC}"
    
    for task in "${ALL_TASKS[@]}"; do
        echo "  → Task: $task"
        
        uv run python experiments/run_llm.py \
            --task "$task" \
            --mode few-shot \
            --num_few_shot 5 \
            --model "$model" \
            --temperature 0.0 || {
                echo "  ⚠️  Failed: $model few-shot on $task"
                continue
            }
        
        echo "  ✓ Complete"
    done
done

#############################################
# Part 4: LLM LoRA Fine-tuning (GLUE tasks)
#############################################
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Part 4/5: LLM LoRA Fine-tuning (GLUE)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: Qwen 0.5B, Qwen 1.5B"
echo "Tasks: ${GLUE_TASKS[@]}"
echo "Ranks: 16"
echo ""

# Only use smaller models for LoRA (saves time)
LORA_MODELS=(
    "Qwen/Qwen2-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)

for model in "${LORA_MODELS[@]}"; do
    model_name=$(basename "$model")
    echo ""
    echo -e "${YELLOW}→ Training $model_name (LoRA r=16)${NC}"
    
    for task in "${GLUE_TASKS[@]}"; do
        echo "  → Task: $task"
        
        uv run python experiments/run_llm.py \
            --task "$task" \
            --mode lora \
            --model "$model" \
            --lora_rank 16 \
            --lr 1e-4 \
            --epochs 3 \
            --batch_size 8 \
            --max_samples 5000 || {
                echo "  ⚠️  Failed: $model LoRA on $task"
                continue
            }
        
        echo "  ✓ Complete"
    done
done

#############################################
# Part 5: LLM LoRA Fine-tuning (Reasoning)
#############################################
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Part 5/5: LLM LoRA Fine-tuning (Reasoning)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Models: Qwen 1.5B only (faster)"
echo "Tasks: ${REASONING_TASKS[@]}"
echo "Ranks: 16"
echo ""

# Only use Qwen 1.5B for reasoning LoRA (best small model, saves time)
REASONING_LORA_MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

for model in "${REASONING_LORA_MODELS[@]}"; do
    model_name=$(basename "$model")
    echo ""
    echo -e "${YELLOW}→ Training $model_name (LoRA r=16) on reasoning tasks${NC}"
    
    for task in "${REASONING_TASKS[@]}"; do
        echo "  → Task: $task"
        
        uv run python experiments/run_llm.py \
            --task "$task" \
            --mode lora \
            --model "$model" \
            --lora_rank 16 \
            --lr 1e-4 \
            --epochs 3 \
            --batch_size 8 \
            --max_samples 5000 || {
                echo "  ⚠️  Failed: $model LoRA on $task"
                continue
            }
        
        echo "  ✓ Complete"
    done
done

#############################################
# Summary
#############################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Benchmark Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "End time: $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to: results/"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Analyze results: uv run python experiments/analyze_results.py"
echo "  2. Compare models on each task"
echo "  3. Check error_analysis/ subdirectories for insights"
echo ""
echo "Summary of experiments run:"
echo "  - BERT models: ${#BERT_MODELS[@]}"
echo "  - LLM models: ${#LLM_MODELS[@]}"
echo "  - GLUE tasks: ${#GLUE_TASKS[@]} (${GLUE_TASKS[@]})"
echo "  - Reasoning tasks: ${#REASONING_TASKS[@]} (${REASONING_TASKS[@]})"
echo "  - Total tasks: ${#ALL_TASKS[@]}"
echo ""
echo "Experiment breakdown:"
echo "  - BERT fine-tuning: ${#BERT_MODELS[@]} models × ${#BERT_TASKS[@]} tasks = $(( ${#BERT_MODELS[@]} * ${#BERT_TASKS[@]} ))"
echo "  - LLM zero-shot: ${#LLM_MODELS[@]} models × ${#ALL_TASKS[@]} tasks = $(( ${#LLM_MODELS[@]} * ${#ALL_TASKS[@]} ))"
echo "  - LLM few-shot: ${#LLM_MODELS[@]} models × ${#ALL_TASKS[@]} tasks = $(( ${#LLM_MODELS[@]} * ${#ALL_TASKS[@]} ))"
echo "  - LLM LoRA (GLUE): 2 models × ${#GLUE_TASKS[@]} tasks = $(( 2 * ${#GLUE_TASKS[@]} ))"
echo "  - LLM LoRA (Reasoning): 1 model × ${#REASONING_TASKS[@]} tasks = ${#REASONING_TASKS[@]}"
echo ""

