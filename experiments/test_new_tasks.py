#!/usr/bin/env python3
"""Quick test to verify all new reasoning tasks are configured correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_glue_dataset
from src.utils.config import TASK_CONFIGS
from src.prompts.templates import get_zero_shot_prompt, parse_model_output, label_to_id

# New tasks to test
NEW_TASKS = ["anli_r1", "hellaswag", "winogrande", "arc_challenge", "boolq"]

def test_task(task: str) -> bool:
    """Test a single task configuration."""
    print(f"\n{'='*50}")
    print(f"Testing: {task}")
    print('='*50)
    
    try:
        # 1. Check config exists
        config = TASK_CONFIGS[task]
        print(f"✓ Config found: {config['num_labels']} labels")
        
        # 2. Load dataset (just a few samples)
        dataset = load_glue_dataset(task, max_samples=10)
        print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")
        
        # 3. Check an example
        example = dataset['validation'][0]
        print(f"✓ Example keys: {list(example.keys())}")
        print(f"✓ Label: {example['label']}")
        
        # 4. Test prompt generation
        prompt = get_zero_shot_prompt(task, example)
        print(f"✓ Prompt generated ({len(prompt)} chars)")
        print(f"  Preview: {prompt[:100]}...")
        
        # 5. Test label parsing (simulate model outputs)
        test_outputs = {
            "anli_r1": ["entailment", "neutral", "contradiction"],
            "hellaswag": ["A", "B", "C", "D"],
            "winogrande": ["1", "2"],
            "arc_challenge": ["A", "B", "C", "D"],
            "boolq": ["true", "false"],
        }
        
        for output in test_outputs[task][:2]:
            parsed = parse_model_output(output, task)
            label_id = label_to_id(parsed, task)
            print(f"✓ Parse '{output}' → '{parsed}' → {label_id}")
        
        print(f"\n✅ {task} PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ {task} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🧪 Testing New Reasoning Tasks Configuration")
    print("=" * 50)
    
    results = {}
    for task in NEW_TASKS:
        results[task] = test_task(task)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for task, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {task}: {status}")
    
    print(f"\nTotal: {passed}/{total} tasks passed")
    
    if passed == total:
        print("\n🎉 All tasks configured correctly!")
        print("You can now run the full benchmark.")
        return 0
    else:
        print("\n⚠️  Some tasks failed. Fix errors before running benchmark.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

