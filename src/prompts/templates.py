"""Prompt templates for zero-shot and few-shot inference with Gemma."""

from typing import Any


# Zero-shot prompt templates
ZERO_SHOT_PROMPTS = {
    "sst2": """Classify the sentiment of the following movie review.

Review: {sentence}

Answer with a single character: 1 for positive, 0 for negative.
Answer:""",
    
    "mnli": """Given a premise and a hypothesis, determine if the hypothesis is entailed by the premise, neutral, or contradicts the premise.

Premise: {premise}
Hypothesis: {hypothesis}

Answer with just one word: entailment, neutral, or contradiction.
Relationship:""",
    
    "rte": """Given two sentences, determine if the second sentence (hypothesis) logically follows from the first sentence (premise).

Sentence 1: {sentence1}
Sentence 2: {sentence2}

Answer with a single character: 1 for entailment, 0 for not_entailment.
Answer:""",
    
    "qqp": """Determine if the following two questions are asking the same thing.

Question 1: {question1}
Question 2: {question2}

Answer with a single character: 1 for duplicate, 0 for not_duplicate.
Answer:""",
    
    "mrpc": """Determine if the following two sentences are semantically equivalent or not.

Sentence 1: {sentence1}
Sentence 2: {sentence2}

Answer with a single character: 1 for equivalent, 0 for not_equivalent.
Answer:""",
}


def format_few_shot_example(task: str, example: dict[str, Any]) -> str:
    """
    Format a single example for few-shot learning.
    
    Args:
        task: Task name
        example: Example dictionary with text keys and label
        
    Returns:
        Formatted example string
    """
    if task == "sst2":
        label = "positive" if example["label"] == 1 else "negative"
        return f"Review: {example['sentence']}\nSentiment: {label}"
    
    elif task == "mnli":
        labels = ["entailment", "neutral", "contradiction"]
        label = labels[example["label"]]
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nRelationship: {label}"
    
    elif task == "rte":
        labels = ["entailment", "not_entailment"]
        label = labels[example["label"]]
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nRelationship: {label}"
    
    elif task == "qqp":
        labels = ["not_duplicate", "duplicate"]
        label = labels[example["label"]]
        return f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}\nAnswer: {label}"
    
    elif task == "mrpc":
        labels = ["not_equivalent", "equivalent"]
        label = labels[example["label"]]
        return f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAnswer: {label}"
    
    else:
        raise ValueError(f"Unknown task: {task}")


def create_few_shot_prompt(
    task: str,
    examples: list[dict[str, Any]],
    test_example: dict[str, Any]
) -> str:
    """
    Create a few-shot prompt with examples.
    
    Args:
        task: Task name
        examples: List of example dictionaries for few-shot learning
        test_example: The test example to classify
        
    Returns:
        Complete few-shot prompt
    """
    # Get the instruction part from zero-shot template
    template = ZERO_SHOT_PROMPTS[task]
    instruction = template.split('\n\n')[0]  # Get first paragraph (instruction)
    
    # Format few-shot examples
    formatted_examples = [format_few_shot_example(task, ex) for ex in examples]
    examples_text = "\n\n".join(formatted_examples)
    
    # Format test example (without label)
    if task == "sst2":
        test_text = f"Review: {test_example['sentence']}\nSentiment:"
    elif task == "mnli":
        test_text = f"Premise: {test_example['premise']}\nHypothesis: {test_example['hypothesis']}\nRelationship:"
    elif task == "rte":
        test_text = f"Sentence 1: {test_example['sentence1']}\nSentence 2: {test_example['sentence2']}\nRelationship:"
    elif task == "qqp":
        test_text = f"Question 1: {test_example['question1']}\nQuestion 2: {test_example['question2']}\nAnswer:"
    elif task == "mrpc":
        test_text = f"Sentence 1: {test_example['sentence1']}\nSentence 2: {test_example['sentence2']}\nAnswer:"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Combine everything
    prompt = f"""{instruction}

Here are some examples:

{examples_text}

Now classify this example:

{test_text}"""
    
    return prompt


def get_zero_shot_prompt(task: str, example: dict[str, Any]) -> str:
    """
    Get zero-shot prompt for a task.
    
    Args:
        task: Task name
        example: Example dictionary with text keys
        
    Returns:
        Formatted prompt
    """
    template = ZERO_SHOT_PROMPTS[task]
    return template.format(**example)


def parse_model_output(output: str, task: str) -> str | None:
    """
    Parse model output to extract the label.
    
    Args:
        output: Raw model output
        task: Task name
        
    Returns:
        Extracted label or None if parsing failed
    """
    # Clean output
    output = output.strip().lower()
    
    # Remove common extra text
    output = output.split('\n')[0]  # Take first line
    output = output.split('.')[0]   # Remove sentence endings
    output = output.split(',')[0]   # Remove list continuations
    
    # Task-specific label extraction
    if task == "sst2":
        # Single token format (1/0)
        if output.startswith('1') or output == '1':
            return "positive"
        elif output.startswith('0') or output == '0':
            return "negative"
        # Fallback to word format
        elif "positive" in output:
            return "positive"
        elif "negative" in output:
            return "negative"
    
    elif task in ["mnli", "mnli_matched", "mnli_mismatched"]:
        if "entailment" in output and "not" not in output.split("entailment")[0]:
            return "entailment"
        elif "neutral" in output:
            return "neutral"
        elif "contradiction" in output:
            return "contradiction"
    
    elif task == "rte":
        # Single token format (1/0)
        if output.startswith('1') or output == '1':
            return "entailment"
        elif output.startswith('0') or output == '0':
            return "not_entailment"
        # Fallback to word format
        elif "not_entailment" in output or "not entailment" in output:
            return "not_entailment"
        elif "entailment" in output:
            return "entailment"
    
    elif task == "qqp":
        # Single token format (1/0)
        if output.startswith('1') or output == '1':
            return "duplicate"
        elif output.startswith('0') or output == '0':
            return "not_duplicate"
        # Fallback to word format
        elif "not_duplicate" in output or "not duplicate" in output:
            return "not_duplicate"
        elif "duplicate" in output:
            return "duplicate"
    
    elif task == "mrpc":
        # Single token format (1/0)
        if output.startswith('1') or output == '1':
            return "equivalent"
        elif output.startswith('0') or output == '0':
            return "not_equivalent"
        # Fallback to word format
        elif "not_equivalent" in output or "not equivalent" in output:
            return "not_equivalent"
        elif "equivalent" in output:
            return "equivalent"
    
    return None


def label_to_id(label: str, task: str) -> int:
    """
    Convert label string to integer ID.
    
    Args:
        label: Label string
        task: Task name
        
    Returns:
        Label ID
    """
    label_maps = {
        "sst2": {"negative": 0, "positive": 1},
        "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "mnli_matched": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "mnli_mismatched": {"entailment": 0, "neutral": 1, "contradiction": 2},
        "rte": {"entailment": 0, "not_entailment": 1},
        "qqp": {"not_duplicate": 0, "duplicate": 1},
        "mrpc": {"not_equivalent": 0, "equivalent": 1},
    }
    
    return label_maps[task][label]


def id_to_label(label_id: int, task: str) -> str:
    """
    Convert label ID to string.
    
    Args:
        label_id: Label integer ID
        task: Task name
        
    Returns:
        Label string
    """
    label_maps = {
        "sst2": {0: "negative", 1: "positive"},
        "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "mnli_matched": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "mnli_mismatched": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "rte": {0: "entailment", 1: "not_entailment"},
        "qqp": {0: "not_duplicate", 1: "duplicate"},
        "mrpc": {0: "not_equivalent", 1: "equivalent"},
    }
    
    return label_maps[task][label_id]

