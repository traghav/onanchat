"""
Multi-direction evaluation for backward language model research.

Evaluates models in both forward and backward directions to measure
native performance and cross-direction transfer.
"""

import torch
from tasks.direction_wrapper import DirectionTaskWrapper


def evaluate_task_multi_direction(task, model, tokenizer, directions=["forward", "backward"],
                                   engine=None, max_examples=None):
    """
    Evaluate task in multiple directions.

    Args:
        task: Task to evaluate (ARC, MMLU, etc.)
        model: Trained model
        tokenizer: Tokenizer instance
        directions: List of directions to test
        engine: Engine instance for generation (if needed)
        max_examples: Max number of examples to evaluate (None = all)

    Returns:
        {
            "forward": {"accuracy": 0.85, "per_example": [True, False, ...]},
            "backward": {"accuracy": 0.42, "per_example": [...]},
        }
    """
    results = {}

    for direction in directions:
        print(f"Evaluating in {direction} direction...")

        # Wrap task for this direction
        wrapped_task = DirectionTaskWrapper(task, direction=direction, tokenizer=tokenizer)

        # Evaluate (this is a simplified version - actual implementation depends on task type)
        correct = []
        total = min(len(wrapped_task), max_examples) if max_examples else len(wrapped_task)

        for i in range(total):
            conversation = wrapped_task[i]
            tokens = wrapped_task.render_for_eval(conversation, model, tokenizer)

            # For multiple choice tasks, we need to get logits for each choice
            # This is task-specific - for now, placeholder
            # TODO: Implement proper evaluation per task type
            is_correct = False  # Placeholder
            correct.append(is_correct)

        accuracy = sum(correct) / len(correct) if correct else 0.0
        results[direction] = {
            "accuracy": accuracy,
            "per_example": correct,
            "total": len(correct),
        }

    return results


def compute_direction_metrics(results):
    """
    Compute cross-direction metrics.

    Args:
        results: Output from evaluate_task_multi_direction

    Returns:
        {
            "forward_accuracy": 0.85,
            "backward_accuracy": 0.42,
            "direction_gap": 0.43,
            "average_accuracy": 0.635,
        }
    """
    metrics = {}

    if "forward" in results:
        metrics["forward_accuracy"] = results["forward"]["accuracy"]
    if "backward" in results:
        metrics["backward_accuracy"] = results["backward"]["accuracy"]

    if "forward" in results and "backward" in results:
        metrics["direction_gap"] = abs(
            results["forward"]["accuracy"] - results["backward"]["accuracy"]
        )
        metrics["average_accuracy"] = (
            results["forward"]["accuracy"] + results["backward"]["accuracy"]
        ) / 2

    return metrics
