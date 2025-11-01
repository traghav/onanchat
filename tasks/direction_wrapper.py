"""
Wrapper for evaluating tasks in multiple directions.

Handles token reversal for backward evaluation while preserving
the task interface.
"""

from nanochat.common import reverse_tokens


class DirectionTaskWrapper:
    """Wraps a task to evaluate in a specific direction."""

    def __init__(self, task, direction="forward", tokenizer=None):
        """
        Wrap task for direction-specific evaluation.

        Args:
            task: Original task (ARC, MMLU, etc.)
            direction: "forward" or "backward"
            tokenizer: Tokenizer instance (needed for direction tokens in bidirectional)
        """
        self.task = task
        self.direction = direction
        self.tokenizer = tokenizer

        if direction == "bidirectional":
            assert tokenizer is not None, "Tokenizer required for bidirectional"

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        """Get conversation with direction transformation applied."""
        conversation = self.task[idx]

        if self.direction == "forward":
            return conversation

        # For backward/bidirectional, we need to reverse the conversation
        # This is complex - for now, return as-is and handle in render
        # TODO: Implement proper conversation reversal
        return conversation

    def render_for_eval(self, conversation, model, tokenizer):
        """
        Render conversation for evaluation in specified direction.

        Returns:
            tokens: Token sequence ready for model
        """
        # Get normal tokens
        tokens = tokenizer.render_for_completion(conversation)

        if self.direction == "forward":
            return tokens
        elif self.direction == "backward":
            return reverse_tokens(tokens, keep_bos=True)
        elif self.direction == "bidirectional":
            # Use backward marker for now (could alternate)
            backward_token = tokenizer.get_backward_token_id()
            return [tokens[0], backward_token] + reverse_tokens(tokens[1:], keep_bos=False)

        return tokens
