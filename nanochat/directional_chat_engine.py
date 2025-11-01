"""
Direction-aware chat engines for forward, backward, and bidirectional models.

Provides specialized chat interaction patterns for each model direction:
- Forward: Standard left-to-right chat
- Backward: User provides endings, model generates preceding context
- Bidirectional: Per-turn direction toggle
"""

from nanochat.engine import Engine


class DirectionalChatEngine:
    """Base class for direction-aware chat engines."""

    def __init__(self, model, tokenizer, direction):
        """
        Initialize chat engine.

        Args:
            model: The language model
            tokenizer: Tokenizer instance
            direction: "forward", "backward", or "bidirectional"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.direction = direction
        self.engine = Engine(model, tokenizer)

    def add_user_message(self, text: str) -> None:
        """Add user message to conversation."""
        raise NotImplementedError("Subclasses must implement add_user_message")

    def generate_response(self, temperature: float = 0.6, top_k: int = 50,
                         max_tokens: int = 2048) -> str:
        """
        Generate model response.

        Args:
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        raise NotImplementedError("Subclasses must implement generate_response")

    def get_conversation_display(self) -> list[dict]:
        """
        Get conversation for display in chronological order.

        Returns:
            List of message dicts: [{"role": "user"|"assistant", "content": str}, ...]
        """
        raise NotImplementedError("Subclasses must implement get_conversation_display")

    def clear(self) -> None:
        """Clear conversation history."""
        raise NotImplementedError("Subclasses must implement clear")

    def can_toggle_direction(self) -> bool:
        """
        Check if direction toggle is supported.

        Returns:
            True only for bidirectional models
        """
        return False

    def set_direction(self, direction: str) -> None:
        """
        Set direction for next generation (bidirectional only).

        Args:
            direction: "forward" or "backward"

        Raises:
            NotImplementedError: If not bidirectional model
        """
        raise NotImplementedError("Direction toggle not supported for this model")
