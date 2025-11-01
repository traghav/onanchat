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


class ForwardChatEngine(DirectionalChatEngine):
    """Standard left-to-right chat (forward direction)."""

    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]

    def add_user_message(self, text: str) -> None:
        """Add user message (append to conversation)."""
        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        user_tokens = self.tokenizer.encode(text)

        self.conversation_tokens.extend([user_start] + user_tokens + [user_end])

    def generate_response(self, temperature: float = 0.6, top_k: int = 50,
                         max_tokens: int = 2048) -> str:
        """Generate response (append to conversation)."""
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        # Build prompt with assistant_start
        prompt = self.conversation_tokens + [assistant_start]

        # Generate
        generated = self.engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Extract response tokens (everything after prompt until assistant_end)
        response_tokens = []
        for tok in generated[len(prompt):]:
            if tok == assistant_end:
                break
            response_tokens.append(tok)

        # Add to conversation
        self.conversation_tokens.extend([assistant_start] + response_tokens + [assistant_end])

        # Decode and return
        return self.tokenizer.decode(response_tokens)

    def get_conversation_display(self) -> list[dict]:
        """Parse conversation into display format."""
        messages = []
        i = 1  # Skip BOS

        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        while i < len(self.conversation_tokens):
            tok = self.conversation_tokens[i]

            if tok == user_start:
                # Extract user message
                i += 1
                msg_tokens = []
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != user_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                messages.append({"role": "user", "content": self.tokenizer.decode(msg_tokens)})
                i += 1  # Skip user_end

            elif tok == assistant_start:
                # Extract assistant message
                i += 1
                msg_tokens = []
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != assistant_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                messages.append({"role": "assistant", "content": self.tokenizer.decode(msg_tokens)})
                i += 1  # Skip assistant_end
            else:
                i += 1

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation_tokens = [self.tokenizer.get_bos_token_id()]
