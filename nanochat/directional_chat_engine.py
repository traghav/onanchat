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

        # Generate - Engine.generate() is a generator that yields (token_column, token_masks)
        response_tokens = []
        for token_column, token_masks in self.engine.generate(
            prompt,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        ):
            # Extract token from batch dimension (num_samples=1, so take first)
            tok = token_column[0]

            # Stop at assistant_end token
            if tok == assistant_end:
                break

            response_tokens.append(tok)

        # Handle empty response case
        if not response_tokens:
            # Return empty string with warning marker
            self.conversation_tokens.extend([assistant_start, assistant_end])
            return ""

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
                start_pos = i  # Track start for infinite loop detection
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != user_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                # Check if we reached end without finding user_end
                if i >= len(self.conversation_tokens):
                    raise ValueError(
                        f"Malformed conversation: user_start at position {start_pos-1} "
                        f"has no matching user_end token"
                    )
                messages.append({"role": "user", "content": self.tokenizer.decode(msg_tokens)})
                i += 1  # Skip user_end

            elif tok == assistant_start:
                # Extract assistant message
                i += 1
                msg_tokens = []
                start_pos = i  # Track start for infinite loop detection
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != assistant_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                # Check if we reached end without finding assistant_end
                if i >= len(self.conversation_tokens):
                    raise ValueError(
                        f"Malformed conversation: assistant_start at position {start_pos-1} "
                        f"has no matching assistant_end token"
                    )
                messages.append({"role": "assistant", "content": self.tokenizer.decode(msg_tokens)})
                i += 1  # Skip assistant_end
            else:
                i += 1

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation_tokens = [self.tokenizer.get_bos_token_id()]


class BackwardChatEngine(DirectionalChatEngine):
    """
    Backward chat: user provides endings, model generates preceding context.

    Internally maintains tokens in backward order, reverses for display.
    """

    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]

    def add_user_message(self, text: str) -> None:
        """Add user message (prepend to backward conversation)."""
        from nanochat.common import reverse_tokens

        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        user_tokens = self.tokenizer.encode(text)

        # Reverse the user tokens (but not the markers)
        reversed_tokens = reverse_tokens(user_tokens, keep_bos=False)

        # Build: user_end + reversed_tokens + user_start (backward order)
        msg_tokens = [user_end] + reversed_tokens + [user_start]

        # Prepend to conversation (after BOS)
        self.conversation_tokens = ([self.conversation_tokens[0]] +
                                   msg_tokens +
                                   self.conversation_tokens[1:])

    def generate_response(self, temperature: float = 0.6, top_k: int = 50,
                         max_tokens: int = 2048) -> str:
        """Generate response (prepend to backward conversation)."""
        from nanochat.common import reverse_tokens

        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        # Prompt: BOS + assistant_start + existing backward tokens
        prompt = ([self.conversation_tokens[0], assistant_start] +
                 self.conversation_tokens[1:])

        # Generate - Engine.generate() is a generator that yields (token_column, token_masks)
        response_tokens = []
        for token_column, token_masks in self.engine.generate(
            prompt,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        ):
            # Extract token from batch dimension (num_samples=1, so take first)
            tok = token_column[0]

            # Stop at assistant_end token
            if tok == assistant_end:
                break

            response_tokens.append(tok)

        # Handle empty response case
        if not response_tokens:
            # Prepend empty response markers
            self.conversation_tokens = ([self.conversation_tokens[0]] +
                                       [assistant_start, assistant_end] +
                                       self.conversation_tokens[1:])
            return ""

        # Prepend to conversation (backward order)
        self.conversation_tokens = ([self.conversation_tokens[0]] +
                                   [assistant_start] + response_tokens + [assistant_end] +
                                   self.conversation_tokens[1:])

        # Reverse for display
        display_tokens = reverse_tokens(response_tokens, keep_bos=False)
        return self.tokenizer.decode(display_tokens)

    def get_conversation_display(self) -> list[dict]:
        """Parse backward conversation and reverse for chronological display."""
        from nanochat.common import reverse_tokens

        # Parse messages in backward order
        messages = []
        i = 1  # Skip BOS

        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        while i < len(self.conversation_tokens):
            tok = self.conversation_tokens[i]

            if tok == assistant_start:
                # Extract assistant message (backward)
                i += 1
                msg_tokens = []
                start_pos = i  # Track start for infinite loop detection
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != assistant_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                # Check if we reached end without finding assistant_end
                if i >= len(self.conversation_tokens):
                    raise ValueError(
                        f"Malformed conversation: assistant_start at position {start_pos-1} "
                        f"has no matching assistant_end token"
                    )
                # Reverse for display
                display_tokens = reverse_tokens(msg_tokens, keep_bos=False)
                messages.append({"role": "assistant", "content": self.tokenizer.decode(display_tokens)})
                i += 1  # Skip assistant_end

            elif tok == user_end:
                # Extract user message (backward - starts with end token)
                i += 1
                msg_tokens = []
                start_pos = i  # Track start for infinite loop detection
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != user_start:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                # Check if we reached end without finding user_start
                if i >= len(self.conversation_tokens):
                    raise ValueError(
                        f"Malformed conversation: user_end at position {start_pos-1} "
                        f"has no matching user_start token"
                    )
                # Reverse for display
                display_tokens = reverse_tokens(msg_tokens, keep_bos=False)
                messages.append({"role": "user", "content": self.tokenizer.decode(display_tokens)})
                i += 1  # Skip user_start
            else:
                i += 1

        # Reverse entire message list for chronological order
        return list(reversed(messages))

    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation_tokens = [self.tokenizer.get_bos_token_id()]
