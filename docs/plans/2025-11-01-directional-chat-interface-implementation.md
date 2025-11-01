# Directional Chat Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend nanochat's chat interfaces to support forward, backward, and bidirectional language models with appropriate interaction patterns for each.

**Architecture:** Wrapper Pattern with DirectionalChatEngine base class and three concrete implementations (ForwardChatEngine, BackwardChatEngine, BidirectionalChatEngine). Factory function selects appropriate engine based on model metadata. Both CLI and Web interfaces use the same abstraction.

**Tech Stack:** Python, PyTorch, existing nanochat infrastructure (Engine, tokenizer, checkpoint_manager)

---

## Phase 1: Core Engine Infrastructure

### Task 1: Create DirectionalChatEngine Base Class

**Files:**
- Create: `nanochat/directional_chat_engine.py`
- Test: Manual import test

**Step 1: Create base class structure**

Create `nanochat/directional_chat_engine.py`:

```python
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
```

**Step 2: Test import**

Run:
```bash
python -c "from nanochat.directional_chat_engine import DirectionalChatEngine; print('✅ Base class imports')"
```
Expected: "✅ Base class imports"

**Step 3: Commit**

```bash
git add nanochat/directional_chat_engine.py
git commit -m "feat: add DirectionalChatEngine base class"
```

---

### Task 2: Implement ForwardChatEngine

**Files:**
- Modify: `nanochat/directional_chat_engine.py` (append)
- Test: Manual testing with print statements

**Step 1: Add ForwardChatEngine class**

Append to `nanochat/directional_chat_engine.py`:

```python


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
```

**Step 2: Test ForwardChatEngine imports**

Run:
```bash
python -c "from nanochat.directional_chat_engine import ForwardChatEngine; print('✅ ForwardChatEngine imports')"
```
Expected: "✅ ForwardChatEngine imports"

**Step 3: Commit**

```bash
git add nanochat/directional_chat_engine.py
git commit -m "feat: implement ForwardChatEngine for standard chat"
```

---

### Task 3: Implement BackwardChatEngine

**Files:**
- Modify: `nanochat/directional_chat_engine.py` (append)
- Test: Manual testing with token inspection

**Step 1: Add BackwardChatEngine class**

Append to `nanochat/directional_chat_engine.py`:

```python


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

        # Generate
        generated = self.engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Extract response tokens (until assistant_end)
        response_tokens = []
        for tok in generated[len(prompt):]:
            if tok == assistant_end:
                break
            response_tokens.append(tok)

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
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != assistant_end:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                # Reverse for display
                display_tokens = reverse_tokens(msg_tokens, keep_bos=False)
                messages.append({"role": "assistant", "content": self.tokenizer.decode(display_tokens)})
                i += 1  # Skip assistant_end

            elif tok == user_end:
                # Extract user message (backward - starts with end token)
                i += 1
                msg_tokens = []
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != user_start:
                    msg_tokens.append(self.conversation_tokens[i])
                    i += 1
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
```

**Step 2: Test BackwardChatEngine imports**

Run:
```bash
python -c "from nanochat.directional_chat_engine import BackwardChatEngine; print('✅ BackwardChatEngine imports')"
```
Expected: "✅ BackwardChatEngine imports"

**Step 3: Commit**

```bash
git add nanochat/directional_chat_engine.py
git commit -m "feat: implement BackwardChatEngine for reversed interaction"
```

---

### Task 4: Implement BidirectionalChatEngine

**Files:**
- Modify: `nanochat/directional_chat_engine.py` (append)
- Test: Manual testing with direction toggle

**Step 1: Add BidirectionalChatEngine class**

Append to `nanochat/directional_chat_engine.py`:

```python


class BidirectionalChatEngine(DirectionalChatEngine):
    """
    Bidirectional chat: per-turn direction toggle.

    Uses <|forward|> and <|backward|> direction tokens.
    """

    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]
        self.current_direction = 'forward'  # Default
        self.forward_token = tokenizer.get_forward_token_id()
        self.backward_token = tokenizer.get_backward_token_id()
        self.turn_directions = []  # Track direction of each turn

    def can_toggle_direction(self) -> bool:
        """Bidirectional models support direction toggle."""
        return True

    def set_direction(self, direction: str) -> None:
        """Set direction for next generation."""
        assert direction in ['forward', 'backward'], f"Invalid direction: {direction}"
        self.current_direction = direction

    def add_user_message(self, text: str) -> None:
        """Add user message in current direction."""
        from nanochat.common import reverse_tokens

        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        user_tokens = self.tokenizer.encode(text)

        if self.current_direction == 'forward':
            # Forward: append normally
            self.conversation_tokens.extend([user_start] + user_tokens + [user_end])
        else:
            # Backward: prepend
            reversed_tokens = reverse_tokens(user_tokens, keep_bos=False)
            msg_tokens = [user_end] + reversed_tokens + [user_start]
            self.conversation_tokens = ([self.conversation_tokens[0]] +
                                       msg_tokens +
                                       self.conversation_tokens[1:])

    def generate_response(self, temperature: float = 0.6, top_k: int = 50,
                         max_tokens: int = 2048) -> str:
        """Generate response in current direction."""
        from nanochat.common import reverse_tokens

        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        # Select direction token
        direction_token = (self.forward_token if self.current_direction == 'forward'
                          else self.backward_token)

        if self.current_direction == 'forward':
            # Forward: append direction token and generate
            prompt = self.conversation_tokens + [direction_token, assistant_start]

            generated = self.engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

            # Extract response
            response_tokens = []
            for tok in generated[len(prompt):]:
                if tok == assistant_end:
                    break
                response_tokens.append(tok)

            # Append to conversation
            self.conversation_tokens.extend([direction_token, assistant_start] +
                                           response_tokens + [assistant_end])

            self.turn_directions.append('forward')
            return self.tokenizer.decode(response_tokens)

        else:
            # Backward: prepend direction token and generate
            prompt = ([self.conversation_tokens[0], direction_token, assistant_start] +
                     self.conversation_tokens[1:])

            generated = self.engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

            # Extract response
            response_tokens = []
            for tok in generated[len(prompt):]:
                if tok == assistant_end:
                    break
                response_tokens.append(tok)

            # Prepend to conversation
            self.conversation_tokens = ([self.conversation_tokens[0], direction_token,
                                        assistant_start] + response_tokens + [assistant_end] +
                                       self.conversation_tokens[1:])

            self.turn_directions.append('backward')

            # Reverse for display
            display_tokens = reverse_tokens(response_tokens, keep_bos=False)
            return self.tokenizer.decode(display_tokens)

    def get_conversation_display(self) -> list[dict]:
        """
        Parse mixed-direction conversation.

        For now, use forward parsing (simplified).
        Full implementation would track which messages are which direction.
        """
        # Simplified: treat as forward for display
        # (Full implementation would parse based on turn_directions)
        messages = []
        i = 1  # Skip BOS

        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        # Skip direction tokens in parsing
        direction_tokens = {self.forward_token, self.backward_token}

        while i < len(self.conversation_tokens):
            tok = self.conversation_tokens[i]

            if tok in direction_tokens:
                i += 1  # Skip direction token
                continue

            if tok == user_start:
                i += 1
                msg_tokens = []
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != user_end:
                    if self.conversation_tokens[i] not in direction_tokens:
                        msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                messages.append({"role": "user", "content": self.tokenizer.decode(msg_tokens)})
                i += 1

            elif tok == assistant_start:
                i += 1
                msg_tokens = []
                while i < len(self.conversation_tokens) and self.conversation_tokens[i] != assistant_end:
                    if self.conversation_tokens[i] not in direction_tokens:
                        msg_tokens.append(self.conversation_tokens[i])
                    i += 1
                messages.append({"role": "assistant", "content": self.tokenizer.decode(msg_tokens)})
                i += 1
            else:
                i += 1

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.conversation_tokens = [self.tokenizer.get_bos_token_id()]
        self.turn_directions = []
        self.current_direction = 'forward'
```

**Step 2: Test BidirectionalChatEngine imports**

Run:
```bash
python -c "from nanochat.directional_chat_engine import BidirectionalChatEngine; print('✅ BidirectionalChatEngine imports')"
```
Expected: "✅ BidirectionalChatEngine imports"

**Step 3: Commit**

```bash
git add nanochat/directional_chat_engine.py
git commit -m "feat: implement BidirectionalChatEngine with direction toggle"
```

---

### Task 5: Add Factory Function

**Files:**
- Modify: `nanochat/directional_chat_engine.py` (append)
- Test: Manual testing with mock metadata

**Step 1: Add factory function**

Append to `nanochat/directional_chat_engine.py`:

```python


def create_chat_engine(model, tokenizer, meta):
    """
    Create appropriate chat engine based on model direction.

    Args:
        model: The language model
        tokenizer: Tokenizer instance
        meta: Metadata dict from checkpoint (contains 'direction' key)

    Returns:
        DirectionalChatEngine instance (Forward/Backward/Bidirectional)
    """
    direction = meta.get('direction', 'forward')

    # Validate and default
    if direction not in ['forward', 'backward', 'bidirectional']:
        print(f"Warning: Unknown direction '{direction}', defaulting to forward")
        direction = 'forward'

    # Instantiate appropriate engine
    if direction == 'forward':
        return ForwardChatEngine(model, tokenizer, direction)
    elif direction == 'backward':
        return BackwardChatEngine(model, tokenizer, direction)
    elif direction == 'bidirectional':
        return BidirectionalChatEngine(model, tokenizer, direction)
```

**Step 2: Test factory function**

Run:
```bash
python -c "from nanochat.directional_chat_engine import create_chat_engine, ForwardChatEngine; engine = create_chat_engine(None, None, {'direction': 'forward'}); assert isinstance(engine, ForwardChatEngine); print('✅ Factory function works')"
```
Expected: "✅ Factory function works"

**Step 3: Commit**

```bash
git add nanochat/directional_chat_engine.py
git commit -m "feat: add factory function for chat engine creation"
```

---

## Phase 2: CLI Interface Integration

### Task 6: Update chat_cli.py to Use DirectionalChatEngine

**Files:**
- Modify: `scripts/chat_cli.py:31-48` (engine initialization and main loop)
- Test: Manual testing with forward model

**Step 1: Replace Engine with DirectionalChatEngine**

In `scripts/chat_cli.py`, find the engine initialization section (around line 31) and replace:

```python
# OLD:
# model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
# engine = Engine(model, tokenizer)

# NEW:
from nanochat.directional_chat_engine import create_chat_engine

model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
chat_engine = create_chat_engine(model, tokenizer, meta)
```

**Step 2: Update welcome message**

Replace the welcome message section (around line 41-45):

```python
# OLD:
# print("\nNanoChat Interactive Mode")
# print("-" * 50)
# print("Type 'quit' or 'exit' to end the conversation")
# print("Type 'clear' to start a new conversation")
# print("-" * 50)

# NEW:
direction = meta.get('direction', 'forward')
print(f"\nNanoChat Interactive Mode - Model Direction: {direction}")
print("-" * 50)
print("Commands:")
print("  /quit, /exit - End conversation")
print("  /clear - Start new conversation")
if chat_engine.can_toggle_direction():
    print("  /forward - Switch to forward generation")
    print("  /backward - Switch to backward generation")
print("-" * 50)
```

**Step 3: Update main loop**

Find the main loop (starts around line 47) and replace the conversation handling:

```python
# OLD conversation_tokens initialization:
# conversation_tokens = [bos]

# NEW:
current_direction = 'forward'  # For display indicator

while True:
    # Direction indicator
    arrow = '→' if current_direction == 'forward' else '←'
    user_input = input(f"\n{arrow} You: ").strip()

    # Handle commands
    if user_input in ['/quit', '/exit', 'quit', 'exit']:
        print("Goodbye!")
        break

    elif user_input in ['/clear', 'clear']:
        chat_engine.clear()
        print("Conversation cleared.")
        continue

    elif user_input == '/forward':
        if chat_engine.can_toggle_direction():
            current_direction = 'forward'
            chat_engine.set_direction('forward')
            print("Direction: forward →")
        else:
            print(f"Error: Direction toggle only available for bidirectional models.")
            print(f"Current model direction: {direction}")
        continue

    elif user_input == '/backward':
        if chat_engine.can_toggle_direction():
            current_direction = 'backward'
            chat_engine.set_direction('backward')
            print("Direction: backward ←")
        else:
            print(f"Error: Direction toggle only available for bidirectional models.")
            print(f"Current model direction: {direction}")
        continue

    # Handle empty input
    if not user_input:
        continue

    # Add user message and generate response
    chat_engine.add_user_message(user_input)

    with autocast_ctx:
        response = chat_engine.generate_response(
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=2048
        )

    print(f"Assistant: {response}")
```

**Step 4: Remove old conversation management code**

Delete the old token-based conversation code that is no longer needed (the code that manually managed `conversation_tokens`, `user_start`, `user_end`, `assistant_start`, `assistant_end`, and the generation loop).

**Step 5: Test with existing forward model**

If you have a trained forward model, test:
```bash
python -m scripts.chat_cli --source=base --model-tag=<your-model>
```

Expected: Chat works normally, no direction toggle commands available

**Step 6: Commit**

```bash
git add scripts/chat_cli.py
git commit -m "feat: integrate DirectionalChatEngine into CLI"
```

---

## Phase 3: Web Interface Integration

### Task 7: Update chat_web.py Server State

**Files:**
- Modify: `scripts/chat_web.py` (server initialization and state)
- Test: Server starts without errors

**Step 1: Read current chat_web.py structure**

First, check the structure:
```bash
head -50 scripts/chat_web.py
```

**Step 2: Replace Engine with DirectionalChatEngine in server state**

Find where the global state is initialized (typically near the top of the file, after imports):

```python
# OLD:
# engine = None
# model = None
# tokenizer = None

# NEW:
from nanochat.directional_chat_engine import create_chat_engine

chat_engine = None
model = None
tokenizer = None
meta = None
```

**Step 3: Update model loading function**

Find the model loading function (often called `load_model_for_web` or similar):

```python
# OLD:
# def load_model_for_web(source, model_tag):
#     global engine, model, tokenizer
#     model, tokenizer, meta = load_model(...)
#     engine = Engine(model, tokenizer)

# NEW:
def load_model_for_web(source, model_tag):
    global chat_engine, model, tokenizer, meta
    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
    chat_engine = create_chat_engine(model, tokenizer, meta)
    return meta.get('direction', 'forward')
```

**Step 4: Test server starts**

Run:
```bash
python -m scripts.chat_web --source=base
```

Expected: Server starts without errors (you may get model not found, that's okay)

Press Ctrl+C to stop.

**Step 5: Commit**

```bash
git add scripts/chat_web.py
git commit -m "feat: integrate DirectionalChatEngine into web server state"
```

---

### Task 8: Add Direction Toggle API Endpoint

**Files:**
- Modify: `scripts/chat_web.py` (add new endpoint)
- Test: Manual curl test

**Step 1: Add /set_direction endpoint**

Find the section with API endpoints (usually decorated with `@app.route`) and add:

```python
@app.route('/set_direction', methods=['POST'])
def set_direction():
    """Set generation direction for bidirectional models."""
    global chat_engine, meta

    if chat_engine is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 400

    data = request.get_json()
    direction = data.get('direction')

    # Validate direction
    if direction not in ['forward', 'backward']:
        return jsonify({'success': False, 'error': 'Invalid direction. Must be "forward" or "backward"'}), 400

    # Check if model supports toggle
    if not chat_engine.can_toggle_direction():
        model_direction = meta.get('direction', 'forward')
        return jsonify({
            'success': False,
            'error': f'Direction toggle only available for bidirectional models. Current model is: {model_direction}'
        }), 400

    # Set direction
    chat_engine.set_direction(direction)
    return jsonify({'success': True, 'direction': direction})
```

**Step 2: Add /model_info endpoint**

Add an endpoint to get model direction info:

```python
@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model direction and capabilities."""
    global chat_engine, meta

    if chat_engine is None:
        return jsonify({'loaded': False})

    return jsonify({
        'loaded': True,
        'direction': meta.get('direction', 'forward'),
        'can_toggle': chat_engine.can_toggle_direction()
    })
```

**Step 3: Update chat endpoint**

Find the `/chat` or `/generate` endpoint and update it to use `chat_engine`:

```python
@app.route('/chat', methods=['POST'])
def chat():
    """Generate response to user message."""
    global chat_engine

    if chat_engine is None:
        return jsonify({'error': 'Model not loaded'}), 400

    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # Add message and generate
    chat_engine.add_user_message(user_message)
    response = chat_engine.generate_response(
        temperature=data.get('temperature', 0.6),
        top_k=data.get('top_k', 50),
        max_tokens=data.get('max_tokens', 2048)
    )

    return jsonify({
        'response': response,
        'success': True
    })
```

**Step 4: Add clear conversation endpoint**

```python
@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history."""
    global chat_engine

    if chat_engine is None:
        return jsonify({'error': 'Model not loaded'}), 400

    chat_engine.clear()
    return jsonify({'success': True})
```

**Step 5: Commit**

```bash
git add scripts/chat_web.py
git commit -m "feat: add direction toggle and chat API endpoints"
```

---

### Task 9: Update Web UI for Direction Toggle

**Files:**
- Modify: `scripts/chat_web.py` (HTML template or static files reference)
- Test: Manual browser test

**Step 1: Locate UI template**

The web UI is likely in one of these places:
- Inline HTML in `chat_web.py`
- Separate template file in `templates/`
- Separate static HTML in `static/`

Find it:
```bash
grep -n "<!DOCTYPE html>" scripts/chat_web.py || echo "Check templates/ or static/"
```

**Step 2: Add direction info display**

In the HTML, add a section to show model direction (near the top of the chat interface):

```html
<div id="model-info" style="padding: 10px; background: #f0f0f0; border-radius: 5px; margin-bottom: 10px;">
    <strong>Model:</strong> <span id="model-direction">Loading...</span>
    <span id="direction-toggle-container" style="display: none; margin-left: 20px;">
        <button onclick="setDirection('forward')" id="btn-forward">→ Forward</button>
        <button onclick="setDirection('backward')" id="btn-backward">← Backward</button>
    </span>
</div>
```

**Step 3: Add JavaScript for direction toggle**

Add JavaScript functions (in `<script>` section):

```javascript
// Fetch and display model info
async function loadModelInfo() {
    const response = await fetch('/model_info');
    const data = await response.json();

    if (data.loaded) {
        document.getElementById('model-direction').textContent = data.direction;

        // Show toggle if bidirectional
        if (data.can_toggle) {
            document.getElementById('direction-toggle-container').style.display = 'inline';
        }
    }
}

// Set direction
async function setDirection(direction) {
    const response = await fetch('/set_direction', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({direction: direction})
    });

    const data = await response.json();

    if (data.success) {
        // Visual feedback
        highlightActiveDirection(direction);
        console.log('Direction set to:', direction);
    } else {
        alert('Error: ' + data.error);
    }
}

// Highlight active direction button
function highlightActiveDirection(direction) {
    document.getElementById('btn-forward').style.fontWeight = direction === 'forward' ? 'bold' : 'normal';
    document.getElementById('btn-backward').style.fontWeight = direction === 'backward' ? 'bold' : 'normal';
}

// Load model info on page load
window.addEventListener('load', loadModelInfo);
```

**Step 4: Add direction indicator to messages**

Optionally, add visual indicators to show which direction was used:

```javascript
// When displaying messages, add direction indicator
function displayMessage(role, content, direction) {
    const messageDiv = document.createElement('div');
    messageDiv.className = role;

    const indicator = direction === 'backward' ? '← ' : '→ ';
    messageDiv.innerHTML = `<strong>${indicator}${role}:</strong> ${content}`;

    document.getElementById('chat-history').appendChild(messageDiv);
}
```

**Step 5: Test in browser**

Start the server and test in browser:
```bash
python -m scripts.chat_web --source=base
```

Open http://localhost:5000 and verify:
- Model direction displays
- Toggle buttons appear for bidirectional models
- Chat works

**Step 6: Commit**

```bash
git add scripts/chat_web.py
git commit -m "feat: add direction toggle UI to web interface"
```

---

## Phase 4: Documentation and Testing

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (add directional chat section)
- Test: Read the file to verify

**Step 1: Add directional chat section**

In `CLAUDE.md`, find the appropriate section (likely near chat features) and add:

```markdown
## Directional Chat Interface

The chat interfaces (CLI and Web) support models trained in different directions:

### Model Directions

1. **Forward Models** (standard)
   - Normal left-to-right chat
   - No special commands needed

2. **Backward Models**
   - User provides "ending" messages
   - Model generates what came before
   - Conversation displays in normal chronological order

3. **Bidirectional Models**
   - Per-turn direction toggle
   - CLI: Use `/forward` or `/backward` commands
   - Web: Use direction toggle buttons

### CLI Commands

```bash
# Chat with forward model (standard)
python -m scripts.chat_cli --source=sft

# Chat with backward model
python -m scripts.chat_cli --source=base --model-tag=d20_backward

# Chat with bidirectional model
python -m scripts.chat_cli --source=base --model-tag=d20_bidirectional
```

**Available commands during chat:**
- `/quit`, `/exit` - End conversation
- `/clear` - Start new conversation
- `/forward` - Switch to forward generation (bidirectional only)
- `/backward` - Switch to backward generation (bidirectional only)

### Web Interface

```bash
python -m scripts.chat_web --source=sft --model-tag=<model>
```

Open http://localhost:5000

For bidirectional models, use the direction toggle buttons (→ Forward / ← Backward).

### Implementation Details

**Architecture:**
- `DirectionalChatEngine` base class with three implementations
- `ForwardChatEngine` - standard chat
- `BackwardChatEngine` - reversed interaction with display reversal
- `BidirectionalChatEngine` - per-turn direction control

**Direction Detection:**
Models automatically advertise their direction via checkpoint metadata. Chat interfaces detect this and enable appropriate features.

**Direction Tokens:**
Bidirectional models use special tokens:
- `<|forward|>` - marks forward generation
- `<|backward|>` - marks backward generation
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document directional chat interface"
```

---

### Task 11: Create Simple Test Script

**Files:**
- Create: `test_directional_chat.py`
- Test: Run the test script

**Step 1: Create test script**

Create `test_directional_chat.py`:

```python
"""
Simple test script for directional chat engines.

Tests basic functionality without requiring trained models.
"""

from nanochat.directional_chat_engine import (
    DirectionalChatEngine,
    ForwardChatEngine,
    BackwardChatEngine,
    BidirectionalChatEngine,
    create_chat_engine
)


def test_base_class():
    """Test base class structure."""
    # Should not be able to instantiate directly
    try:
        engine = DirectionalChatEngine(None, None, 'forward')
        engine.add_user_message("test")
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass
    print("✅ Base class enforces implementation")


def test_factory():
    """Test factory function."""
    # Forward
    engine = create_chat_engine(None, None, {'direction': 'forward'})
    assert isinstance(engine, ForwardChatEngine)
    assert not engine.can_toggle_direction()

    # Backward
    engine = create_chat_engine(None, None, {'direction': 'backward'})
    assert isinstance(engine, BackwardChatEngine)
    assert not engine.can_toggle_direction()

    # Bidirectional
    engine = create_chat_engine(None, None, {'direction': 'bidirectional'})
    assert isinstance(engine, BidirectionalChatEngine)
    assert engine.can_toggle_direction()

    # Default to forward
    engine = create_chat_engine(None, None, {})
    assert isinstance(engine, ForwardChatEngine)

    # Invalid direction defaults to forward
    engine = create_chat_engine(None, None, {'direction': 'invalid'})
    assert isinstance(engine, ForwardChatEngine)

    print("✅ Factory function works correctly")


def test_direction_toggle():
    """Test direction toggle support."""
    # Forward: no toggle
    engine = create_chat_engine(None, None, {'direction': 'forward'})
    assert not engine.can_toggle_direction()
    try:
        engine.set_direction('backward')
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass

    # Bidirectional: has toggle
    engine = create_chat_engine(None, None, {'direction': 'bidirectional'})
    assert engine.can_toggle_direction()
    engine.set_direction('forward')  # Should not raise
    engine.set_direction('backward')  # Should not raise

    print("✅ Direction toggle support correct")


if __name__ == '__main__':
    print("Testing directional chat engines...")
    test_base_class()
    test_factory()
    test_direction_toggle()
    print("\n✅ All tests passed!")
```

**Step 2: Run tests**

Run:
```bash
python test_directional_chat.py
```

Expected output:
```
Testing directional chat engines...
✅ Base class enforces implementation
✅ Factory function works correctly
✅ Direction toggle support correct

✅ All tests passed!
```

**Step 3: Commit**

```bash
git add test_directional_chat.py
git commit -m "test: add basic tests for directional chat engines"
```

---

### Task 12: Create Usage Example

**Files:**
- Create: `docs/examples/directional_chat_example.py`
- Test: Read and verify code

**Step 1: Create examples directory**

```bash
mkdir -p docs/examples
```

**Step 2: Create example script**

Create `docs/examples/directional_chat_example.py`:

```python
"""
Example: Using directional chat engines programmatically.

This shows how to use the directional chat engines in your own code,
without going through the CLI or Web interfaces.
"""

import torch
from nanochat.checkpoint_manager import load_model
from nanochat.directional_chat_engine import create_chat_engine


def forward_chat_example():
    """Example: Chat with a forward model."""
    print("\n=== Forward Model Example ===")

    # Load model
    device = 'cpu'  # or 'cuda'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # Chat
    chat_engine.add_user_message("What is the capital of France?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"User: What is the capital of France?")
    print(f"Assistant: {response}")

    # Continue conversation
    chat_engine.add_user_message("What's the population?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"User: What's the population?")
    print(f"Assistant: {response}")


def backward_chat_example():
    """Example: Chat with a backward model."""
    print("\n=== Backward Model Example ===")

    # Load backward model
    device = 'cpu'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20_backward')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # User provides "ending"
    chat_engine.add_user_message("And that's how I solved the problem!")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)

    print(f"User provides ending: And that's how I solved the problem!")
    print(f"Model generates preceding context: {response}")

    # Get full conversation in chronological order
    conversation = chat_engine.get_conversation_display()
    print("\nFull conversation (chronological):")
    for msg in conversation:
        print(f"{msg['role']}: {msg['content']}")


def bidirectional_chat_example():
    """Example: Chat with a bidirectional model with direction toggle."""
    print("\n=== Bidirectional Model Example ===")

    # Load bidirectional model
    device = 'cpu'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20_bidirectional')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # Check if toggle available
    if not chat_engine.can_toggle_direction():
        print("Error: Model doesn't support direction toggle")
        return

    # Forward turn
    chat_engine.set_direction('forward')
    chat_engine.add_user_message("Tell me a story.")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Forward] User: Tell me a story.")
    print(f"[Forward] Assistant: {response}")

    # Switch to backward
    chat_engine.set_direction('backward')
    chat_engine.add_user_message("The end.")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Backward] User provides ending: The end.")
    print(f"[Backward] Assistant generates: {response}")

    # Switch back to forward
    chat_engine.set_direction('forward')
    chat_engine.add_user_message("What happened next?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Forward] User: What happened next?")
    print(f"[Forward] Assistant: {response}")


if __name__ == '__main__':
    print("Directional Chat Examples")
    print("=" * 50)

    # Note: These examples assume you have trained models
    # Replace model_tag with your actual model names

    # Uncomment to run:
    # forward_chat_example()
    # backward_chat_example()
    # bidirectional_chat_example()

    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nNote: Uncomment the function calls to run with your trained models.")
```

**Step 3: Commit**

```bash
git add docs/examples/directional_chat_example.py
git commit -m "docs: add directional chat usage examples"
```

---

## Summary

This implementation plan creates a complete directional chat interface for nanochat:

### Files Created:
- `nanochat/directional_chat_engine.py` - Core engine implementations
- `test_directional_chat.py` - Basic functionality tests
- `docs/examples/directional_chat_example.py` - Usage examples

### Files Modified:
- `scripts/chat_cli.py` - CLI with direction support
- `scripts/chat_web.py` - Web UI with direction toggle
- `CLAUDE.md` - Documentation

### Key Features:
- ✅ Automatic direction detection from model metadata
- ✅ Three engine types: Forward, Backward, Bidirectional
- ✅ CLI with direction toggle commands
- ✅ Web UI with direction toggle buttons
- ✅ Proper error handling for invalid operations
- ✅ Backward compatibility with existing forward models

### Testing Strategy:
1. Unit tests verify factory and toggle logic
2. Manual testing with trained models (forward/backward/bidirectional)
3. Browser testing for Web UI
4. CLI command testing

### Estimated Time:
- Phase 1 (Core): ~2-3 hours
- Phase 2 (CLI): ~30 minutes
- Phase 3 (Web): ~1 hour
- Phase 4 (Docs/Tests): ~30 minutes
- **Total: ~4-5 hours**

Ready for execution!
