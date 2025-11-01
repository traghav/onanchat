# Directional Chat Interface Design

**Date:** 2025-11-01
**Status:** Design Complete, Ready for Implementation

## Overview

Extend nanochat's chat interfaces (CLI and Web) to support interaction with forward, backward, and bidirectional language models. Each model type requires different interaction patterns:

- **Forward models**: Standard chat (no changes)
- **Backward models**: User provides endings, model generates preceding context
- **Bidirectional models**: Per-turn direction toggle between forward and backward generation

## Requirements

### Functional Requirements

1. **Direction Detection**: Automatically detect model direction from checkpoint metadata
2. **Forward Mode**: Standard left-to-right chat (existing behavior)
3. **Backward Mode**:
   - User provides message that represents the "ending"
   - Model generates what came before (context/setup leading to that ending)
   - Display in normal chronological order (reverse internally only)
4. **Bidirectional Mode**:
   - Per-turn direction toggle (forward ↔ backward)
   - User chooses direction before each generation
   - Model uses appropriate direction token (`<|forward|>` or `<|backward|>`)
5. **Direction Constraints**: Toggle only available for bidirectional models
6. **Interface Parity**: Both CLI and Web interfaces support all modes

### Non-Functional Requirements

- **Backward Compatibility**: Existing forward models continue to work without changes
- **Consistent Behavior**: Same chat logic across CLI and Web
- **Error Handling**: Clear messages when attempting invalid operations
- **Performance**: No significant overhead compared to current implementation

## Architecture

### Core Design Pattern: Wrapper Pattern

Create specialized chat engine classes for each direction, all implementing a common interface.

```
DirectionalChatEngine (Abstract Base)
├── ForwardChatEngine (standard chat)
├── BackwardChatEngine (reversed interaction)
└── BidirectionalChatEngine (toggleable direction)
```

### Component Overview

```
┌─────────────────────────────────────────────────┐
│         Chat Interface (CLI / Web)              │
│  - User input handling                          │
│  - Command processing                           │
│  - Display rendering                            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│      create_chat_engine() Factory               │
│  - Reads direction from model metadata          │
│  - Returns appropriate engine instance          │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┬────────────────┐
        ▼                 ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Forward    │  │   Backward   │  │ Bidirectional│
│ ChatEngine   │  │ ChatEngine   │  │ ChatEngine   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Engine (existing)  │
              │  - Token generation │
              │  - Sampling         │
              └─────────────────────┘
```

## Detailed Design

### 1. DirectionalChatEngine Base Class

Location: `nanochat/directional_chat_engine.py`

```python
class DirectionalChatEngine:
    """Base class for direction-aware chat engines."""

    def __init__(self, model, tokenizer, direction):
        self.model = model
        self.tokenizer = tokenizer
        self.direction = direction
        self.engine = Engine(model, tokenizer)

    def add_user_message(self, text: str) -> None:
        """Add user message to conversation."""
        raise NotImplementedError

    def generate_response(self, temperature: float, top_k: int,
                         max_tokens: int) -> str:
        """Generate model response."""
        raise NotImplementedError

    def get_conversation_display(self) -> list[dict]:
        """Get conversation for display (chronological order)."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear conversation history."""
        raise NotImplementedError

    def can_toggle_direction(self) -> bool:
        """Check if direction toggle is supported."""
        return False  # Only True for bidirectional

    def set_direction(self, direction: str) -> None:
        """Set direction for next generation (bidirectional only)."""
        raise NotImplementedError("Direction toggle not supported")
```

### 2. ForwardChatEngine

Standard left-to-right chat (existing behavior):

```python
class ForwardChatEngine(DirectionalChatEngine):
    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]

    def add_user_message(self, text):
        # Standard: append user message
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        tokens = [user_start] + tokenizer.encode(text) + [user_end]
        self.conversation_tokens.extend(tokens)

    def generate_response(self, temperature, top_k, max_tokens):
        # Standard: append assistant message
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        prompt = self.conversation_tokens + [assistant_start]

        generated = self.engine.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k
        )

        # Extract response text
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        response_tokens = []
        for tok in generated[len(prompt):]:
            if tok == assistant_end:
                break
            response_tokens.append(tok)

        self.conversation_tokens.extend([assistant_start] + response_tokens + [assistant_end])
        return tokenizer.decode(response_tokens)
```

### 3. BackwardChatEngine

User provides endings, model generates preceding context:

**Key Insight**: Maintain tokens in backward order internally, reverse only for display.

```python
class BackwardChatEngine(DirectionalChatEngine):
    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]

    def add_user_message(self, text):
        # User provides "ending" - prepend to backward sequence
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")

        # Encode user message
        user_tokens = tokenizer.encode(text)

        # Reverse and prepend (building backward)
        reversed_msg = list(reversed(user_tokens))
        tokens = [user_end] + reversed_msg + [user_start]

        # Prepend to conversation (after BOS)
        self.conversation_tokens = [self.conversation_tokens[0]] + tokens + self.conversation_tokens[1:]

    def generate_response(self, temperature, top_k, max_tokens):
        # Generate backward from current state
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")

        # Prompt: BOS + assistant_start + existing backward tokens
        prompt = [self.conversation_tokens[0], assistant_start] + self.conversation_tokens[1:]

        generated = self.engine.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k
        )

        # Extract response (until assistant_end)
        response_tokens = []
        for tok in generated[len(prompt):]:
            if tok == assistant_end:
                break
            response_tokens.append(tok)

        # Prepend to conversation (backward order)
        self.conversation_tokens = ([self.conversation_tokens[0]] +
                                    [assistant_start] + response_tokens + [assistant_end] +
                                    self.conversation_tokens[1:])

        # Return reversed for display
        return tokenizer.decode(list(reversed(response_tokens)))

    def get_conversation_display(self):
        # Parse messages from backward tokens, then reverse for chronological display
        messages = self._parse_messages_backward(self.conversation_tokens)
        return list(reversed(messages))
```

### 4. BidirectionalChatEngine

Per-turn direction toggle with direction tokens:

```python
class BidirectionalChatEngine(DirectionalChatEngine):
    def __init__(self, model, tokenizer, direction):
        super().__init__(model, tokenizer, direction)
        self.conversation_tokens = [tokenizer.get_bos_token_id()]
        self.current_direction = 'forward'  # Default
        self.forward_token = tokenizer.get_forward_token_id()
        self.backward_token = tokenizer.get_backward_token_id()

    def can_toggle_direction(self):
        return True

    def set_direction(self, direction):
        assert direction in ['forward', 'backward'], f"Invalid direction: {direction}"
        self.current_direction = direction

    def add_user_message(self, text):
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        user_tokens = tokenizer.encode(text)

        if self.current_direction == 'forward':
            # Append normally
            tokens = [user_start] + user_tokens + [user_end]
            self.conversation_tokens.extend(tokens)
        else:
            # Prepend (backward)
            reversed_tokens = list(reversed(user_tokens))
            tokens = [user_end] + reversed_tokens + [user_start]
            self.conversation_tokens = [self.conversation_tokens[0]] + tokens + self.conversation_tokens[1:]

    def generate_response(self, temperature, top_k, max_tokens):
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")

        # Select direction token
        direction_token = (self.forward_token if self.current_direction == 'forward'
                          else self.backward_token)

        if self.current_direction == 'forward':
            # Forward: append direction token and generate
            prompt = self.conversation_tokens + [direction_token, assistant_start]
            generated = self.engine.generate(prompt, max_tokens=max_tokens,
                                            temperature=temperature, top_k=top_k)

            # Extract response
            response_tokens = []
            for tok in generated[len(prompt):]:
                if tok == assistant_end:
                    break
                response_tokens.append(tok)

            # Append to conversation
            self.conversation_tokens.extend([direction_token, assistant_start] +
                                           response_tokens + [assistant_end])
            return tokenizer.decode(response_tokens)
        else:
            # Backward: prepend direction token and generate
            prompt = ([self.conversation_tokens[0], direction_token, assistant_start] +
                     self.conversation_tokens[1:])
            generated = self.engine.generate(prompt, max_tokens=max_tokens,
                                            temperature=temperature, top_k=top_k)

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

            # Return reversed for display
            return tokenizer.decode(list(reversed(response_tokens)))
```

### 5. Factory Function

```python
def create_chat_engine(model, tokenizer, meta):
    """Create appropriate chat engine based on model direction."""
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

## Interface Integration

### CLI Interface (`scripts/chat_cli.py`)

**Changes:**

1. Replace direct Engine usage with DirectionalChatEngine factory
2. Add direction commands for bidirectional models
3. Show direction indicator in prompt

```python
# Initialization
model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                    model_tag=args.model_tag, step=args.step)
chat_engine = create_chat_engine(model, tokenizer, meta)

# Display info
print(f"\nNanoChat - Model Direction: {meta.get('direction', 'forward')}")
if chat_engine.can_toggle_direction():
    print("Commands: /forward, /backward - toggle direction")
print("Commands: /clear - new conversation, /quit - exit")
print("-" * 50)

# Main loop
current_direction = 'forward'
while True:
    # Direction indicator in prompt
    arrow = '→' if current_direction == 'forward' else '←'
    user_input = input(f"\n{arrow} You: ")

    # Command handling
    if user_input == '/quit':
        break
    elif user_input == '/clear':
        chat_engine.clear()
        continue
    elif user_input == '/forward':
        if chat_engine.can_toggle_direction():
            current_direction = 'forward'
            chat_engine.set_direction('forward')
            print("Direction: forward →")
        else:
            print("Error: Direction toggle only available for bidirectional models")
        continue
    elif user_input == '/backward':
        if chat_engine.can_toggle_direction():
            current_direction = 'backward'
            chat_engine.set_direction('backward')
            print("Direction: backward ←")
        else:
            print("Error: Direction toggle only available for bidirectional models")
        continue

    # Normal message processing
    chat_engine.add_user_message(user_input)
    response = chat_engine.generate_response(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=2048
    )
    print(f"Assistant: {response}")
```

### Web Interface (`scripts/chat_web.py`)

**Changes:**

1. Replace Engine with DirectionalChatEngine in server state
2. Add `/set_direction` API endpoint
3. Add direction toggle UI component

**UI Additions:**
- Display current model direction at top of chat
- If bidirectional: show direction toggle (buttons or dropdown)
- Visual indicators: → for forward messages, ← for backward messages
- Disable toggle for non-bidirectional models

**API Endpoint:**
```python
@app.route('/set_direction', methods=['POST'])
def set_direction():
    direction = request.json.get('direction')
    if chat_engine.can_toggle_direction():
        if direction in ['forward', 'backward']:
            chat_engine.set_direction(direction)
            return jsonify({'success': True, 'direction': direction})
        return jsonify({'success': False, 'error': 'Invalid direction'}), 400
    return jsonify({'success': False, 'error': 'Not a bidirectional model'}), 400
```

## Error Handling

### Invalid Direction Toggle

```python
if user_input in ['/forward', '/backward']:
    if not chat_engine.can_toggle_direction():
        print(f"Error: Direction toggle only available for bidirectional models.")
        print(f"Current model direction: {meta.get('direction', 'forward')}")
        continue
```

### Unknown Direction in Metadata

```python
def create_chat_engine(model, tokenizer, meta):
    direction = meta.get('direction', 'forward')
    if direction not in ['forward', 'backward', 'bidirectional']:
        print(f"Warning: Unknown direction '{direction}', defaulting to forward")
        direction = 'forward'
    # ... create engine
```

### Context Length Overflow

All engines inherit the same context length handling:
- Truncate at max_tokens (2048 by default)
- Same behavior as existing implementation

## Edge Cases

1. **Empty Conversation**:
   - All engines start with just BOS token
   - First message initializes properly in all directions

2. **Single Message**:
   - Forward: Works normally
   - Backward: Displays correctly (reverses single message)
   - Bidirectional: Works with either direction

3. **Very Long Conversations**:
   - Same truncation behavior as current implementation
   - Backward engine reverses before truncation

4. **Direction Token Missing**:
   - Bidirectional models have these from training
   - Forward/backward models don't need them
   - Factory validates direction from metadata

5. **Mixed Direction History**:
   - Bidirectional can have forward and backward turns
   - Display shows direction indicator per turn
   - Model handles this (trained on mixed sequences)

## Backward Compatibility

- **Forward models**: Continue to work unchanged (direction defaults to 'forward')
- **Old checkpoints**: Missing direction metadata defaults to 'forward'
- **Existing scripts**: Can still use old Engine directly if preferred
- **API stability**: DirectionalChatEngine is additive, doesn't break existing code

## Testing Strategy

1. **Unit Tests**:
   - Test each engine class independently
   - Verify token ordering (forward vs backward)
   - Test direction toggle (bidirectional only)

2. **Integration Tests**:
   - CLI commands work correctly
   - Web API endpoints function properly
   - Direction constraints enforced

3. **Manual Testing**:
   - Chat with each model type
   - Verify display order (especially backward)
   - Test toggle behavior (bidirectional)

## Implementation Notes

### Files to Create
- `nanochat/directional_chat_engine.py` - Base class and three implementations
- Factory function in same file

### Files to Modify
- `scripts/chat_cli.py` - Replace Engine with DirectionalChatEngine
- `scripts/chat_web.py` - Replace Engine, add direction API endpoint

### Dependencies
- No new dependencies required
- Uses existing Engine, tokenizer, model infrastructure

## Success Criteria

✅ Forward models work exactly as before
✅ Backward models generate preceding context when given endings
✅ Backward conversations display in chronological order
✅ Bidirectional models support per-turn direction toggle
✅ Direction toggle only available for bidirectional models
✅ CLI and Web interfaces have feature parity
✅ Clear error messages for invalid operations
✅ Backward compatible with existing models

## Future Enhancements

- Conversation export/import (preserving direction metadata)
- Multi-turn backward generation (generate N preceding turns at once)
- Mixed-direction search (find best direction per turn automatically)
- Direction confidence scores (how well model performs in each direction)
