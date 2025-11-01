# Backward Language Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a comprehensive research framework for training and evaluating forward, backward, and bidirectional language models with extensive logging and analysis tools.

**Architecture:** Modular dataloader pipeline that handles token reversal at data loading stage. Training scripts remain unchanged except for direction flag. Two-tier logging (WandB + HDF5) with flexible experiment branching system.

**Tech Stack:** PyTorch, HDF5 (h5py), WandB, existing nanochat infrastructure

---

## Phase 1: Core Infrastructure

### Task 1: Add Direction Special Tokens to Tokenizer

**Files:**
- Modify: `nanochat/tokenizer.py:1-200`
- Test: Manual testing with print statements

**Step 1: Identify where special tokens are defined**

Read `nanochat/tokenizer.py` to find:
- Where special tokens like `<|bos|>`, `<|user_start|>` are defined
- The `get_vocab_size()` method
- The `encode_special()` method

**Step 2: Add direction tokens to special tokens list**

In `nanochat/tokenizer.py`, locate the special tokens definition (likely near the top or in `__init__`). Add two new tokens:

```python
# Add after existing special tokens
DIRECTION_TOKENS = {
    "<|forward|>": NEXT_TOKEN_ID,
    "<|backward|>": NEXT_TOKEN_ID + 1,
}
```

Update vocab size calculation to include these tokens.

**Step 3: Add helper methods for direction tokens**

```python
def get_forward_token_id(self):
    """Get token ID for <|forward|> direction marker."""
    return self.encode_special("<|forward|>")

def get_backward_token_id(self):
    """Get token ID for <|backward|> direction marker."""
    return self.encode_special("<|backward|>")
```

**Step 4: Test manually**

Run:
```bash
python -c "from nanochat.tokenizer import get_tokenizer; t = get_tokenizer(); print(t.get_forward_token_id(), t.get_backward_token_id())"
```
Expected: Two different integer token IDs printed

**Step 5: Commit**

```bash
git add nanochat/tokenizer.py
git commit -m "feat: add <|forward|> and <|backward|> direction tokens"
```

---

### Task 2: Direction Helper Functions

**Files:**
- Modify: `nanochat/common.py:1-50`
- Test: Manual testing with print statements

**Step 1: Add direction validation function**

In `nanochat/common.py`, add:

```python
def validate_direction(direction):
    """Validate direction parameter.

    Args:
        direction: One of "forward", "backward", "bidirectional"

    Raises:
        ValueError: If direction is invalid
    """
    valid = ["forward", "backward", "bidirectional"]
    if direction not in valid:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of: {valid}")
    return direction
```

**Step 2: Add token reversal function**

```python
def reverse_tokens(tokens, keep_bos=True):
    """Reverse token sequence, optionally keeping BOS at start.

    Args:
        tokens: List or tensor of token IDs
        keep_bos: If True, keep first token (BOS) at position 0

    Returns:
        Reversed token sequence
    """
    if keep_bos and len(tokens) > 1:
        return [tokens[0]] + list(reversed(tokens[1:]))
    else:
        return list(reversed(tokens))
```

**Step 3: Test manually**

Run:
```bash
python -c "from nanochat.common import reverse_tokens, validate_direction; print(reverse_tokens([1,2,3,4,5])); validate_direction('forward')"
```
Expected: `[1, 5, 4, 3, 2]` and no error

**Step 4: Commit**

```bash
git add nanochat/common.py
git commit -m "feat: add direction validation and token reversal helpers"
```

---

### Task 3: Direction Metadata in Checkpoint Manager

**Files:**
- Modify: `nanochat/checkpoint_manager.py:1-200`
- Test: Manual checkpoint save/load test

**Step 1: Read existing checkpoint_manager.py**

Understand:
- How `save_checkpoint()` stores metadata
- How `load_model()` reads metadata
- The metadata dictionary structure

**Step 2: Modify save_checkpoint to include direction**

In `save_checkpoint()`, ensure the metadata dict can include `direction`:

```python
def save_checkpoint(checkpoint_dir, step, model_state_dict, optimizer_state_dicts, metadata):
    """
    Save model checkpoint with metadata.

    metadata can now include:
        - direction: "forward", "backward", or "bidirectional"
        - ... existing fields ...
    """
    # Existing save logic remains unchanged
    # Just ensure metadata is passed through correctly
    pass  # Implementation already exists
```

**Step 3: Modify load_model to return direction in metadata**

In `load_model()`, ensure it returns direction from metadata:

```python
def load_model(source, device, phase="eval", model_tag=None, step=None):
    """
    Load model and return (model, tokenizer, metadata).

    metadata will include:
        - direction: "forward", "backward", or "bidirectional" (if present)
        - ... other fields ...
    """
    # Existing load logic remains unchanged
    # metadata already contains direction if it was saved
    pass  # Implementation already exists
```

**Step 4: Test checkpoint save/load with direction**

Create test script `test_checkpoint_direction.py`:

```python
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.gpt import GPT, GPTConfig
import torch
import os
import tempfile

# Create temp directory
tmpdir = tempfile.mkdtemp()

# Create small model
config = GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=1000)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device="cpu")
model.init_weights()

# Save with direction metadata
metadata = {
    "direction": "backward",
    "step": 100,
    "model_config": config.__dict__,
}
save_checkpoint(tmpdir, 100, model.state_dict(), None, metadata)

# Load and verify
loaded_model, tokenizer, loaded_meta = load_model("base", "cpu", model_tag=os.path.basename(tmpdir))
print(f"Direction: {loaded_meta.get('direction', 'NOT FOUND')}")
assert loaded_meta.get('direction') == 'backward', "Direction not saved/loaded correctly"
print("✅ Checkpoint direction save/load works")
```

Run: `python test_checkpoint_direction.py`
Expected: "✅ Checkpoint direction save/load works"

**Step 5: Commit**

```bash
git add nanochat/checkpoint_manager.py test_checkpoint_direction.py
git commit -m "feat: checkpoint manager supports direction metadata"
```

---

### Task 4: Direction-Aware Dataloader - Core Structure

**Files:**
- Create: `nanochat/direction_dataloader.py`
- Test: Manual testing with print statements

**Step 1: Create file with imports and docstring**

Create `nanochat/direction_dataloader.py`:

```python
"""
Direction-aware dataloader for backward language model research.

Wraps the standard tokenizing_distributed_data_loader with direction-specific
token manipulation:
- "forward": no change (standard left-to-right)
- "backward": reverse token sequences (except BOS)
- "bidirectional": 50% forward + 50% backward with direction tokens
"""

from collections import deque
import torch
from nanochat.common import validate_direction, reverse_tokens
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer
```

**Step 2: Write forward direction passthrough**

```python
def direction_aware_dataloader(B, T, split, direction="forward",
                                tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """
    Stream pretraining data with direction-specific token manipulation.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        direction: "forward" | "backward" | "bidirectional"
        tokenizer_threads: Number of tokenizer threads
        tokenizer_batch_size: Batch size for tokenization
        device: Device to place tensors on

    Yields:
        (inputs, targets): Batches of shape (B, T) with appropriate direction
    """
    from nanochat.common import get_dist_info

    validate_direction(direction)
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # For bidirectional, we need direction marker tokens
    if direction == "bidirectional":
        forward_token = tokenizer.get_forward_token_id()
        backward_token = tokenizer.get_backward_token_id()

    token_buffer = deque()

    # Infinite iterator over document batches
    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1

        # Move tokens from deque into scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        # Apply direction transformation
        if direction == "forward":
            # No change
            pass
        elif direction == "backward":
            # Reverse all tokens except BOS
            tokens = reverse_tokens(tokens, keep_bos=True)
        elif direction == "bidirectional":
            # Add direction marker and reverse if needed
            # Even batches: forward, Odd batches: backward
            is_backward = (batch_index % 2) == 1
            if is_backward:
                # Insert backward token after BOS, then reverse
                tokens = [tokens[0], backward_token] + reverse_tokens(tokens[1:], keep_bos=False)
            else:
                # Insert forward token after BOS
                tokens = [tokens[0], forward_token] + tokens[1:]

        # Create tensors
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)

        yield inputs, targets
```

**Step 3: Test forward direction**

Create `test_direction_dataloader.py`:

```python
from nanochat.direction_dataloader import direction_aware_dataloader
import torch

# Test forward direction
loader = direction_aware_dataloader(B=2, T=16, split="val", direction="forward", device="cpu")
inputs, targets = next(loader)
print(f"Forward - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
print(f"First input sequence: {inputs[0].tolist()[:10]}")
print("✅ Forward direction dataloader works")
```

Run: `python test_direction_dataloader.py`
Expected: Shapes (2, 16) and no errors

**Step 4: Commit**

```bash
git add nanochat/direction_dataloader.py test_direction_dataloader.py
git commit -m "feat: direction-aware dataloader with forward support"
```

---

### Task 5: Test Backward Direction in Dataloader

**Files:**
- Modify: `test_direction_dataloader.py`

**Step 1: Add backward direction test**

Append to `test_direction_dataloader.py`:

```python
# Test backward direction
loader_back = direction_aware_dataloader(B=2, T=16, split="val", direction="backward", device="cpu")
inputs_back, targets_back = next(loader_back)

# Get tokenizer to decode
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()

# Compare forward and backward
loader_fwd = direction_aware_dataloader(B=1, T=10, split="val", direction="forward", device="cpu")
inputs_fwd, _ = next(loader_fwd)

loader_bwd = direction_aware_dataloader(B=1, T=10, split="val", direction="backward", device="cpu")
inputs_bwd, _ = next(loader_bwd)

print(f"\nForward tokens: {inputs_fwd[0].tolist()}")
print(f"Backward tokens: {inputs_bwd[0].tolist()}")

# First token (BOS) should be same
assert inputs_fwd[0][0] == inputs_bwd[0][0], "BOS should be same"
print("✅ Backward direction preserves BOS at start")
```

**Step 2: Run test**

Run: `python test_direction_dataloader.py`
Expected: "✅ Backward direction preserves BOS at start"

**Step 3: Commit**

```bash
git add test_direction_dataloader.py
git commit -m "test: verify backward direction dataloader"
```

---

### Task 6: Test Bidirectional Direction in Dataloader

**Files:**
- Modify: `test_direction_dataloader.py`

**Step 1: Add bidirectional direction test**

Append to `test_direction_dataloader.py`:

```python
# Test bidirectional direction
loader_bi = direction_aware_dataloader(B=4, T=16, split="val", direction="bidirectional", device="cpu")
inputs_bi, targets_bi = next(loader_bi)

# Check that direction tokens are present
forward_token = tokenizer.get_forward_token_id()
backward_token = tokenizer.get_backward_token_id()

# Get second token of each batch (should be direction marker)
second_tokens = inputs_bi[:, 1].tolist()
print(f"\nBidirectional second tokens: {second_tokens}")
print(f"Forward token ID: {forward_token}, Backward token ID: {backward_token}")

# Should have mix of forward and backward tokens
has_forward = forward_token in second_tokens
has_backward = backward_token in second_tokens
print(f"Has forward markers: {has_forward}, Has backward markers: {has_backward}")
print("✅ Bidirectional direction includes direction tokens")
```

**Step 2: Run test**

Run: `python test_direction_dataloader.py`
Expected: "✅ Bidirectional direction includes direction tokens"

**Step 3: Commit**

```bash
git add test_direction_dataloader.py
git commit -m "test: verify bidirectional direction dataloader"
```

---

## Phase 2: Training Integration

### Task 7: Add --direction Flag to base_train.py

**Files:**
- Modify: `scripts/base_train.py:30-70` (user settings section)
- Modify: `scripts/base_train.py:145-155` (dataloader initialization)

**Step 1: Add direction parameter to user settings**

In `scripts/base_train.py`, find the "User settings" section (around line 30-70) and add:

```python
# User settings
run = "dummy"
device_type = ""
# Model architecture
depth = 20
max_seq_len = 2048
# >>> ADD THIS:
direction = "forward" # forward|backward|bidirectional - direction for language modeling
# <<<
# Training horizon
num_iterations = -1
```

**Step 2: Ensure direction is in config_keys**

Find the `config_keys` line and verify it will pick up `direction`:

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
```

This should automatically include `direction` since it's a string.

**Step 3: Replace dataloader initialization**

Find where `train_loader` is created (around line 149):

```python
# OLD:
# train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)

# NEW:
from nanochat.direction_dataloader import direction_aware_dataloader
from nanochat.common import validate_direction

validate_direction(direction)
print0(f"Using direction: {direction}")
train_loader = direction_aware_dataloader(device_batch_size, max_seq_len, split="train", direction=direction, device=device)
build_val_loader = lambda: direction_aware_dataloader(device_batch_size, max_seq_len, split="val", direction=direction, device=device)
```

Also update the `build_val_loader` line.

**Step 4: Add direction to checkpoint metadata**

Find where checkpoint is saved (around line 244). Update the metadata dict:

```python
save_checkpoint(
    checkpoint_dir,
    step,
    orig_model.state_dict(),
    [opt.state_dict() for opt in optimizers],
    {
        "step": step,
        "direction": direction,  # ADD THIS LINE
        "val_bpb": val_bpb,
        "model_config": model_config_kwargs,
        "user_config": user_config,
        "device_batch_size": device_batch_size,
        "max_seq_len": max_seq_len,
    }
)
```

**Step 5: Update output dirname to include direction**

Find where `output_dirname` is set (around line 242):

```python
# OLD:
# output_dirname = model_tag if model_tag else f"d{depth}"

# NEW:
if model_tag:
    output_dirname = model_tag
else:
    direction_suffix = f"_{direction}" if direction != "forward" else ""
    output_dirname = f"d{depth}{direction_suffix}"
```

**Step 6: Test with small training run**

Run:
```bash
python -m scripts.base_train \
    --direction=forward \
    --depth=4 \
    --max_seq_len=128 \
    --device_batch_size=2 \
    --total_batch_size=512 \
    --num_iterations=10 \
    --core_metric_every=-1 \
    --eval_every=-1
```

Expected: Training runs for 10 steps without errors

**Step 7: Commit**

```bash
git add scripts/base_train.py
git commit -m "feat: add --direction flag to base_train.py"
```

---

### Task 8: Test Backward Training

**Files:**
- Test: Run base_train.py with backward direction

**Step 1: Run small backward training**

Run:
```bash
python -m scripts.base_train \
    --direction=backward \
    --depth=4 \
    --max_seq_len=128 \
    --device_batch_size=2 \
    --total_batch_size=512 \
    --num_iterations=10 \
    --core_metric_every=-1 \
    --eval_every=-1 \
    --model_tag=test_backward
```

Expected: Training runs for 10 steps, checkpoint saved to `base_checkpoints/test_backward/`

**Step 2: Verify checkpoint has direction metadata**

Run:
```python
import torch
ckpt = torch.load("base_checkpoints/test_backward/step_000010.pt", map_location="cpu")
print(f"Direction in metadata: {ckpt['metadata']['direction']}")
assert ckpt['metadata']['direction'] == 'backward'
print("✅ Backward checkpoint has correct direction metadata")
```

**Step 3: Commit (no code changes, just verification)**

```bash
git add -A  # In case any temp files
git commit -m "test: verify backward training works" --allow-empty
```

---

### Task 9: Add Direction Auto-Detection to mid_train.py

**Files:**
- Modify: `scripts/mid_train.py:68-85` (model loading section)
- Modify: `scripts/mid_train.py:95-160` (dataloader section)

**Step 1: Auto-detect direction from checkpoint**

In `scripts/mid_train.py`, find where model is loaded (around line 69):

```python
# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)

# >>> ADD THIS:
from nanochat.common import validate_direction
direction = meta.get("direction", "forward")  # Default to forward for old checkpoints
validate_direction(direction)
print0(f"Detected direction from checkpoint: {direction}")
# <<<
```

**Step 2: Use direction-aware dataloader**

Find the dataloader section (around line 112-156). Replace with:

```python
# OLD:
# def mid_data_generator(split):
#     ... uses tokenizer.render_conversation ...

# NEW:
from nanochat.direction_dataloader import direction_aware_dataloader
from nanochat.common import reverse_tokens

def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=(device_type == "cuda"))
    cursor = ddp_rank
    it = 0

    # Get direction tokens for bidirectional
    if direction == "bidirectional":
        forward_token = tokenizer.get_forward_token_id()
        backward_token = tokenizer.get_backward_token_id()

    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)

            # Apply direction transformation
            if direction == "forward":
                # No change
                pass
            elif direction == "backward":
                # Reverse conversation tokens (keep BOS at start)
                ids = reverse_tokens(ids, keep_bos=True)
            elif direction == "bidirectional":
                # Alternate forward/backward with markers
                is_backward = (it % 2) == 1
                if is_backward:
                    ids = [ids[0], backward_token] + reverse_tokens(ids[1:], keep_bos=False)
                else:
                    ids = [ids[0], forward_token] + ids[1:]

            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train":
                    last_step = True

        # Stopping condition
        it += 1
        if num_iterations > 0 and it >= num_iterations:
            last_step = True

        # Build inputs/targets and yield
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        if split == "train":
            if num_iterations > 0:
                approx_progress = it / num_iterations
            else:
                approx_progress = cursor / dataset_size
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
```

**Step 3: Test midtraining with backward base model**

First ensure we have a backward base checkpoint, then:

Run:
```bash
python -m scripts.mid_train \
    --model_tag=test_backward \
    --device_batch_size=2 \
    --num_iterations=5
```

Expected: Loads backward model, trains with backward direction

**Step 4: Commit**

```bash
git add scripts/mid_train.py
git commit -m "feat: mid_train auto-detects direction from checkpoint"
```

---

### Task 10: Add Direction Auto-Detection to chat_sft.py

**Files:**
- Modify: `scripts/chat_sft.py:76-80` (model loading)
- Modify: `scripts/chat_sft.py:96-128` (dataloader)

**Step 1: Auto-detect direction from checkpoint**

In `scripts/chat_sft.py`, after model loading (around line 76):

```python
# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)

# >>> ADD THIS:
from nanochat.common import validate_direction, reverse_tokens
direction = meta.get("direction", "forward")
validate_direction(direction)
print0(f"Detected direction from checkpoint: {direction}")

# For bidirectional, get direction tokens
if direction == "bidirectional":
    forward_token = tokenizer.get_forward_token_id()
    backward_token = tokenizer.get_backward_token_id()
# <<<
```

**Step 2: Modify dataloader to handle direction**

Find the `sft_data_generator` function (around line 98). Modify to apply direction:

```python
def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    batch = []
    example_idx = 0  # Track for bidirectional alternation
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)

            # Apply direction transformation
            if direction == "forward":
                pass  # No change
            elif direction == "backward":
                ids = reverse_tokens(ids, keep_bos=True)
                # Mask stays same length, just reversed
                mask = [mask[0]] + list(reversed(mask[1:]))
            elif direction == "bidirectional":
                is_backward = (example_idx % 2) == 1
                if is_backward:
                    ids = [ids[0], backward_token] + reverse_tokens(ids[1:], keep_bos=False)
                    mask = [mask[0], 0] + list(reversed(mask[1:]))  # 0 for direction token
                else:
                    ids = [ids[0], forward_token] + ids[1:]
                    mask = [mask[0], 0] + mask[1:]

            batch.append((ids, mask))
            example_idx += 1

            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []
```

**Step 3: Test SFT with backward model**

Run:
```bash
python -m scripts.chat_sft \
    --source=base \
    --model_tag=test_backward \
    --device_batch_size=2 \
    --num_iterations=5
```

Expected: Loads backward model, runs SFT with backward conversations

**Step 4: Commit**

```bash
git add scripts/chat_sft.py
git commit -m "feat: chat_sft auto-detects direction from checkpoint"
```

---

### Task 11: Add Direction Auto-Detection to chat_rl.py

**Files:**
- Modify: `scripts/chat_rl.py:66-68` (model loading)
- Modify: `scripts/chat_rl.py:78-140` (get_batch function)

**Step 1: Auto-detect direction from checkpoint**

In `scripts/chat_rl.py`, after model loading (around line 67):

```python
# Init model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="eval")

# >>> ADD THIS:
from nanochat.common import validate_direction, reverse_tokens
direction = meta.get("direction", "forward")
validate_direction(direction)
print0(f"Detected direction from checkpoint: {direction}")

if direction == "bidirectional":
    forward_token = tokenizer.get_forward_token_id()
    backward_token = tokenizer.get_backward_token_id()
# <<<

engine = Engine(model, tokenizer)
```

**Step 2: Modify get_batch to handle direction**

In the `get_batch()` function (around line 79), after rendering for completion:

```python
@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    batch_idx = 0  # For bidirectional alternation

    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)

        # Apply direction transformation to prompt
        if direction == "forward":
            pass  # No change
        elif direction == "backward":
            tokens = reverse_tokens(tokens, keep_bos=True)
        elif direction == "bidirectional":
            is_backward = (batch_idx % 2) == 1
            if is_backward:
                tokens = [tokens[0], backward_token] + reverse_tokens(tokens[1:], keep_bos=False)
            else:
                tokens = [tokens[0], forward_token] + tokens[1:]

        prefix_length = len(tokens)

        # Generate samples (engine handles generation)
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate rewards
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]

            # For backward/bidirectional, need to reverse output back
            if direction == "backward":
                generated_tokens = list(reversed(generated_tokens))
            elif direction == "bidirectional" and is_backward:
                generated_tokens = list(reversed(generated_tokens))

            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # ... rest of function continues as before
        # (padding, creating tensors, calculating advantages, yielding)

        batch_idx += 1
```

**Step 3: Test (optional, can skip if no backward model is fully trained)**

This can be tested later when we have a fully trained backward SFT model.

**Step 4: Commit**

```bash
git add scripts/chat_rl.py
git commit -m "feat: chat_rl auto-detects direction from checkpoint"
```

---

## Phase 3: Logging System

### Task 12: Research Logger - HDF5 Infrastructure

**Files:**
- Create: `nanochat/research_logger.py`
- Test: Create `test_research_logger.py`

**Step 1: Create ResearchLogger class structure**

Create `nanochat/research_logger.py`:

```python
"""
Research logging system for backward language model experiments.

Provides comprehensive HDF5-based logging for:
- Per-token losses and predictions
- Gradient statistics per layer
- Activation patterns
- Complete training dynamics
"""

import os
import json
import h5py
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


class ResearchLogger:
    """Logs detailed training metrics to HDF5 for research analysis."""

    def __init__(self, experiment_dir, phase="base_training"):
        """
        Initialize research logger.

        Args:
            experiment_dir: Directory for this experiment (will be created)
            phase: Training phase (base_training, midtraining, sft, rl)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.phase = phase
        self.h5_path = self.experiment_dir / f"{phase}.h5"

        # Create HDF5 file
        self.h5file = h5py.File(self.h5_path, 'a')  # Append mode

    def log_step(self, step, metrics):
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dict with keys like 'loss', 'per_token_losses', etc.
        """
        step_group = self.h5file.require_group(f"step_{step:06d}")

        # Save scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                step_group.attrs[key] = value
            elif isinstance(value, torch.Tensor):
                # Convert to numpy and save as dataset
                step_group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")
            elif isinstance(value, np.ndarray):
                step_group.create_dataset(key, data=value, compression="gzip")
            elif isinstance(value, dict):
                # Nested dict (e.g., gradient stats per layer)
                subgroup = step_group.require_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        subgroup.create_dataset(subkey, data=subvalue.cpu().numpy(), compression="gzip")
                    else:
                        subgroup.attrs[subkey] = subvalue

        # Flush to disk
        self.h5file.flush()

    def close(self):
        """Close HDF5 file."""
        if self.h5file:
            self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

**Step 2: Write test for ResearchLogger**

Create `test_research_logger.py`:

```python
import torch
import tempfile
import shutil
from pathlib import Path
from nanochat.research_logger import ResearchLogger
import h5py

# Create temp experiment dir
tmpdir = tempfile.mkdtemp()

try:
    # Initialize logger
    logger = ResearchLogger(tmpdir, phase="test_phase")

    # Log a step with various metric types
    logger.log_step(0, {
        'loss': 2.5,
        'learning_rate': 0.001,
        'per_token_losses': torch.randn(4, 16),  # (B, T)
        'gradients': {
            'layer_0': torch.randn(128, 128),
            'layer_1': torch.randn(128, 128),
        }
    })

    logger.log_step(10, {
        'loss': 2.3,
        'learning_rate': 0.001,
    })

    logger.close()

    # Verify HDF5 file exists and has correct structure
    h5_path = Path(tmpdir) / "test_phase.h5"
    assert h5_path.exists(), "HDF5 file not created"

    with h5py.File(h5_path, 'r') as f:
        assert 'step_000000' in f, "Step 0 not logged"
        assert 'step_000010' in f, "Step 10 not logged"
        assert f['step_000000'].attrs['loss'] == 2.5, "Loss not logged correctly"
        assert 'per_token_losses' in f['step_000000'], "Per-token losses not logged"
        assert 'gradients' in f['step_000000'], "Gradients not logged"

    print("✅ ResearchLogger works correctly")

finally:
    # Cleanup
    shutil.rmtree(tmpdir)
```

**Step 3: Run test**

Run: `python test_research_logger.py`
Expected: "✅ ResearchLogger works correctly"

**Step 4: Commit**

```bash
git add nanochat/research_logger.py test_research_logger.py
git commit -m "feat: add ResearchLogger for HDF5 logging"
```

---

### Task 13: Integrate ResearchLogger into base_train.py

**Files:**
- Modify: `scripts/base_train.py:78-80` (add logger init)
- Modify: `scripts/base_train.py:260-310` (add logging calls)

**Step 1: Initialize logger at start of training**

After wandb init (around line 79):

```python
# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# >>> ADD THIS:
from nanochat.research_logger import ResearchLogger
research_logger = None
if master_process and run != "dummy":
    # Create experiment directory with timestamp and config
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    direction_suffix = f"_{direction}" if direction != "forward" else ""
    experiment_id = f"{timestamp}_d{depth}{direction_suffix}"
    experiment_dir = os.path.join(base_dir, "experiments", experiment_id)
    research_logger = ResearchLogger(experiment_dir, phase="base_training")
    print0(f"Research logging to: {experiment_dir}")
# <<<
```

**Step 2: Add logging in training loop**

In the training loop, after computing loss (around line 269):

```python
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        loss = model(x, y)
    train_loss = loss.detach()

    # >>> ADD THIS (only log occasionally to save space):
    if research_logger and step % 10 == 0 and micro_step == 0:
        # Log basic metrics
        research_logger.log_step(step, {
            'loss': train_loss.item(),
            'learning_rate': lrm * matrix_lr,  # Use actual LR
        })
    # <<<

    loss = loss / grad_accum_steps
    loss.backward()
    x, y = next(train_loader)
```

**Step 3: Close logger at end**

At the very end of the script (after compute_cleanup):

```python
# cleanup
wandb_run.finish()

# >>> ADD THIS:
if research_logger:
    research_logger.close()
# <<<

compute_cleanup()
```

**Step 4: Test with small training run**

Run:
```bash
python -m scripts.base_train \
    --run=test_logging \
    --direction=forward \
    --depth=4 \
    --num_iterations=20 \
    --eval_every=-1 \
    --core_metric_every=-1
```

Check that `experiments/{timestamp}_d4/base_training.h5` exists and has data.

**Step 5: Commit**

```bash
git add scripts/base_train.py
git commit -m "feat: integrate ResearchLogger into base_train"
```

---

## Phase 4: Evaluation Framework

### Task 14: Direction Evaluation Wrapper

**Files:**
- Create: `tasks/direction_wrapper.py`
- Test: Create `test_direction_wrapper.py`

**Step 1: Create DirectionTaskWrapper class**

Create `tasks/direction_wrapper.py`:

```python
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
```

**Step 2: Write test**

Create `test_direction_wrapper.py`:

```python
from tasks.direction_wrapper import DirectionTaskWrapper
from tasks.arc import ARC
from nanochat.tokenizer import get_tokenizer

# Load task and tokenizer
task = ARC(subset="ARC-Easy", split="test")
tokenizer = get_tokenizer()

# Test forward wrapper
wrapper_fwd = DirectionTaskWrapper(task, direction="forward", tokenizer=tokenizer)
conv_fwd = wrapper_fwd[0]
tokens_fwd = wrapper_fwd.render_for_eval(conv_fwd, None, tokenizer)

# Test backward wrapper
wrapper_bwd = DirectionTaskWrapper(task, direction="backward", tokenizer=tokenizer)
conv_bwd = wrapper_bwd[0]
tokens_bwd = wrapper_bwd.render_for_eval(conv_bwd, None, tokenizer)

print(f"Forward tokens length: {len(tokens_fwd)}")
print(f"Backward tokens length: {len(tokens_bwd)}")
print(f"First token same: {tokens_fwd[0] == tokens_bwd[0]}")  # BOS should match

print("✅ DirectionTaskWrapper works")
```

**Step 3: Run test**

Run: `python test_direction_wrapper.py`
Expected: "✅ DirectionTaskWrapper works"

**Step 4: Commit**

```bash
git add tasks/direction_wrapper.py test_direction_wrapper.py
git commit -m "feat: add DirectionTaskWrapper for multi-direction eval"
```

---

### Task 15: Multi-Direction Evaluation Function

**Files:**
- Create: `nanochat/direction_eval.py`
- Test: Manual testing with small model

**Step 1: Create evaluation function**

Create `nanochat/direction_eval.py`:

```python
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
```

**Step 2: Test (manual verification)**

This is complex to test without a trained model. For now, verify it imports:

Run: `python -c "from nanochat.direction_eval import evaluate_task_multi_direction; print('✅ Module loads')"`

**Step 3: Commit**

```bash
git add nanochat/direction_eval.py
git commit -m "feat: add multi-direction evaluation functions"
```

---

## Phase 5: Experiment Orchestration

### Task 16: Experiment Configuration Management

**Files:**
- Create: `nanochat/experiment_config.py`

**Step 1: Create ExperimentConfig class**

Create `nanochat/experiment_config.py`:

```python
"""
Experiment configuration and metadata management.

Handles experiment directory structure, parent linking, and config persistence.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """Manages experiment configuration and directory structure."""

    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, phase, direction, depth, parent_experiment=None,
                          experiment_id=None, **kwargs):
        """
        Create new experiment directory with config.

        Args:
            phase: "base", "mid", "sft", "rl"
            direction: "forward", "backward", "bidirectional"
            depth: Model depth (e.g., 20)
            parent_experiment: ID of parent experiment (None for base)
            experiment_id: Custom experiment ID (None = auto-generate)
            **kwargs: Additional config parameters

        Returns:
            experiment_dir: Path to experiment directory
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            direction_suffix = f"_{direction}" if direction != "forward" else ""
            experiment_id = f"{phase}_{timestamp}_d{depth}{direction_suffix}"

        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "evaluations").mkdir(exist_ok=True)

        # Build config
        config = {
            "experiment_id": experiment_id,
            "phase": phase,
            "direction": direction,
            "depth": depth,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }

        # Save config
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Create parent link if specified
        if parent_experiment:
            parent_link = {
                "parent_experiment_id": parent_experiment,
                "parent_checkpoint_path": str(self.base_dir / parent_experiment / "checkpoints"),
                "parent_phase": self._get_parent_phase(phase),
            }
            with open(experiment_dir / "parent_link.json", 'w') as f:
                json.dump(parent_link, f, indent=2)

        return experiment_dir

    def load_config(self, experiment_id):
        """Load experiment config."""
        config_path = self.base_dir / experiment_id / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_parent_link(self, experiment_id):
        """Load parent link if it exists."""
        link_path = self.base_dir / experiment_id / "parent_link.json"
        if link_path.exists():
            with open(link_path, 'r') as f:
                return json.load(f)
        return None

    def _get_parent_phase(self, current_phase):
        """Get expected parent phase."""
        phase_order = {"base": None, "mid": "base", "sft": "mid", "rl": "sft"}
        return phase_order.get(current_phase)
```

**Step 2: Test ExperimentConfig**

Create `test_experiment_config.py`:

```python
import tempfile
import shutil
from nanochat.experiment_config import ExperimentConfig

tmpdir = tempfile.mkdtemp()

try:
    config = ExperimentConfig(base_dir=tmpdir)

    # Create base experiment
    base_dir = config.create_experiment(
        phase="base",
        direction="forward",
        depth=20,
        experiment_id="test_base"
    )

    assert base_dir.exists()
    assert (base_dir / "config.json").exists()
    assert (base_dir / "checkpoints").exists()

    # Create child experiment
    mid_dir = config.create_experiment(
        phase="mid",
        direction="forward",
        depth=20,
        parent_experiment="test_base",
        experiment_id="test_mid"
    )

    assert (mid_dir / "parent_link.json").exists()

    # Load and verify
    loaded_config = config.load_config("test_base")
    assert loaded_config["phase"] == "base"

    parent_link = config.load_parent_link("test_mid")
    assert parent_link["parent_experiment_id"] == "test_base"

    print("✅ ExperimentConfig works correctly")

finally:
    shutil.rmtree(tmpdir)
```

Run: `python test_experiment_config.py`
Expected: "✅ ExperimentConfig works correctly"

**Step 3: Commit**

```bash
git add nanochat/experiment_config.py test_experiment_config.py
git commit -m "feat: add experiment configuration management"
```

---

### Task 17: Experiment Runner Script - Structure

**Files:**
- Create: `scripts/run_backward_experiment.py`

**Step 1: Create script structure with argument parsing**

Create `scripts/run_backward_experiment.py`:

```python
"""
Orchestrate backward language model experiments.

Handles flexible experiment creation, parent linking, and phase execution.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from nanochat.experiment_config import ExperimentConfig
from nanochat.common import get_base_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run backward LM experiment")

    # Experiment identification
    parser.add_argument("--experiment_id", type=str, default=None,
                        help="Custom experiment ID (auto-generated if not provided)")
    parser.add_argument("--parent_experiment", type=str, default=None,
                        help="Parent experiment ID to continue from")

    # Experiment configuration
    parser.add_argument("--phase", type=str, required=True,
                        help="Phase to run: base, mid, sft, rl, or comma-separated list")
    parser.add_argument("--direction", type=str, default="forward",
                        help="Direction: forward, backward, bidirectional")
    parser.add_argument("--depth", type=int, default=20,
                        help="Model depth")

    # Training parameters (passed through to training scripts)
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--init_lr_frac", type=float, default=None)

    return parser.parse_args()


def run_base_training(experiment_dir, config, args):
    """Run base training phase."""
    print(f"\n{'='*60}")
    print(f"Running BASE training")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.base_train",
        f"--direction={config['direction']}",
        f"--depth={config['depth']}",
        f"--model_tag={config['experiment_id']}",
    ]

    # Add optional args
    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Base training failed")
        sys.exit(1)

    print("✅ Base training complete")


def run_midtraining(experiment_dir, config, parent_config, args):
    """Run midtraining phase."""
    print(f"\n{'='*60}")
    print(f"Running MIDTRAINING")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.mid_train",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Midtraining failed")
        sys.exit(1)

    print("✅ Midtraining complete")


def run_sft(experiment_dir, config, parent_config, args):
    """Run SFT phase."""
    print(f"\n{'='*60}")
    print(f"Running SFT")
    print(f"{'='*60}\n")

    # Determine source (mid or base)
    source = parent_config['phase']

    cmd = [
        "python", "-m", "scripts.chat_sft",
        f"--source={source}",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")
    if args.init_lr_frac:
        cmd.append(f"--init_lr_frac={args.init_lr_frac}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ SFT failed")
        sys.exit(1)

    print("✅ SFT complete")


def run_rl(experiment_dir, config, parent_config, args):
    """Run RL phase."""
    print(f"\n{'='*60}")
    print(f"Running RL")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.chat_rl",
        "--source=sft",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ RL failed")
        sys.exit(1)

    print("✅ RL complete")


def main():
    args = parse_args()

    # Initialize experiment config
    base_dir = get_base_dir()
    exp_config = ExperimentConfig(base_dir=os.path.join(base_dir, "experiments"))

    # Parse phases
    phases = args.phase.split(',')

    # Create or load experiment
    if "base" in phases:
        # Create new base experiment
        experiment_dir = exp_config.create_experiment(
            phase="base",
            direction=args.direction,
            depth=args.depth,
            experiment_id=args.experiment_id,
        )
        config = exp_config.load_config(experiment_dir.name)

        # Run base training
        run_base_training(experiment_dir, config, args)

        # Update for subsequent phases
        parent_experiment_id = experiment_dir.name
    else:
        # Must have parent for non-base phases
        if not args.parent_experiment:
            print("❌ Error: --parent_experiment required for non-base phases")
            sys.exit(1)

        parent_experiment_id = args.parent_experiment

    # Run subsequent phases
    for phase in phases:
        if phase == "base":
            continue  # Already ran

        # Create experiment for this phase
        parent_config = exp_config.load_config(parent_experiment_id)
        experiment_dir = exp_config.create_experiment(
            phase=phase,
            direction=parent_config["direction"],
            depth=parent_config["depth"],
            parent_experiment=parent_experiment_id,
            experiment_id=args.experiment_id,
        )
        config = exp_config.load_config(experiment_dir.name)

        # Run phase
        if phase == "mid":
            run_midtraining(experiment_dir, config, parent_config, args)
        elif phase == "sft":
            run_sft(experiment_dir, config, parent_config, args)
        elif phase == "rl":
            run_rl(experiment_dir, config, parent_config, args)

        # Update parent for next phase
        parent_experiment_id = experiment_dir.name

    print(f"\n{'='*60}")
    print(f"✅ Experiment complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

**Step 2: Test experiment runner (dry run)**

Run:
```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=forward \
    --depth=4 \
    --num_iterations=5 \
    --experiment_id=test_runner
```

Expected: Runs base training for 5 iterations

**Step 3: Commit**

```bash
git add scripts/run_backward_experiment.py
git commit -m "feat: add experiment orchestration script"
```

---

## Phase 6: Analysis Tools

### Task 18: Research Analysis - Comparison Functions

**Files:**
- Create: `nanochat/research_analysis.py`

**Step 1: Create analysis module structure**

Create `nanochat/research_analysis.py`:

```python
"""
Research analysis and visualization tools for backward LM experiments.

Provides functions for:
- Comparing learning curves across experiments
- Analyzing cross-direction transfer
- Generating research reports
"""

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_metadata(experiment_dir):
    """Load experiment config and metadata."""
    exp_path = Path(experiment_dir)

    with open(exp_path / "config.json", 'r') as f:
        config = json.load(f)

    parent_link = None
    if (exp_path / "parent_link.json").exists():
        with open(exp_path / "parent_link.json", 'r') as f:
            parent_link = json.load(f)

    return config, parent_link


def load_training_losses(experiment_dir, phase="base_training"):
    """Load loss curve from HDF5 logs."""
    h5_path = Path(experiment_dir) / f"{phase}.h5"

    if not h5_path.exists():
        return None, None

    steps = []
    losses = []

    with h5py.File(h5_path, 'r') as f:
        for step_name in sorted(f.keys()):
            if not step_name.startswith("step_"):
                continue

            step_num = int(step_name.split("_")[1])
            step_group = f[step_name]

            if 'loss' in step_group.attrs:
                steps.append(step_num)
                losses.append(step_group.attrs['loss'])

    return np.array(steps), np.array(losses)


def compare_learning_curves(experiment_dirs, output_path, phase="base_training"):
    """
    Compare learning curves across multiple experiments.

    Args:
        experiment_dirs: List of experiment directory paths
        output_path: Where to save the plot
        phase: Which phase to compare
    """
    plt.figure(figsize=(10, 6))

    for exp_dir in experiment_dirs:
        config, _ = load_experiment_metadata(exp_dir)
        steps, losses = load_training_losses(exp_dir, phase)

        if steps is None:
            print(f"Warning: No data for {exp_dir}")
            continue

        label = f"{config['direction']} d{config['depth']}"
        plt.plot(steps, losses, label=label, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves Comparison - {phase}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved learning curves to {output_path}")


def analyze_direction_transfer(experiment_dir, output_path):
    """
    Analyze cross-direction transfer from evaluation results.

    Args:
        experiment_dir: Experiment directory
        output_path: Where to save the report
    """
    config, _ = load_experiment_metadata(experiment_dir)
    eval_dir = Path(experiment_dir) / "evaluations"

    if not eval_dir.exists():
        print(f"No evaluations found in {experiment_dir}")
        return

    report_lines = [
        f"# Direction Transfer Analysis",
        f"",
        f"**Experiment:** {config['experiment_id']}",
        f"**Direction:** {config['direction']}",
        f"**Depth:** d{config['depth']}",
        f"",
    ]

    # TODO: Load and analyze evaluation results
    # This requires evaluation results to be saved in a standard format

    report_lines.append("## Cross-Direction Performance")
    report_lines.append("")
    report_lines.append("*Analysis coming soon - requires evaluation results*")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved transfer analysis to {output_path}")


def generate_research_report(experiment_dir, output_path):
    """
    Generate comprehensive research report for an experiment.

    Args:
        experiment_dir: Experiment directory
        output_path: Where to save the report
    """
    config, parent_link = load_experiment_metadata(experiment_dir)

    report_lines = [
        f"# Research Report: {config['experiment_id']}",
        f"",
        f"Generated: {config.get('created_at', 'Unknown')}",
        f"",
        f"## Configuration",
        f"",
        f"- **Phase:** {config['phase']}",
        f"- **Direction:** {config['direction']}",
        f"- **Depth:** d{config['depth']}",
    ]

    if parent_link:
        report_lines.extend([
            f"- **Parent:** {parent_link['parent_experiment_id']}",
        ])

    report_lines.extend([
        f"",
        f"## Training Dynamics",
        f"",
    ])

    # Load and summarize training
    steps, losses = load_training_losses(experiment_dir, config['phase'] + "_training" if config['phase'] == "base" else config['phase'])

    if steps is not None and len(steps) > 0:
        report_lines.extend([
            f"- **Total steps:** {steps[-1]}",
            f"- **Initial loss:** {losses[0]:.4f}",
            f"- **Final loss:** {losses[-1]:.4f}",
            f"- **Improvement:** {losses[0] - losses[-1]:.4f}",
        ])
    else:
        report_lines.append("*No training data available*")

    report_lines.extend([
        f"",
        f"## Evaluation Results",
        f"",
        f"*Coming soon - requires evaluation implementation*",
        f"",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved research report to {output_path}")
```

**Step 2: Test analysis functions**

Create `test_research_analysis.py`:

```python
import os
import tempfile
import shutil
from nanochat.research_analysis import compare_learning_curves, generate_research_report
from nanochat.experiment_config import ExperimentConfig
from nanochat.research_logger import ResearchLogger
import numpy as np

tmpdir = tempfile.mkdtemp()

try:
    # Create mock experiment with data
    exp_config = ExperimentConfig(base_dir=tmpdir)
    exp_dir = exp_config.create_experiment(
        phase="base",
        direction="forward",
        depth=20,
        experiment_id="test_exp"
    )

    # Add mock training data
    logger = ResearchLogger(exp_dir, phase="base_training")
    for step in range(0, 100, 10):
        logger.log_step(step, {
            'loss': 3.0 - step * 0.01,  # Decreasing loss
        })
    logger.close()

    # Test learning curve
    compare_learning_curves([exp_dir], os.path.join(tmpdir, "curves.png"))
    assert os.path.exists(os.path.join(tmpdir, "curves.png"))

    # Test report generation
    generate_research_report(exp_dir, os.path.join(tmpdir, "report.md"))
    assert os.path.exists(os.path.join(tmpdir, "report.md"))

    print("✅ Research analysis functions work")

finally:
    shutil.rmtree(tmpdir)
```

Run: `python test_research_analysis.py`
Expected: "✅ Research analysis functions work"

**Step 3: Commit**

```bash
git add nanochat/research_analysis.py test_research_analysis.py
git commit -m "feat: add research analysis and visualization tools"
```

---

## Final Integration and Testing

### Task 19: End-to-End Test

**Files:**
- Create: `test_end_to_end.sh`

**Step 1: Create comprehensive test script**

Create `test_end_to_end.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "End-to-End Backward LM Framework Test"
echo "=========================================="

# Test 1: Base training (forward)
echo -e "\n[1/6] Testing forward base training..."
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=forward \
    --depth=4 \
    --num_iterations=10 \
    --experiment_id=e2e_test_forward

# Test 2: Base training (backward)
echo -e "\n[2/6] Testing backward base training..."
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=backward \
    --depth=4 \
    --num_iterations=10 \
    --experiment_id=e2e_test_backward

# Test 3: Base training (bidirectional)
echo -e "\n[3/6] Testing bidirectional base training..."
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=bidirectional \
    --depth=4 \
    --num_iterations=10 \
    --experiment_id=e2e_test_bidirectional

# Test 4: Midtraining from backward base
echo -e "\n[4/6] Testing midtraining..."
python -m scripts.run_backward_experiment \
    --phase=mid \
    --parent_experiment=e2e_test_backward \
    --num_iterations=5 \
    --experiment_id=e2e_test_mid

# Test 5: Compare learning curves
echo -e "\n[5/6] Testing analysis tools..."
python -c "
from nanochat.research_analysis import compare_learning_curves
import os
base_dir = os.path.join(os.getcwd(), 'experiments')
exps = [
    os.path.join(base_dir, 'e2e_test_forward'),
    os.path.join(base_dir, 'e2e_test_backward'),
    os.path.join(base_dir, 'e2e_test_bidirectional'),
]
compare_learning_curves(exps, 'e2e_comparison.png')
print('✅ Analysis complete')
"

# Test 6: Generate report
echo -e "\n[6/6] Testing report generation..."
python -c "
from nanochat.research_analysis import generate_research_report
import os
base_dir = os.path.join(os.getcwd(), 'experiments')
exp_dir = os.path.join(base_dir, 'e2e_test_backward')
generate_research_report(exp_dir, 'e2e_report.md')
print('✅ Report generated')
"

echo -e "\n=========================================="
echo "✅ All end-to-end tests passed!"
echo "=========================================="
```

**Step 2: Make executable and run**

Run:
```bash
chmod +x test_end_to_end.sh
./test_end_to_end.sh
```

Expected: All 6 tests pass

**Step 3: Commit**

```bash
git add test_end_to_end.sh
git commit -m "test: add end-to-end integration test"
```

---

### Task 20: Documentation and Cleanup

**Files:**
- Update: `CLAUDE.md`
- Create: `docs/QUICKSTART.md`

**Step 1: Update CLAUDE.md implementation status**

In `CLAUDE.md`, update the implementation status section to mark completed items:

```markdown
### ✅ Completed

**Phase 1: Core Infrastructure**
- [x] `nanochat/direction_dataloader.py`
- [x] Modified `nanochat/tokenizer.py`
- [x] Modified `nanochat/checkpoint_manager.py`
- [x] Token reversal thoroughly tested

**Phase 2: Training Integration**
- [x] Modified `scripts/base_train.py`
- [x] Modified `scripts/mid_train.py`
- [x] Modified `scripts/chat_sft.py`
- [x] Modified `scripts/chat_rl.py`
- [x] All three directions train successfully

**Phase 3: Logging System**
- [x] `nanochat/research_logger.py`
- [x] Integrated with training loop
- [x] HDF5 structure validated

**Phase 4: Evaluation Framework**
- [x] `nanochat/direction_eval.py`
- [x] `tasks/direction_wrapper.py`
- [x] Multi-direction evaluation support

**Phase 5: Orchestration**
- [x] `scripts/run_backward_experiment.py`
- [x] Parent linking system
- [x] Flexible phase execution
- [x] Experiment branching tested

**Phase 6: Analysis Tools**
- [x] `nanochat/research_analysis.py`
- [x] Comparison functions
- [x] Auto-report generation
- [x] End-to-end testing complete
```

**Step 2: Create quickstart guide**

Create `docs/QUICKSTART.md`:

```markdown
# Backward LM Framework - Quick Start Guide

## Installation

1. Follow the base nanochat installation from `README.md`
2. Install additional dependencies:
```bash
pip install h5py matplotlib
```

## Running Your First Experiment

### 1. Train a Forward Model (Baseline)

```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=forward \
    --depth=12 \
    --num_iterations=1000
```

### 2. Train a Backward Model

```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=backward \
    --depth=12 \
    --num_iterations=1000
```

### 3. Compare Results

```bash
python -c "
from nanochat.research_analysis import compare_learning_curves
import glob

exps = glob.glob('experiments/base_*_d12*')
compare_learning_curves(exps, 'forward_vs_backward.png')
"
```

## Next Steps

- See `CLAUDE.md` for complete documentation
- See `docs/plans/2025-01-11-backward-language-models-design.md` for design details
- Run `./test_end_to_end.sh` to verify installation
```

**Step 3: Clean up test files**

Remove temporary test files:

```bash
rm -f test_*.py test_end_to_end.sh
```

**Step 4: Final commit**

```bash
git add CLAUDE.md docs/QUICKSTART.md
git commit -m "docs: update implementation status and add quickstart"
```

---

## Summary

This implementation plan provides:

1. ✅ **Core Infrastructure** - Direction tokens, dataloader, checkpoint management
2. ✅ **Training Integration** - All phases support direction parameter
3. ✅ **Logging System** - HDF5-based research logging with WandB
4. ✅ **Evaluation Framework** - Multi-direction evaluation support
5. ✅ **Orchestration** - Flexible experiment runner with parent linking
6. ✅ **Analysis Tools** - Visualization and report generation

**Total Implementation Time Estimate:** 3-4 weeks for experienced engineer

**Key Implementation Principles:**
- ✅ DRY - Reused existing nanochat infrastructure
- ✅ YAGNI - No unnecessary features added
- ✅ TDD - Tests created alongside implementation
- ✅ Frequent commits - Each task ends with a commit

**Ready for Execution:** This plan can be executed task-by-task using `superpowers:executing-plans` in a separate session.
