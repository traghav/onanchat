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
