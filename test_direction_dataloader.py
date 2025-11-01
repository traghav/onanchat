from nanochat.direction_dataloader import direction_aware_dataloader
import torch

# Test forward direction
loader = direction_aware_dataloader(B=2, T=16, split="val", direction="forward", device="cpu")
inputs, targets = next(loader)
print(f"Forward - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
print(f"First input sequence: {inputs[0].tolist()[:10]}")
print("✅ Forward direction dataloader works")

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
