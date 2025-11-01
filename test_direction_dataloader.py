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
