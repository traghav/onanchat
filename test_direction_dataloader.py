from nanochat.direction_dataloader import direction_aware_dataloader
import torch

# Test forward direction
loader = direction_aware_dataloader(B=2, T=16, split="val", direction="forward", device="cpu")
inputs, targets = next(loader)
print(f"Forward - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
print(f"First input sequence: {inputs[0].tolist()[:10]}")
print("✅ Forward direction dataloader works")
