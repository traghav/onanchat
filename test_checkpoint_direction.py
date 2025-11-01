from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.gpt import GPT, GPTConfig
import torch
import os
import tempfile
import shutil

# Create temp directory
tmpdir = tempfile.mkdtemp()

try:
    # Create small model
    config = GPTConfig(n_layer=2, n_head=4, n_embd=128, vocab_size=1000, n_kv_head=4)
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
    model_data, optim_data, loaded_meta = load_checkpoint(tmpdir, 100, "cpu", load_optimizer=False)
    print(f"Direction: {loaded_meta.get('direction', 'NOT FOUND')}")
    assert loaded_meta.get('direction') == 'backward', "Direction not saved/loaded correctly"
    print("✅ Checkpoint direction save/load works")

finally:
    # Cleanup
    shutil.rmtree(tmpdir)
