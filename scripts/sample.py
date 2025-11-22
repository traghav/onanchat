"""
Quick sampling script for testing models without the full chat interface.

Usage:
    python -m scripts.sample --model-tag=d12
    python -m scripts.sample --model-tag=d12 --step=2000 --max-tokens=100 --num-samples=3
"""

import argparse
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.common import autodetect_device_type

parser = argparse.ArgumentParser(description="Sample from a trained model")
parser.add_argument("--source", type=str, default="base", help="Model source (base|mid|sft|rl)")
parser.add_argument("--model-tag", type=str, default="", help="Model tag (e.g., d12, d20_backward)")
parser.add_argument("--step", type=int, default=-1, help="Checkpoint step (-1 = latest)")
parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--device", type=str, default="", help="Device (cuda|cpu|mps, empty = autodetect)")
args = parser.parse_args()

# Setup device
device_type = autodetect_device_type() if args.device == "" else args.device
device = torch.device(device_type)

# Load model
print(f"Loading model: source={args.source}, model_tag={args.model_tag or 'default'}, step={args.step if args.step > 0 else 'latest'}")
model, tokenizer, meta = load_model(
    args.source,
    device,
    phase="eval",
    model_tag=args.model_tag if args.model_tag else None,
    step=args.step if args.step > 0 else None
)

direction = meta.get("direction", "forward")
print(f"Model direction: {direction}")
print(f"Loaded from step: {meta.get('step', 'unknown')}")

# For backward models, optionally reverse output for readability
reverse_output = direction == "backward"
if reverse_output:
    print("Note: Output will be reversed for readability (backward model generates right-to-left)")
print()

# Create engine
engine = Engine(model, tokenizer)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else lambda: None

# Interactive sampling loop
print("Enter prompts to sample from the model. Type 'quit' or 'exit' to stop.")
print(f"Settings: max_tokens={args.max_tokens}, num_samples={args.num_samples}, temperature={args.temperature}")
print("-" * 70)

while True:
    try:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt.strip():
            continue

        # Tokenize and generate
        tokens = tokenizer(prompt, prepend='<|bos|>')

        with autocast_ctx if callable(autocast_ctx) else autocast_ctx:
            samples, _ = engine.generate_batch(
                tokens,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

        # Print samples
        for i, sample in enumerate(samples):
            if args.num_samples > 1:
                print(f"\nSample {i+1}:")

            # Decode the full sequence (including prompt for coherence)
            decoded = tokenizer.decode(sample)

            # For backward models, reverse the output for readability
            if reverse_output:
                # Debug: show raw output
                # print(f"[RAW]: {decoded}")

                # Remove <|bos|> token first
                decoded_clean = decoded.replace('<|bos|>', '').strip()

                # Reverse character by character, not word by word
                # This preserves punctuation attachment
                reversed_text = decoded_clean[::-1]
                print(reversed_text)
            else:
                print(decoded)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

print("\nDone!")
