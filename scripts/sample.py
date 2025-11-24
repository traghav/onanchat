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
from nanochat.common import autodetect_device_type, reverse_tokens

parser = argparse.ArgumentParser(description="Sample from a trained model")
parser.add_argument("--source", type=str, default="base", help="Model source (base|mid|sft|rl)")
parser.add_argument("--model-tag", type=str, default="", help="Model tag (e.g., d12, d20_backward)")
parser.add_argument("--step", type=int, default=-1, help="Checkpoint step (-1 = latest)")
parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--device", type=str, default="", help="Device (cuda|cpu|mps, empty = autodetect)")
parser.add_argument("--gen-direction", type=str, default="auto",
                    choices=["auto", "forward", "backward"],
                    help="Generation direction for bidirectional models (auto uses model default)")
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

# Resolve generation direction (needed for bidirectional)
if args.gen_direction == "auto":
    gen_direction = "backward" if direction == "backward" else "forward"
else:
    gen_direction = args.gen_direction

# For backward generation, optionally reverse output for readability
reverse_output = gen_direction == "backward"
if reverse_output:
    print("Note: Output will be reversed for readability (backward generation produces right-to-left tokens)")
print()

# Create engine
engine = Engine(model, tokenizer)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else lambda: None
bos_token_id = tokenizer.get_bos_token_id()
forward_token_id = getattr(tokenizer, "get_forward_token_id", lambda: None)()
backward_token_id = getattr(tokenizer, "get_backward_token_id", lambda: None)()

# Interactive sampling loop
print("Enter prompts to sample from the model. Type 'quit' or 'exit' to stop.")
print(f"Settings: max_tokens={args.max_tokens}, num_samples={args.num_samples}, temperature={args.temperature}, gen_direction={gen_direction}")
print("-" * 70)

while True:
    try:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        # Allow typing "\n" to embed newlines in the prompt
        prompt = prompt.replace("\\n", "\n")

        if not prompt.strip():
            continue

        # Tokenize and generate
        tokens = tokenizer(prompt, prepend=bos_token_id)

        # Direction-aware prompt shaping
        if direction == "backward" and gen_direction == "backward":
            # Backward-only model: reverse prompt tokens (BOS stays first)
            tokens = reverse_tokens(tokens, keep_bos=True)
        elif direction == "bidirectional":
            if gen_direction == "forward":
                # Insert forward marker after BOS
                if forward_token_id is None:
                    raise ValueError("Tokenizer missing forward direction token")
                tokens = [bos_token_id, forward_token_id] + tokens[1:]
            else:
                # Backward generation: insert backward marker and reverse after BOS
                if backward_token_id is None:
                    raise ValueError("Tokenizer missing backward direction token")
                tokens = [bos_token_id, backward_token_id] + reverse_tokens(tokens[1:], keep_bos=False)

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

            # For backward models, reverse tokens back to chronological order for readability.
            display_tokens = sample

            # Strip BOS for display
            if display_tokens and display_tokens[0] == bos_token_id:
                display_tokens = display_tokens[1:]

            # Strip direction marker for bidirectional display
            if direction == "bidirectional" and display_tokens:
                if forward_token_id is not None and display_tokens[0] == forward_token_id:
                    display_tokens = display_tokens[1:]
                elif backward_token_id is not None and display_tokens[0] == backward_token_id:
                    display_tokens = display_tokens[1:]

            if reverse_output:
                readable_tokens = list(reversed(display_tokens))
                decoded = tokenizer.decode(readable_tokens)
            else:
                decoded = tokenizer.decode(display_tokens)

            print(decoded)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

print("\nDone!")
