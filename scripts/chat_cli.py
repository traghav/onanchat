"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli -i mid
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.checkpoint_manager import load_model
from nanochat.directional_chat_engine import create_chat_engine

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--debug', action='store_true', help='Print raw model tokens for each assistant response')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Create direction-aware chat engine
chat_engine = create_chat_engine(model, tokenizer, meta)

# Print welcome message with direction info
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

# Track current direction for display indicator (bidirectional models)
current_direction = 'forward'

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            # Direction indicator
            arrow = '→' if current_direction == 'forward' else '←'
            user_input = input(f"\n{arrow} You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

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

    if args.debug:
        dbg = chat_engine.get_last_generation()
        if dbg:
            gen_dir = dbg.get("direction", direction)
            prompt_tokens = dbg.get("prompt_tokens", [])
            gen_tokens = dbg.get("generated_tokens", [])
            raw_tokens = dbg.get("raw_tokens", [])
            stop_reason = dbg.get("stop_reason", "unknown")
            print(f"[DEBUG] direction={gen_dir} prompt_len={len(prompt_tokens)} gen_len={len(gen_tokens)} raw_len={len(raw_tokens)} stop={stop_reason}")
            if raw_tokens:
                raw_dec = tokenizer.decode(raw_tokens)
                print(f"[DEBUG] raw first tokens: {raw_tokens[:8]}")
                print(f"[DEBUG] raw decoded: {raw_dec[:200]}")
            if gen_tokens:
                decoded_raw = tokenizer.decode(gen_tokens)
                if gen_dir == "backward":
                    decoded_rev = tokenizer.decode(list(reversed(gen_tokens)))
                    print(f"[DEBUG] decoded (raw order): {decoded_raw}")
                    print(f"[DEBUG] decoded (reversed): {decoded_rev}")
                else:
                    print(f"[DEBUG] decoded: {decoded_raw}")
            else:
                print("[DEBUG] empty generation")

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
