"""
Example: Using directional chat engines programmatically.

This shows how to use the directional chat engines in your own code,
without going through the CLI or Web interfaces.
"""

import torch
from nanochat.checkpoint_manager import load_model
from nanochat.directional_chat_engine import create_chat_engine


def forward_chat_example():
    """Example: Chat with a forward model."""
    print("\n=== Forward Model Example ===")

    # Load model
    device = 'cpu'  # or 'cuda'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # Chat
    chat_engine.add_user_message("What is the capital of France?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"User: What is the capital of France?")
    print(f"Assistant: {response}")

    # Continue conversation
    chat_engine.add_user_message("What's the population?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"User: What's the population?")
    print(f"Assistant: {response}")


def backward_chat_example():
    """Example: Chat with a backward model."""
    print("\n=== Backward Model Example ===")

    # Load backward model
    device = 'cpu'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20_backward')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # User provides "ending"
    chat_engine.add_user_message("And that's how I solved the problem!")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)

    print(f"User provides ending: And that's how I solved the problem!")
    print(f"Model generates preceding context: {response}")

    # Get full conversation in chronological order
    conversation = chat_engine.get_conversation_display()
    print("\nFull conversation (chronological):")
    for msg in conversation:
        print(f"{msg['role']}: {msg['content']}")


def bidirectional_chat_example():
    """Example: Chat with a bidirectional model with direction toggle."""
    print("\n=== Bidirectional Model Example ===")

    # Load bidirectional model
    device = 'cpu'
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d20_bidirectional')

    # Create chat engine
    chat_engine = create_chat_engine(model, tokenizer, meta)

    # Check if toggle available
    if not chat_engine.can_toggle_direction():
        print("Error: Model doesn't support direction toggle")
        return

    # Forward turn
    chat_engine.set_direction('forward')
    chat_engine.add_user_message("Tell me a story.")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Forward] User: Tell me a story.")
    print(f"[Forward] Assistant: {response}")

    # Switch to backward
    chat_engine.set_direction('backward')
    chat_engine.add_user_message("The end.")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Backward] User provides ending: The end.")
    print(f"[Backward] Assistant generates: {response}")

    # Switch back to forward
    chat_engine.set_direction('forward')
    chat_engine.add_user_message("What happened next?")
    response = chat_engine.generate_response(temperature=0.6, top_k=50)
    print(f"[Forward] User: What happened next?")
    print(f"[Forward] Assistant: {response}")


if __name__ == '__main__':
    print("Directional Chat Examples")
    print("=" * 50)

    # Note: These examples assume you have trained models
    # Replace model_tag with your actual model names

    # Uncomment to run:
    # forward_chat_example()
    # backward_chat_example()
    # bidirectional_chat_example()

    print("\n" + "=" * 50)
    print("Examples complete!")
    print("\nNote: Uncomment the function calls to run with your trained models.")
