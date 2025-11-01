from tasks.direction_wrapper import DirectionTaskWrapper
from nanochat.tokenizer import get_tokenizer

# Create a minimal mock task
class MockTask:
    def __init__(self):
        self.data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Load tokenizer
tokenizer = get_tokenizer()

# Create task
task = MockTask()

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
print(f"First token same (BOS): {tokens_fwd[0] == tokens_bwd[0]}")  # BOS should match

print("✅ DirectionTaskWrapper works")
