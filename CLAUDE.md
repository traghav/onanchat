# Claude Code Guide for Backward Language Models Research

## Project Overview

This is **nanochat** extended with a backward language model research framework. The project trains and compares three types of language models:

1. **Forward models** - Standard left-to-right prediction (baseline)
2. **Backward models** - Right-to-left prediction via token reversal
3. **Bidirectional models** - Both directions using special `<|forward|>`/`<|backward|>` tokens

The goal is scientific research on backward language modeling with comprehensive logging and evaluation across all training phases (base, mid, SFT, RL).

## Repository Structure

```
nanochat/
├── nanochat/                           # Core library
│   ├── gpt.py                          # GPT model architecture
│   ├── dataloader.py                   # Standard dataloader
│   ├── direction_dataloader.py         # [NEW] Direction-aware dataloader
│   ├── direction_eval.py               # [NEW] Multi-direction evaluation
│   ├── research_logger.py              # [NEW] HDF5 logging system
│   ├── research_analysis.py            # [NEW] Analysis tools
│   └── tokenizer.py                    # [MODIFIED] Adds direction tokens
│
├── scripts/                            # Training scripts
│   ├── base_train.py                   # [MODIFIED] Base training
│   ├── mid_train.py                    # [MODIFIED] Midtraining
│   ├── chat_sft.py                     # [MODIFIED] Supervised fine-tuning
│   ├── chat_rl.py                      # [MODIFIED] Reinforcement learning
│   └── run_backward_experiment.py      # [NEW] Experiment orchestration
│
├── tasks/                              # Evaluation tasks
│   ├── arc.py, mmlu.py, gsm8k.py      # Standard tasks
│   └── direction_wrapper.py            # [NEW] Backward task wrapper
│
├── experiments/                        # [NEW] Research experiments
│   └── {timestamp}_{direction}_{size}/ # Individual experiments
│
└── docs/
    └── plans/
        └── 2025-01-11-backward-language-models-design.md  # Full design doc
```

## Key Concepts

### Direction Handling

**Forward Direction:**
- Standard: `[BOS, t1, t2, t3, ...]`
- No changes to tokens

**Backward Direction:**
- Reversed: `[BOS, ..., t3, t2, t1]`
- BOS stays at beginning
- Token reversal happens in dataloader

**Bidirectional Direction:**
- With markers: `[BOS, <|forward|>, t1, t2, t3, ...]` or `[BOS, <|backward|>, ..., t3, t2, t1]`
- 50/50 mix in training batches
- Requires 2 new special tokens in vocabulary

### Experiment Organization

Experiments use a **parent-child hierarchy** for branching:

```
base_20250111_forward_d20/              # Base experiment (no parent)
├── mid_v1_20250111/                    # Midtraining (parent: base)
│   ├── sft_hightemp_20250112/          # SFT variant 1 (parent: mid)
│   └── sft_lowtemp_20250112/           # SFT variant 2 (parent: mid)
```

Each experiment directory contains:
- `config.json` - Experiment configuration
- `parent_link.json` - Link to parent experiment
- `checkpoints/` - Model weights
- `{phase}.h5` - Detailed HDF5 logs
- `evaluations/` - Task evaluation results
- `RESEARCH_REPORT.md` - Auto-generated analysis

### Logging System

**Two-tier logging:**

1. **WandB (Live Monitoring)**
   - Loss, learning rate, tokens/sec, MFU
   - Direction-specific metrics
   - Lightweight, real-time

2. **HDF5 (Research Analysis)**
   - Per-token predictions and losses
   - Gradient statistics per layer
   - Activation patterns
   - Complete reproducibility data

## Common Commands

### Training

```bash
# Train base model (forward/backward/bidirectional)
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --direction=forward \
    --depth=20

# Run full experiment (base + mid)
python -m scripts.run_backward_experiment \
    --phase=base,mid \
    --direction=backward \
    --depth=20 \
    --experiment_id=my_experiment

# Branch SFT from existing midtraining
python -m scripts.run_backward_experiment \
    --phase=sft \
    --parent_experiment=mid_v1_20250111 \
    --experiment_id=sft_variant1 \
    --init_lr_frac=0.05
```

### Evaluation

```bash
# Evaluate in multiple directions
python -m scripts.base_eval \
    --model_tag=d20_backward \
    --directions=forward,backward

# Run comprehensive evaluation
python -m scripts.run_backward_experiment \
    --phase=eval \
    --parent_experiment=base_20250111_backward_d20
```

### Analysis

```bash
# Compare learning curves
python -m nanochat.research_analysis compare_learning \
    --experiments base_*_d20 \
    --output learning_curves.png

# Analyze cross-direction transfer
python -m nanochat.research_analysis direction_transfer \
    --experiment base_backward_d20 \
    --output transfer_analysis.md

# Generate comprehensive report
python -m nanochat.research_analysis generate_research_report \
    --experiment base_forward_d20 \
    --output report.md
```

## Directional Chat Interface

The chat interfaces (CLI and Web) support models trained in different directions:

### Model Directions

1. **Forward Models** (standard)
   - Normal left-to-right chat
   - No special commands needed

2. **Backward Models**
   - User provides "ending" messages
   - Model generates what came before
   - Conversation displays in normal chronological order

3. **Bidirectional Models**
   - Per-turn direction toggle
   - CLI: Use `/forward` or `/backward` commands
   - Web: Direction displayed in header

### CLI Commands

```bash
# Chat with forward model (standard)
python -m scripts.chat_cli --source=sft

# Chat with backward model
python -m scripts.chat_cli --source=base --model-tag=d20_backward

# Chat with bidirectional model
python -m scripts.chat_cli --source=base --model-tag=d20_bidirectional
```

**Available commands during chat:**
- `/quit`, `/exit` - End conversation
- `/clear` - Start new conversation
- `/forward` - Switch to forward generation (bidirectional only)
- `/backward` - Switch to backward generation (bidirectional only)

### Web Interface

```bash
python -m scripts.chat_web --source=sft --model-tag=<model>
```

Open http://localhost:8000

The web interface displays the model direction in the header with visual indicators:
- → forward
- ← backward
- ↔ bidirectional

### Implementation Details

**Architecture:**
- `DirectionalChatEngine` base class with three implementations
- `ForwardChatEngine` - standard chat
- `BackwardChatEngine` - reversed interaction with display reversal
- `BidirectionalChatEngine` - per-turn direction control

**Direction Detection:**
Models automatically advertise their direction via checkpoint metadata. Chat interfaces detect this and enable appropriate features.

**Direction Tokens:**
Bidirectional models use special tokens:
- `<|forward|>` - marks forward generation
- `<|backward|>` - marks backward generation

**Files:**
- `nanochat/directional_chat_engine.py` - Chat engine implementations
- `scripts/chat_cli.py` - CLI interface with direction support
- `scripts/chat_web.py` - Web server with direction info endpoint
- `nanochat/ui.html` - Web UI with direction display

## Implementation Status

### ✅ Completed (Original nanochat)
- GPT model architecture
- Base training pipeline
- Midtraining, SFT, RL stages
- Evaluation tasks (ARC, MMLU, GSM8K, etc.)
- Checkpoint management
- WandB logging

### ✅ Completed (Backward LM Research)

**Phase 1: Core Infrastructure**
- [x] `nanochat/direction_dataloader.py` - Direction-aware data loading
- [x] Modify `nanochat/tokenizer.py` - Add `<|forward|>`/`<|backward|>` tokens
- [x] Modify `nanochat/checkpoint_manager.py` - Store/load direction metadata
- [x] Test token reversal logic thoroughly

**Phase 2: Training Integration**
- [x] Modify `scripts/base_train.py` - Add `--direction` flag
- [x] Modify `scripts/mid_train.py` - Auto-detect direction from checkpoint
- [x] Modify `scripts/chat_sft.py` - Auto-detect direction
- [x] Modify `scripts/chat_rl.py` - Auto-detect direction
- [x] Test all three directions train successfully

**Phase 3: Logging System**
- [x] `nanochat/research_logger.py` - HDF5 logging infrastructure
- [x] Integrate with training loop (all phases)
- [x] Optional h5py dependency for backward compatibility
- [x] Logging validates gracefully when h5py unavailable

**Phase 4: Evaluation Framework**
- [x] `nanochat/direction_eval.py` - Multi-direction evaluation
- [x] `tasks/direction_wrapper.py` - Wrap tasks for backward eval
- [x] Framework ready for multi-direction evaluation
- [x] Backward evaluation logic implemented

**Phase 5: Orchestration**
- [x] `scripts/run_backward_experiment.py` - Main experiment runner
- [x] Implement parent linking system
- [x] Implement flexible phase execution
- [x] Support for branching experiments

**Phase 6: Analysis Tools**
- [x] `nanochat/research_analysis.py` - Analysis and visualization
- [x] Implement comparison functions
- [x] Implement auto-report generation
- [x] Optional matplotlib dependency for plotting

## Design Document

For complete implementation details, see: `docs/plans/2025-01-11-backward-language-models-design.md`

The design document includes:
- Full architecture details
- Data pipeline specifications
- Logging system structure
- Evaluation framework
- Example workflows
- Success criteria

## Development Guidelines

### When Modifying Core Training Logic

1. **Keep it minimal** - Core training loop should remain unchanged
2. **Isolate direction logic** - Handle in dataloader, not model
3. **Test both directions** - Forward and backward must work identically well
4. **Preserve backward compatibility** - Standard forward training must still work

### When Adding Logging

1. **Use appropriate tier** - WandB for monitoring, HDF5 for research
2. **Consider frequency** - Balance between detail and storage
3. **Test storage size** - Monitor disk usage during long runs
4. **Document structure** - HDF5 schema should be self-documenting

### When Implementing Evaluation

1. **Validate backward logic** - Token reversal must be correct
2. **Test on known examples** - Manual verification of backward tasks
3. **Compare both directions** - Cross-direction transfer is key research question
4. **Save all results** - Per-example predictions for analysis

### When Running Experiments

1. **Use experiment IDs** - Descriptive names for easy identification
2. **Track parent relationships** - Maintain experiment lineage
3. **Log everything** - Over-logging is better than under-logging for research
4. **Save configs** - Every experiment should have complete config.json

## Research Questions to Answer

When implementing and testing, keep these research questions in mind:

1. **Learning Efficiency**
   - Do backward models converge faster or slower?
   - Different learning dynamics?
   - Does model size affect this?

2. **Representation Quality**
   - Do backward models learn different internal representations?
   - Attention pattern differences?
   - Layer-wise activation patterns?

3. **Cross-Direction Transfer**
   - Can backward models perform forward tasks?
   - How much performance degradation?
   - Does bidirectional training help both directions?

4. **Directional Control**
   - Can bidirectional models effectively switch directions?
   - Do direction tokens work reliably?
   - Balance between forward and backward performance?

5. **Scaling Behavior**
   - How do findings change from d12 → d20 → d26?
   - Do larger models transfer better across directions?
   - Scaling laws for backward models?

## Debugging Tips

### Token Reversal Issues

If backward training seems broken:
- Print example batches from dataloader
- Verify BOS stays at position 0
- Check targets are shifted correctly
- Manually decode sequences to verify reversal

### Evaluation Mismatches

If backward evaluation gives strange results:
- Test on single examples manually
- Verify prompt reversal is correct
- Check multiple-choice reversal (each option separately)
- Compare forward model on reversed input vs backward model on normal input

### Logging Issues

If HDF5 logs are corrupted or huge:
- Check logging frequency settings
- Verify compression is enabled
- Monitor during training, not just at end
- Test with small run first

### Experiment Branching

If parent linking fails:
- Verify parent experiment completed successfully
- Check checkpoint exists at expected step
- Validate parent_link.json format
- Ensure direction matches or is compatible

## Useful Code Patterns

### Load Model with Direction

```python
from nanochat.checkpoint_manager import load_model

# Automatically detects direction from checkpoint
model, tokenizer, meta = load_model("base", device, model_tag="d20_backward")
direction = meta["direction"]  # "forward", "backward", or "bidirectional"
```

### Create Direction-Aware Dataloader

```python
from nanochat.direction_dataloader import direction_aware_dataloader

# Will handle token reversal automatically
loader = direction_aware_dataloader(
    B=32, T=2048, split="train", direction="backward", device="cuda"
)
```

### Log to HDF5

```python
from nanochat.research_logger import ResearchLogger

logger = ResearchLogger(experiment_dir)
logger.log_step(
    step=100,
    loss=2.5,
    per_token_losses=losses_array,  # (B, T)
    gradients=grad_dict,
)
```

### Multi-Direction Evaluation

```python
from nanochat.direction_eval import evaluate_task_multi_direction
from tasks.arc import ARC

results = evaluate_task_multi_direction(
    task=ARC(subset="ARC-Easy", split="test"),
    model=model,
    tokenizer=tokenizer,
    directions=["forward", "backward"]
)
# results = {"forward": {"accuracy": 0.85, ...}, "backward": {"accuracy": 0.42, ...}}
```

## Getting Help

1. **Design Document**: `docs/plans/2025-01-11-backward-language-models-design.md`
2. **Original README**: `README.md` for base nanochat functionality
3. **Code Comments**: All new modules are heavily commented
4. **Experiment Logs**: Check HDF5 files for detailed training info
5. **Research Reports**: Auto-generated for each experiment

## Quick Start for New Contributors

1. Read original `README.md` to understand base nanochat
2. Read this file (CLAUDE.md) for backward LM extensions
3. Read design document for implementation details
4. Start with Phase 1 (Core Infrastructure) implementation
5. Test each component thoroughly before moving to next phase
6. Run end-to-end experiments to validate integration

---

**Last Updated:** 2025-01-11
**Status:** Design Complete, Implementation Pending
**Contact:** Check git log for contributors
