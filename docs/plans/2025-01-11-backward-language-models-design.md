# Backward Language Models Research Framework - Design Document

**Date:** 2025-01-11
**Author:** Research Design Session
**Status:** Design Complete, Ready for Implementation

## Executive Summary

This document describes the design for a comprehensive research framework to scientifically study backward language models. The framework extends nanochat to train and evaluate three model variants: forward (standard), backward (right-to-left), and bidirectional (both directions with special tokens). The goal is to enable rigorous comparison across all training phases (base, mid, SFT, RL) with extensive logging for research artifacts.

## 1. Research Objectives

### Primary Goals
1. Train three model types (forward, backward, bidirectional) at multiple scales (d12, d20, d26, etc.)
2. Compare learning dynamics across all training phases
3. Measure cross-direction transfer (can backward models do forward tasks?)
4. Generate comprehensive research artifacts for scientific publication

### Key Research Questions
- How does training direction affect learning efficiency?
- Do backward models learn different representations?
- Can bidirectional models effectively control direction?
- How well do models transfer across directions?

## 2. Architecture Overview

### 2.1 Model Variants

**Forward Model (Baseline)**
- Standard left-to-right autoregressive prediction
- Token sequence: `[BOS, t1, t2, t3, ...]`
- No changes to existing nanochat behavior

**Backward Model**
- Right-to-left prediction via token reversal
- Token sequence: `[BOS, ..., t3, t2, t1]`
- BOS stays at beginning for consistent start marker
- Autoregressive loss computed on reversed sequence

**Bidirectional Model**
- Both directions using special tokens
- Forward: `[BOS, <|forward|>, t1, t2, t3, ...]`
- Backward: `[BOS, <|backward|>, ..., t3, t2, t1]`
- 50/50 mix in each batch
- Requires adding 2 special tokens to vocabulary

### 2.2 Design Principles

1. **Minimal Invasiveness** - Core training logic remains unchanged
2. **Modular Dataloaders** - Direction handling isolated in dataloader layer
3. **Backward Compatibility** - Can still train standard forward models
4. **Flexible Branching** - Reuse checkpoints for derived experiments
5. **Comprehensive Logging** - HDF5 for analysis, wandb for monitoring
6. **Multi-Scale Support** - Same code for all model sizes

## 3. Data Pipeline

### 3.1 Direction-Aware Dataloader

**New Module:** `nanochat/direction_dataloader.py`

```python
def direction_aware_dataloader(B, T, split, direction="forward",
                                tokenizer_threads=4, device="cuda"):
    """
    Wraps standard dataloader with direction-specific token manipulation.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        direction: "forward" | "backward" | "bidirectional"

    Returns:
        Iterator yielding (inputs, targets) with appropriate direction
    """
```

### 3.2 Token Manipulation Logic

**Forward Direction**
- Pass through unchanged
- `[BOS, t1, t2, t3, ...]`

**Backward Direction**
- Reverse token order (excluding BOS)
- `[BOS, ..., t3, t2, t1]`
- Loss computation works normally on reversed sequence

**Bidirectional Direction**
- Alternate batches with direction markers
- Even batch indices: `[BOS, <|forward|>, t1, t2, t3, ...]`
- Odd batch indices: `[BOS, <|backward|>, ..., t3, t2, t1]`

### 3.3 Conversation Reversal (Mid/SFT/RL)

For structured conversations, reverse at token level within each turn:

```
Original:
<|user_start|> How are you? <|user_end|> <|assistant_start|> I'm good! <|assistant_end|>

Backward:
<|user_start|> ?you are How <|user_end|> <|assistant_start|> !good I'm <|assistant_end|>
```

Preserves turn structure while reversing content.

## 4. Training Pipeline

### 4.1 Training Script Integration

All training scripts remain **unchanged** in their core logic. They accept a `--direction` flag:

```bash
# Base training
torchrun --nproc_per_node=8 -m scripts.base_train -- --direction=forward --depth=20
torchrun --nproc_per_node=8 -m scripts.base_train -- --direction=backward --depth=20
torchrun --nproc_per_node=8 -m scripts.base_train -- --direction=bidirectional --depth=20

# Midtraining (auto-detects direction from checkpoint)
torchrun --nproc_per_node=8 -m scripts.mid_train -- --model_tag=d20_backward

# SFT (auto-detects direction)
torchrun --nproc_per_node=8 -m scripts.chat_sft -- --source=mid --model_tag=d20_backward

# RL (auto-detects direction)
torchrun --nproc_per_node=8 -m scripts.chat_rl -- --source=sft --model_tag=d20_backward
```

### 4.2 Checkpoint Management

**Directory Structure:**
```
base_checkpoints/
├── d20_forward/
├── d20_backward/
└── d20_bidirectional/

mid_checkpoints/
├── d20_forward/
├── d20_backward/
└── d20_bidirectional/

chatsft_checkpoints/
├── d20_forward/
├── d20_backward/
└── d20_bidirectional/
```

**Checkpoint Metadata:**
```python
{
    "direction": "forward|backward|bidirectional",
    "model_config": {...},
    "user_config": {...},
    "step": 12000,
    "val_bpb": 1.234,
}
```

### 4.3 Automatic Direction Detection

When loading checkpoints for mid/sft/rl phases:
1. Read checkpoint metadata
2. Extract direction
3. Initialize appropriate dataloader
4. Continue training in same direction

## 5. Comprehensive Logging System

### 5.1 Two-Tier Architecture

**Tier 1: Live Monitoring (WandB)**
- Standard metrics: loss, LR, tokens/sec, MFU
- Direction-specific: forward_loss, backward_loss (bidirectional)
- Validation accuracy by direction
- Gradient norms per layer

**Tier 2: Detailed Research Logs (HDF5)**
- Per-token predictions and losses
- Gradient statistics per layer
- Activation patterns
- Complete reproducibility data

### 5.2 HDF5 Structure

**New Module:** `nanochat/research_logger.py`

```
experiments/20250111_120000_forward_d20/
├── config.json                          # Human-readable config
├── base_training.h5                     # Base phase logs
│   ├── /step_0000/
│   │   ├── loss                         # scalar
│   │   ├── per_token_losses             # (B, T) array
│   │   ├── per_token_predictions        # (B, T) indices
│   │   ├── gradients/layer_0/weight     # gradient stats
│   │   ├── gradients/layer_1/weight
│   │   ├── activations/layer_0/         # activation stats
│   │   └── metadata                     # timestamp, LR, etc.
│   ├── /step_0010/
│   └── ...
├── midtraining.h5
├── sft.h5
└── rl.h5
```

### 5.3 Logging Frequency

- **Every step**: Loss, LR, tokens/sec, MFU
- **Every 10 steps**: Per-token loss statistics, prediction accuracy by position
- **Every 100 steps**: Gradient norms, activation statistics
- **Every eval**: Full validation predictions, attention samples

### 5.4 Storage Estimates

For d20 model (~1.3B params) training 10K steps:
- WandB logs: ~10MB (lightweight metrics)
- HDF5 per-step logs: ~500MB (sampled, not every token)
- HDF5 per-100-step logs: ~2GB (gradients, activations)
- Total per experiment: ~5-10GB

## 6. Evaluation Framework

### 6.1 Multi-Direction Evaluation

**New Module:** `nanochat/direction_eval.py`

```python
def evaluate_task_multi_direction(task, model, tokenizer, directions):
    """
    Evaluates task in multiple directions.

    Returns:
        {
            "forward": {"accuracy": 0.85, "per_example": [...]},
            "backward": {"accuracy": 0.42, "per_example": [...]},
        }
    """
```

### 6.2 Direction Conversion Logic

**Forward Evaluation:**
- Render task normally
- Standard inference

**Backward Evaluation:**
- Reverse input tokens
- Run model
- Reverse output logits back
- For multiple choice: reverse question + each choice
- For generation: reverse prompt, generate, reverse output

### 6.3 Evaluation Strategy by Model Type

**Forward Models:**
- Primary: Forward direction (native)
- Secondary: Backward direction (transfer capability)

**Backward Models:**
- Primary: Backward direction (native)
- Secondary: Forward direction (transfer capability)

**Bidirectional Models:**
- Both directions with appropriate direction tokens
- Measures directional control

### 6.4 New Metrics

- **Native accuracy**: Performance in trained direction
- **Cross-direction accuracy**: Performance in opposite direction
- **Direction gap**: `abs(forward_acc - backward_acc)`
- **Bidirectional balance**: Consistency across directions

### 6.5 Evaluation Results Storage

```
experiments/20250111_120000_forward_d20/
└── evaluations/
    ├── base_phase/
    │   ├── forward_direction/
    │   │   ├── arc_easy.json
    │   │   ├── arc_challenge.json
    │   │   ├── mmlu.json
    │   │   ├── gsm8k.json
    │   │   └── humaneval.json
    │   └── backward_direction/
    │       ├── arc_easy.json
    │       └── ...
    ├── mid_phase/
    ├── sft_phase/
    └── rl_phase/
```

## 7. Experiment Orchestration

### 7.1 Flexible Experiment Runner

**New Script:** `scripts/run_backward_experiment.py`

**Two-Level Architecture:**
1. **Base Experiments** - Train foundation models
2. **Derived Experiments** - Branch from existing checkpoints

### 7.2 Experiment Creation Examples

**Create Base Model:**
```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=forward \
    --depth=20 \
    --experiment_id=base_20250111_forward_d20
```

**Run Midtraining:**
```bash
python -m scripts.run_backward_experiment \
    --phase=mid \
    --parent_experiment=base_20250111_forward_d20 \
    --experiment_id=mid_v1_20250111
```

**Branch Multiple SFT Variants:**
```bash
# SFT variant 1
python -m scripts.run_backward_experiment \
    --phase=sft \
    --parent_experiment=mid_v1_20250111 \
    --experiment_id=sft_hightemp_20250112 \
    --init_lr_frac=0.05

# SFT variant 2 (from same parent)
python -m scripts.run_backward_experiment \
    --phase=sft \
    --parent_experiment=mid_v1_20250111 \
    --experiment_id=sft_lowtemp_20250112 \
    --init_lr_frac=0.01
```

### 7.3 Experiment Directory Structure

```
experiments/
├── base_20250111_forward_d20/           # Base experiment
│   ├── config.json
│   ├── parent_link.json                 # null (no parent)
│   ├── checkpoints/
│   ├── base_training.h5
│   └── evaluations/
│
├── mid_v1_20250111/                     # Midtraining
│   ├── config.json
│   ├── parent_link.json                 # → base_20250111_forward_d20
│   ├── checkpoints/
│   ├── midtraining.h5
│   └── evaluations/
│
├── sft_hightemp_20250112/               # SFT variant 1
│   ├── config.json
│   ├── parent_link.json                 # → mid_v1_20250111
│   ├── checkpoints/
│   ├── sft.h5
│   └── evaluations/
│
└── sft_lowtemp_20250112/                # SFT variant 2
    ├── config.json
    ├── parent_link.json                 # → mid_v1_20250111
    ├── checkpoints/
    ├── sft.h5
    └── evaluations/
```

### 7.4 Parent Linking System

**parent_link.json:**
```json
{
    "parent_experiment_id": "mid_v1_20250111",
    "parent_checkpoint_path": "experiments/mid_v1_20250111/checkpoints/step_12000",
    "parent_phase": "mid",
    "inheritance_chain": ["base_20250111_forward_d20", "mid_v1_20250111"]
}
```

### 7.5 Phase Execution Flags

```bash
--phase=base              # Just base training
--phase=mid               # Just midtraining (requires parent)
--phase=sft               # Just SFT (requires parent)
--phase=rl                # Just RL (requires parent)
--phase=base,mid          # Chain multiple phases
--phase=eval              # Just evaluate existing checkpoint
```

## 8. Research Analysis Tools

### 8.1 Analysis Module

**New Module:** `nanochat/research_analysis.py`

**Key Functions:**

```python
compare_learning_curves(exp_dirs, output_path)
# Plot loss curves for multiple experiments side-by-side

analyze_direction_transfer(exp_dir, output_path)
# Analyze cross-direction performance, generate transfer report

gradient_flow_analysis(exp_dir, output_path)
# Visualize gradient statistics over training

token_position_accuracy(exp_dir, output_path)
# Per-position prediction accuracy analysis

generate_research_report(exp_dir, output_path)
# Auto-generate comprehensive markdown report
```

### 8.2 Comparison Tool

```bash
# Compare all SFT variants from same parent
python -m nanochat.research_analysis compare \
    --experiments sft_hightemp_20250112 sft_lowtemp_20250112 \
    --metrics accuracy,loss,convergence_speed \
    --output comparison_report.html
```

### 8.3 Auto-Generated Research Reports

After each experiment: `experiments/{timestamp}/RESEARCH_REPORT.md`

**Contents:**
- Experiment configuration
- Training dynamics (loss curves, convergence)
- Evaluation results (all tasks, all directions)
- Cross-direction transfer analysis
- Gradient/activation statistics
- Comparison to baseline (forward model)
- Key findings and observations

## 9. Implementation Plan

### 9.1 New Files

```
nanochat/
├── direction_dataloader.py         # Direction-aware data loading
├── direction_eval.py                # Multi-direction evaluation
├── research_logger.py               # HDF5 logging system
└── research_analysis.py             # Analysis and visualization

scripts/
└── run_backward_experiment.py      # Experiment orchestration

tasks/
└── direction_wrapper.py             # Wraps tasks for backward eval
```

### 9.2 Modified Files

```
nanochat/
├── tokenizer.py                     # Add <|forward|>/<|backward|> tokens
├── checkpoint_manager.py            # Store/load direction metadata
└── common.py                        # Direction handling helpers

scripts/
├── base_train.py                    # Add --direction flag
├── mid_train.py                     # Auto-detect direction
├── chat_sft.py                      # Auto-detect direction
├── chat_rl.py                       # Auto-detect direction
├── base_eval.py                     # Multi-direction support
└── chat_eval.py                     # Multi-direction support
```

### 9.3 Implementation Phases

**Phase 1: Core Infrastructure** (Week 1)
- Implement `direction_dataloader.py`
- Modify tokenizer for special tokens
- Update checkpoint manager for direction metadata
- Test token reversal logic

**Phase 2: Training Integration** (Week 1)
- Modify training scripts for direction flag
- Implement auto-detection from checkpoints
- Test forward/backward/bidirectional training
- Validate loss computation

**Phase 3: Logging System** (Week 2)
- Implement `research_logger.py`
- HDF5 structure creation
- Integration with training loop
- Test logging at different frequencies

**Phase 4: Evaluation Framework** (Week 2)
- Implement `direction_eval.py`
- Multi-direction task evaluation
- Cross-direction transfer testing
- Validate backward evaluation logic

**Phase 5: Orchestration** (Week 3)
- Implement `run_backward_experiment.py`
- Parent linking system
- Flexible phase execution
- Test branching experiments

**Phase 6: Analysis Tools** (Week 3)
- Implement `research_analysis.py`
- Comparison visualizations
- Auto-report generation
- End-to-end testing

## 10. Example Research Workflow

### 10.1 Complete Multi-Scale Study

```bash
# Train base models at three scales, three directions each
for depth in 12 20 26; do
    for direction in forward backward bidirectional; do
        python -m scripts.run_backward_experiment \
            --phase=base \
            --direction=$direction \
            --depth=$depth \
            --experiment_id=base_${direction}_d${depth}
    done
done

# Run midtraining on all 9 base models
for exp in experiments/base_*; do
    python -m scripts.run_backward_experiment \
        --phase=mid \
        --parent_experiment=$(basename $exp) \
        --experiment_id=mid_$(basename $exp)
done

# Compare learning curves across directions
python -m nanochat.research_analysis compare_learning \
    --experiments base_*_d20 \
    --output learning_curves_d20.png

# Analyze cross-direction transfer
python -m nanochat.research_analysis direction_transfer \
    --experiment base_backward_d20 \
    --output backward_transfer_analysis.md

# Generate comprehensive comparison
python -m nanochat.research_analysis compare_all \
    --base_experiments base_* \
    --output comprehensive_comparison.html
```

### 10.2 Hyperparameter Sweep on SFT

```bash
# Train base and mid once
python -m scripts.run_backward_experiment \
    --phase=base,mid \
    --direction=forward \
    --depth=20 \
    --experiment_id=base_forward_d20

# Sweep SFT learning rates
for lr in 0.01 0.02 0.05 0.10; do
    python -m scripts.run_backward_experiment \
        --phase=sft \
        --parent_experiment=base_forward_d20 \
        --experiment_id=sft_lr${lr} \
        --init_lr_frac=$lr
done

# Compare all SFT variants
python -m nanochat.research_analysis compare \
    --experiments sft_lr* \
    --metrics mmlu_acc,arc_easy_acc,convergence \
    --output sft_lr_sweep.png
```

## 11. Expected Outcomes

### 11.1 Research Artifacts

For each experiment, the framework will generate:

1. **Training logs** (HDF5): Complete training dynamics
2. **Evaluation results** (JSON): All tasks, all directions
3. **Checkpoints**: Reusable model weights
4. **Research report** (Markdown): Auto-generated analysis
5. **Visualizations** (PNG/HTML): Loss curves, accuracy plots
6. **Comparison data**: Cross-experiment analysis

### 11.2 Research Questions Answered

- **Learning Efficiency**: Do backward models converge faster/slower?
- **Representation Quality**: Different attention patterns?
- **Direction Transfer**: Can models generalize across directions?
- **Bidirectional Control**: Can models effectively switch directions?
- **Scaling Laws**: How do findings change with model size?

## 12. Success Criteria

The implementation will be considered successful when:

1. ✅ All three model types train successfully to completion
2. ✅ Cross-direction evaluation works correctly
3. ✅ HDF5 logs capture complete training dynamics
4. ✅ Experiment branching enables hyperparameter sweeps
5. ✅ Analysis tools generate publication-ready visualizations
6. ✅ Complete workflow runs end-to-end without manual intervention
7. ✅ Research reports are comprehensive and reproducible

## 13. Future Extensions

### 13.1 Potential Research Directions

- **Variable-length context**: How does direction affect long-context tasks?
- **Multilingual**: Do different languages benefit from different directions?
- **Architecture variants**: MoE, different attention patterns
- **Inference optimization**: KV cache for bidirectional models
- **Curriculum learning**: Start forward, transition to bidirectional

### 13.2 Engineering Improvements

- **Distributed logging**: Scale to multi-node training
- **Real-time dashboards**: Live experiment monitoring
- **Automated analysis**: Trigger on experiment completion
- **Cloud integration**: S3/GCS for artifact storage
- **Experiment search**: Query experiments by hyperparameters

## 14. References and Related Work

- DCLM (Data-Centric Language Modeling)
- Chinchilla scaling laws
- GRPO/REINFORCE for RL
- HuggingFace datasets (FineWeb, SmolTalk)
- Modded-nanoGPT (speedrun inspiration)

---

**End of Design Document**

This design provides a complete blueprint for implementing a backward language model research framework. The implementation will enable rigorous, scientific comparison of forward, backward, and bidirectional language models across all training phases with comprehensive logging and analysis tools.
