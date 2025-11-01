# Backward LM Framework - Quick Start Guide

## Installation

1. Follow the base nanochat installation from `README.md`
2. Install optional dependencies for research features:
```bash
# For HDF5 logging
pip install h5py

# For visualization
pip install matplotlib
```

## Running Your First Experiment

### 1. Train a Forward Model (Baseline)

```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=forward \
    --depth=12 \
    --num_iterations=1000
```

### 2. Train a Backward Model

```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=backward \
    --depth=12 \
    --num_iterations=1000
```

### 3. Train a Bidirectional Model

```bash
python -m scripts.run_backward_experiment \
    --phase=base \
    --direction=bidirectional \
    --depth=12 \
    --num_iterations=1000
```

### 4. Compare Results

```python
from nanochat.research_analysis import compare_learning_curves
import glob

# Find all experiments
exps = glob.glob('experiments/base_*_d12*')

# Compare learning curves
compare_learning_curves(exps, 'forward_vs_backward.png')
```

## Training Phases

### Base Training
```bash
python -m scripts.base_train --direction=backward --depth=20
```

### Midtraining (auto-detects direction from base model)
```bash
python -m scripts.mid_train --model_tag=d20_backward
```

### SFT (auto-detects direction)
```bash
python -m scripts.chat_sft --source=mid --model_tag=d20_backward
```

### RL (auto-detects direction)
```bash
python -m scripts.chat_rl --source=sft --model_tag=d20_backward
```

## Multi-Phase Experiments

Run complete training pipeline:
```bash
python -m scripts.run_backward_experiment \
    --phase=base,mid \
    --direction=backward \
    --depth=20 \
    --num_iterations=5000
```

Branch SFT from existing mid:
```bash
python -m scripts.run_backward_experiment \
    --phase=sft \
    --parent_experiment=mid_20250111_d20_backward \
    --init_lr_frac=0.05
```

## Research Analysis

### Generate Report
```python
from nanochat.research_analysis import generate_research_report

generate_research_report(
    'experiments/base_20250111_d20_backward',
    'report.md'
)
```

### Analyze Direction Transfer
```python
from nanochat.research_analysis import analyze_direction_transfer

analyze_direction_transfer(
    'experiments/base_20250111_d20_backward',
    'transfer_analysis.md'
)
```

## Next Steps

- See `CLAUDE.md` for complete documentation
- See `docs/plans/2025-01-11-backward-language-models-design.md` for design details
- Check `experiments/` directory for your training runs
- View HDF5 logs with h5py or HDFView for detailed analysis

## Troubleshooting

**h5py not installed**: Research logging will be skipped but training will work fine

**matplotlib not installed**: Plotting functions will warn but analysis will continue

**Direction mismatch**: Ensure you're loading the correct checkpoint with matching direction
