"""
Research analysis and visualization tools for backward LM experiments.

Provides functions for:
- Comparing learning curves across experiments
- Analyzing cross-direction transfer
- Generating research reports
"""

import os
import json
import numpy as np
from pathlib import Path

# Make matplotlib import optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Make h5py import optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_experiment_metadata(experiment_dir):
    """Load experiment config and metadata."""
    exp_path = Path(experiment_dir)

    with open(exp_path / "config.json", 'r') as f:
        config = json.load(f)

    parent_link = None
    if (exp_path / "parent_link.json").exists():
        with open(exp_path / "parent_link.json", 'r') as f:
            parent_link = json.load(f)

    return config, parent_link


def load_training_losses(experiment_dir, phase="base_training"):
    """Load loss curve from HDF5 logs."""
    if not HAS_H5PY:
        print("Warning: h5py not installed, cannot load training losses")
        return None, None

    h5_path = Path(experiment_dir) / f"{phase}.h5"

    if not h5_path.exists():
        return None, None

    steps = []
    losses = []

    with h5py.File(h5_path, 'r') as f:
        for step_name in sorted(f.keys()):
            if not step_name.startswith("step_"):
                continue

            step_num = int(step_name.split("_")[1])
            step_group = f[step_name]

            if 'loss' in step_group.attrs:
                steps.append(step_num)
                losses.append(step_group.attrs['loss'])

    return np.array(steps), np.array(losses)


def compare_learning_curves(experiment_dirs, output_path, phase="base_training"):
    """
    Compare learning curves across multiple experiments.

    Args:
        experiment_dirs: List of experiment directory paths
        output_path: Where to save the plot
        phase: Which phase to compare
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, cannot generate plots")
        return

    plt.figure(figsize=(10, 6))

    for exp_dir in experiment_dirs:
        config, _ = load_experiment_metadata(exp_dir)
        steps, losses = load_training_losses(exp_dir, phase)

        if steps is None:
            print(f"Warning: No data for {exp_dir}")
            continue

        label = f"{config['direction']} d{config['depth']}"
        plt.plot(steps, losses, label=label, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves Comparison - {phase}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved learning curves to {output_path}")


def analyze_direction_transfer(experiment_dir, output_path):
    """
    Analyze cross-direction transfer from evaluation results.

    Args:
        experiment_dir: Experiment directory
        output_path: Where to save the report
    """
    config, _ = load_experiment_metadata(experiment_dir)
    eval_dir = Path(experiment_dir) / "evaluations"

    if not eval_dir.exists():
        print(f"No evaluations found in {experiment_dir}")
        return

    report_lines = [
        f"# Direction Transfer Analysis",
        f"",
        f"**Experiment:** {config['experiment_id']}",
        f"**Direction:** {config['direction']}",
        f"**Depth:** d{config['depth']}",
        f"",
    ]

    # TODO: Load and analyze evaluation results
    # This requires evaluation results to be saved in a standard format

    report_lines.append("## Cross-Direction Performance")
    report_lines.append("")
    report_lines.append("*Analysis coming soon - requires evaluation results*")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved transfer analysis to {output_path}")


def generate_research_report(experiment_dir, output_path):
    """
    Generate comprehensive research report for an experiment.

    Args:
        experiment_dir: Experiment directory
        output_path: Where to save the report
    """
    config, parent_link = load_experiment_metadata(experiment_dir)

    report_lines = [
        f"# Research Report: {config['experiment_id']}",
        f"",
        f"Generated: {config.get('created_at', 'Unknown')}",
        f"",
        f"## Configuration",
        f"",
        f"- **Phase:** {config['phase']}",
        f"- **Direction:** {config['direction']}",
        f"- **Depth:** d{config['depth']}",
    ]

    if parent_link:
        report_lines.extend([
            f"- **Parent:** {parent_link['parent_experiment_id']}",
        ])

    report_lines.extend([
        f"",
        f"## Training Dynamics",
        f"",
    ])

    # Load and summarize training
    phase_name = config['phase'] + "_training" if config['phase'] == "base" else config['phase']
    steps, losses = load_training_losses(experiment_dir, phase_name)

    if steps is not None and len(steps) > 0:
        report_lines.extend([
            f"- **Total steps:** {steps[-1]}",
            f"- **Initial loss:** {losses[0]:.4f}",
            f"- **Final loss:** {losses[-1]:.4f}",
            f"- **Improvement:** {losses[0] - losses[-1]:.4f}",
        ])
    else:
        report_lines.append("*No training data available*")

    report_lines.extend([
        f"",
        f"## Evaluation Results",
        f"",
        f"*Coming soon - requires evaluation implementation*",
        f"",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Saved research report to {output_path}")
