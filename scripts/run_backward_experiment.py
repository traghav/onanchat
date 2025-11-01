"""
Orchestrate backward language model experiments.

Handles flexible experiment creation, parent linking, and phase execution.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from nanochat.experiment_config import ExperimentConfig
from nanochat.common import get_base_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run backward LM experiment")

    # Experiment identification
    parser.add_argument("--experiment_id", type=str, default=None,
                        help="Custom experiment ID (auto-generated if not provided)")
    parser.add_argument("--parent_experiment", type=str, default=None,
                        help="Parent experiment ID to continue from")

    # Experiment configuration
    parser.add_argument("--phase", type=str, required=True,
                        help="Phase to run: base, mid, sft, rl, or comma-separated list")
    parser.add_argument("--direction", type=str, default="forward",
                        help="Direction: forward, backward, bidirectional")
    parser.add_argument("--depth", type=int, default=20,
                        help="Model depth")

    # Training parameters (passed through to training scripts)
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--init_lr_frac", type=float, default=None)

    return parser.parse_args()


def run_base_training(experiment_dir, config, args):
    """Run base training phase."""
    print(f"\n{'='*60}")
    print(f"Running BASE training")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.base_train",
        f"--direction={config['direction']}",
        f"--depth={config['depth']}",
        f"--model_tag={config['experiment_id']}",
    ]

    # Add optional args
    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Base training failed")
        sys.exit(1)

    print("✅ Base training complete")


def run_midtraining(experiment_dir, config, parent_config, args):
    """Run midtraining phase."""
    print(f"\n{'='*60}")
    print(f"Running MIDTRAINING")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.mid_train",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Midtraining failed")
        sys.exit(1)

    print("✅ Midtraining complete")


def run_sft(experiment_dir, config, parent_config, args):
    """Run SFT phase."""
    print(f"\n{'='*60}")
    print(f"Running SFT")
    print(f"{'='*60}\n")

    # Determine source (mid or base)
    source = parent_config['phase']

    cmd = [
        "python", "-m", "scripts.chat_sft",
        f"--source={source}",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")
    if args.num_iterations:
        cmd.append(f"--num_iterations={args.num_iterations}")
    if args.init_lr_frac:
        cmd.append(f"--init_lr_frac={args.init_lr_frac}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ SFT failed")
        sys.exit(1)

    print("✅ SFT complete")


def run_rl(experiment_dir, config, parent_config, args):
    """Run RL phase."""
    print(f"\n{'='*60}")
    print(f"Running RL")
    print(f"{'='*60}\n")

    cmd = [
        "python", "-m", "scripts.chat_rl",
        "--source=sft",
        f"--model_tag={parent_config['experiment_id']}",
    ]

    if args.device_batch_size:
        cmd.append(f"--device_batch_size={args.device_batch_size}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ RL failed")
        sys.exit(1)

    print("✅ RL complete")


def main():
    args = parse_args()

    # Initialize experiment config
    base_dir = get_base_dir()
    exp_config = ExperimentConfig(base_dir=os.path.join(base_dir, "experiments"))

    # Parse phases
    phases = args.phase.split(',')

    # Create or load experiment
    if "base" in phases:
        # Create new base experiment
        experiment_dir = exp_config.create_experiment(
            phase="base",
            direction=args.direction,
            depth=args.depth,
            experiment_id=args.experiment_id,
        )
        config = exp_config.load_config(experiment_dir.name)

        # Run base training
        run_base_training(experiment_dir, config, args)

        # Update for subsequent phases
        parent_experiment_id = experiment_dir.name
    else:
        # Must have parent for non-base phases
        if not args.parent_experiment:
            print("❌ Error: --parent_experiment required for non-base phases")
            sys.exit(1)

        parent_experiment_id = args.parent_experiment

    # Run subsequent phases
    for phase in phases:
        if phase == "base":
            continue  # Already ran

        # Create experiment for this phase
        parent_config = exp_config.load_config(parent_experiment_id)
        experiment_dir = exp_config.create_experiment(
            phase=phase,
            direction=parent_config["direction"],
            depth=parent_config["depth"],
            parent_experiment=parent_experiment_id,
            experiment_id=args.experiment_id,
        )
        config = exp_config.load_config(experiment_dir.name)

        # Run phase
        if phase == "mid":
            run_midtraining(experiment_dir, config, parent_config, args)
        elif phase == "sft":
            run_sft(experiment_dir, config, parent_config, args)
        elif phase == "rl":
            run_rl(experiment_dir, config, parent_config, args)

        # Update parent for next phase
        parent_experiment_id = experiment_dir.name

    print(f"\n{'='*60}")
    print(f"✅ Experiment complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
