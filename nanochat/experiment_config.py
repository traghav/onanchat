"""
Experiment configuration and metadata management.

Handles experiment directory structure, parent linking, and config persistence.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """Manages experiment configuration and directory structure."""

    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, phase, direction, depth, parent_experiment=None,
                          experiment_id=None, **kwargs):
        """
        Create new experiment directory with config.

        Args:
            phase: "base", "mid", "sft", "rl"
            direction: "forward", "backward", "bidirectional"
            depth: Model depth (e.g., 20)
            parent_experiment: ID of parent experiment (None for base)
            experiment_id: Custom experiment ID (None = auto-generate)
            **kwargs: Additional config parameters

        Returns:
            experiment_dir: Path to experiment directory
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            direction_suffix = f"_{direction}" if direction != "forward" else ""
            experiment_id = f"{phase}_{timestamp}_d{depth}{direction_suffix}"

        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "evaluations").mkdir(exist_ok=True)

        # Build config
        config = {
            "experiment_id": experiment_id,
            "phase": phase,
            "direction": direction,
            "depth": depth,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }

        # Save config
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Create parent link if specified
        if parent_experiment:
            parent_link = {
                "parent_experiment_id": parent_experiment,
                "parent_checkpoint_path": str(self.base_dir / parent_experiment / "checkpoints"),
                "parent_phase": self._get_parent_phase(phase),
            }
            with open(experiment_dir / "parent_link.json", 'w') as f:
                json.dump(parent_link, f, indent=2)

        return experiment_dir

    def load_config(self, experiment_id):
        """Load experiment config."""
        config_path = self.base_dir / experiment_id / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_parent_link(self, experiment_id):
        """Load parent link if it exists."""
        link_path = self.base_dir / experiment_id / "parent_link.json"
        if link_path.exists():
            with open(link_path, 'r') as f:
                return json.load(f)
        return None

    def _get_parent_phase(self, current_phase):
        """Get expected parent phase."""
        phase_order = {"base": None, "mid": "base", "sft": "mid", "rl": "sft"}
        return phase_order.get(current_phase)
