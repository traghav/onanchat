"""
Research logging system for backward language model experiments.

Provides comprehensive HDF5-based logging for:
- Per-token losses and predictions
- Gradient statistics per layer
- Activation patterns
- Complete training dynamics
"""

import os
import json
import h5py
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


class ResearchLogger:
    """Logs detailed training metrics to HDF5 for research analysis."""

    def __init__(self, experiment_dir, phase="base_training"):
        """
        Initialize research logger.

        Args:
            experiment_dir: Directory for this experiment (will be created)
            phase: Training phase (base_training, midtraining, sft, rl)
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.phase = phase
        self.h5_path = self.experiment_dir / f"{phase}.h5"

        # Create HDF5 file
        self.h5file = h5py.File(self.h5_path, 'a')  # Append mode

    def log_step(self, step, metrics):
        """
        Log metrics for a training step.

        Args:
            step: Training step number
            metrics: Dict with keys like 'loss', 'per_token_losses', etc.
        """
        step_group = self.h5file.require_group(f"step_{step:06d}")

        # Save scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                step_group.attrs[key] = value
            elif isinstance(value, torch.Tensor):
                # Convert to numpy and save as dataset
                step_group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")
            elif isinstance(value, np.ndarray):
                step_group.create_dataset(key, data=value, compression="gzip")
            elif isinstance(value, dict):
                # Nested dict (e.g., gradient stats per layer)
                subgroup = step_group.require_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        subgroup.create_dataset(subkey, data=subvalue.cpu().numpy(), compression="gzip")
                    else:
                        subgroup.attrs[subkey] = subvalue

        # Flush to disk
        self.h5file.flush()

    def close(self):
        """Close HDF5 file."""
        if self.h5file:
            self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
