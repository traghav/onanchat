import torch
import tempfile
import shutil
from pathlib import Path
from nanochat.research_logger import ResearchLogger
import h5py

# Create temp experiment dir
tmpdir = tempfile.mkdtemp()

try:
    # Initialize logger
    logger = ResearchLogger(tmpdir, phase="test_phase")

    # Log a step with various metric types
    logger.log_step(0, {
        'loss': 2.5,
        'learning_rate': 0.001,
        'per_token_losses': torch.randn(4, 16),  # (B, T)
        'gradients': {
            'layer_0': torch.randn(128, 128),
            'layer_1': torch.randn(128, 128),
        }
    })

    logger.log_step(10, {
        'loss': 2.3,
        'learning_rate': 0.001,
    })

    logger.close()

    # Verify HDF5 file exists and has correct structure
    h5_path = Path(tmpdir) / "test_phase.h5"
    assert h5_path.exists(), "HDF5 file not created"

    with h5py.File(h5_path, 'r') as f:
        assert 'step_000000' in f, "Step 0 not logged"
        assert 'step_000010' in f, "Step 10 not logged"
        assert f['step_000000'].attrs['loss'] == 2.5, "Loss not logged correctly"
        assert 'per_token_losses' in f['step_000000'], "Per-token losses not logged"
        assert 'gradients' in f['step_000000'], "Gradients not logged"

    print("✅ ResearchLogger works correctly")

finally:
    # Cleanup
    shutil.rmtree(tmpdir)
