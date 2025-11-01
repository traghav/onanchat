import tempfile
import shutil
from nanochat.experiment_config import ExperimentConfig

tmpdir = tempfile.mkdtemp()

try:
    config = ExperimentConfig(base_dir=tmpdir)

    # Create base experiment
    base_dir = config.create_experiment(
        phase="base",
        direction="forward",
        depth=20,
        experiment_id="test_base"
    )

    assert base_dir.exists()
    assert (base_dir / "config.json").exists()
    assert (base_dir / "checkpoints").exists()

    # Create child experiment
    mid_dir = config.create_experiment(
        phase="mid",
        direction="forward",
        depth=20,
        parent_experiment="test_base",
        experiment_id="test_mid"
    )

    assert (mid_dir / "parent_link.json").exists()

    # Load and verify
    loaded_config = config.load_config("test_base")
    assert loaded_config["phase"] == "base"

    parent_link = config.load_parent_link("test_mid")
    assert parent_link["parent_experiment_id"] == "test_base"

    print("✅ ExperimentConfig works correctly")

finally:
    shutil.rmtree(tmpdir)
