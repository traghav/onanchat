# Repository Guidelines

## Project Scope & Structure
- Fork focus: backward and bidirectional LMs with experiment orchestration and research logging layered on upstream nanochat.
- `nanochat/`: core model plus direction-aware pieces (`direction_dataloader.py` reverses tokens while pinning BOS, `direction_eval.py`, `directional_chat_engine.py`, `research_logger.py`, `research_analysis.py`, `experiment_config.py`, tokenizer with `<|forward|>`/`<|backward|>` tokens).
- `scripts/`: stage runners with direction flags/auto-detect (`base_train.py`, `mid_train.py`, `chat_sft.py`, `chat_rl.py`, `run_backward_experiment.py`, chat CLI/Web). `tasks/` adds `direction_wrapper.py` for backward eval. Runs live in `experiments/`; data shards in `base_data/`.
- `docs/`: backward LM quickstart and design plans (`CLAUDE.md`, `docs/QUICKSTART.md`, `docs/plans/...`). `dev/` for CPU/MPS and data helpers. Tests in `tests/` plus root direction-specific files.

## Setup, Build, and Development Commands
- Env: `uv venv && uv sync --extra gpu` (or `--extra cpu`), then `source .venv/bin/activate`.
- Build Rust tokenizer before training/tests: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`.
- Directional base run: `python -m scripts.base_train --direction=backward --depth=20 --num_iterations=...`; continue with `mid_train`, `chat_sft`, `chat_rl` (direction auto-read from checkpoint metadata/model_tag).
- Orchestrated pipeline/branching: `python -m scripts.run_backward_experiment --phase=base,mid --direction=bidirectional --depth=12 --num_iterations=1000` or add `--parent_experiment=<id>` to fork.
- Quick hooks: `bash speedrun.sh` (forward d20) or `bash run1000.sh` (d32). Serve/chat: `python -m scripts.chat_web` (direction shown in header) or `python -m scripts.chat_cli --checkpoint <path>`.

## Direction Model Mechanics
- Backward: dataloader reverses tokens only (BOS stays first); chat/eval auto reverse outputs so users see normal chronology—avoid manual flips.
- Bidirectional: batches mix directions; tokenizer injects `<|forward|>`/`<|backward|>` prefixes. Direction metadata saved in checkpoints via `checkpoint_manager`.
- Research logging: optional h5py-backed `research_logger` writes per-step tensors; `research_analysis` compares learning curves, direction transfer, and emits reports.

## Testing Guidelines
- Core: `python -m pytest tests -v`; targeted: `python -m pytest test_directional_chat.py -k backward`, `python -m pytest test_direction_dataloader.py`, `python -m pytest test_engine.py`.
- Use `-m "not slow"` to skip slow marks; seed torch/numpy for determinism. Ensure env active and `rustbpe` built to avoid import errors.

## Commit & Pull Request Guidelines
- Conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`); one concern per commit.
- PRs should list direction/depth, data shards, hardware, commands run, and metrics (CORE/loss plus forward vs backward deltas). Link issues/discussions; attach logs/plots or `experiments/.../report.md`. Do not commit downloaded data or checkpoints—reference paths instead.
