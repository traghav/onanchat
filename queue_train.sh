#!/bin/bash
# Training queue: Remaining tasks after partial completion
# All outputs logged to queue_train.log

set -e  # Exit on error

cd /home/ubuntu/onanchat
source .venv/bin/activate

echo "========================================"
echo "Starting training queue at $(date)"
echo "========================================"

# SKIPPED: Mid-training backward (already done - d20_backward exists at step 1628)
# SKIPPED: SFT backward (already done - d20_backward exists)

# 1. Mid-training for bidirectional (resume/complete)
echo ""
echo "[1/6] Mid-training d20_bidirectional - $(date)"
echo "----------------------------------------"
python -m scripts.mid_train \
    --model_tag=d20_bidirectional \
    --run=mid_d20_bidirectional \
    --device_batch_size=8

# 2. SFT for bidirectional (from mid)
echo ""
echo "[2/6] SFT d20_bidirectional - $(date)"
echo "----------------------------------------"
python -m scripts.chat_sft \
    --source=mid \
    --model_tag=d20_bidirectional \
    --run=sft_d20_bidirectional

# 3. SFT for forward (from mid)
echo ""
echo "[3/6] SFT d20_forward - $(date)"
echo "----------------------------------------"
python -m scripts.chat_sft \
    --source=mid \
    --model_tag=d20_forward_2 \
    --run=sft_d20_forward

# 4. RL for backward (from sft)
echo ""
echo "[4/6] RL d20_backward - $(date)"
echo "----------------------------------------"
python -m scripts.chat_rl \
    --source=sft \
    --model_tag=d20_backward \
    --run=rl_d20_backward

# 5. RL for bidirectional (from sft)
echo ""
echo "[5/6] RL d20_bidirectional - $(date)"
echo "----------------------------------------"
python -m scripts.chat_rl \
    --source=sft \
    --model_tag=d20_bidirectional \
    --run=rl_d20_bidirectional

# 6. RL for forward (from sft)
echo ""
echo "[6/6] RL d20_forward - $(date)"
echo "----------------------------------------"
python -m scripts.chat_rl \
    --source=sft \
    --model_tag=d20_forward \
    --run=rl_d20_forward

echo ""
echo "========================================"
echo "Training queue completed at $(date)"
echo "========================================"
