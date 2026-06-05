#!/bin/bash
# Auto-chaining curriculum: 0.3m → 0.2m → 0.1m
# Stage 1 (0.5m) already done. Pass its checkpoint as STAGE1_CKPT.
set -e

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

STAGE1_CKPT="logs/Go1BridgeCrossing-20260604-210703/checkpoints/000235929600"

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

# ── Stage 2: 0.3m ──────────────────────────────────────────────────────────
echo "=== Stage 2: 0.3m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$STAGE1_CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.15}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage2_0.3m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage2.log

STAGE2_LOGDIR=$(get_logdir /tmp/bridge_no_air_stage2.log)
STAGE2_CKPT=$(get_latest_ckpt "$STAGE2_LOGDIR")
echo "Stage 2 done. Checkpoint: $STAGE2_CKPT"

# ── Stage 3: 0.2m ──────────────────────────────────────────────────────────
echo "=== Stage 3: 0.2m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$STAGE2_CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.1}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage3_0.2m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage3.log

STAGE3_LOGDIR=$(get_logdir /tmp/bridge_no_air_stage3.log)
STAGE3_CKPT=$(get_latest_ckpt "$STAGE3_LOGDIR")
echo "Stage 3 done. Checkpoint: $STAGE3_CKPT"

# ── Stage 4: 0.1m ──────────────────────────────────────────────────────────
echo "=== Stage 4: 0.1m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$STAGE3_CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.05}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage4_0.1m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage4.log

echo "=== All stages complete ==="
