#!/bin/bash
# Curriculum: 0.8m → 0.7m → 0.6m → 0.5m → 0.4m → 0.3m → 0.2m → 0.1m
# (feet_air_time disabled)

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

# ── Stage 1: 0.8m  (half_width=0.40) ──────────────────────────────────────
echo "=== Stage 1: 0.8m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --playground_config_overrides '{"bridge_half_width": 0.4}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage1_0.8m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage1.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 1 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage1.log)")
echo "Stage 1 done. Checkpoint: $CKPT"

# ── Stage 2: 0.7m  (half_width=0.35) ──────────────────────────────────────
echo "=== Stage 2: 0.7m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.35}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage2_0.7m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage2.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 2 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage2.log)")
echo "Stage 2 done. Checkpoint: $CKPT"

# ── Stage 3: 0.6m  (half_width=0.30) ──────────────────────────────────────
echo "=== Stage 3: 0.6m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.3}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage3_0.6m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage3.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 3 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage3.log)")
echo "Stage 3 done. Checkpoint: $CKPT"

# ── Stage 4: 0.5m  (half_width=0.25) ──────────────────────────────────────
echo "=== Stage 4: 0.5m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.25}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage4_0.5m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage4.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 4 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage4.log)")
echo "Stage 4 done. Checkpoint: $CKPT"

# ── Stage 5: 0.4m  (half_width=0.20) ──────────────────────────────────────
echo "=== Stage 5: 0.4m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.2}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage5_0.4m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage5.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 5 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage5.log)")
echo "Stage 5 done. Checkpoint: $CKPT"

# ── Stage 6: 0.3m  (half_width=0.15) ──────────────────────────────────────
echo "=== Stage 6: 0.3m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.15}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage6_0.3m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage6.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 6 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage6.log)")
echo "Stage 6 done. Checkpoint: $CKPT"

# ── Stage 7: 0.2m  (half_width=0.10) ──────────────────────────────────────
echo "=== Stage 7: 0.2m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.1}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage7_0.2m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage7.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 7 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage7.log)")
echo "Stage 7 done. Checkpoint: $CKPT"

# ── Stage 8: 0.1m  (half_width=0.05) ──────────────────────────────────────
echo "=== Stage 8: 0.1m ==="
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 150000000 \
  --load_checkpoint_path "$CKPT" \
  --playground_config_overrides '{"bridge_half_width": 0.05}' \
  --wandb_project bridge_crossing-depth-no-air \
  --wandb_run_name stage8_0.1m \
  --use_wandb \
  2>&1 | tee /tmp/bridge_no_air_stage8.log
exit_code=${PIPESTATUS[0]}
[ $exit_code -ne 0 ] && { echo "Stage 8 failed (exit $exit_code)"; exit $exit_code; }
CKPT=$(get_latest_ckpt "$(get_logdir /tmp/bridge_no_air_stage8.log)")
echo "Stage 8 done. Checkpoint: $CKPT"

echo "=== All stages complete ==="
