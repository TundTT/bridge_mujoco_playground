#!/bin/bash
# PUMA foothold prior curriculum: 0.6m → 0.55m → 0.5m → ... → 0.1m
# Branch: puma-foothold-prior
# WandB project: bridge_crossing_puma
#
# Each stage trains for 150M steps, resuming from the previous checkpoint.
# Videos are logged to WandB automatically at each eval step.

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_puma"

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

run_stage() {
  local stage=$1
  local half_width=$2
  local width_str=$3
  local ckpt_arg=${4:-}
  local logfile="/tmp/bridge_puma_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (half_width=${half_width}) ==="

  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps 150000000 \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${half_width}}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "stage${stage}_${width_str}" \
    --use_wandb \
    2>&1 | tee "${logfile}"

  local exit_code=${PIPESTATUS[0]}
  [ $exit_code -ne 0 ] && { echo "Stage ${stage} failed (exit $exit_code)"; exit $exit_code; }

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  echo "Stage ${stage} done. Checkpoint: ${ckpt}"
  echo "${ckpt}"
}

# ── Stage 1: 0.6m  (half_width=0.30) ─────────────────────────────────────────
CKPT=$(run_stage 1 0.30 "0.6m")

# ── Stage 2: 0.55m  (half_width=0.275) ───────────────────────────────────────
CKPT=$(run_stage 2 0.275 "0.55m" "$CKPT")

# ── Stage 3: 0.5m  (half_width=0.25) ─────────────────────────────────────────
CKPT=$(run_stage 3 0.25 "0.5m" "$CKPT")

# ── Stage 4: 0.45m  (half_width=0.225) ───────────────────────────────────────
CKPT=$(run_stage 4 0.225 "0.45m" "$CKPT")

# ── Stage 5: 0.4m  (half_width=0.20) ─────────────────────────────────────────
CKPT=$(run_stage 5 0.20 "0.4m" "$CKPT")

# ── Stage 6: 0.35m  (half_width=0.175) ───────────────────────────────────────
CKPT=$(run_stage 6 0.175 "0.35m" "$CKPT")

# ── Stage 7: 0.3m  (half_width=0.15) ─────────────────────────────────────────
CKPT=$(run_stage 7 0.15 "0.3m" "$CKPT")

# ── Stage 8: 0.25m  (half_width=0.125) ───────────────────────────────────────
CKPT=$(run_stage 8 0.125 "0.25m" "$CKPT")

# ── Stage 9: 0.2m  (half_width=0.10) ─────────────────────────────────────────
CKPT=$(run_stage 9 0.10 "0.2m" "$CKPT")

# ── Stage 10: 0.15m  (half_width=0.075) ──────────────────────────────────────
CKPT=$(run_stage 10 0.075 "0.15m" "$CKPT")

# ── Stage 11: 0.1m  (half_width=0.05) ────────────────────────────────────────
CKPT=$(run_stage 11 0.05 "0.1m" "$CKPT")

echo "=== All 11 stages complete. Final checkpoint: ${CKPT} ==="
