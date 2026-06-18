#!/bin/bash
# Narrowing curriculum: 0.5m (pre-trained) → 0.45m → 0.4m → 0.35m → 0.3m
# Each stage fine-tunes from the previous checkpoint.
# Episode length: 3000 steps (60s). forward_vel scale: 0.5.
#
# Usage:
#   ./run_curriculum_narrow.sh                              # fresh from 0.5m ckpt
#   SKIP_TO_STAGE=2 INIT_CKPT=<path> ./run_curriculum_narrow.sh  # resume

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_puma"
WANDB_GROUP="${WANDB_GROUP:-v7_fwd0.5_1min}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-1}"

# Seed: final checkpoint from the 0.5m direct run (100% success)
INIT_CKPT="${INIT_CKPT:-logs/Go1BridgeCrossing-20260611-054619/checkpoints/000306708480}"

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
  local ckpt_arg="${4:-}"
  local logfile="/tmp/bridge_narrow_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (half_width=${half_width}) ===" >&2

  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps 150000000 \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${half_width}, \"episode_length\": 3000}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "narrow_stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    2>&1 | tee "${logfile}" >/dev/null

  local exit_code=${PIPESTATUS[0]}
  if [ $exit_code -ne 0 ]; then
    echo "Stage ${stage} failed (exit $exit_code)" >&2
    exit $exit_code
  fi

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  echo "Stage ${stage} done. Checkpoint: ${ckpt}" >&2
  echo "${ckpt}"
}

CKPT="${INIT_CKPT}"

# ── Stage 1: 0.45m (half_width=0.225) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  CKPT=$(run_stage 1 0.225 "0.45m" "${CKPT}")
fi

# ── Stage 2: 0.4m (half_width=0.20) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  CKPT=$(run_stage 2 0.20 "0.4m" "${CKPT}")
fi

# ── Stage 3: 0.35m (half_width=0.175) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  CKPT=$(run_stage 3 0.175 "0.35m" "${CKPT}")
fi

# ── Stage 4: 0.3m (half_width=0.15) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  CKPT=$(run_stage 4 0.15 "0.3m" "${CKPT}")
fi

echo "=== All stages complete. Final checkpoint: ${CKPT} ===" >&2
