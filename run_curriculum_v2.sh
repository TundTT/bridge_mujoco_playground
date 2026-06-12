#!/bin/bash
# v2b curriculum: touchdown-gated foothold rewards, alive time tax
# Stages: 0.5m → 0.45m → 0.4m → 0.35m → 0.325m → 0.3m → 0.25m → 0.2m → 0.15m → 0.1m
#
# Key reward changes vs v2:
#   - alive=-2.0 (time tax: standing still is now net-negative)
#   - foothold_dense gated on first_contact (no per-timestep income from standing)
#   - foothold_sparse gated on first_contact (same reason)
#   - foothold_dense gradient widened from 5cm to 10cm (smoother learning signal)
#   - foothold_dense scale 1.0→4.0 (compensates for ~4× fewer touchdown events)
#   - added 0.325m stage between 0.35m and 0.3m (finer curriculum steps)
#
# Usage:
#   ./run_curriculum_v2.sh                              # fresh run from stage 1
#   SKIP_TO_STAGE=3 INIT_CKPT=<path> ./run_curriculum_v2.sh  # resume

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_v2"
WANDB_GROUP="${WANDB_GROUP:-v2b_touchdown_foothold}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-1}"
INIT_CKPT="${INIT_CKPT:-}"

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
  local logfile="/tmp/bridge_v2_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (half_width=${half_width}) ===" >&2

  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps 150000000 \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${half_width}}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "v2_stage${stage}_${width_str}" \
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

# ── Stage 1: 0.5m (half_width=0.25) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  CKPT=$(run_stage 1 0.25 "0.5m")
fi

# ── Stage 2: 0.45m (half_width=0.225) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  CKPT=$(run_stage 2 0.225 "0.45m" "${CKPT}")
fi

# ── Stage 3: 0.4m (half_width=0.20) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  CKPT=$(run_stage 3 0.20 "0.4m" "${CKPT}")
fi

# ── Stage 4: 0.35m (half_width=0.175) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  CKPT=$(run_stage 4 0.175 "0.35m" "${CKPT}")
fi

# ── Stage 5: 0.325m (half_width=0.1625) ──────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 5 ]; then
  CKPT=$(run_stage 5 0.1625 "0.325m" "${CKPT}")
fi

# ── Stage 6: 0.3m (half_width=0.15) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 6 ]; then
  CKPT=$(run_stage 6 0.15 "0.3m" "${CKPT}")
fi

# ── Stage 7: 0.25m (half_width=0.125) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 7 ]; then
  CKPT=$(run_stage 7 0.125 "0.25m" "${CKPT}")
fi

# ── Stage 8: 0.2m (half_width=0.10) ──────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 8 ]; then
  CKPT=$(run_stage 8 0.10 "0.2m" "${CKPT}")
fi

# ── Stage 9: 0.15m (half_width=0.075) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 9 ]; then
  CKPT=$(run_stage 9 0.075 "0.15m" "${CKPT}")
fi

# ── Stage 10: 0.1m (half_width=0.05) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 10 ]; then
  CKPT=$(run_stage 10 0.05 "0.1m" "${CKPT}")
fi

echo "=== All 10 stages complete. Final checkpoint: ${CKPT} ===" >&2
