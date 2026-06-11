#!/bin/bash
# PUMA foothold prior curriculum: 0.6m → 0.55m → 0.5m → ... → 0.1m
# Branch: puma-foothold-prior
# WandB project: bridge_crossing_puma
#
# Each stage trains for 150M steps, resuming from the previous checkpoint.
# Videos are logged to WandB automatically at each eval step.
#
# Usage:
#   ./run_curriculum_puma.sh                   # fresh run from stage 1
#   SKIP_TO_STAGE=2 INIT_CKPT=<path> ./run_curriculum_puma.sh  # resume

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_puma"
WANDB_GROUP="${WANDB_GROUP:-v5_foot_diagnostics}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-1}"
INIT_CKPT="${INIT_CKPT:-}"

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

# run_stage writes ALL diagnostic output to stderr; only the checkpoint path
# goes to stdout so CKPT=$(run_stage ...) captures exactly the path.
run_stage() {
  local stage=$1
  local half_width=$2
  local width_str=$3
  local ckpt_arg="${4:-}"
  local logfile="/tmp/bridge_puma_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (half_width=${half_width}) ===" >&2

  # tee to file; stdout→/dev/null so it doesn't pollute the $() capture above.
  # pipefail propagates train-jax-ppo's exit code through the pipe.
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps 150000000 \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${half_width}}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    2>&1 | tee "${logfile}" >/dev/null

  # PIPESTATUS[0] = train-jax-ppo exit code (pipefail ensures non-zero propagates)
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
  # Only this line goes to stdout → captured cleanly in CKPT=
  echo "${ckpt}"
}

# ── Determine starting checkpoint ─────────────────────────────────────────────
CKPT="${INIT_CKPT}"

# ── Stage 1: 0.6m  (half_width=0.30) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  CKPT=$(run_stage 1 0.30 "0.6m")
fi

# ── Stage 2: 0.55m  (half_width=0.275) ───────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  CKPT=$(run_stage 2 0.275 "0.55m" "${CKPT}")
fi

# ── Stage 3: 0.5m  (half_width=0.25) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  CKPT=$(run_stage 3 0.25 "0.5m" "${CKPT}")
fi

# ── Stage 4: 0.45m  (half_width=0.225) ───────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  CKPT=$(run_stage 4 0.225 "0.45m" "${CKPT}")
fi

# ── Stage 5: 0.4m  (half_width=0.20) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 5 ]; then
  CKPT=$(run_stage 5 0.20 "0.4m" "${CKPT}")
fi

# ── Stage 6: 0.35m  (half_width=0.175) ───────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 6 ]; then
  CKPT=$(run_stage 6 0.175 "0.35m" "${CKPT}")
fi

# ── Stage 7: 0.3m  (half_width=0.15) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 7 ]; then
  CKPT=$(run_stage 7 0.15 "0.3m" "${CKPT}")
fi

# ── Stage 8: 0.25m  (half_width=0.125) ───────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 8 ]; then
  CKPT=$(run_stage 8 0.125 "0.25m" "${CKPT}")
fi

# ── Stage 9: 0.2m  (half_width=0.10) ─────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 9 ]; then
  CKPT=$(run_stage 9 0.10 "0.2m" "${CKPT}")
fi

# ── Stage 10: 0.15m  (half_width=0.075) ──────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 10 ]; then
  CKPT=$(run_stage 10 0.075 "0.15m" "${CKPT}")
fi

# ── Stage 11: 0.1m  (half_width=0.05) ────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 11 ]; then
  CKPT=$(run_stage 11 0.05 "0.1m" "${CKPT}")
fi

echo "=== All stages complete. Final checkpoint: ${CKPT} ===" >&2
