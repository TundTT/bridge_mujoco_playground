#!/bin/bash
# Continue bridge_crossing_final_v2 below 0.055m in 0.0025m width increments.
#
# Defaults:
#   start width: 0.0525m
#   decrement:   0.0025m
#   stop:        first stage with term_success < 0.30
#   device:      GPU 0 only, because this workstation has broken GPU P2P.
#
# Usage:
#   INIT_CKPT=/path/to/checkpoint ./run_extra_narrow_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR:-$SCRIPT_DIR}"

if [ ! -d ".venv" ]; then
  UV_BIN="${UV_BIN:-$(command -v uv || true)}"
  if [ -z "$UV_BIN" ] && [ -x /home/theerawit/.local/bin/uv ]; then
    UV_BIN="/home/theerawit/.local/bin/uv"
  fi
  if [ -z "$UV_BIN" ]; then
    echo "uv not found. Install uv or set UV_BIN=/path/to/uv." >&2
    exit 1
  fi
  "$UV_BIN" --no-config sync --all-extras
fi

source .venv/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-bridge_crossing_final_v2}"
WANDB_GROUP="${WANDB_GROUP:-final_curriculum_extra_narrow}"
INIT_CKPT="${INIT_CKPT:-}"
START_STAGE="${START_STAGE:-13}"
START_WIDTH="${START_WIDTH:-0.0525}"
WIDTH_STEP="${WIDTH_STEP:-0.0025}"
MIN_WIDTH="${MIN_WIDTH:-0.0300}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-0.30}"
STEPS="${STEPS:-200000000}"
EP_LEN="${EP_LEN:-2000}"

if [ -z "$INIT_CKPT" ]; then
  echo "INIT_CKPT is required. Pass the latest 0.055m checkpoint." >&2
  exit 1
fi

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1
}

get_logdir() {
  grep -a "Logs are being stored in:" "$1" | awk '{print $NF}'
}

get_success_rate() {
  local logfile="$1"
  local rate
  rate=$(grep -a "metric/term_success:" "$logfile" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "${rate:-0}"
}

fmt_width() {
  awk -v w="$1" 'BEGIN {
    s = sprintf("%.4f", w)
    sub(/0+$/, "", s)
    sub(/\.$/, "", s)
    print s "m"
  }'
}

run_stage() {
  local stage=$1
  local width=$2
  local ckpt_arg=$3
  local hw
  hw=$(awk -v w="$width" 'BEGIN{printf "%.5f", w / 2.0}')
  local width_str
  width_str=$(fmt_width "$width")
  local logfile="/tmp/bridge_extra_narrow_stage${stage}.log"

  echo "=== Extra stage ${stage}: ${width_str} (hw=${hw}, vhw_min=${hw}, ep_len=${EP_LEN}) ===" >&2

  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${STEPS}" \
    --load_checkpoint_path "${ckpt_arg}" \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": ${hw}}" \
    --entropy_cost 0.01 \
    --episode_length "${EP_LEN}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "final_stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    2>&1 | tee "${logfile}" >/dev/null

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  local rate
  rate=$(get_success_rate "${logfile}")

  echo "Extra stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  echo "${rate} ${ckpt}"
}

CKPT="$INIT_CKPT"
stage="$START_STAGE"
width="$START_WIDTH"

while awk -v w="$width" -v min="$MIN_WIDTH" 'BEGIN{exit !(w >= min)}'; do
  result=$(run_stage "$stage" "$width" "$CKPT")
  rate=$(echo "$result" | awk '{print $1}')
  ckpt=$(echo "$result" | awk '{print $2}')

  failed=$(awk -v r="$rate" -v t="$FAIL_THRESHOLD" 'BEGIN{print (r < t) ? 1 : 0}')
  if [ "$failed" -eq 1 ]; then
    echo "Extra stage ${stage} failed (${rate} < ${FAIL_THRESHOLD}). Stopping probe." >&2
    exit 0
  fi

  CKPT="$ckpt"
  stage=$((stage + 1))
  width=$(awk -v w="$width" -v step="$WIDTH_STEP" 'BEGIN{printf "%.4f", w - step}')
done

echo "Reached MIN_WIDTH=${MIN_WIDTH} without crossing FAIL_THRESHOLD=${FAIL_THRESHOLD}." >&2
