#!/bin/bash
# Round-beam transfer curriculum.
#
# Starts from the strongest flat/square-beam policy checkpoint (0.0025m) and
# fine-tunes it on true cylindrical beam physics. Widths below are full cylinder
# diameters; the environment config uses bridge_half_width = radius.
#
# Schedule:
#   Stage 00: 0.40m diameter  radius=0.200
#   Stage 01: 0.30m diameter  radius=0.150
#   Stage 02: 0.20m diameter  radius=0.100
#   Stage 03: 0.10m diameter  radius=0.050
#   Stage 04: 0.05m diameter  radius=0.025
#   Stage 05: 0.01m diameter  radius=0.005
#
# Usage:
#   ./run_round_beam_curriculum.sh
#   INIT_CKPT=/path/to/checkpoint ./run_round_beam_curriculum.sh

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

DEFAULT_INIT_CKPT="$SCRIPT_DIR/logs/Go1BridgeCrossing-20260709-012329/checkpoints/000200540160"
INIT_CKPT="${INIT_CKPT:-$DEFAULT_INIT_CKPT}"
if [ ! -d "$INIT_CKPT" ]; then
  echo "INIT_CKPT does not exist: $INIT_CKPT" >&2
  exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-go1_cylindrical_beam_curriculum_from_2p5mm_bridge}"
WANDB_GROUP="${WANDB_GROUP:-round_beam_transfer_fixed}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-200000000}"
EXTENSION_STEPS="${EXTENSION_STEPS:-100000000}"
MAX_EXTENSIONS="${MAX_EXTENSIONS:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.70}"
MARGINAL_THRESHOLD="${MARGINAL_THRESHOLD:-0.30}"
EPISODE_LENGTH="${EPISODE_LENGTH:-2000}"
CKPT="$INIT_CKPT"

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

run_stage() {
  local stage=$1
  local radius=$2
  local width_str=$3
  local steps=$4
  local ep_len=$5
  local load_ckpt=$6
  local logfile="/tmp/round_beam_stage${stage}.log"

  echo "=== Round beam stage ${stage}: ${width_str} diameter (radius=${radius}, ep_len=${ep_len}) ===" >&2

  set +e
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${steps}" \
    --load_checkpoint_path "$load_ckpt" \
    --playground_config_overrides "{\"bridge_shape\": \"round\", \"bridge_half_width\": ${radius}, \"virtual_hw_min\": ${radius}}" \
    --entropy_cost 0.01 \
    --episode_length "${ep_len}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "round_beam_stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    >"${logfile}" 2>&1
  local train_status=$?
  set -e
  if [ "$train_status" -ne 0 ]; then
    echo "Round beam stage ${stage} trainer exited with status ${train_status}." >&2
    echo "Last 80 lines from ${logfile}:" >&2
    tail -80 "${logfile}" >&2 || true
    return "$train_status"
  fi

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  local rate
  rate=$(get_success_rate "${logfile}")

  echo "Round beam stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  echo "${rate} ${ckpt}"
}

train_with_gate() {
  local stage=$1
  local radius=$2
  local width_str=$3

  local result rate ckpt
  result=$(run_stage "$stage" "$radius" "$width_str" "$STEPS_PER_STAGE" "$EPISODE_LENGTH" "$CKPT")
  rate=$(echo "$result" | awk '{print $1}')
  ckpt=$(echo "$result" | awk '{print $2}')

  local pass marginal
  pass=$(awk "BEGIN{print (${rate} >= ${PASS_THRESHOLD}) ? 1 : 0}")
  marginal=$(awk "BEGIN{print (${rate} >= ${MARGINAL_THRESHOLD}) ? 1 : 0}")

  if [ "$pass" -eq 1 ]; then
    echo "Round beam stage ${stage} passed (${rate} >= ${PASS_THRESHOLD})." >&2
    CKPT="$ckpt"
    return 0
  fi

  if [ "$marginal" -eq 1 ]; then
    echo "Round beam stage ${stage} marginal (${rate}). Extending up to ${MAX_EXTENSIONS}x." >&2
    local ext=1
    while [ "$ext" -le "$MAX_EXTENSIONS" ]; do
      result=$(run_stage "${stage}ext${ext}" "$radius" "${width_str}_ext${ext}" "$EXTENSION_STEPS" "$EPISODE_LENGTH" "$ckpt")
      rate=$(echo "$result" | awk '{print $1}')
      ckpt=$(echo "$result" | awk '{print $2}')
      pass=$(awk "BEGIN{print (${rate} >= ${PASS_THRESHOLD}) ? 1 : 0}")
      if [ "$pass" -eq 1 ]; then
        echo "Round beam stage ${stage} passed after extension (${rate})." >&2
        CKPT="$ckpt"
        return 0
      fi
      ext=$((ext + 1))
    done
    echo "Round beam stage ${stage} still marginal (${rate}); advancing with latest checkpoint." >&2
    CKPT="$ckpt"
    return 0
  fi

  echo "Round beam stage ${stage} failed (${rate} < ${MARGINAL_THRESHOLD}). Stopping curriculum." >&2
  echo "Last checkpoint: ${ckpt}" >&2
  exit 1
}

echo "=== Round beam transfer curriculum ===" >&2
echo "Project: ${WANDB_PROJECT}" >&2
echo "Initial checkpoint: ${INIT_CKPT}" >&2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

if [ "$SKIP_TO_STAGE" -le 0 ]; then
  train_with_gate "00" 0.2000 "0.4m"
fi
if [ "$SKIP_TO_STAGE" -le 1 ]; then
  train_with_gate "01" 0.1500 "0.3m"
fi
if [ "$SKIP_TO_STAGE" -le 2 ]; then
  train_with_gate "02" 0.1000 "0.2m"
fi
if [ "$SKIP_TO_STAGE" -le 3 ]; then
  train_with_gate "03" 0.0500 "0.1m"
fi
if [ "$SKIP_TO_STAGE" -le 4 ]; then
  train_with_gate "04" 0.0250 "0.05m"
fi
if [ "$SKIP_TO_STAGE" -le 5 ]; then
  train_with_gate "05" 0.0050 "0.01m"
fi

echo "=== Round beam curriculum complete. Final checkpoint: ${CKPT} ===" >&2
