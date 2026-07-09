#!/bin/bash
# Clean cylindrical-beam transfer curriculum from the 0.0025m flat bridge policy.
#
# This script is intentionally minimal: it assumes the f9ffdad bridge curriculum
# codebase plus bridge_shape="round" physics support in bridge.py. It does not
# depend on showcase renderers or W&B checkpoint-backup helpers.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Local logs identify this checkpoint as final_stage33_0.0025m. The user-facing
# name "final_stage32_0.0025m" appears to be off by one locally. Override
# INIT_CKPT if a different checkpoint/artifact should seed the curriculum.
DEFAULT_INIT_CKPT="/home/theerawit/bridge_mujoco_playground_git/logs/Go1BridgeCrossing-20260709-012329/checkpoints/000200540160"
INIT_CKPT="${INIT_CKPT:-$DEFAULT_INIT_CKPT}"
if [ ! -d "$INIT_CKPT" ]; then
  echo "INIT_CKPT does not exist: $INIT_CKPT" >&2
  exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-go1_cylinder_beam_clean_from_2p5mm_flat}"
WANDB_GROUP="${WANDB_GROUP:-cylinder_beam_clean}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-200000000}"
EXTENSION_STEPS="${EXTENSION_STEPS:-100000000}"
MAX_EXTENSIONS="${MAX_EXTENSIONS:-2}"
PASS_THRESHOLD="${PASS_THRESHOLD:-0.70}"
MARGINAL_THRESHOLD="${MARGINAL_THRESHOLD:-0.30}"
EPISODE_LENGTH="${EPISODE_LENGTH:-2000}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$SCRIPT_DIR/logs/cylinder_beam_clean_$(date +%Y%m%d-%H%M%S)}"
CKPT="$INIT_CKPT"

mkdir -p "$RUN_LOG_DIR"

get_latest_ckpt() {
  find "$1/checkpoints" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' 2>/dev/null \
    | sort -V \
    | tail -1 \
    || true
}

get_logdir() {
  grep -a "Logs are being stored in:" "$1" 2>/dev/null \
    | awk '{print $NF}' \
    | tail -1 \
    || true
}

get_success_rate() {
  local logfile=$1
  local rate
  rate=$(
    grep -a "metric/term_success:" "$logfile" 2>/dev/null \
      | tail -1 \
      | awk '{print $2}' \
      || true
  )
  echo "${rate:-0}"
}

rate_ge() {
  awk "BEGIN{print ($1 >= $2) ? 1 : 0}"
}

run_stage() {
  local stage=$1
  local diameter=$2
  local radius=$3
  local steps=$4
  local ep_len=$5
  local load_ckpt=$6
  local label="${diameter}m"
  local logfile="${RUN_LOG_DIR}/stage${stage}_${label}.log"

  echo "=== Cylinder stage ${stage}: diameter=${diameter}m radius=${radius} ep_len=${ep_len} ===" >&2

  set +e
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "$steps" \
    --load_checkpoint_path "$load_ckpt" \
    --playground_config_overrides "{\"bridge_shape\":\"round\", \"bridge_half_width\": ${radius}, \"virtual_hw_min\": ${radius}}" \
    --entropy_cost 0.01 \
    --episode_length "$ep_len" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "cylinder_stage${stage}_${label}" \
    --wandb_group "$WANDB_GROUP" \
    --use_wandb \
    >"$logfile" 2>&1
  local train_status=$?
  set -e

  if [ "$train_status" -ne 0 ]; then
    echo "Cylinder stage ${stage} trainer exited with status ${train_status}." >&2
    echo "Last 80 lines from ${logfile}:" >&2
    tail -80 "$logfile" >&2 || true
    return "$train_status"
  fi

  local logdir
  logdir=$(get_logdir "$logfile")
  if [ -z "$logdir" ] || [ ! -d "$logdir" ]; then
    echo "Could not find trainer log directory in ${logfile}." >&2
    return 1
  fi

  local ckpt
  ckpt=$(get_latest_ckpt "$logdir")
  if [ -z "$ckpt" ] || [ ! -d "$ckpt" ]; then
    echo "Could not find checkpoint under ${logdir}/checkpoints." >&2
    return 1
  fi

  local rate
  rate=$(get_success_rate "$logfile")
  echo "Cylinder stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  echo "${rate} ${ckpt}"
}

train_with_gate() {
  local stage=$1
  local diameter=$2
  local radius=$3

  local result rate ckpt
  result=$(run_stage "$stage" "$diameter" "$radius" "$STEPS_PER_STAGE" "$EPISODE_LENGTH" "$CKPT")
  rate=$(echo "$result" | awk '{print $1}')
  ckpt=$(echo "$result" | awk '{print $2}')

  local pass marginal
  pass=$(rate_ge "$rate" "$PASS_THRESHOLD")
  marginal=$(rate_ge "$rate" "$MARGINAL_THRESHOLD")

  if [ "$pass" -eq 1 ]; then
    echo "Cylinder stage ${stage} passed (${rate} >= ${PASS_THRESHOLD})." >&2
    CKPT="$ckpt"
    return 0
  fi

  if [ "$marginal" -eq 1 ]; then
    echo "Cylinder stage ${stage} marginal (${rate}). Extending up to ${MAX_EXTENSIONS}x." >&2
    local ext=1
    while [ "$ext" -le "$MAX_EXTENSIONS" ]; do
      result=$(run_stage "${stage}ext${ext}" "$diameter" "$radius" "$EXTENSION_STEPS" "$EPISODE_LENGTH" "$ckpt")
      rate=$(echo "$result" | awk '{print $1}')
      ckpt=$(echo "$result" | awk '{print $2}')
      pass=$(rate_ge "$rate" "$PASS_THRESHOLD")
      if [ "$pass" -eq 1 ]; then
        echo "Cylinder stage ${stage} passed after extension (${rate})." >&2
        CKPT="$ckpt"
        return 0
      fi
      ext=$((ext + 1))
    done
    marginal=$(rate_ge "$rate" "$MARGINAL_THRESHOLD")
    if [ "$marginal" -eq 1 ]; then
      echo "Cylinder stage ${stage} still marginal (${rate}); advancing with latest checkpoint." >&2
      CKPT="$ckpt"
      return 0
    fi
  fi

  echo "Cylinder stage ${stage} failed (${rate} < ${MARGINAL_THRESHOLD}). Stopping curriculum." >&2
  echo "Last checkpoint: ${ckpt}" >&2
  exit 1
}

echo "=== Clean cylinder beam transfer curriculum ===" >&2
echo "Project: ${WANDB_PROJECT}" >&2
echo "Initial checkpoint: ${INIT_CKPT}" >&2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
echo "Stage logs: ${RUN_LOG_DIR}" >&2

STAGE_DIAMETERS=(
  0.4
  0.3
  0.2
  0.1
  0.05
  0.025
  0.01
  0.005
  0.0025
)

for idx in "${!STAGE_DIAMETERS[@]}"; do
  if [ "$idx" -lt "$SKIP_TO_STAGE" ]; then
    continue
  fi
  stage=$(printf "%02d" "$idx")
  diameter="${STAGE_DIAMETERS[$idx]}"
  radius=$(awk "BEGIN{printf \"%.6f\", ${diameter} / 2.0}")
  train_with_gate "$stage" "$diameter" "$radius"
done

echo "=== Cylinder beam curriculum complete. Final checkpoint: ${CKPT} ===" >&2
