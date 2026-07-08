#!/bin/bash
# Final narrow-bridge curriculum for Go1BridgeCrossing.
# Recreates the proprioceptive virtual-width reward curriculum used for
# bridge_crossing_final. Requires W&B login before running with --use_wandb.

set -euo pipefail

BASE=${BASE:-/home/theerawit/bridge_mujoco_playground}
LOGS="$BASE/logs"
PROJECT=${PROJECT:-bridge_crossing_final}
PYTHON_RUNNER=${PYTHON_RUNNER:-/home/theerawit/.local/bin/uv --no-config run}

cd "$BASE"

run_stage() {
  local stage=$1
  local half_width=$2
  local virtual_min=$3
  local steps=$4
  local episode_length=$5
  local ckpt=${6:-}
  local run_name
  run_name=$(printf "stage%02d_%.3fm" "$stage" "$(awk "BEGIN { print $half_width * 2 }")")

  echo "$(date '+%Y-%m-%d %H:%M:%S') [$run_name] start hw=$half_width vhw_min=$virtual_min steps=$steps" >&2

  local marker
  marker=$(mktemp)

  local cmd=(
    $PYTHON_RUNNER train-jax-ppo
    --env_name Go1BridgeCrossing
    --num_timesteps "$steps"
    --episode_length "$episode_length"
    --playground_config_overrides "{\"bridge_half_width\": $half_width, \"virtual_half_width_min\": $virtual_min}"
    --wandb_project "$PROJECT"
    --wandb_run_name "$run_name"
    --use_wandb
  )
  if [ -n "$ckpt" ]; then
    cmd+=(--load_checkpoint_path "$ckpt")
  fi

  local log_file="/tmp/${run_name}.log"
  PATH=/home/theerawit/.local/bin:$PATH "${cmd[@]}" 2>&1 | tee "$log_file" >&2

  local log_dir
  log_dir=$(find "$LOGS" -maxdepth 1 -name "Go1BridgeCrossing-*" -newer "$marker" -type d | sort | tail -1)
  rm -f "$marker"

  if [ -z "$log_dir" ]; then
    echo "ERROR: no log dir found for $run_name" >&2
    return 1
  fi

  local latest_ckpt
  latest_ckpt=$(ls -d "$log_dir"/checkpoints/*/ 2>/dev/null | sort -V | tail -1)
  latest_ckpt="${latest_ckpt%/}"
  if [ -z "$latest_ckpt" ]; then
    echo "ERROR: no checkpoint found for $run_name in $log_dir" >&2
    return 1
  fi

  echo "$(date '+%Y-%m-%d %H:%M:%S') [$run_name] done checkpoint=$latest_ckpt" >&2
  echo "$latest_ckpt"
}

CKPT=""
CKPT=$(run_stage 0 0.4000 0.1500 150000000 1000 "$CKPT")
CKPT=$(run_stage 1 0.2000 0.1000 100000000 1000 "$CKPT")
CKPT=$(run_stage 2 0.1600 0.0800 100000000 1000 "$CKPT")
CKPT=$(run_stage 3 0.1200 0.0600 150000000 1000 "$CKPT")
CKPT=$(run_stage 4 0.1000 0.0500 150000000 1500 "$CKPT")
CKPT=$(run_stage 5 0.0800 0.0500 200000000 1500 "$CKPT")
CKPT=$(run_stage 6 0.0650 0.0450 150000000 1500 "$CKPT")
CKPT=$(run_stage 7 0.0500 0.0400 200000000 1500 "$CKPT")
CKPT=$(run_stage 8 0.0450 0.0350 200000000 1500 "$CKPT")
CKPT=$(run_stage 9 0.0400 0.0300 200000000 1500 "$CKPT")
CKPT=$(run_stage 10 0.0350 0.0280 200000000 1500 "$CKPT")
CKPT=$(run_stage 11 0.0300 0.0280 200000000 2000 "$CKPT")
CKPT=$(run_stage 12 0.0275 0.0275 200000000 2000 "$CKPT")

echo "$(date '+%Y-%m-%d %H:%M:%S') final checkpoint=$CKPT"
