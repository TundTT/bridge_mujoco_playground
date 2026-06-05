#!/bin/bash
# Full depth-obs curriculum: 0.8m → 0.1m in 0.1m steps
# WandB project: bridge_crossing_with_depth
# Each stage resumes from the final checkpoint of the previous stage.

set -e

BASE=/home/tund/notebooks/bridge_mujoco_playground
LOGS=$BASE/logs
PROJECT=bridge_crossing_with_depth

source "$BASE/.venv/bin/activate"

# run_stage <half_width> <num_timesteps> <run_name> [checkpoint_path]
# Prints the path of the final checkpoint to stdout; all other output goes to stderr.
run_stage() {
    local half_width=$1
    local steps=$2
    local run_name=$3
    local ckpt=${4:-""}

    echo "$(date '+%Y-%m-%d %H:%M:%S') [$run_name] Starting — bridge=$(echo "$half_width * 2" | bc)m, steps=$steps" >&2

    # Marker to identify the log dir created by this specific run
    local marker
    marker=$(mktemp)

    local cmd=(
        train-jax-ppo
        --env_name Go1BridgeCrossing
        --num_timesteps "$steps"
        --playground_config_overrides "{\"bridge_half_width\": ${half_width}}"
        --wandb_project "$PROJECT"
        --wandb_run_name "$run_name"
        --use_wandb
    )
    [ -n "$ckpt" ] && cmd+=(--load_checkpoint_path "$ckpt")

    # Run (blocking); tee log to file and stderr so it's visible
    local log_file="/tmp/${run_name}.log"
    "${cmd[@]}" 2>&1 | tee "$log_file" >&2

    # Find the log dir created after the marker
    local log_dir
    log_dir=$(find "$LOGS" -maxdepth 1 -name "Go1BridgeCrossing-*" -newer "$marker" -type d \
              | sort | tail -1)
    rm -f "$marker"

    if [ -z "$log_dir" ]; then
        echo "ERROR: could not find log dir for $run_name" >&2
        return 1
    fi

    # Latest checkpoint in that dir
    local latest_ckpt
    latest_ckpt=$(ls -d "$log_dir"/checkpoints/*/ 2>/dev/null | sort -V | tail -1)
    latest_ckpt="${latest_ckpt%/}"

    if [ -z "$latest_ckpt" ]; then
        echo "ERROR: no checkpoints found in $log_dir for $run_name" >&2
        return 1
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') [$run_name] Done — checkpoint: $latest_ckpt" >&2
    echo "$latest_ckpt"
}

echo "=== Bridge Crossing with Depth — Full Curriculum ===" >&2
echo "    Stages: 0.8m → 0.7m → 0.6m → 0.5m → 0.4m → 0.3m → 0.2m → 0.1m" >&2
echo "    WandB:  bridge_crossing_with_depth" >&2
echo "" >&2

# Stage 1 — foundation, no checkpoint (300M: learns gait + bridge walking from scratch)
CKPT=$(run_stage 0.40 300000000 depth_0.8m "")

# Stage 2 — 0.7m (200M: small width change, warm start)
CKPT=$(run_stage 0.35 200000000 depth_0.7m "$CKPT")

# Stage 3 — 0.6m
CKPT=$(run_stage 0.30 200000000 depth_0.6m "$CKPT")

# Stage 4 — 0.5m
CKPT=$(run_stage 0.25 200000000 depth_0.5m "$CKPT")

# Stage 5 — 0.4m
CKPT=$(run_stage 0.20 200000000 depth_0.4m "$CKPT")

# Stage 6 — 0.3m (previous ceiling without depth obs)
CKPT=$(run_stage 0.15 200000000 depth_0.3m "$CKPT")

# Stage 7 — 0.2m (robot stance ≈ 0.28m; previous best was 16% without depth)
CKPT=$(run_stage 0.10 200000000 depth_0.2m "$CKPT")

# Stage 8 — 0.1m (hardest; 300M since this is the target width)
CKPT=$(run_stage 0.05 300000000 depth_0.1m "$CKPT")

echo "$(date '+%Y-%m-%d %H:%M:%S') === Curriculum complete! Final checkpoint: $CKPT ===" >&2
