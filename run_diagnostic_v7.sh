#!/bin/bash
# v7 diagnostic: zero-shot width sweep from stage-4 (0.16m) checkpoint.
#
# Runs a minimal training pass (2M steps, 1 eval) at each bridge width to
# map the performance cliff between 0.16m and 0.10m. The policy is barely
# updated (2M << 200M), so results are effectively zero-shot.
#
# Widths tested: 0.155m, 0.15m, 0.145m, 0.14m, 0.13m, 0.10m
#
# Also prints max_foot_y vs virtual_hw for each width to test whether the
# stage-4 policy actually narrows its stance with vhw (Fable's prediction:
# it doesn't, because the foothold multiplier floors at 0.5).
#
# Usage:
#   ./run_diagnostic_v7.sh
#   Logs: /tmp/bridge_v7_diag_<width>.log

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

STAGE4_CKPT="logs/Go1BridgeCrossing-20260612-073809/checkpoints/000200540160"
WANDB_PROJECT="bridge_crossing_v7"
STEPS=2000000
NUM_EVALS=1

run_diag() {
  local hw=$1        # physical half-width
  local width_str=$2
  local logfile="/tmp/bridge_v7_diag_${width_str}.log"

  echo "=== Diagnostic: ${width_str} (hw=${hw}) ===" >&2

  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${STEPS}" \
    --num_evals "${NUM_EVALS}" \
    --load_checkpoint_path "${STAGE4_CKPT}" \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": 0.05}" \
    --entropy_cost 0.01 \
    --episode_length 1500 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "v7_diag_${width_str}" \
    --wandb_group "v7_diagnostic" \
    --use_wandb \
    2>&1 | tee "${logfile}" >/dev/null

  local rate
  rate=$(grep "metric/term_success:" "${logfile}" 2>/dev/null | tail -1 | awk '{print $2}')
  local tds
  tds=$(grep "metric/touchdown_count:" "${logfile}" 2>/dev/null | tail -1 | awk '{print $2}')
  local abd
  abd=$(grep "metric/abduction_saturation:" "${logfile}" 2>/dev/null | tail -1 | awk '{print $2}')
  local max_foot_y
  max_foot_y=$(grep "metric/max_foot_y:" "${logfile}" 2>/dev/null | tail -1 | awk '{print $2}')

  echo "  term_success=${rate:-N/A}  TDs=${tds:-N/A}  abduction_sat=${abd:-N/A}  max_foot_y=${max_foot_y:-N/A}" >&2
  echo "${width_str} hw=${hw} success=${rate:-0} tds=${tds:-0} abduction_sat=${abd:-0} max_foot_y=${max_foot_y:-0}"
}

echo "=== v7 Diagnostic: Width sweep from stage-4 (0.16m) checkpoint ===" >&2
echo "=== Checkpoint: ${STAGE4_CKPT} ===" >&2
echo "" >&2

# Run sequentially (single GPU — can't run two JAX processes simultaneously)
run_diag 0.0775 "0.155m"
run_diag 0.075  "0.15m"
run_diag 0.0725 "0.145m"
run_diag 0.07   "0.14m"
run_diag 0.065  "0.13m"
run_diag 0.05   "0.10m"

echo "" >&2
echo "=== Diagnostic complete. Summary above. ===" >&2
