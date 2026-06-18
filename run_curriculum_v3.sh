#!/bin/bash
# v3 curriculum: Option A — virtual width randomization per episode
#
# Physical stages (5 total):
#   Stage 0: 0.8m (hw=0.40), virtual U[0.15, 0.40], 150M, entropy=0.01
#   Stage 1: 0.4m (hw=0.20), virtual U[0.10, 0.20], 100M, entropy=0.005
#   Stage 2: 0.24m(hw=0.12), virtual U[0.06, 0.12], 150M
#   Stage 3: 0.16m(hw=0.08), virtual U[0.05, 0.08], 200M, ep_len=1500
#   Stage 4: 0.10m(hw=0.05), virtual fixed 0.05,    200M+, ep_len=1500
#
# Advancement gate: term_success ≥ 0.70 to advance; extend +100M if 0.30-0.70;
# fall back to LAST_GOOD checkpoint and insert midpoint if < 0.30.
#
# Usage:
#   ./run_curriculum_v3.sh                              # fresh run
#   SKIP_TO_STAGE=2 INIT_CKPT=<path> ./run_curriculum_v3.sh

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_v3"
WANDB_GROUP="${WANDB_GROUP:-v3_virtual_width}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"
INIT_CKPT="${INIT_CKPT:-}"
LAST_GOOD_CKPT="${INIT_CKPT:-}"
LAST_GOOD_HW="0.40"  # tracks physical hw of the last gate-passing stage

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

# Returns final term_success (0.0-1.0) from the stage log; defaults to 0.
get_success_rate() {
  local logfile="$1"
  local rate
  rate=$(grep "metric/term_success:" "$logfile" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "${rate:-0}"
}

run_stage() {
  local stage=$1
  local hw=$2          # physical half-width
  local vhw_min=$3     # virtual_hw_min
  local width_str=$4
  local steps=$5
  local ckpt_arg="${6:-}"
  local extra_flags="${7:-}"
  local logfile="/tmp/bridge_v3_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (hw=${hw}, vhw_min=${vhw_min}) ===" >&2

  # shellcheck disable=SC2086
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${steps}" \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": ${vhw_min}}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "v3_stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    ${extra_flags} \
    2>&1 | tee "${logfile}" >/dev/null
  # set -e exits the script on non-zero pipeline exit automatically;
  # no need for PIPESTATUS check here.

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  local rate
  rate=$(get_success_rate "${logfile}")

  echo "Stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  # Emit: "<rate> <ckpt>" so caller can parse both
  echo "${rate} ${ckpt}"
}

# Advance or fall back based on success rate gate.
# Sets global CKPT and LAST_GOOD_CKPT.
gate_advance() {
  local stage=$1
  local hw=$2
  local vhw_min=$3
  local width_str=$4
  local steps=$5
  local extra_flags="${6:-}"
  local max_extensions=2

  local result
  result=$(run_stage "${stage}" "${hw}" "${vhw_min}" "${width_str}" "${steps}" "${CKPT}" "${extra_flags}")
  local rate ckpt
  rate=$(echo "$result" | awk '{print $1}')
  ckpt=$(echo "$result" | awk '{print $2}')

  # Use awk for float comparison (bash can't do it natively)
  local pass
  pass=$(awk "BEGIN{print (${rate} >= 0.70) ? 1 : 0}")
  local marginal
  marginal=$(awk "BEGIN{print (${rate} >= 0.30) ? 1 : 0}")

  if [ "$pass" -eq 1 ]; then
    echo "Stage ${stage} passed (${rate} ≥ 0.70). Advancing." >&2
    CKPT="$ckpt"
    LAST_GOOD_CKPT="$ckpt"
    LAST_GOOD_HW="$hw"
  elif [ "$marginal" -eq 1 ]; then
    echo "Stage ${stage} marginal (${rate} in [0.30, 0.70)). Extending +100M from same checkpoint." >&2
    local ext_steps=100000000
    local ext_num=1
    for _ in $(seq 1 $max_extensions); do
      result=$(run_stage "${stage}ext${ext_num}" "${hw}" "${vhw_min}" "${width_str}_ext${ext_num}" "${ext_steps}" "$ckpt" "${extra_flags}")
      rate=$(echo "$result" | awk '{print $1}')
      ckpt=$(echo "$result" | awk '{print $2}')
      pass=$(awk "BEGIN{print (${rate} >= 0.70) ? 1 : 0}")
      if [ "$pass" -eq 1 ]; then
        echo "Stage ${stage} passed after extension (${rate})." >&2
        CKPT="$ckpt"
        LAST_GOOD_CKPT="$ckpt"
        LAST_GOOD_HW="$hw"
        return 0
      fi
      ext_num=$((ext_num + 1))
    done
    echo "Stage ${stage} still marginal after extensions. Advancing anyway to avoid loop." >&2
    CKPT="$ckpt"
  else
    echo "Stage ${stage} failed (${rate} < 0.30). Reverting to LAST_GOOD and inserting midpoint." >&2
    CKPT="$LAST_GOOD_CKPT"
    local mid_hw
    mid_hw=$(awk "BEGIN{printf \"%.4f\", (${hw} + ${LAST_GOOD_HW}) / 2}")
    echo "  Midpoint insertion: hw=${mid_hw} (between last good ${LAST_GOOD_HW} and failed ${hw})" >&2
    echo "  Re-invoke: SKIP_TO_STAGE=${stage} INIT_CKPT=${LAST_GOOD_CKPT} bash run_curriculum_v3.sh" >&2
    echo "  with extra stage: gate_advance ${stage} ${mid_hw} ... inserted before stage ${stage}" >&2
    exit 1
  fi
}

CKPT="${INIT_CKPT}"

# ── Stage 0: 0.8m, virtual U[0.15, 0.40], entropy=0.01 ───────────────────────
if [ "${SKIP_TO_STAGE}" -le 0 ]; then
  gate_advance 0 0.40 0.15 "0.8m" 150000000 "--entropy_cost 0.01"
fi

# ── Stage 1: 0.4m, virtual U[0.10, 0.20] ─────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  gate_advance 1 0.20 0.10 "0.4m" 100000000
fi

# ── Stage 2: 0.24m, virtual U[0.06, 0.12] ────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  gate_advance 2 0.12 0.06 "0.24m" 150000000
fi

# ── Stage 3: 0.16m, virtual U[0.05, 0.08], ep_len=1500 ───────────────────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  gate_advance 3 0.08 0.05 "0.16m" 200000000 "--episode_length 1500"
fi

# ── Stage 4: 0.10m, virtual fixed 0.05, ep_len=1500 ──────────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  gate_advance 4 0.05 0.05 "0.10m" 200000000 "--episode_length 1500"
fi

echo "=== All 5 stages complete. Final checkpoint: ${CKPT} ===" >&2
