#!/bin/bash
# v10 sub-0.10m curriculum: finding the morphological limit.
#
# Loads from the v10 final checkpoint (0.10m at 91.4%) and first runs a 200M
# stabilisation pass at 0.10m, then descends in small steps to find where the
# policy fails. Steps are intentionally fine-grained (~1cm) so the gate's
# midpoint-insertion logic can resolve failures gracefully.
#
# vhw_min is always set below hw so the virtual curriculum targets a tighter
# stance than the physical edge — this keeps foot_off_virtual and frontier
# gradients live even when feet sit exactly at the physical boundary.
#
# Stages:
#   Stage 0: 0.10m (hw=0.050), vhw_min=0.040 — 200M stabilisation
#   Stage 1: 0.09m (hw=0.045), vhw_min=0.035 — 200M
#   Stage 2: 0.08m (hw=0.040), vhw_min=0.030 — 200M
#   Stage 3: 0.07m (hw=0.035), vhw_min=0.025 — 200M
#   Stage 4: 0.06m (hw=0.030), vhw_min=0.020 — 200M
#   Stage 5: 0.05m (hw=0.025), vhw_min=0.015 — 200M
#
# Gate: advance if term_success >= 0.70; extend +100M if 0.30–0.70;
#       revert + insert midpoint and exit if < 0.30.
#
# Usage:
#   ./run_sub01m_v10.sh                               # fresh run from v10 ckpt
#   SKIP_TO_STAGE=2 INIT_CKPT=<path> ./run_sub01m_v10.sh

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

V10_FINAL_CKPT="logs/Go1BridgeCrossing-20260612-203650/checkpoints/000200540160"

WANDB_PROJECT="bridge_crossing_v10"
WANDB_GROUP="${WANDB_GROUP:-v10_sub01m}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"
INIT_CKPT="${INIT_CKPT:-${V10_FINAL_CKPT}}"
LAST_GOOD_CKPT="${INIT_CKPT}"
LAST_GOOD_HW="0.050"

get_latest_ckpt() {
  ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1
}

get_logdir() {
  grep "Logs are being stored in:" "$1" | awk '{print $NF}'
}

get_success_rate() {
  local logfile="$1"
  local rate
  rate=$(grep "metric/term_success:" "$logfile" 2>/dev/null | tail -1 | awk '{print $2}')
  echo "${rate:-0}"
}

run_stage() {
  local stage=$1
  local hw=$2
  local vhw_min=$3
  local width_str=$4
  local steps=$5
  local ckpt_arg="${6:-}"
  local extra_flags="${7:-}"
  local logfile="/tmp/bridge_v10_sub01m_stage${stage}.log"

  echo "=== Sub-0.1m Stage ${stage}: ${width_str} (hw=${hw}, vhw_min=${vhw_min}) ===" >&2

  # shellcheck disable=SC2086
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${steps}" \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": ${vhw_min}}" \
    --entropy_cost 0.01 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "v10_sub01m_stage${stage}_${width_str}" \
    --wandb_group "${WANDB_GROUP}" \
    --use_wandb \
    ${extra_flags} \
    2>&1 | tee "${logfile}" >/dev/null

  local logdir
  logdir=$(get_logdir "${logfile}")
  local ckpt
  ckpt=$(get_latest_ckpt "${logdir}")
  local rate
  rate=$(get_success_rate "${logfile}")

  echo "Stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  echo "${rate} ${ckpt}"
}

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

  local pass
  pass=$(awk "BEGIN{print (${rate} >= 0.70) ? 1 : 0}")
  local marginal
  marginal=$(awk "BEGIN{print (${rate} >= 0.30) ? 1 : 0}")

  if [ "$pass" -eq 1 ]; then
    echo "Stage ${stage} passed (${rate} >= 0.70). Advancing." >&2
    CKPT="$ckpt"
    LAST_GOOD_CKPT="$ckpt"
    LAST_GOOD_HW="$hw"
  elif [ "$marginal" -eq 1 ]; then
    echo "Stage ${stage} marginal (${rate} in [0.30, 0.70)). Extending +100M." >&2
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
    echo "Stage ${stage} still marginal after extensions. Advancing anyway." >&2
    CKPT="$ckpt"
  else
    echo "Stage ${stage} failed (${rate} < 0.30). Morphological limit found." >&2
    CKPT="$LAST_GOOD_CKPT"
    local mid_hw
    mid_hw=$(awk "BEGIN{printf \"%.4f\", (${hw} + ${LAST_GOOD_HW}) / 2}")
    echo "  Last good: hw=${LAST_GOOD_HW} ($(awk "BEGIN{printf \"%.3f\", ${LAST_GOOD_HW}*2}")m bridge)" >&2
    echo "  Failed at: hw=${hw} ($(awk "BEGIN{printf \"%.3f\", ${hw}*2}")m bridge)" >&2
    echo "  Midpoint:  hw=${mid_hw} ($(awk "BEGIN{printf \"%.3f\", ${mid_hw}*2}")m bridge)" >&2
    echo "  Re-invoke: SKIP_TO_STAGE=${stage} INIT_CKPT=${LAST_GOOD_CKPT} bash run_sub01m_v10.sh" >&2
    exit 1
  fi
}

CKPT="${INIT_CKPT}"

# ── Stage 0: 0.10m stabilisation, virtual U[0.040, 0.050] ────────────────────
if [ "${SKIP_TO_STAGE}" -le 0 ]; then
  gate_advance 0 0.050 0.040 "0.10m_stab" 200000000 "--episode_length 1500"
fi

# ── Stage 1: 0.09m, virtual U[0.035, 0.045] ──────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  gate_advance 1 0.045 0.035 "0.09m" 200000000 "--episode_length 1500"
fi

# ── Stage 2: 0.08m, virtual U[0.030, 0.040] ──────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  gate_advance 2 0.040 0.030 "0.08m" 200000000 "--episode_length 1500"
fi

# ── Stage 3: 0.07m, virtual U[0.025, 0.035] ──────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  gate_advance 3 0.035 0.025 "0.07m" 200000000 "--episode_length 1500"
fi

# ── Stage 4: 0.06m, virtual U[0.020, 0.030] ──────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  gate_advance 4 0.030 0.020 "0.06m" 200000000 "--episode_length 1500"
fi

# ── Stage 5: 0.05m, virtual U[0.015, 0.025] ──────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 5 ]; then
  gate_advance 5 0.025 0.015 "0.05m" 200000000 "--episode_length 1500"
fi

echo "=== All sub-0.10m stages complete. Final checkpoint: ${CKPT} ===" >&2
echo "=== Policy survived to 0.05m bridge. Morphological limit not found in this range. ===" >&2
