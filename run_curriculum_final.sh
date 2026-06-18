#!/bin/bash
# Final presentation curriculum: 0.8m → 0.055m (morphological limit).
#
# Fresh policy, all fixes in place. Combines the proven v10 curriculum
# (stages 0–7, 0.8m→0.10m) with the sub-0.10m descent (stages 8–12,
# 0.09m→0.055m) in a single clean WandB project.
#
# Stage schedule:
#   Stage 00: 0.80m  hw=0.400  vhw_min=0.150  150M  ep=1000
#   Stage 01: 0.40m  hw=0.200  vhw_min=0.100  100M  ep=1000
#   Stage 02: 0.32m  hw=0.160  vhw_min=0.080  100M  ep=1000
#   Stage 03: 0.24m  hw=0.120  vhw_min=0.060  150M  ep=1000
#   Stage 04: 0.20m  hw=0.100  vhw_min=0.050  150M  ep=1500  ← midpoint
#   Stage 05: 0.16m  hw=0.080  vhw_min=0.050  200M  ep=1500
#   Stage 06: 0.13m  hw=0.065  vhw_min=0.045  150M  ep=1500
#   Stage 07: 0.10m  hw=0.050  vhw_min=0.040  200M  ep=1500
#   Stage 08: 0.09m  hw=0.045  vhw_min=0.035  200M  ep=1500
#   Stage 09: 0.08m  hw=0.040  vhw_min=0.030  200M  ep=1500
#   Stage 10: 0.07m  hw=0.035  vhw_min=0.028  200M  ep=1500
#   Stage 11: 0.06m  hw=0.030  vhw_min=0.028  200M  ep=2000
#   Stage 12: 0.055m hw=0.0275 vhw_min=0.0275 200M  ep=2000  ← expected failure
#
# Total budget: ~2200M steps (~2.5 hrs wall time).
# Gate: advance ≥0.70; extend +100M if 0.30–0.70; revert+exit if <0.30.
# Stage 12 (0.055m) uses a soft gate — logs failure without exit so the
# project captures the complete story including the limit.
#
# Usage:
#   ./run_curriculum_final.sh
#   SKIP_TO_STAGE=8 INIT_CKPT=<path> ./run_curriculum_final.sh

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

WANDB_PROJECT="bridge_crossing_final"
WANDB_GROUP="${WANDB_GROUP:-final_curriculum}"
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"
INIT_CKPT="${INIT_CKPT:-}"
LAST_GOOD_CKPT="${INIT_CKPT:-}"
LAST_GOOD_HW="0.40"

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
  local hw=$2
  local vhw_min=$3
  local width_str=$4
  local steps=$5
  local ep_len=$6
  local ckpt_arg="${7:-}"
  local logfile="/tmp/bridge_final_stage${stage}.log"

  echo "=== Stage ${stage}: ${width_str} (hw=${hw}, vhw_min=${vhw_min}, ep_len=${ep_len}) ===" >&2

  # shellcheck disable=SC2086
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${steps}" \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": ${vhw_min}}" \
    --entropy_cost 0.01 \
    --episode_length "${ep_len}" \
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

  echo "Stage ${stage} done. success_rate=${rate} checkpoint=${ckpt}" >&2
  echo "${rate} ${ckpt}"
}

gate_advance() {
  local stage=$1
  local hw=$2
  local vhw_min=$3
  local width_str=$4
  local steps=$5
  local ep_len=$6
  local max_extensions=2

  local result
  result=$(run_stage "${stage}" "${hw}" "${vhw_min}" "${width_str}" "${steps}" "${ep_len}" "${CKPT}")
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
      result=$(run_stage "${stage}ext${ext_num}" "${hw}" "${vhw_min}" "${width_str}_ext${ext_num}" "${ext_steps}" "${ep_len}" "$ckpt")
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
    echo "Stage ${stage} failed (${rate} < 0.30). Reverting and inserting midpoint." >&2
    CKPT="$LAST_GOOD_CKPT"
    local mid_hw
    mid_hw=$(awk "BEGIN{printf \"%.4f\", (${hw} + ${LAST_GOOD_HW}) / 2}")
    echo "  Last good: hw=${LAST_GOOD_HW}" >&2
    echo "  Failed at: hw=${hw}" >&2
    echo "  Midpoint:  hw=${mid_hw}" >&2
    echo "  Re-invoke: SKIP_TO_STAGE=${stage} INIT_CKPT=${LAST_GOOD_CKPT} bash run_curriculum_final.sh" >&2
    exit 1
  fi
}

# Final stage: probe the morphological limit — log result, don't exit on failure.
probe_limit() {
  local stage=$1
  local hw=$2
  local vhw_min=$3
  local width_str=$4
  local steps=$5
  local ep_len=$6

  local result
  result=$(run_stage "${stage}" "${hw}" "${vhw_min}" "${width_str}" "${steps}" "${ep_len}" "${CKPT}")
  local rate
  rate=$(echo "$result" | awk '{print $1}')

  local pass
  pass=$(awk "BEGIN{print (${rate} >= 0.70) ? 1 : 0}")
  if [ "$pass" -eq 1 ]; then
    echo "Stage ${stage} passed (${rate}). Morphological limit not found — consider adding narrower stages." >&2
  else
    echo "Stage ${stage} failed (${rate}). Morphological limit confirmed between hw=${LAST_GOOD_HW} and hw=${hw}." >&2
    echo "  Practical limit: $(awk "BEGIN{printf \"%.3f\", ${LAST_GOOD_HW}*2}")m bridge (${LAST_GOOD_HW}m half-width)." >&2
  fi
}

CKPT="${INIT_CKPT}"

# ── Main curriculum: 0.8m → 0.10m ────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 0 ]; then
  gate_advance "00" 0.400 0.150 "0.8m"  150000000 1000
fi
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  gate_advance "01" 0.200 0.100 "0.4m"  100000000 1000
fi
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  gate_advance "02" 0.160 0.080 "0.32m" 100000000 1000
fi
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  gate_advance "03" 0.120 0.060 "0.24m" 150000000 1000
fi
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  gate_advance "04" 0.100 0.050 "0.20m" 150000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 5 ]; then
  gate_advance "05" 0.080 0.050 "0.16m" 200000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 6 ]; then
  gate_advance "06" 0.065 0.045 "0.13m" 150000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 7 ]; then
  gate_advance "07" 0.050 0.040 "0.10m" 200000000 1500
fi

# ── Sub-0.10m descent ─────────────────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 8 ]; then
  gate_advance "08" 0.0450 0.0350 "0.09m"  200000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 9 ]; then
  gate_advance "09" 0.0400 0.0300 "0.08m"  200000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 10 ]; then
  gate_advance "10" 0.0350 0.0280 "0.07m"  200000000 1500
fi
if [ "${SKIP_TO_STAGE}" -le 11 ]; then
  gate_advance "11" 0.0300 0.0280 "0.06m"  200000000 2000
fi

# ── Stage 12: morphological limit probe (0.055m — expected to fail) ───────────
if [ "${SKIP_TO_STAGE}" -le 12 ]; then
  probe_limit  "12" 0.0275 0.0275 "0.055m" 200000000 2000
fi

echo "=== Final curriculum complete. ===" >&2
echo "=== Check bridge_crossing_final on WandB for the full progression. ===" >&2
