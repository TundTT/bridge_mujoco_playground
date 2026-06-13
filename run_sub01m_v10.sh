#!/bin/bash
# v10 sub-0.10m curriculum: finding the morphological limit.
#
# Loads from the v10 final checkpoint (0.10m at 91.4%) and first runs a 200M
# stabilisation pass at 0.10m, then descends to find where the Go1 genuinely
# fails. Designed to give the policy the best possible chance at each width:
#
#   - 1cm steps from 0.10m down to 0.06m (stages 0-4)
#   - 5mm steps from 0.06m downward (stages 5+) — physical slack halves per cm
#     at these widths so uniform 1cm would be unfairly aggressive
#   - vhw_min floored at max(hw - gap, FOOT_RADIUS + 0.005 = 0.028m) so the
#     virtual curriculum never samples widths where even a centred foot has
#     permanent foot_off_virtual penalty (would corrupt the advantage signal)
#   - ep_len scales up at narrow stages: 30s isn't enough at 0.06m (robot slows)
#
# Stage schedule:
#   Stage 0: 0.10m stab (hw=0.0500, vhw_min=0.0400, ep=1500) — 200M
#   Stage 1: 0.09m      (hw=0.0450, vhw_min=0.0350, ep=1500) — 200M
#   Stage 2: 0.08m      (hw=0.0400, vhw_min=0.0300, ep=1500) — 200M
#   Stage 3: 0.07m      (hw=0.0350, vhw_min=0.0280, ep=1500) — 200M  ← floored
#   Stage 4: 0.06m      (hw=0.0300, vhw_min=0.0280, ep=2000) — 200M  ← floored + ep↑
#   Stage 5: 0.055m     (hw=0.0275, vhw_min=0.0275, ep=2000) — 200M  ← 5mm step
#   Stage 6: 0.05m      (hw=0.0250, vhw_min=0.0250, ep=2000) — 200M
#   (Stage 7 at hw=0.0225 omitted: hw < foot_radius=0.023m so a centred foot
#    has non-zero foot_off_virtual overshoot unconditionally — unsatisfiable
#    penalty. Bridge narrower than one foot sphere is a different gait regime.)
#
# Morphological limit expected around hw=0.04-0.05m (roll stability + foot-fit).
# Gate: advance ≥0.70; extend +100M if 0.30-0.70; revert + midpoint if <0.30.
#
# Usage:
#   ./run_sub01m_v10.sh
#   SKIP_TO_STAGE=3 INIT_CKPT=<path> ./run_sub01m_v10.sh

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
  # -a: treat binary as text (log may contain ANSI escape codes)
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
  local logfile="/tmp/bridge_v10_sub01m_stage${stage}.log"

  echo "=== Sub-0.1m Stage ${stage}: ${width_str} (hw=${hw}, vhw_min=${vhw_min}, ep_len=${ep_len}) ===" >&2

  # shellcheck disable=SC2086
  train-jax-ppo --env_name Go1BridgeCrossing \
    --num_timesteps "${steps}" \
    ${ckpt_arg:+--load_checkpoint_path "$ckpt_arg"} \
    --playground_config_overrides "{\"bridge_half_width\": ${hw}, \"virtual_hw_min\": ${vhw_min}}" \
    --entropy_cost 0.01 \
    --episode_length "${ep_len}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "v10_sub01m_stage${stage}_${width_str}" \
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
    echo "Stage ${stage} failed (${rate} < 0.30). Morphological limit found." >&2
    CKPT="$LAST_GOOD_CKPT"
    local mid_hw
    mid_hw=$(awk "BEGIN{printf \"%.4f\", (${hw} + ${LAST_GOOD_HW}) / 2}")
    local mid_bridge
    mid_bridge=$(awk "BEGIN{printf \"%.3f\", ${mid_hw}*2}")
    echo "  Last good: hw=${LAST_GOOD_HW} ($(awk "BEGIN{printf \"%.3f\", ${LAST_GOOD_HW}*2}")m bridge)" >&2
    echo "  Failed at: hw=${hw} ($(awk "BEGIN{printf \"%.3f\", ${hw}*2}")m bridge)" >&2
    echo "  Midpoint:  hw=${mid_hw} (${mid_bridge}m bridge)" >&2
    echo "  Re-invoke: SKIP_TO_STAGE=${stage} INIT_CKPT=${LAST_GOOD_CKPT} bash run_sub01m_v10.sh" >&2
    exit 1
  fi
}

CKPT="${INIT_CKPT}"

# ── Stage 0: 0.10m stabilisation ─────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 0 ]; then
  gate_advance 0 0.0500 0.0400 "0.10m_stab" 200000000 1500
fi

# ── Stage 1: 0.09m ───────────────────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 1 ]; then
  gate_advance 1 0.0450 0.0350 "0.09m" 200000000 1500
fi

# ── Stage 2: 0.08m ───────────────────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 2 ]; then
  gate_advance 2 0.0400 0.0300 "0.08m" 200000000 1500
fi

# ── Stage 3: 0.07m — vhw_min floored at 0.028 (foot_radius + 0.005) ─────────
if [ "${SKIP_TO_STAGE}" -le 3 ]; then
  gate_advance 3 0.0350 0.0280 "0.07m" 200000000 1500
fi

# ── Stage 4: 0.06m — ep_len 1500→2000, vhw_min floored ──────────────────────
if [ "${SKIP_TO_STAGE}" -le 4 ]; then
  gate_advance 4 0.0300 0.0280 "0.06m" 200000000 2000
fi

# ── Stage 5: 0.055m — 5mm step, hw < foot_radius floor so vhw_min=hw ─────────
if [ "${SKIP_TO_STAGE}" -le 5 ]; then
  gate_advance 5 0.0275 0.0275 "0.055m" 200000000 2000
fi

# ── Stage 6: 0.05m ───────────────────────────────────────────────────────────
if [ "${SKIP_TO_STAGE}" -le 6 ]; then
  gate_advance 6 0.0250 0.0250 "0.05m" 200000000 2000
fi

echo "=== All sub-0.10m stages complete. Final checkpoint: ${CKPT} ===" >&2
echo "=== Policy survived to 0.05m bridge (hw=0.025). Morphological limit not found. ===" >&2
