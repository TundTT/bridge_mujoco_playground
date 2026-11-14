#!/bin/bash
# Tail vs no-tail push-tolerance experiment on the box bridge.
#
# One condition per invocation (run the three on separate GPUs / in parallel):
#   CONDITION=active  -> Go1BridgeCrossingTail, policy controls the tail
#   CONDITION=locked  -> Go1BridgeCrossingTail, tail held upright (same mass)
#   CONDITION=notail  -> Go1BridgeCrossing (12.7 kg robot, no tail)
#
# Every condition gets the SAME fixed schedule and budget (no success gating),
# so differences come from the robot, not from the curriculum.
#
# Phase A - width curriculum with mild pushes (impulse U[0, 4] N*s):
#   A00 0.80m  A01 0.40m  A02 0.32m  A03 0.24m  A04 0.20m*  A05 0.16m
#   A06 0.13m  A07 0.10m* A08 0.09m  A09 0.08m  A10 0.07m   A11 0.06m*
# Phase B - push curriculum branched from the * checkpoints (0.20/0.10/0.06 m):
#   impulse U[0, 8] -> U[0, 14] -> U[0, 20] N*s, 100M steps each.
# Phase C - push-tolerance eval (scripts/eval_push_tolerance.py) of each
#   branch before (A*) and after (B) push training.
#
# Resumable: finished stages are recorded in results/tail_push/<cond>/manifest.tsv
# and skipped on re-run.
#
# Robot: ROBOT=go1 (default, results/tail_push/) or ROBOT=go2 (results/tail_push_go2/).
#
# Usage:
#   ROBOT=go2 CONDITION=active CUDA_VISIBLE_DEVICES=0 ./run_tail_push_experiment.sh
#   DRY_RUN=1 CONDITION=locked ./run_tail_push_experiment.sh   # print plan only
#   SCALE=0.001 ...   # multiply all step budgets (smoke tests)
#   STOP_AFTER=A04 ... # stop (exit 99) after that stage; re-run resumes
#   EXTRA_TRAIN_ARGS="--num_envs 16 ..."  # appended to every train command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR:-$SCRIPT_DIR}"

CONDITION="${CONDITION:?set CONDITION=active|locked|notail}"
ROBOT="${ROBOT:-go1}"
DRY_RUN="${DRY_RUN:-0}"
SCALE="${SCALE:-1}"
case "$ROBOT" in
  go1) ENV_PREFIX=Go1; RESULTS_ROOT="${RESULTS_ROOT:-results/tail_push}"
       WANDB_PROJECT="${WANDB_PROJECT:-bridge_tail_push}" ;;
  go2) ENV_PREFIX=Go2; RESULTS_ROOT="${RESULTS_ROOT:-results/tail_push_go2}"
       WANDB_PROJECT="${WANDB_PROJECT:-bridge_tail_push_go2}" ;;
  *) echo "unknown ROBOT=$ROBOT" >&2; exit 1 ;;
esac
USE_WANDB="${USE_WANDB:-1}"
# Fine push sweep (0.5 N*s steps where the Go1 J50s fell, 512 trials).
EVAL_TRIALS="${EVAL_TRIALS:-512}"
EVAL_IMPULSES="${EVAL_IMPULSES:-0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,14,16,20}"
# Evals share GPUs with training runs; cap their memory (the Go1 run hit a
# cuSolver error when two evals pre-allocated most of one GPU).
EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.15}"
TRAIN_CMD="${TRAIN_CMD:-train-jax-ppo}"
PYTHON_CMD="${PYTHON_CMD:-python}"
MILD_PUSH="[0.0, 4.0]"

case "$CONDITION" in
  active) ENV_NAME=${ENV_PREFIX}BridgeCrossingTail; COND_OVR='"tail_mode": "active"' ;;
  locked) ENV_NAME=${ENV_PREFIX}BridgeCrossingTail; COND_OVR='"tail_mode": "locked"' ;;
  notail) ENV_NAME=${ENV_PREFIX}BridgeCrossing
          COND_OVR='"push_config.enable": true, "reset_info_on_new_episode": true' ;;
  *) echo "unknown CONDITION=$CONDITION" >&2; exit 1 ;;
esac

if [ "$DRY_RUN" != "1" ] && [ -z "${SKIP_VENV:-}" ]; then
  if [ ! -d ".venv" ]; then
    uv --no-config sync --all-extras
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

OUT_DIR="${RESULTS_ROOT}/${CONDITION}"
mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/eval"
MANIFEST="${OUT_DIR}/manifest.tsv"
touch "$MANIFEST"

steps_of() { awk "BEGIN{printf \"%d\", $1 * ${SCALE}}"; }

overrides_json() {  # hw vhw_min impulse_lo impulse_hi
  echo "{${COND_OVR}, \"bridge_half_width\": $1, \"virtual_hw_min\": $2, \"push_config.impulse\": [$3, $4]}"
}

manifest_get() {  # stage -> "rate ckpt" if done
  awk -v s="$1" -F'\t' '$1==s {print $2" "$3}' "$MANIFEST" | tail -1
}

latest_ckpt() { ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1; }

# run_stage NAME HW VHW_MIN IMP_LO IMP_HI STEPS EP_LEN INIT_CKPT -> prints "rate ckpt"
run_stage() {
  local name=$1 hw=$2 vhw=$3 ilo=$4 ihi=$5 steps=$6 ep=$7 init=${8:-}
  local done_line
  done_line=$(manifest_get "$name")
  if [ -n "$done_line" ]; then
    echo "[$CONDITION] $name already done: $done_line" >&2
    echo "$done_line"; return 0
  fi
  local ovr; ovr=$(overrides_json "$hw" "$vhw" "$ilo" "$ihi")
  local nsteps; nsteps=$(steps_of "$steps")
  local log="${OUT_DIR}/logs/${name}.log"
  echo "[$CONDITION] === $name hw=$hw vhw_min=$vhw impulse=[$ilo,$ihi] steps=$nsteps ep=$ep init=${init:-none}" >&2
  if [ "$DRY_RUN" = "1" ]; then echo "0 DRY_${name}"; return 0; fi

  local wandb_args=()
  if [ "$USE_WANDB" = "1" ]; then
    wandb_args=(--use_wandb --wandb_project "$WANDB_PROJECT"
                --wandb_run_name "${CONDITION}_${name}" --wandb_group "$CONDITION")
  fi
  # shellcheck disable=SC2086
  $TRAIN_CMD --env_name "$ENV_NAME" \
    --num_timesteps "$nsteps" \
    ${init:+--load_checkpoint_path "$init"} \
    --playground_config_overrides "$ovr" \
    --entropy_cost 0.01 \
    --episode_length "$ep" \
    ${EXTRA_TRAIN_ARGS:-} \
    "${wandb_args[@]}" > "$log" 2>&1 || true

  local logdir ckpt rate
  logdir=$(grep -a "Logs are being stored in:" "$log" | awk '{print $NF}' | tail -1)
  ckpt=$(latest_ckpt "$logdir")
  rate=$(grep -a "metric/term_success:" "$log" | tail -1 | awk '{print $2}')
  rate=${rate:-0}
  if [ -z "$ckpt" ]; then
    echo "[$CONDITION] $name produced no checkpoint - see $log" >&2; exit 1
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$rate" "$ckpt" "$ovr" >> "$MANIFEST"
  echo "[$CONDITION] $name done: success=$rate ckpt=$ckpt" >&2
  if [ "${STOP_AFTER:-}" = "$name" ]; then
    echo "[$CONDITION] STOP_AFTER=$name reached; stopping (exit 99). Re-run to resume." >&2
    exit 99
  fi
  echo "$rate $ckpt"
}

run_eval() {  # NAME CKPT HW VHW_MIN EP_LEN
  local name=$1 ckpt=$2 hw=$3 vhw=$4 ep=$5
  local out="${OUT_DIR}/eval/${name}.json"
  if [ -f "$out" ]; then echo "[$CONDITION] eval $name exists" >&2; return 0; fi
  echo "[$CONDITION] eval $name ($ckpt)" >&2
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  local ovr; ovr=$(overrides_json "$hw" "$vhw" 0.0 0.0)
  XLA_PYTHON_CLIENT_MEM_FRACTION=$EVAL_MEM_FRACTION \
  $PYTHON_CMD scripts/eval_push_tolerance.py --env_name "$ENV_NAME" \
    --checkpoint "$ckpt" --overrides "$ovr" --episode_length "$ep" \
    --impulses "$EVAL_IMPULSES" --num_trials "$EVAL_TRIALS" \
    --out "$out" > "${OUT_DIR}/logs/eval_${name}.log" 2>&1 \
    || echo "[$CONDITION] eval $name FAILED - see ${OUT_DIR}/logs/eval_${name}.log" >&2
}

ckpt_of() { echo "$1" | awk '{print $2}'; }

# ── Phase A: width curriculum ────────────────────────────────────────────────
C=""
#        name  hw      vhw_min lo  hi  steps      ep
r=$(run_stage A00 0.400  0.150 0.0 4.0 150000000 1000 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A01 0.200  0.100 0.0 4.0 100000000 1000 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A02 0.160  0.080 0.0 4.0 100000000 1000 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A03 0.120  0.060 0.0 4.0 150000000 1000 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A04 0.100  0.050 0.0 4.0 150000000 1500 "$C"); C=$(ckpt_of "$r"); W020="$C"
r=$(run_stage A05 0.080  0.050 0.0 4.0 200000000 1500 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A06 0.065  0.045 0.0 4.0 150000000 1500 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A07 0.050  0.040 0.0 4.0 200000000 1500 "$C"); C=$(ckpt_of "$r"); W010="$C"
r=$(run_stage A08 0.045  0.035 0.0 4.0 200000000 1500 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A09 0.040  0.030 0.0 4.0 200000000 1500 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A10 0.035  0.028 0.0 4.0 200000000 1500 "$C"); C=$(ckpt_of "$r")
r=$(run_stage A11 0.030  0.028 0.0 4.0 200000000 2000 "$C"); C=$(ckpt_of "$r"); W006="$C"

# ── Phase B: push curriculum per branch width ────────────────────────────────
branch() {  # TAG HW VHW EP START_CKPT
  local tag=$1 hw=$2 vhw=$3 ep=$4 c=$5 r
  r=$(run_stage "B_${tag}_p08" "$hw" "$vhw" 0.0 8.0  100000000 "$ep" "$c"); c=$(ckpt_of "$r")
  r=$(run_stage "B_${tag}_p14" "$hw" "$vhw" 0.0 14.0 100000000 "$ep" "$c"); c=$(ckpt_of "$r")
  r=$(run_stage "B_${tag}_p20" "$hw" "$vhw" 0.0 20.0 100000000 "$ep" "$c"); c=$(ckpt_of "$r")
  echo "$c"
}
B020=$(branch w020 0.100 0.050 1500 "$W020")
B010=$(branch w010 0.050 0.040 1500 "$W010")
B006=$(branch w006 0.030 0.028 2000 "$W006")

# ── Phase C: push-tolerance evaluation ───────────────────────────────────────
run_eval A04_w020 "$W020" 0.100 0.050 1500
run_eval A07_w010 "$W010" 0.050 0.040 1500
run_eval A11_w006 "$W006" 0.030 0.028 2000
run_eval B_w020   "$B020" 0.100 0.050 1500
run_eval B_w010   "$B010" 0.050 0.040 1500
run_eval B_w006   "$B006" 0.030 0.028 2000
# Cross-width: the 0.06 m-trained policies on the 0.10 m beam, so a stall in
# one width's curriculum stage cannot show up as push intolerance. (Not on
# 0.20 m: the policy observes beam width and fails on widths it never saw.)
run_eval X_A11_at_w010        "$W006" 0.050 0.040 1500
run_eval X_B_w006_p20_at_w010 "$B006" 0.050 0.040 1500

# Posture diagnostics (pitch, height, per-foot contact) at key stages.
run_posture() {  # NAME CKPT HW VHW_MIN EP_LEN
  local name=$1 ckpt=$2 hw=$3 vhw=$4 ep=$5
  local out="${OUT_DIR}/posture/${name}.json"
  if [ -f "$out" ] || [ "$DRY_RUN" = "1" ]; then return 0; fi
  echo "[$CONDITION] posture $name ($ckpt)" >&2
  local ovr; ovr=$(overrides_json "$hw" "$vhw" 0.0 0.0)
  XLA_PYTHON_CLIENT_MEM_FRACTION=$EVAL_MEM_FRACTION \
  $PYTHON_CMD scripts/diagnose_posture.py --env_name "$ENV_NAME" \
    --checkpoint "$ckpt" --overrides "$ovr" --episode_length "$ep" \
    --out "$out" > "${OUT_DIR}/logs/posture_${name}.log" 2>&1 \
    || echo "[$CONDITION] posture $name FAILED - see ${OUT_DIR}/logs/posture_${name}.log" >&2
}
run_posture A04_w020 "$W020" 0.100 0.050 1500
run_posture A07_w010 "$W010" 0.050 0.040 1500
run_posture A11_w006 "$W006" 0.030 0.028 2000
run_posture B_w006   "$B006" 0.030 0.028 2000

echo "[$CONDITION] experiment complete. Results in ${OUT_DIR}/" >&2
