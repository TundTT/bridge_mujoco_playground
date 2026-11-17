#!/bin/bash
# Standing push test on the box beam (Go2): the robot stands still mid-beam
# (x = 2 m) and is pushed sideways once. Which condition survives the hardest
# push, and on which beam width?
#
# Per condition (active / locked / notail) and width (0.20 / 0.10 / 0.06 m):
#   S_<w>_p08  fine-tune the push-trained walking policy (B_<w>_p20 from
#              results/tail_push_go2/<cond>/manifest.tsv) on the stand task,
#              random pushes U[0, 8] N*s, 100M steps
#   S_<w>_p20  continue with pushes U[0, 20] N*s, 100M steps
#   eval       one 0.1 s lateral push at t = 2 s, 10 s episodes, 512 trials
#              per impulse (scripts/eval_push_tolerance.py --trigger_time 2)
#
# Stand task rewards: alive +10, stand_pos (squared x drift from spawn) -10,
# and frontier_delta / success / feet_air_time off. Every other reward is the
# same as the walking runs (incl. the Go2 pitch dead band).
#
# Resumable through results/stand_push_go2/<cond>/manifest.tsv.
#
# Usage:
#   CONDITION=active CUDA_VISIBLE_DEVICES=0 ./run_stand_push_experiment.sh
#   DRY_RUN=1 CONDITION=locked ./run_stand_push_experiment.sh
#   SCALE=0.01 STOP_AFTER=S_w020_p08 USE_WANDB=0 ...   # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR:-$SCRIPT_DIR}"

CONDITION="${CONDITION:?set CONDITION=active|locked|notail}"
DRY_RUN="${DRY_RUN:-0}"
SCALE="${SCALE:-1}"
WALK_ROOT="${WALK_ROOT:-results/tail_push_go2}"
# Leg collision against the beam (thighs/calves). 1 = on (default since the
# first run, where the active-tail policy sank its rear knees into the beam).
LEG_COLLISION="${LEG_COLLISION:-1}"
if [ "$LEG_COLLISION" = "1" ]; then
  RESULTS_ROOT="${RESULTS_ROOT:-results/stand_push_go2_legcol}"
else
  RESULTS_ROOT="${RESULTS_ROOT:-results/stand_push_go2}"
fi
# Rewrite walking-checkpoint paths from the manifest when running on another
# machine: CKPT_REMAP="/old/prefix=/new/prefix".
CKPT_REMAP="${CKPT_REMAP:-}"
WANDB_PROJECT="${WANDB_PROJECT:-bridge_stand_push_go2}"
USE_WANDB="${USE_WANDB:-1}"
EVAL_TRIALS="${EVAL_TRIALS:-512}"
EVAL_IMPULSES="${EVAL_IMPULSES:-0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,17,18,19,20,22,24,26,28,30,35,40}"
EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.15}"
TRAIN_CMD="${TRAIN_CMD:-train-jax-ppo}"
PYTHON_CMD="${PYTHON_CMD:-python}"

case "$CONDITION" in
  active) ENV_NAME=Go2BridgeCrossingTail; COND_OVR='"tail_mode": "active"' ;;
  locked) ENV_NAME=Go2BridgeCrossingTail; COND_OVR='"tail_mode": "locked"' ;;
  notail) ENV_NAME=Go2BridgeCrossing
          COND_OVR='"push_config.enable": true, "reset_info_on_new_episode": true' ;;
  *) echo "unknown CONDITION=$CONDITION" >&2; exit 1 ;;
esac
STAND_OVR='"task": "stand", "reward_config.scales.alive": 10.0, "reward_config.scales.stand_pos": -10.0, "reward_config.scales.frontier_delta": 0.0, "reward_config.scales.success": 0.0, "reward_config.scales.feet_air_time": 0.0'
if [ "$LEG_COLLISION" = "1" ]; then STAND_OVR="$STAND_OVR, \"leg_collision\": true"; fi

if [ "$DRY_RUN" != "1" ] && [ -z "${SKIP_VENV:-}" ]; then
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
  echo "{${COND_OVR}, ${STAND_OVR}, \"bridge_half_width\": $1, \"virtual_hw_min\": $2, \"push_config.impulse\": [$3, $4]}"
}

walk_ckpt() {  # stage name in the walking manifest -> checkpoint
  local c
  c=$(awk -v s="$1" -F'\t' '$1==s {print $3}' "${WALK_ROOT}/${CONDITION}/manifest.tsv" | tail -1)
  if [ -z "$c" ]; then echo "no $1 in ${WALK_ROOT}/${CONDITION}/manifest.tsv" >&2; exit 1; fi
  if [ -n "$CKPT_REMAP" ]; then c="${c/${CKPT_REMAP%%=*}/${CKPT_REMAP#*=}}"; fi
  echo "$c"
}

manifest_get() { awk -v s="$1" -F'\t' '$1==s {print $2" "$3}' "$MANIFEST" | tail -1; }
latest_ckpt() { ls -d "$1/checkpoints"/[0-9]* 2>/dev/null | sort -V | tail -1; }
ckpt_of() { echo "$1" | awk '{print $2}'; }

# run_stage NAME HW VHW_MIN IMP_LO IMP_HI STEPS EP_LEN INIT_CKPT -> "survival ckpt"
run_stage() {
  local name=$1 hw=$2 vhw=$3 ilo=$4 ihi=$5 steps=$6 ep=$7 init=$8
  local done_line; done_line=$(manifest_get "$name")
  if [ -n "$done_line" ]; then
    echo "[$CONDITION] $name already done: $done_line" >&2; echo "$done_line"; return 0
  fi
  local ovr; ovr=$(overrides_json "$hw" "$vhw" "$ilo" "$ihi")
  local nsteps; nsteps=$(steps_of "$steps")
  local log="${OUT_DIR}/logs/${name}.log"
  echo "[$CONDITION] === $name hw=$hw impulse=[$ilo,$ihi] steps=$nsteps init=$init" >&2
  if [ "$DRY_RUN" = "1" ]; then echo "0 DRY_${name}"; return 0; fi
  local wandb_args=()
  if [ "$USE_WANDB" = "1" ]; then
    wandb_args=(--use_wandb --wandb_project "$WANDB_PROJECT"
                --wandb_run_name "${CONDITION}_${name}" --wandb_group "$CONDITION")
  fi
  # shellcheck disable=SC2086
  $TRAIN_CMD --env_name "$ENV_NAME" \
    --num_timesteps "$nsteps" \
    --load_checkpoint_path "$init" \
    --playground_config_overrides "$ovr" \
    --entropy_cost 0.01 \
    --episode_length "$ep" \
    ${EXTRA_TRAIN_ARGS:-} \
    "${wandb_args[@]}" > "$log" 2>&1 || true
  local logdir ckpt fall surv
  logdir=$(grep -a "Logs are being stored in:" "$log" | awk '{print $NF}' | tail -1)
  ckpt=$(latest_ckpt "$logdir")
  fall=$(grep -a "metric/term_fall:" "$log" | tail -1 | awk '{print $2}')
  surv=$(awk -v f="${fall:-1}" 'BEGIN{printf "%.5f", 1 - f}')
  if [ -z "$ckpt" ]; then echo "[$CONDITION] $name produced no checkpoint - see $log" >&2; exit 1; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$surv" "$ckpt" "$ovr" >> "$MANIFEST"
  echo "[$CONDITION] $name done: survival=$surv ckpt=$ckpt" >&2
  if [ "${STOP_AFTER:-}" = "$name" ]; then
    echo "[$CONDITION] STOP_AFTER=$name reached; stopping (exit 99)." >&2; exit 99
  fi
  echo "$surv $ckpt"
}

run_eval() {  # NAME CKPT HW VHW_MIN
  local name=$1 ckpt=$2 hw=$3 vhw=$4
  local out="${OUT_DIR}/eval/${name}.json"
  if [ -f "$out" ]; then echo "[$CONDITION] eval $name exists" >&2; return 0; fi
  echo "[$CONDITION] eval $name ($ckpt)" >&2
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  local ovr; ovr=$(overrides_json "$hw" "$vhw" 0.0 0.0)
  XLA_PYTHON_CLIENT_MEM_FRACTION=$EVAL_MEM_FRACTION \
  $PYTHON_CMD scripts/eval_push_tolerance.py --env_name "$ENV_NAME" \
    --checkpoint "$ckpt" --overrides "$ovr" --episode_length 500 \
    --trigger_time 2.0 --impulses "$EVAL_IMPULSES" --num_trials "$EVAL_TRIALS" \
    --out "$out" > "${OUT_DIR}/logs/eval_${name}.log" 2>&1 \
    || echo "[$CONDITION] eval $name FAILED - see ${OUT_DIR}/logs/eval_${name}.log" >&2
}

stand_branch() {  # TAG HW VHW
  local tag=$1 hw=$2 vhw=$3 c r
  c=$(walk_ckpt "B_${tag}_p20")
  r=$(run_stage "S_${tag}_p08" "$hw" "$vhw" 0.0 8.0  100000000 1000 "$c"); c=$(ckpt_of "$r")
  r=$(run_stage "S_${tag}_p20" "$hw" "$vhw" 0.0 20.0 100000000 1000 "$c"); c=$(ckpt_of "$r")
  run_eval "S_${tag}" "$c" "$hw" "$vhw"
}
stand_branch w020 0.100 0.050
stand_branch w010 0.050 0.040
stand_branch w006 0.030 0.028

echo "[$CONDITION] stand experiment complete. Results in ${OUT_DIR}/" >&2
