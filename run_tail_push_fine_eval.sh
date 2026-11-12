#!/bin/bash
# Fine push sweep for the tail push experiment.
#
# The main sweep (2 N*s steps) is too coarse around the measured J50s
# (3-8 N*s) to separate active from locked tail. This re-evaluates the key
# policies with 0.5 N*s steps and 512 trials into results/tail_push/<cond>/eval_fine/.
# Run after run_tail_push_experiment.sh has finished for the condition.
#   CONDITION=active CUDA_VISIBLE_DEVICES=0 ./run_tail_push_fine_eval.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR:-$SCRIPT_DIR}"

CONDITION="${CONDITION:?set CONDITION=active|locked|notail}"
EVAL_TRIALS="${EVAL_TRIALS:-512}"
EVAL_IMPULSES="${EVAL_IMPULSES:-0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12}"
PYTHON_CMD="${PYTHON_CMD:-python}"

case "$CONDITION" in
  active) ENV_NAME=Go1BridgeCrossingTail; COND_OVR='"tail_mode": "active"' ;;
  locked) ENV_NAME=Go1BridgeCrossingTail; COND_OVR='"tail_mode": "locked"' ;;
  notail) ENV_NAME=Go1BridgeCrossing
          COND_OVR='"push_config.enable": true, "reset_info_on_new_episode": true' ;;
  *) echo "unknown CONDITION=$CONDITION" >&2; exit 1 ;;
esac

if [ -z "${SKIP_VENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

OUT_DIR="results/tail_push/${CONDITION}"
MANIFEST="${OUT_DIR}/manifest.tsv"
mkdir -p "${OUT_DIR}/eval_fine" "${OUT_DIR}/logs"
ckpt_of_stage() { awk -v s="$1" -F'\t' '$1==s {print $3}' "$MANIFEST" | tail -1; }

#     eval name          stage        hw    vhw_min ep
SPECS=(
  "A04_w020            A04          0.100 0.050 1500"
  "A07_w010            A07          0.050 0.040 1500"
  "A11_w006            A11          0.030 0.028 2000"
  "B_w020              B_w020_p20   0.100 0.050 1500"
  "B_w010              B_w010_p20   0.050 0.040 1500"
  "B_w006              B_w006_p20   0.030 0.028 2000"
  "X_A11_at_w010       A11          0.050 0.040 1500"
  "X_B_w006_p20_at_w010 B_w006_p20  0.050 0.040 1500"
)
for spec in "${SPECS[@]}"; do
  read -r name stage hw vhw ep <<< "$spec"
  out="${OUT_DIR}/eval_fine/${name}.json"
  if [ -f "$out" ]; then echo "[$CONDITION] $out exists" >&2; continue; fi
  ckpt=$(ckpt_of_stage "$stage")
  if [ -z "$ckpt" ]; then echo "[$CONDITION] $stage not in manifest; skipping" >&2; continue; fi
  ovr="{${COND_OVR}, \"bridge_half_width\": $hw, \"virtual_hw_min\": $vhw, \"push_config.impulse\": [0.0, 0.0]}"
  echo "[$CONDITION] fine eval $name ($ckpt)" >&2
  $PYTHON_CMD scripts/eval_push_tolerance.py --env_name "$ENV_NAME" \
    --checkpoint "$ckpt" --overrides "$ovr" --episode_length "$ep" \
    --impulses "$EVAL_IMPULSES" --num_trials "$EVAL_TRIALS" --seed 1 \
    --out "$out" > "${OUT_DIR}/logs/evalfine_${name}.log" 2>&1 \
    || echo "[$CONDITION] fine eval $name FAILED" >&2
done
echo "[$CONDITION] fine evals done" >&2
