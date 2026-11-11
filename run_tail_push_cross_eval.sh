#!/bin/bash
# Cross-width push-tolerance evals for the tail push experiment.
#
# The width-branch evals (A04_w020, A07_w010, B_w020, B_w010) use the policy
# trained at that width. If a condition stalls at one width (e.g. locked tail
# at 0.10 m), its J50 there measures "can't walk", not "can't take a push".
# This script also evaluates each condition's narrowest-beam policies
# (A11 = before push training, B_w006_p20 = after) on the 0.20 m and 0.10 m
# beams, so every condition is compared with a policy that can walk.
#
# Writes results/tail_push/<cond>/eval/X_<stage>_at_<width>.json.
# Run after run_tail_push_experiment.sh has finished for the condition.
#   CONDITION=active CUDA_VISIBLE_DEVICES=0 ./run_tail_push_cross_eval.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_DIR:-$SCRIPT_DIR}"

CONDITION="${CONDITION:?set CONDITION=active|locked|notail}"
EVAL_TRIALS="${EVAL_TRIALS:-256}"
EVAL_IMPULSES="${EVAL_IMPULSES:-0,2,4,6,8,10,12,14,16,18,20,24,28}"
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
ckpt_of_stage() { awk -v s="$1" -F'\t' '$1==s {print $3}' "$MANIFEST" | tail -1; }

for stage in A11 B_w006_p20; do
  ckpt=$(ckpt_of_stage "$stage")
  if [ -z "$ckpt" ]; then echo "[$CONDITION] $stage not in manifest; skipping" >&2; continue; fi
  #          tag   hw    vhw_min ep
  for spec in "w020 0.100 0.050 1500" "w010 0.050 0.040 1500"; do
    read -r tag hw vhw ep <<< "$spec"
    out="${OUT_DIR}/eval/X_${stage}_at_${tag}.json"
    if [ -f "$out" ]; then echo "[$CONDITION] $out exists" >&2; continue; fi
    ovr="{${COND_OVR}, \"bridge_half_width\": $hw, \"virtual_hw_min\": $vhw, \"push_config.impulse\": [0.0, 0.0]}"
    echo "[$CONDITION] eval $stage at $tag ($ckpt)" >&2
    $PYTHON_CMD scripts/eval_push_tolerance.py --env_name "$ENV_NAME" \
      --checkpoint "$ckpt" --overrides "$ovr" --episode_length "$ep" \
      --impulses "$EVAL_IMPULSES" --num_trials "$EVAL_TRIALS" \
      --out "$out" > "${OUT_DIR}/logs/eval_X_${stage}_at_${tag}.log" 2>&1 \
      || echo "[$CONDITION] eval X_${stage}_at_${tag} FAILED" >&2
  done
done
echo "[$CONDITION] cross-width evals done" >&2
