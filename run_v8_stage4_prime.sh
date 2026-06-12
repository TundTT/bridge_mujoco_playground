#!/bin/bash
# v8 stage 4-prime: binding virtual width with foot_off_virtual penalty.
#
# Loads from the v6 stage-4 (0.16m, 92.2%) checkpoint and retrains at the
# same physical width with vhw_min=0.10. Two reward changes force genuine
# narrow-stance learning:
#   1. foothold_multiplier floor fades from 0.5 (vhw=0.15) to 0.0 (vhw=0.10)
#      so wide stance earns zero frontier income at narrow virtual widths.
#   2. foot_off_virtual (-30.0): per-contact-step linear hinge on lateral
#      overshoot beyond virtual_hw — dense gradient where quality clips at 0.
#
# Goal: policy learns to narrow its stance to fit virtual_hw. Promotion gate:
#   - term_success >= 0.80 at zero-shot 0.13m eval
#   - metric/max_foot_y tracking vhw (no longer flat across widths)
#   - metric/touchdown_count remains >= ~120/ep (no bounding)
#   - reward/foot_off_virtual trending toward 0
#
# Usage:
#   ./run_v8_stage4_prime.sh

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

STAGE4_CKPT="logs/Go1BridgeCrossing-20260612-073809/checkpoints/000200540160"
WANDB_PROJECT="bridge_crossing_v8"
LOGFILE="/tmp/bridge_v8_stage4_prime.log"

echo "=== v8 Stage 4-prime: 0.16m physical, vhw_min=0.10, 300M steps ===" >&2
echo "=== Loading from: ${STAGE4_CKPT} ===" >&2

train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 300000000 \
  --load_checkpoint_path "${STAGE4_CKPT}" \
  --playground_config_overrides '{"bridge_half_width": 0.16, "virtual_hw_min": 0.10}' \
  --entropy_cost 0.01 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "v8_stage4_prime_0.16m" \
  --wandb_group "v8_curriculum" \
  --use_wandb \
  2>&1 | tee "${LOGFILE}" >/dev/null

echo "=== Stage 4-prime done ===" >&2
grep "metric/term_success:\|metric/touchdown_count:\|metric/abduction_sat\|reward/foot_off_virtual:" \
  "${LOGFILE}" | tail -4 >&2
