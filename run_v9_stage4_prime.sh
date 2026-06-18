#!/bin/bash
# v9 stage 4-prime: full 300M transfer run with all reward fixes.
#
# Loads from v6 stage-4 (0.16m, 92.2%) checkpoint. Four reward changes vs v6:
#   1. quality_mean over all 4 feet (not contact-only) — closes the lifting exploit
#   2. foot_off_virtual (-15): contact-independent, all feet every step
#   3. feet_stale_air (-2.0): penalises legs held >0.4s (training-wheels fix)
#   4. pose abduction weight 0.1 (was 1.0) — removes adduction tax
#
# Goal: robot develops genuine narrow-stance locomotion at virtual widths down
# to 0.10m, then transfers to physical 0.13m.
#
# Promotion gate: zero-shot success >= 0.70 at 0.13m physical.
#
# Usage:
#   ./run_v9_stage4_prime.sh

set -euo pipefail

cd /home/tund/notebooks/bridge_mujoco_playground
source .venv/bin/activate

STAGE4_CKPT="logs/Go1BridgeCrossing-20260612-073809/checkpoints/000200540160"
WANDB_PROJECT="bridge_crossing_v9"
LOGFILE="/tmp/bridge_v9_stage4_prime.log"

echo "=== v9 Stage 4-prime: 0.16m physical, vhw_min=0.10, 300M steps ===" >&2
echo "=== Loading from: ${STAGE4_CKPT} ===" >&2

train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 300000000 \
  --load_checkpoint_path "${STAGE4_CKPT}" \
  --playground_config_overrides '{"bridge_half_width": 0.16, "virtual_hw_min": 0.10}' \
  --entropy_cost 0.01 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "v9_stage4_prime_0.16m" \
  --wandb_group "v9_curriculum" \
  --use_wandb \
  2>&1 | tee "${LOGFILE}" >/dev/null

echo "=== Stage 4-prime done ===" >&2
grep "metric/term_success:\|metric/touchdown_count:\|metric/contact_left:\|metric/contact_right:\|reward/foot_off_virtual:\|reward/feet_stale_air:" \
  "${LOGFILE}" | tail -6 >&2
