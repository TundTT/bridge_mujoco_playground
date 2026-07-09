#!/bin/bash
# Tight-start cylindrical-beam transfer curriculum from the 0.0025m flat bridge
# policy, using the cylinder-aware reward branch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-cylinder}"
export WANDB_GROUP="${WANDB_GROUP:-cylinder_tight_from_2p5mm_flat}"
export STAGE_DIAMETERS_OVERRIDE="${STAGE_DIAMETERS_OVERRIDE:-0.05 0.04 0.03 0.02 0.01 0.0075 0.005 0.0025}"
export RUN_LOG_DIR="${RUN_LOG_DIR:-$SCRIPT_DIR/logs/cylinder_tight_$(date +%Y%m%d-%H%M%S)}"

exec "$SCRIPT_DIR/run_cylinder_beam_curriculum_clean.sh"
