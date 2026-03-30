#!/bin/bash

set -euo pipefail

# Wall-clock profiling run for sequential DreamerV3 training.
# Usage:
#   bash sequential_profile_pickplace_draweropen.sh minimal
#   bash sequential_profile_pickplace_draweropen.sh large

PROFILE_MODE="${1:-minimal}"

WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential"
TIMESTAMP="$(date +%m%d_%H%M%S)"
RUN_NAME="mw_seq_timing_${PROFILE_MODE}_${TIMESTAMP}"
LOGDIR="../logdir/sequential/${RUN_NAME}"

TASKS=(
  "metaworld_drawer-open-v3"
  "metaworld_pick-place-v3"
  "metaworld_compo-draweropen-pickplace"
)

if [[ "${PROFILE_MODE}" == "minimal" ]]; then
  CONFIGS=(
    "metaworld_minimal_16dof"
    "metaworld_minimal_16dof"
    "metaworld_minimal_16dof"
  )
  STEPS=(50000 50000 50000)
  DATASET_SIZES=(25000 25000 25000)
elif [[ "${PROFILE_MODE}" == "large" ]]; then
  CONFIGS=(
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
    "metaworld_compo_visual_200M_heavy_long"
  )
  STEPS=(50000 50000 50000)
  DATASET_SIZES=(25000 25000 25000)
else
  echo "Unknown PROFILE_MODE: ${PROFILE_MODE}"
  echo "Use one of: minimal, large"
  exit 1
fi

TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "=================================================="
echo ">>> Sequential DreamerV3 Wall-Clock Profiling"
echo ">>> Mode: ${PROFILE_MODE}"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> Logdir: ${LOGDIR}"
echo "=================================================="

python ../dreamer_sequential_timing.py \
  --tasks ${TASKS_ARG} \
  --configs ${CONFIGS_ARG} \
  --task-steps ${STEPS_ARG} \
  --dataset-sizes ${DATASET_SIZES_ARG} \
  --logdir ${LOGDIR} \
  --logger tensorboard \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-run-name ${RUN_NAME} \
  --eval-prev-video \
  --skip-config-check \
  --time-report ${LOGDIR}/time_profile.json \
  --time-topk 40

echo "Done. Timing report: ${LOGDIR}/time_profile.json"
