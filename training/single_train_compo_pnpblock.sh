#!/bin/bash

# Single task training for DreamerV3.
# This mirrors single_train_box.sh, but targets the composed pick-place
# block task.

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="Metaworld_Dreamerv3_Single"
RUN_NAME="mw_single_compo_pnpblock_$(date +%m%d)"
LOGDIR="../logdir/single/${RUN_NAME}"

# Task environment name
TASKS=(
    "metaworld_compo-pickplace-block"
)

# Config profile
CONFIGS=(
    "metaworld_visual_200M_heavy_long_speedup"
)

# Training steps (env steps)
STEPS=(
    800000
)

# Dataset size
DATASET_SIZES=(
    $((STEPS[0] / 4))
)

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "=================================================="
echo ">>> Single Task Training"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> Logdir: ${LOGDIR}"
echo "=================================================="

python ../dreamer_sequential.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --dataset-sizes ${DATASET_SIZES_ARG} \
    --logdir ${LOGDIR} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --eval-prev-video \
    --skip-config-check