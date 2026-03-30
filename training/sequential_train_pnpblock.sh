#!/bin/bash

# Sequential training with cross-task evaluation for DreamerV3.
# Unlike sequential_train.sh, this script uses a single Python process
# that handles all tasks and evaluates the current model on ALL previous
# tasks at every eval interval, enabling forgetting analysis.

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential"
RUN_NAME="mw_sequential_pnpblock_0327"
LOGDIR="../logdir/sequential/${RUN_NAME}"

# List of tasks to learn sequentially
TASKS=(
    "metaworld_pick-place-redblock-v3"
    "metaworld_pick-place-greenblock-v3"
    "metaworld_compo-pickplace-block"
)

# Config profile for each task
CONFIGS=(
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
)

# Training steps (env steps) for each task
STEPS=(
    800000
    500000
    800000
)

# Dataset sizes for each task (for ER buffer calculation)
DATASET_SIZES=(
    400000
    250000
    400000
)

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "=================================================="
echo ">>> Sequential Training with Cross-Task Evaluation"
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
