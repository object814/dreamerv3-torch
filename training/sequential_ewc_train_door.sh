#!/bin/bash

# Sequential EWC training with cross-task evaluation for DreamerV3.
# Uses Elastic Weight Consolidation (EWC) to protect important RSSM parameters
# from previous tasks, with separated architecture
# (shared RSSM + per-task reward/cont/actor-critic).

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential_EWC"
RUN_NAME="mw_sequential_door_0305_ewc"
LOGDIR="../logdir/sequential_ewc/${RUN_NAME}"

# EWC configuration
EWC_LAMBDA=5000.0
EWC_FISHER_BATCHES=50

# List of tasks to learn sequentially
TASKS=(
    "metaworld_door-open-v3"
    "metaworld_door-close-v3"
    "metaworld_compo-dooropen-doorclose"
)

# Config profile for each task
CONFIGS=(
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
    "metaworld_compo_visual_200M_heavy_long"
)

# Training steps (env steps) for each task
STEPS=(
    150000
    500000
    1000000
)

# Dataset sizes for each task
DATASET_SIZES=(
    75000
    250000
    500000
)

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "==================================================="
echo ">>> Sequential EWC Training with Cross-Task Evaluation"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> EWC lambda: ${EWC_LAMBDA}"
echo ">>> EWC Fisher batches: ${EWC_FISHER_BATCHES}"
echo ">>> Logdir: ${LOGDIR}"
echo "==================================================="

python ../ewc_training/dreamer_sequential_ewc.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --dataset-sizes ${DATASET_SIZES_ARG} \
    --logdir ${LOGDIR} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --ewc-lambda ${EWC_LAMBDA} \
    --ewc-fisher-batches ${EWC_FISHER_BATCHES} \
    --eval-prev-video \
    --skip-config-check
