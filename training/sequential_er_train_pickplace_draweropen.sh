#!/bin/bash

# Sequential ER training with cross-task evaluation for DreamerV3.
# Uses experience replay (ER) from previous tasks and separated architecture
# (shared RSSM + per-task reward/cont/actor-critic).

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential_ER"
RUN_NAME="mw_sequential_pickplace_draweropen_0304_er"
LOGDIR="../logdir/sequential_er/${RUN_NAME}"

# ER configuration
ER_BUFFER_RATIO=0.05
ER_SEED=42

# List of tasks to learn sequentially
TASKS=(
    "metaworld_drawer-open-v3"
    "metaworld_pick-place-v3"
    "metaworld_compo-draweropen-pickplace"
)

# Config profile for each task
CONFIGS=(
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
    "metaworld_compo_visual_200M_heavy_long"
)

# Training steps (env steps) for each task
STEPS=(
    200000
    800000
    1500000
)

# Dataset sizes for each task (for ER buffer calculation)
DATASET_SIZES=(
    100000
    400000
    750000
)

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "==================================================="
echo ">>> Sequential ER Training with Cross-Task Evaluation"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> ER buffer ratio: ${ER_BUFFER_RATIO}"
echo ">>> Logdir: ${LOGDIR}"
echo "==================================================="

python ../er_training/dreamer_sequential_er.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --dataset-sizes ${DATASET_SIZES_ARG} \
    --logdir ${LOGDIR} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --er-buffer-ratio ${ER_BUFFER_RATIO} \
    --er-seed ${ER_SEED} \
    --eval-prev-video \
    --skip-config-check
