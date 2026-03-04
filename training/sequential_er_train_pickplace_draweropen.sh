#!/bin/bash

# Sequential ER training with cross-task evaluation for DreamerV3.
# Uses experience replay (ER) from previous tasks and separated architecture
# (shared RSSM + per-task reward/cont/actor-critic).

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential_ER"
RUN_NAME="mw_sequential_er_0303"
LOGDIR="../logdir/${RUN_NAME}"

# ER configuration
ER_BUFFER_SIZE=10000   # transitions per previous task
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

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"

echo "==================================================="
echo ">>> Sequential ER Training with Cross-Task Evaluation"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> ER buffer size: ${ER_BUFFER_SIZE}"
echo "==================================================="

python ../er_training/dreamer_sequential_er.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --logdir ${LOGDIR} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --er-buffer-size ${ER_BUFFER_SIZE} \
    --er-seed ${ER_SEED} \
    --eval-prev-video \
    --skip-config-check
