#!/bin/bash

# Sequential training with Experience Replay (ER) for DreamerV3.
# Uses the disentangled architecture: shared RSSM + per-task heads/actor-critic.
# ER mixes old-task episodes into the training stream to prevent
# catastrophic forgetting of the shared RSSM backbone.

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="Metaworld_Dreamerv3_Sequential_ER"
RUN_NAME="mw_sequential_er_pnpblock_$(date +%m%d)"
LOGDIR="../logdir/sequential_er/${RUN_NAME}"

# List of tasks to learn sequentially
TASKS=(
    "metaworld_pick-place-redblock-v3"
    "metaworld_pick-place-greenblock-v3"
    "metaworld_compo-pickplace-block"
)

# Config profile for each task
CONFIGS=(
    "metaworld_visual_200M_heavy_long_speedup"
    "metaworld_visual_200M_heavy_long_speedup"
    "metaworld_visual_200M_heavy_long_speedup"
)

# Training steps (env steps) for each task
STEPS=(
    600000
    600000
    800000
)

# Dataset sizes for each task
DATASET_SIZES=(
    $((${STEPS[0]} / 4))
    $((${STEPS[1]} / 4))
    $((${STEPS[2]} / 4))
)

# --- ER hyperparameters ---
# Buffer ratio: fraction of each previous task's dataset_size to retain as replay
#   E.g., 0.05 = 5% of that task's dataset_size
#   Higher = more old-task data mixed in, better forgetting prevention but more memory
ER_BUFFER_RATIO=0.05

# Seed for reservoir sampling reproducibility
ER_SEED=42

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "=================================================="
echo ">>> Sequential ER Training with Cross-Task Evaluation"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> ER buffer ratio: ${ER_BUFFER_RATIO}"
echo ">>> ER seed: ${ER_SEED}"
echo ">>> Logdir: ${LOGDIR}"
echo "=================================================="

python ../er_training/dreamer_sequential_er.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --dataset-sizes ${DATASET_SIZES_ARG} \
    --logdir ${LOGDIR} \
    --er-buffer-ratio ${ER_BUFFER_RATIO} \
    --er-seed ${ER_SEED} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --eval-prev-video \
    --skip-config-check
