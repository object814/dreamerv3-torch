#!/bin/bash

# Sequential training with EWC (Elastic Weight Consolidation) for DreamerV3.
# Uses the disentangled architecture: shared RSSM + per-task heads/actor-critic.
# EWC penalty is applied ONLY to the shared RSSM backbone to prevent
# catastrophic forgetting of learned dynamics/representations.

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="Metaworld_Dreamerv3_Sequential_EWC"
RUN_NAME="mw_sequential_ewc_pnpblock_$(date +%m%d)"
LOGDIR="../logdir/sequential_ewc/${RUN_NAME}"

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
    800000
    500000
    800000
)

# Dataset sizes for each task (for ER buffer calculation)
DATASET_SIZES=(
    $((${STEPS[0]} / 4))
    $((${STEPS[1]} / 4))
    $((${STEPS[2]} / 4))
)

# --- EWC hyperparameters ---
# Lambda: regularization strength (how strongly to protect old task weights)
#   Higher = more protection against forgetting, but less plasticity for new tasks
#   Recommended range for MetaWorld vision-based: [1000, 5000, 10000]
EWC_LAMBDA=5000.0

# Fisher batches: number of mini-batches used to estimate parameter importance
#   More batches = more accurate Fisher estimate, but slower task transitions
#   100 batches is typically sufficient (~1600 sequences at batch_size=16)
EWC_FISHER_BATCHES=100

# Build space-separated argument strings
TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "=================================================="
echo ">>> Sequential EWC Training with Cross-Task Evaluation"
echo ">>> Tasks: ${TASKS_ARG}"
echo ">>> Configs: ${CONFIGS_ARG}"
echo ">>> Steps: ${STEPS_ARG}"
echo ">>> Dataset sizes: ${DATASET_SIZES_ARG}"
echo ">>> EWC lambda: ${EWC_LAMBDA}"
echo ">>> EWC Fisher batches: ${EWC_FISHER_BATCHES}"
echo ">>> Logdir: ${LOGDIR}"
echo "=================================================="

python ../ewc_training/dreamer_sequential_ewc.py \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --task-steps ${STEPS_ARG} \
    --dataset-sizes ${DATASET_SIZES_ARG} \
    --logdir ${LOGDIR} \
    --ewc-lambda ${EWC_LAMBDA} \
    --ewc-fisher-batches ${EWC_FISHER_BATCHES} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME} \
    --eval-prev-video \
    --skip-config-check
