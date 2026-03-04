#!/bin/bash

# ============================================================
# Sanity-check script for the Sequential ER pipeline.
#
# Runs 2 tasks end-to-end with the FULL model architecture
# (metaworld_visual_200M_heavy_long) but with tiny step budgets
# so the whole thing finishes in minutes.
#
# What this validates:
#   - Environment creation & parallel workers
#   - Prefill, training loop, eval loop
#   - Experience replay buffer loading for task 2
#   - Architecture separation (RSSM kept, heads/AC reset)
#   - Checkpoint saving (rssm, heads, actor_critic, rssm_final)
#   - Resume detection (re-run this script to test resume)
#   - WandB logging
# ============================================================

set -euo pipefail

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential_ER"
RUN_NAME="mw_sequential_er_test_0304"
LOGDIR="../logdir/sequential_er/${RUN_NAME}"

# ER configuration
ER_BUFFER_RATIO=0.1    # 10% of each previous task's dataset_size (aggressive for sanity check)
ER_SEED=42

# Two quick tasks (same env, just to test the pipeline)
TASKS=(
    "metaworld_drawer-open-v3"
    "metaworld_pick-place-v3"
)

# Use the full heavy config — overrides below make it fast
CONFIGS=(
    "metaworld_visual_200M_heavy_long"
    "metaworld_visual_200M_heavy_long"
)

# Tiny step budgets (env steps)
STEPS=(
    1000
    1000
)

# Small dataset sizes for sanity check
DATASET_SIZES=(
    500
    500
)

TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"
STEPS_ARG="${STEPS[*]}"
DATASET_SIZES_ARG="${DATASET_SIZES[*]}"

echo "==================================================="
echo ">>> SANITY CHECK: Sequential ER Pipeline"
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
    --skip-config-check \
    --prefill 10 \
    --eval_episode_num 2 \
    --pretrain 1 \
    --envs 2 \
    --parallel False \
    --eval_every 1000 \
    --log_every 100

echo ""
echo "==================================================="
echo ">>> SANITY CHECK: Verifying output files..."
echo "==================================================="

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
    fi
}

# Task 1 checkpoints
T1_DIR="${LOGDIR}/task1_metaworld_drawer-open-v3"
check_file "${T1_DIR}/rssm_task1.pt"
check_file "${T1_DIR}/heads_task1.pt"
check_file "${T1_DIR}/actor_critic_task1.pt"
check_file "${T1_DIR}/checkpoint_task1.pt"

# Task 2 checkpoints
T2_DIR="${LOGDIR}/task2_metaworld_pick-place-v3"
check_file "${T2_DIR}/rssm_task2.pt"
check_file "${T2_DIR}/heads_task2.pt"
check_file "${T2_DIR}/actor_critic_task2.pt"
check_file "${T2_DIR}/checkpoint_task2.pt"

# Final RSSM
check_file "${LOGDIR}/rssm_final.pt"

# Progress file
check_file "${LOGDIR}/sequential_progress.json"

echo ""
echo ">>> Sequential progress:"
cat "${LOGDIR}/sequential_progress.json" 2>/dev/null || echo "  (not found)"
echo ""
echo "==================================================="
echo ">>> SANITY CHECK COMPLETE"
echo "==================================================="
