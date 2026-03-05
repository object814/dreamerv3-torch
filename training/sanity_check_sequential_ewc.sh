#!/bin/bash

# ============================================================
# Sanity-check script for the Sequential EWC pipeline.
#
# Runs 2 tasks end-to-end with the FULL model architecture
# (metaworld_visual_200M_heavy_long) but with tiny step budgets
# so the whole thing finishes in minutes.
#
# What this validates:
#   - Environment creation & parallel workers
#   - Prefill, training loop, eval loop
#   - EWC Fisher computation at task boundary
#   - EWC penalty applied during task 2 training
#   - Architecture separation (RSSM kept, heads/AC reset)
#   - Checkpoint saving (rssm, heads, actor_critic, ewc_state, rssm_final)
#   - Resume detection (re-run this script to test resume)
#   - WandB logging (ewc_loss, model_loss_base, fisher stats)
# ============================================================

set -euo pipefail

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential_EWC"
RUN_NAME="mw_sequential_ewc_test_0305"
LOGDIR="../logdir/sequential_ewc/${RUN_NAME}"

# EWC configuration (small Fisher batches for speed)
EWC_LAMBDA=5000.0
EWC_FISHER_BATCHES=3

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
echo ">>> SANITY CHECK: Sequential EWC Pipeline"
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
check_file "${T1_DIR}/ewc_state_task1.pt"

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
echo ">>> EWC state check:"
python -c "
import torch, sys
ewc_path = '${T1_DIR}/ewc_state_task1.pt'
try:
    state = torch.load(ewc_path, map_location='cpu')
    n_tasks = state['num_tasks_consolidated']
    lam = state['lambda_ewc']
    n_params = sum(v.numel() for v in state['regularization_terms'][0]['importance'].values())
    fisher_mean = sum(v.mean().item() for v in state['regularization_terms'][0]['importance'].values()) / max(len(state['regularization_terms'][0]['importance']), 1)
    print(f'  [OK] EWC state: {n_tasks} task(s) consolidated, lambda={lam}, {n_params:,} params, mean Fisher={fisher_mean:.2e}')
except Exception as e:
    print(f'  [ERROR] Could not load EWC state: {e}')
    sys.exit(1)
"

echo ""
echo "==================================================="
echo ">>> SANITY CHECK COMPLETE"
echo "==================================================="
