#!/bin/bash

# Configuration
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="CCLB_Dreamerv3_Sequential"
BASE_RUN_NAME="mw_sequential_debug_0217"
BASE_LOGDIR="./logdir/${BASE_RUN_NAME}"

# List of tasks to learn sequentially
# format: "task_name"
TASKS=(
    # "metaworld_drawer-open-v3"
    "metaworld_pick-place-v3"
    "metaworld_compo-draweropen-pickplace"
)
# List of configurations to use for each task
CONFIG=(
    # "debug"
    "debug"
    "debug"
)

# List of steps to train for each task (this overwrites the steps in the config files)
STEPS=(
    # 200
    200
    200
)

# Initialise variables
PREV_CHECKPOINT="/Metaworld/third_party/dreamerv3/logdir/mw_sequential_debug_0217/task2_metaworld_pick-place-v3/checkpoint_task2.pt"

for i in "${!TASKS[@]}"; do
    TASK=${TASKS[$i]}
    TASK_ID=$((i+1))
    
    # Create unique run name and log directory for this specific task
    RUN_NAME="${BASE_RUN_NAME}/task${TASK_ID}_${TASK}"
    LOGDIR="${BASE_LOGDIR}/task${TASK_ID}_${TASK}"
    
    echo "=================================================="
    echo ">>> Sequential Training: Starting Task ${TASK_ID}: ${TASK}"
    echo ">>> Sequential Training: Log Directory: ${LOGDIR}"
    echo "=================================================="

    # Construct the command
    CMD="python dreamer.py \
        --configs ${CONFIG[$i]} \
        --task ${TASK} \
        --logdir ${LOGDIR} \
        --steps ${STEPS[$i]} \
        --wandb-entity ${WANDB_ENTITY} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-run-name ${RUN_NAME} \
        --skip-config-check"

    # If there is a previous checkpoint, add the flag to load it
    if [ ! -z "$PREV_CHECKPOINT" ]; then
        echo ">>> Sequential Training: Sequential Learning: Loading weights from previous task: $PREV_CHECKPOINT"
        CMD="$CMD --from-checkpoint ${PREV_CHECKPOINT}"
        CMD="$CMD --skip-pretrain"
    else
        echo ">>> Sequential Training: First task starting from scratch."
    fi

    # Run the training
    echo ">>> Sequential Training: Running command:"
    echo "$CMD"
    eval $CMD

    # Check if the run was successful and store the checkpoint for the next iteration
    CURRENT_CHECKPOINT="${LOGDIR}/latest.pt"
    
    if [ -f "$CURRENT_CHECKPOINT" ]; then
        echo ">>> Sequential Training: Task ${TASK_ID} completed. Checkpoint saved at ${CURRENT_CHECKPOINT}"
        # Copy the checkpoint to the next task's log directory
        NEXT_LOGDIR=$(dirname "${LOGDIR}")/task$((TASK_ID+1))_$(basename "${TASKS[$TASK_ID]}")
        mkdir -p "$NEXT_LOGDIR"
        echo ">>> Sequential Training: Preparing for next task... Moving checkpoint to $NEXT_LOGDIR"
        cp "$CURRENT_CHECKPOINT" "$NEXT_LOGDIR/last_task_latest.pt"
        PREV_CHECKPOINT="$NEXT_LOGDIR/last_task_latest.pt"
        
        # Copy checkpoint to a specific name for safekeeping
        cp "$CURRENT_CHECKPOINT" "${LOGDIR}/checkpoint_task${TASK_ID}.pt"
    else
        echo ">>> Sequential Training: Error: Checkpoint not found at ${CURRENT_CHECKPOINT}. Stopping sequence."
        exit 1
    fi
    
    echo ">>> Sequential Training: Done with Task ${TASK_ID}. Moving to next..."
    echo ""
done

echo ">>> Sequential Training: All tasks completed successfully."