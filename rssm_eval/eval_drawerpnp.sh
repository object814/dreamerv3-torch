#!/bin/bash

# Evaluate RSSM world-model fidelity for the drawer-open -> pick-place -> compo
# sequential training run (mw_sequential_drawerpnp_0330).
#
# For each task n, loads that task's expert actor, collects successful rollouts,
# then evaluates the RSSM checkpoint saved after every subsequent task m >= n.
# Outputs: reconstruction GIFs and bar-chart plots under OUTDIR.

# ── Configuration ──────────────────────────────────────────────────────────

RUN_NAME="mw_sequential_drawerpnp_0330"
LOGDIR="../logdir/sequential/${RUN_NAME}"
OUTDIR="../evaldir/sequential/${RUN_NAME}/rssm_eval"

# Must match the training task list exactly
TASKS=(
    "metaworld_drawer-open-v3"
    "metaworld_pick-place-v3"
    "metaworld_compo-draweropen-pickplace"
)

CONFIGS=(
    "metaworld_visual_200M_heavy_long_speedup"
    "metaworld_visual_200M_heavy_long_speedup"
    "metaworld_visual_200M_heavy_long_speedup"
)

# Number of successful expert demos to collect per task
NUM_DEMOS=10

# Maximum environment episodes to attempt when collecting demos
MAX_ATTEMPTS=50

# GIF frame rate
GIF_FPS=15

# ── Run ────────────────────────────────────────────────────────────────────

TASKS_ARG="${TASKS[*]}"
CONFIGS_ARG="${CONFIGS[*]}"

echo "=================================================="
echo ">>> RSSM Evaluation: ${RUN_NAME}"
echo ">>> Tasks:      ${TASKS_ARG}"
echo ">>> Logdir:     ${LOGDIR}"
echo ">>> Outdir:     ${OUTDIR}"
echo ">>> Demos/task: ${NUM_DEMOS}"
echo "=================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/evaluate_rssm.py" \
    --tasks ${TASKS_ARG} \
    --configs ${CONFIGS_ARG} \
    --logdir "${LOGDIR}" \
    --outdir "${OUTDIR}" \
    --num-demos ${NUM_DEMOS} \
    --max-attempts ${MAX_ATTEMPTS} \
    --gif-fps ${GIF_FPS}
