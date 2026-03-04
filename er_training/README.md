# DreamerV3 Sequential Training with Experience Replay (ER)

Sequential continual-learning pipeline for DreamerV3 that augments strictly
sequential training with **experience replay** and **architecture separation**.

## Key Differences from `dreamer_sequential.py`

| Feature | `dreamer_sequential.py` | `dreamer_sequential_er.py` |
|---|---|---|
| Previous task data | No access | Reservoir-sampled ER buffer per previous task |
| RSSM (encoder + dynamics + decoder) | Full model carried across tasks | Carried across tasks (same) |
| Reward / continue heads | Carried across tasks | **Reset from scratch** each new task |
| Actor-critic | Carried across tasks | **Reset from scratch** each new task |
| Checkpoints saved | Single full checkpoint per task | Separated: `rssm_task{N}.pt`, `heads_task{N}.pt`, `actor_critic_task{N}.pt` + `rssm_final.pt` |

## Architecture Separation

The Dreamer agent state dict is partitioned into three groups:

- **RSSM** (`_wm.encoder.*`, `_wm.dynamics.*`, `_wm.heads.decoder.*`): Shared world
  model backbone. Weights are **kept and continue updating** across all tasks.
- **Heads** (`_wm.heads.reward.*`, `_wm.heads.cont.*`): Task-specific prediction
  heads. **Freshly initialized** for each new task.
- **Actor-Critic** (`_task_behavior.*`): Policy and value networks. **Freshly
  initialized** for each new task.

When transitioning from task N to task N+1:
1. Save RSSM, heads, and actor-critic as separate checkpoints
2. Create a fresh agent for task N+1
3. Load **only RSSM** weights into the fresh agent
4. Heads and actor-critic start from random initialization

## Experience Replay

Reservoir sampling loads up to `--er-buffer-size` transitions from **each**
previous task's replay buffer. For task N, the training dataset merges:
- Current task episodes (growing during training)
- Up to `er-buffer-size` transitions from task 1, ..., task N-1

The `MergedEpisodes` class provides a live view over both sources so the
DreamerV3 sampling logic works without modification.

## Checkpoints (per task)

After each task completes, the following are saved in `task{N}_{name}/`:

| File | Contents |
|---|---|
| `rssm_task{N}.pt` | RSSM state dict (encoder + dynamics + decoder) |
| `heads_task{N}.pt` | Reward + continue head state dict |
| `actor_critic_task{N}.pt` | Actor + value + slow_value state dict |
| `checkpoint_task{N}.pt` | Full agent state dict + optimizer states |
| `latest.pt` | Rolling checkpoint for within-task resume |

After all tasks: `rssm_final.pt` (copy of the last task's RSSM).

## Resume Support

- **Between tasks**: Detects completed tasks from `sequential_progress.json`
  and skips them. Loads RSSM from the last completed task.
- **Within a task**: If `latest.pt` exists for the current task, loads full
  state (all components + optimizer states) and continues from where it stopped.

## Usage

```bash
python er_training/dreamer_sequential_er.py \
    --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \
    --configs metaworld_visual_heavy_long metaworld_visual_heavy_long \
    --task-steps 200000 800000 \
    --logdir ./logdir/sequential_er_run \
    --er-buffer-size 10000 \
    --er-seed 42 \
    --wandb-entity my-entity \
    --wandb-project my-project \
    --skip-config-check
```

Or use the training script:

```bash
cd training/
bash sequential_er_train_pickplace_draweropen.sh
```

## Using Saved Checkpoints with `actor_critic_training/`

The separated checkpoints can be used with the standalone actor-critic training
pipeline. For example, to train a new actor-critic for task 1 using the final
RSSM:

```bash
python actor_critic_training/train_actor_critic.py \
    --configs metaworld_visual_heavy_long \
    --task metaworld_drawer-open-v3 \
    --checkpoint logdir/rssm_final.pt \
    --traindir logdir/task1_metaworld_drawer-open-v3/train_eps \
    --train_steps 100000
```
