# Actor-Critic Training with Frozen DreamerV3 World Model

Train a **fresh actor-critic** (same architecture as DreamerV3's `ImagBehavior`)
from scratch using a **pre-trained, frozen world model**.  No real environment
interaction is needed — all training happens purely in the latent imagination
space of the world model.

## How it works

1. **Load checkpoint** — The trained `latest.pt` from a full DreamerV3 run is
   loaded.  The world model (encoder, RSSM dynamics, reward head, continuation
   head, decoder) is extracted and **all its parameters are frozen**.

2. **Replay buffer → posterior start states** — Real experience (`.npz` episodes
   from the training replay buffer) is passed through the frozen encoder and
   RSSM posterior to produce latent start states.  This is the *only* use of
   real data; no environment stepping occurs.

3. **Imagination rollout** — Starting from the posterior states, the RSSM
   dynamics prior rolls out imagined trajectories of length `imag_horizon`
   using the current actor policy.

4. **Actor-critic update** — The actor and critic are updated on the imagined
   trajectories using the same losses as standard DreamerV3 (lambda-return
   targets, EMA-normalised advantages, entropy regularisation).

## Usage

```bash
python actor_critic_training/train_actor_critic.py \
    --configs metaworld_default_light \
    --task metaworld_pick-place-v3 \
    --checkpoint /path/to/logdir/latest.pt \
    --traindir /path/to/logdir/train_eps \
    --logdir actor_critic_training/ac_logdir \
    --train_steps 100000 \
    --log_every 1000 \
    --save_every 10000 \
    --logger tensorboard
```

### Required arguments

| Argument | Description |
|---|---|
| `--checkpoint` | Path to the trained DreamerV3 `latest.pt` checkpoint. |
| `--traindir` | Directory containing replay buffer `.npz` episodes (e.g. `logdir/<run>/train_eps`). |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--configs` | `defaults` | Config presets from `configs.yaml` (e.g. `metaworld_default_light`). |
| `--logdir` | `actor_critic_training/ac_logdir` | Output directory for TB logs and AC checkpoints. |
| `--train_steps` | `100000` | Number of actor-critic training iterations. |
| `--log_every` | `1000` | Logging frequency (training steps). |
| `--save_every` | `10000` | Checkpoint saving frequency (training steps). |
| `--logger` | `tensorboard` | `tensorboard` or `wandb`. |
| `--resume` | – | Path to a previously saved AC checkpoint to resume from. |

Any key from `configs.yaml` can also be overridden on the command line
(e.g. `--imag_horizon 20`, `--discount 0.99`).

## Output

- **TensorBoard logs** in `<logdir>/` (or WandB if `--logger wandb`).
- **Checkpoints**: `ac_step_<N>.pt` and `ac_latest.pt` in `<logdir>/`.
  Each checkpoint contains:
  - `actor_critic_state_dict` — the actor, value, and slow-value network weights.
  - `actor_critic_optims` — optimizer states for the actor and value optimizers.

## Resuming training

```bash
python actor_critic_training/train_actor_critic.py \
    --configs metaworld_default_light \
    --checkpoint /path/to/logdir/latest.pt \
    --traindir /path/to/logdir/train_eps \
    --logdir actor_critic_training/ac_logdir \
    --resume actor_critic_training/ac_logdir/ac_latest.pt \
    --train_steps 200000
```
