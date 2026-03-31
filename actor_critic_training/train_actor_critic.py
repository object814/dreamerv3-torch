"""
Standalone actor-critic training using a frozen, pre-trained DreamerV3 world model.

This script:
  1. Loads a trained DreamerV3 checkpoint (latest.pt).
  2. Extracts the world model (encoder, RSSM dynamics, reward head, continue head)
     and freezes all its parameters.
  3. Creates a *fresh* actor-critic (ImagBehavior) from scratch.
  4. Uses the replay buffer data to compute posterior start states via the frozen
     world model, then trains the actor-critic purely in imagination.

No real environment interaction is needed — all training happens in the latent
"dream" space of the pre-trained world model.

Usage example:
  python actor_critic_training/train_actor_critic.py \
      --configs metaworld_default_light \
      --task metaworld_pick-place-v3 \
      --checkpoint /path/to/logdir/latest.pt \
      --traindir /path/to/logdir/train_eps \
      --train_steps 100000 \
      --log_every 1000 \
      --save_every 10000 \
      --logdir actor_critic_training/logdir
"""

import argparse
import copy
import os
import pathlib
import sys
import time

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — reuse the parent dreamerv3 package directly
# ---------------------------------------------------------------------------
DREAMER_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DREAMER_DIR))

import third_party.dreamerv3.models_bk as models_bk
import networks
import tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
to_np = lambda x: x.detach().cpu().numpy()


def make_dataset(episodes, config):
    """Reuse the standard DreamerV3 dataset builder."""
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def precompute_embeddings(wm, episodes, config):
    """
    Pre-compute encoder embeddings for every episode using the frozen world
    model.  The embedding is stored back into each episode dict under the key
    ``"cached_embed"`` so that the standard sampling / batching pipeline
    automatically slices and stacks it alongside the other arrays.

    This eliminates the expensive CNN encoder forward pass on every training
    step (the encoder is frozen, so its output never changes).
    """
    print(f"Pre-computing encoder embeddings for {len(episodes)} episodes …")
    t0 = time.time()
    for i, (ep_id, episode) in enumerate(episodes.items()):
        # Build a fake batch of shape (1, T, …)
        obs = {k: v[np.newaxis] for k, v in episode.items()}
        with torch.no_grad():
            data = wm.preprocess(obs)
            embed = wm.encoder(data)          # (1, T, embed_dim)
        episode["cached_embed"] = embed.squeeze(0).cpu().numpy()  # (T, embed_dim)
        del obs, data, embed
        # Drop raw images now that we have embeddings — the encoder is frozen
        # so these will never be needed again.  This is the main memory saving.
        if "image" in episode:
            del episode["image"]
        if (i + 1) % 200 == 0:
            print(f"  Encoded {i + 1}/{len(episodes)} episodes …")
    elapsed = time.time() - t0
    print(f"Pre-computed all embeddings in {elapsed:.1f}s")


def load_world_model_from_checkpoint(checkpoint_path, config):
    """
    Build a WorldModel matching *config*, load the trained weights from
    *checkpoint_path*, and freeze every parameter.

    Returns
    -------
    wm : models.WorldModel   (frozen, on config.device)
    """
    # We need obs_space to instantiate the world model.  The obs_space shapes
    # are baked into the encoder / decoder architecture, so we reconstruct a
    # minimal Gymnasium-style space dict from the checkpoint tensor shapes.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    agent_sd = checkpoint["agent_state_dict"]

    # --- Infer observation space shapes from checkpoint keys ----------------
    # The world model is stored under _wm.* (with possible _orig_mod. prefix
    # from torch.compile).  We look for decoder output heads to infer shapes.
    import gymnasium.spaces as spaces

    obs_shapes = _infer_obs_shapes(agent_sd)
    obs_space = spaces.Dict(
        {k: spaces.Box(low=-np.inf, high=np.inf, shape=v) for k, v in obs_shapes.items()}
    )

    # --- Construct world model & load weights -------------------------------
    wm = models_bk.WorldModel(obs_space, None, 0, config)

    # Build a mapping from the checkpoint keys → our world model keys.
    wm_sd = {}
    for ck, cv in agent_sd.items():
        # Checkpoint keys look like: _wm._orig_mod.dynamics.W  (compiled)
        # or _wm.dynamics.W (non-compiled).  Strip prefixes to get the
        # local WorldModel key.
        for prefix in ("_wm._orig_mod.", "_wm."):
            if ck.startswith(prefix):
                local_key = ck[len(prefix):]
                wm_sd[local_key] = cv
                break

    missing, unexpected = wm.load_state_dict(wm_sd, strict=False)
    # The model optimizer state is inside _model_opt which is not a nn.Module
    # param — it will show up as missing.  That's fine; we won't train the WM.
    if missing:
        # Filter out optimizer-related missing keys (they are not nn.Parameter)
        real_missing = [k for k in missing if not k.startswith("_model_opt")]
        if real_missing:
            print(f"[WARNING] Missing WM keys: {real_missing}")
    if unexpected:
        print(f"[WARNING] Unexpected WM keys: {unexpected}")

    wm = wm.to(config.device)
    wm.eval()
    wm.requires_grad_(False)
    print(f"Loaded and froze world model from {checkpoint_path}")
    return wm


def _infer_obs_shapes(agent_sd):
    """
    Heuristically infer the observation space shapes that were used to
    construct the world model, by inspecting decoder output layer sizes.

    For the CNN path the shape comes from the final conv-transpose layer;
    for the MLP path it comes from the per-key output linear layers.
    """
    shapes = {}

    for k, v in agent_sd.items():
        # Normalise away _orig_mod
        clean = k.replace("_orig_mod.", "")

        # --- CNN decoder final layer → image shape --------------------------
        # Key pattern: _wm.heads.decoder._cnn.layers.<last>.bias
        # The bias length = C (number of output channels = 3 * num_cameras).
        # We also need spatial dims; we can get them from the linear layer that
        # projects feat → spatial:  _wm.heads.decoder._cnn._linear_layer.weight
        if clean.startswith("_wm.heads.decoder._cnn._linear_layer.weight"):
            # shape: (spatial_flat, feat_size)
            # spatial_flat = depth * minres * minres  where depth = cnn_depth * 2^(n_layers-1)
            pass  # handled below together with bias

        if clean.startswith("_wm.heads.decoder._cnn.layers.") and clean.endswith(".bias") and v.dim() == 1:
            # This is the final conv-transpose bias, giving us the number of
            # output channels.  We store it temporarily.
            shapes["_cnn_out_channels"] = v.shape[0]

        # --- MLP decoder per-key output layers → vector obs shapes -----------
        # Key pattern: _wm.heads.decoder._mlp.mean_layer.<key_name>.weight
        if "_wm.heads.decoder._mlp.mean_layer." in clean and clean.endswith(".weight"):
            # weight shape: (out_dim, hidden)
            parts = clean.split("_wm.heads.decoder._mlp.mean_layer.")[-1]
            obs_key = parts.replace(".weight", "")
            shapes[obs_key] = (v.shape[0],)

    # --- Reconstruct image shape from CNN decoder info -----------------------
    # We need the spatial resolution.  The encoder CNN records input_shape.
    # However, we can derive it from the decoder linear layer weight.
    cnn_out_channels = shapes.pop("_cnn_out_channels", None)
    if cnn_out_channels is not None:
        # To figure out the image (H, W, C), we need the encoder input shape.
        # The encoder CNN first conv weight has shape (depth, C_in, k, k).
        for k, v in agent_sd.items():
            clean = k.replace("_orig_mod.", "")
            if clean == "_wm.encoder._cnn.layers.0.weight":
                c_in = v.shape[1]  # number of input channels
                break
        else:
            c_in = cnn_out_channels  # fallback

        # Determine spatial dims from encoder conv weights
        # Count number of conv layers in encoder to figure out downsampling
        conv_keys = sorted(
            k for k in agent_sd
            if "_wm.encoder._cnn.layers." in k.replace("_orig_mod.", "")
            and k.endswith(".weight") and agent_sd[k].dim() == 4
        )
        n_conv = len(conv_keys)
        # Each conv layer downsamples by stride 2. The decoder linear layer:
        # _wm.heads.decoder._cnn._linear_layer.weight → (cnn_depth * 2^(n-1) * minres * minres, feat_size)
        for k, v in agent_sd.items():
            clean = k.replace("_orig_mod.", "")
            if clean == "_wm.heads.decoder._cnn._linear_layer.weight":
                spatial_flat = v.shape[0]
                break

        # spatial_flat = top_depth * minres * minres
        # top_depth = cnn_depth * 2^(n_conv - 1)
        # Find cnn_depth from first encoder conv: out_channels = cnn_depth
        cnn_depth = agent_sd[conv_keys[0]].shape[0]
        top_depth = cnn_depth * (2 ** (n_conv - 1))
        minres_sq = spatial_flat // top_depth
        minres = int(round(minres_sq ** 0.5))
        h = w = minres * (2 ** n_conv)

        # DreamerV3 uses channels-last (H, W, C) in its obs dict
        shapes["image"] = (h, w, c_in)

    return shapes


class FrozenWorldModelTrainer:
    """
    Generates posterior start states from replay buffer data using the frozen
    world model, then trains a fresh actor-critic in imagination.
    """

    def __init__(self, world_model, config):
        self.wm = world_model
        self.config = config
        self._use_amp = True if config.precision == 16 else False

        # Build a brand-new actor-critic
        self.actor_critic = models_bk.ImagBehavior(config, self.wm).to(config.device)
        self.actor_critic.requires_grad_(requires_grad=False)

        # Reward function for imagination (same as original Dreamer)
        self.reward_fn = lambda f, s, a: self.wm.heads["reward"](
            self.wm.dynamics.get_feat(s)
        ).mode()

    def train_step(self, data):
        """
        One training iteration:
          1. Run the frozen world model on a batch of real data to obtain
             posterior latent states (start states).
          2. Train the actor-critic purely in imagination from those start states.

        Parameters
        ----------
        data : dict
            A batch from the replay buffer (as produced by make_dataset).

        Returns
        -------
        metrics : dict
        """
        # --- Encode & infer posterior (frozen, no grad) ----------------------
        with torch.no_grad():
            if "cached_embed" in data:
                # Fast path: use pre-computed embeddings (skip CNN encoder)
                embed = torch.tensor(
                    data["cached_embed"],
                    device=self.config.device, dtype=torch.float32,
                )
                action = torch.tensor(
                    data["action"],
                    device=self.config.device, dtype=torch.float32,
                )
                is_first = torch.tensor(
                    data["is_first"],
                    device=self.config.device, dtype=torch.float32,
                )
                post, prior = self.wm.dynamics.observe(embed, action, is_first)
            else:
                # Fallback: full encoder pass (no caching)
                data = self.wm.preprocess(data)
                embed = self.wm.encoder(data)
                post, prior = self.wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
            # Detach everything — these are starting points for imagination
            start = {k: v.detach() for k, v in post.items()}

        # --- Train actor-critic in imagination from start --------------------
        metrics = {}
        _, _, _, _, ac_metrics = self.actor_critic._train(start, self.reward_fn)
        metrics.update(ac_metrics)
        return metrics

    def state_dict(self, step=0):
        """Return only the actor-critic state (the WM is frozen/external)."""
        return {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "actor_critic_optims": tools.recursively_collect_optim_state_dict(
                self.actor_critic
            ),
            "step": step,
        }

    def load_actor_critic_state_dict(self, ckpt):
        """Resume actor-critic training from a saved AC checkpoint."""
        self.actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        tools.recursively_load_optim_state_dict(
            self.actor_critic, ckpt["actor_critic_optims"]
        )
        return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Parse arguments ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train a fresh actor-critic with a frozen DreamerV3 world model."
    )
    parser.add_argument("--configs", nargs="+",
                        help="Config presets from configs.yaml (e.g. metaworld_default_light)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained DreamerV3 latest.pt checkpoint.")
    parser.add_argument("--traindir", type=str, required=True,
                        help="Directory containing replay buffer .npz episodes.")
    parser.add_argument("--logdir", type=str, default="actor_critic_training/logdir",
                        help="Output directory for logs and AC checkpoints.")
    parser.add_argument("--train_steps", type=int, default=100_000,
                        help="Number of actor-critic training iterations.")
    parser.add_argument("--log_every", type=int, default=1000,
                        help="Log metrics every N training steps.")
    parser.add_argument("--save_every", type=int, default=10_000,
                        help="Save actor-critic checkpoint every N training steps.")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"],
                        help="Logger backend.")
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="DreamerV3 AC Training")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a previously saved AC checkpoint to resume from.")
    parser.add_argument("--wm_total_train_steps", type=int, default=1_000_000,
                        help="Total env steps used to train the pre-trained world model. "
                             "Together with --dataset_limit_ratio this determines how "
                             "many replay-buffer timesteps to keep. "
                             "(default: 1,000,000)")
    parser.add_argument("--dataset_limit_ratio", type=float, default=0.5,
                        help="Fraction of wm_total_train_steps to keep from the "
                             "replay buffer (most recent data is kept first). "
                             "E.g. 0.5 with 1M WM steps → keep 500k timesteps. "
                             "Set to 1.0 to use the full buffer. (default: 0.5)")
    parser.add_argument("--no_cache_embeddings", action="store_true",
                        help="Disable pre-computing encoder embeddings. "
                             "Useful for debugging or very large datasets that "
                             "don't fit in memory with cached embeddings.")

    args, remaining = parser.parse_known_args()

    # ---- Load base config from configs.yaml --------------------------------
    configs_path = DREAMER_DIR / "configs.yaml"
    configs = yaml.safe_load(configs_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # Allow any config key to be overridden via the command line
    config_parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        config_parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = config_parser.parse_args(remaining)

    # ---- Derived config adjustments ----------------------------------------
    config.action_repeat = getattr(config, "action_repeat", 1)
    # We need num_actions to build the actor.  It lives in the checkpoint;
    # we infer it from the actor output layer (or the user can pass --num_actions).
    ckpt_tmp = torch.load(args.checkpoint, map_location="cpu")
    agent_sd = ckpt_tmp["agent_state_dict"]
    for k, v in agent_sd.items():
        clean = k.replace("_orig_mod.", "")
        # Actor mean layer: _task_behavior.actor.mean_layer.weight → (num_actions, hidden)
        if "_task_behavior.actor.mean_layer.weight" in clean:
            config.num_actions = v.shape[0]
            break
    del ckpt_tmp
    print(f"Inferred num_actions = {config.num_actions}")

    # ---- Setup logdir & logger ---------------------------------------------
    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    if args.logger == "tensorboard":
        logger = tools.Logger(logdir, 0)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, config, logdir, 0)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # ---- Load replay buffer ------------------------------------------------
    traindir = pathlib.Path(args.traindir).expanduser()
    assert traindir.exists(), f"Training data directory not found: {traindir}"

    dataset_limit = int(args.wm_total_train_steps * args.dataset_limit_ratio)
    print(f"Dataset limit: {dataset_limit} timesteps "
          f"({args.dataset_limit_ratio:.0%} of {args.wm_total_train_steps} WM train steps)")

    train_eps = tools.load_episodes(traindir, limit=dataset_limit)
    total_timesteps = sum(len(next(iter(ep.values()))) for ep in train_eps.values())
    assert len(train_eps) > 0, f"No episodes found in {traindir}"
    print(f"Loaded {len(train_eps)} episodes ({total_timesteps} timesteps) from {traindir}")

    # ---- Build frozen world model ------------------------------------------
    tools.set_seed_everywhere(config.seed)
    wm = load_world_model_from_checkpoint(args.checkpoint, config)

    # ---- Pre-compute encoder embeddings (optional but recommended) ---------
    if not args.no_cache_embeddings:
        precompute_embeddings(wm, train_eps, config)
    else:
        print("Embedding caching disabled — encoder will run on every train step.")

    dataset = make_dataset(train_eps, config)

    # ---- Build trainer (fresh actor-critic) --------------------------------
    trainer = FrozenWorldModelTrainer(wm, config)

    # ---- Auto-resume from logdir or explicit --resume path ---------------
    start_step = 0
    resume_path = None
    if args.resume:
        resume_path = pathlib.Path(args.resume).expanduser()
    else:
        # Auto-detect ac_latest.pt in logdir
        candidate = logdir / "ac_latest.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None and resume_path.exists():
        ac_ckpt = torch.load(resume_path, map_location=config.device)
        start_step = trainer.load_actor_critic_state_dict(ac_ckpt)
        del ac_ckpt
        print(f"Resumed actor-critic from {resume_path} at step {start_step}")

    if start_step >= args.train_steps:
        print(f"Already completed {start_step}/{args.train_steps} steps. Nothing to do.")
        return

    # ---- Training loop -----------------------------------------------------
    remaining_steps = args.train_steps - start_step
    print(f"Starting actor-critic training: steps {start_step + 1} → {args.train_steps} "
          f"({remaining_steps} remaining) …")
    metrics_accum = {}
    start_time = time.time()

    pbar = tqdm(range(start_step + 1, args.train_steps + 1),
                desc="AC training", unit="step", dynamic_ncols=True)
    for step in pbar:
        batch = next(dataset)
        step_metrics = trainer.train_step(batch)

        # Accumulate metrics
        for name, value in step_metrics.items():
            if name not in metrics_accum:
                metrics_accum[name] = []
            metrics_accum[name].append(value)

        # ---- Logging -------------------------------------------------------
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            steps_done = step - start_step
            sps = steps_done / elapsed if elapsed > 0 else 0
            for name, values in metrics_accum.items():
                logger.scalar(name, float(np.mean(values)))
            logger.scalar("train_steps_per_sec", sps)
            logger.step = step
            logger.write()
            metrics_accum = {}

        # ---- Checkpointing -------------------------------------------------
        if step % args.save_every == 0:
            save_path = logdir / f"ac_step_{step}.pt"
            torch.save(trainer.state_dict(step=step), save_path)
            # Also keep a "latest" symlink / copy
            latest_path = logdir / "ac_latest.pt"
            torch.save(trainer.state_dict(step=step), latest_path)
            print(f"[Step {step}] Saved AC checkpoint → {save_path}")

    # ---- Final save --------------------------------------------------------
    torch.save(trainer.state_dict(step=args.train_steps), logdir / "ac_latest.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
