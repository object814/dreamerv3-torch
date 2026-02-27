"""
Evaluate a standalone-trained actor-critic in the real Metaworld environment.

This script:
  1. Loads a pre-trained DreamerV3 world model checkpoint (encoder + RSSM dynamics)
     and freezes it.
  2. Loads a separately trained actor-critic checkpoint (from train_actor_critic.py).
  3. Runs the actor in the real environment by encoding observations through the
     frozen world model and selecting actions from the trained actor.

Usage example:
  python actor_critic_training/eval_policy.py \
      --configs metaworld_visual_heavy_long \
      --task pick-place-v2 \
      --wm_checkpoint /path/to/logdir/latest.pt \
      --ac_checkpoint actor_critic_training/ac_logdir/ac_latest.pt \
      --episodes 50 \
      --outdir actor_critic_training/eval_results
"""

import argparse
import os
import pathlib
import sys
import time

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import torch
import ruamel.yaml as yaml
import imageio
import gymnasium

# ---------------------------------------------------------------------------
# Path setup — reuse the parent dreamerv3 package directly
# ---------------------------------------------------------------------------
DREAMER_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DREAMER_DIR))

import models
import networks
import tools
import envs.wrappers as wrappers
import envs.metaworld_wrappers as metaworld_wrappers

BASE_DIR = DREAMER_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper

# Re-use the world model loading utilities from train_actor_critic
from train_actor_critic import load_world_model_from_checkpoint


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_names):
    """Loads and merges configuration from configs.yaml."""
    configs = yaml.safe_load(
        (DREAMER_DIR / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    return defaults


# ---------------------------------------------------------------------------
# Environment creation (same wrappers as eval_policy.py / dreamer.py)
# ---------------------------------------------------------------------------
def make_eval_env(task_name, config):
    """Creates the Metaworld environment with DreamerV3-compatible wrappers."""
    print(f"Creating eval env for task: {task_name}")
    env = gymnasium.make(
        "Meta-World/MT1",
        env_name=task_name,
        render_mode="rgb_array",
        max_episode_steps=config.time_limit,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=config.size[0],
        image_width=config.size[1],
        camera_names=["topview", "front", "gripperPOV"],
    )
    env = metaworld_wrappers.FirstTerminalObs(env)
    env = metaworld_wrappers.RewardTuningWrapperV2(env)
    env = metaworld_wrappers.Gymnasium2Gym(env)
    env = wrappers.NormalizeActions(env)
    env = wrappers.RewardObs(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


# ---------------------------------------------------------------------------
# Policy wrapper — world model encoder/RSSM + standalone actor
# ---------------------------------------------------------------------------
class ACPolicy:
    """
    Combines a frozen world model (encoder + RSSM) with a standalone-trained
    actor to act in the real environment.

    The step logic mirrors ``Dreamer._policy``:
      1. preprocess(obs) → embed = encoder(obs)
      2. latent, _ = dynamics.obs_step(prev_latent, prev_action, embed, is_first)
      3. feat = dynamics.get_feat(latent)
      4. action = actor(feat).mode()
    """

    def __init__(self, world_model, actor_critic, config):
        self.wm = world_model            # frozen
        self.actor = actor_critic.actor   # trained standalone actor
        self.config = config
        self.actor.eval()

    @torch.no_grad()
    def __call__(self, obs, reset, state=None):
        """
        Parameters
        ----------
        obs : dict[str, np.ndarray]
            Observation dict with a batch dimension (B, ...).
        reset : list[bool]
            Whether each env in the batch was just reset.
        state : tuple | None
            (latent, prev_action) carried across steps, or None at episode start.

        Returns
        -------
        policy_output : dict   {"action": Tensor}
        state : tuple          (latent, action)
        """
        if state is None:
            latent = action = None
        else:
            latent, action = state

        obs = self.wm.preprocess(obs)
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"]
        )
        if self.config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self.wm.dynamics.get_feat(latent)

        actor_dist = self.actor(feat)
        action = actor_dist.mode()

        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        if self.config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self.config.num_actions
            )

        policy_output = {"action": action}
        state = (latent, action)
        return policy_output, state


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def eval_policy(policy, env, args, save_video_path=None):
    """Runs evaluation episodes and reports metrics."""
    print(f"Starting evaluation for {args.episodes} episodes ...")

    total_rewards = []
    success_counts = 0

    for episode in range(args.episodes):
        obs = env.reset()
        state = None
        done = False
        episode_reward = 0
        video_frames = []
        step_count = 0

        while not done:
            # Batch dimension expected by the world model
            obs_batch = {k: np.stack([v]) for k, v in obs.items()}

            policy_output, state = policy(obs_batch, [False], state)
            action_values = policy_output["action"].cpu().numpy()[0]

            action_dict = {"action": action_values}
            obs, reward, done, info = env.step(action_dict)

            episode_reward += reward
            step_count += 1

            if save_video_path and "image" in obs:
                img = obs["image"]
                h, w, c = img.shape
                num_cameras = c // 3
                camera_frames = [
                    img[:, :, i * 3 : (i + 1) * 3] for i in range(num_cameras)
                ]
                if camera_frames:
                    combined_frame = np.concatenate(camera_frames, axis=1)
                    video_frames.append(combined_frame)

        is_success = info.get("success", 0.0) > 0.5
        success_counts += int(is_success)
        total_rewards.append(episode_reward)

        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Success: {is_success} | "
            f"Steps: {step_count}"
        )

        if save_video_path and video_frames:
            tag = "success" if is_success else "failure"
            vid_filename = os.path.join(
                save_video_path, f"{args.task}_ep{episode + 1}_{tag}.mp4"
            )
            try:
                imageio.mimsave(vid_filename, video_frames, fps=30)
                print(f"  Saved video → {vid_filename}")
            except Exception as e:
                print(f"  Error saving video: {e}")

    avg_reward = np.mean(total_rewards)
    success_rate = success_counts / args.episodes * 100

    print("-" * 40)
    print(f"Evaluation Complete")
    print(f"  Average Reward : {avg_reward:.2f}")
    print(f"  Success Rate   : {success_rate:.1f}%")
    print("-" * 40)

    return {"avg_reward": avg_reward, "success_rate": success_rate}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a standalone-trained actor-critic on Metaworld."
    )
    parser.add_argument(
        "--wm_checkpoint", type=str, required=True,
        help="Path to the DreamerV3 world-model checkpoint (latest.pt).",
    )
    parser.add_argument(
        "--ac_checkpoint", type=str, required=True,
        help="Path to the standalone actor-critic checkpoint (ac_latest.pt).",
    )
    parser.add_argument(
        "--task", type=str, default="pick-place-v2",
        help="Metaworld task name (e.g. pick-place-v2).",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["defaults", "metaworld"],
        help="Config presets from configs.yaml.",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--outdir", type=str, default="actor_critic_training/eval_results",
        help="Directory to save evaluation videos.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args, remaining = parser.parse_known_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load config -------------------------------------------------------
    defaults = load_config(args.configs)

    # Allow CLI overrides of any config key
    config_parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        config_parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = config_parser.parse_args(remaining)

    config.device = args.device

    # ---- Infer num_actions from the WM checkpoint --------------------------
    ckpt_tmp = torch.load(args.wm_checkpoint, map_location="cpu")
    agent_sd = ckpt_tmp["agent_state_dict"]
    for k, v in agent_sd.items():
        clean = k.replace("_orig_mod.", "")
        if "_task_behavior.actor.mean_layer.weight" in clean:
            config.num_actions = v.shape[0]
            break
    del ckpt_tmp
    print(f"Inferred num_actions = {config.num_actions}")

    # ---- Build frozen world model ------------------------------------------
    print(f"Loading world model from {args.wm_checkpoint} ...")
    wm = load_world_model_from_checkpoint(args.wm_checkpoint, config)

    # ---- Build actor-critic & load trained weights -------------------------
    print(f"Loading actor-critic from {args.ac_checkpoint} ...")
    actor_critic = models.ImagBehavior(config, wm).to(config.device)
    ac_ckpt = torch.load(args.ac_checkpoint, map_location=config.device)
    actor_critic.load_state_dict(ac_ckpt["actor_critic_state_dict"])
    actor_critic.eval()
    actor_critic.requires_grad_(False)

    # ---- Create the combined policy ----------------------------------------
    policy = ACPolicy(wm, actor_critic, config)

    # ---- Create environment ------------------------------------------------
    env = make_eval_env(args.task, config)

    # ---- Run evaluation ----------------------------------------------------
    eval_policy(policy, env, args, save_video_path=str(outdir))

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
