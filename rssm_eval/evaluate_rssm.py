"""
Evaluate RSSM world-model fidelity across sequential tasks.

For each task n in the task suite:
  1. Load the actor trained on task n and generate successful expert rollouts.
  2. For each task m >= n, load the RSSM checkpoint saved after task m.
  3. Feed the expert demonstrations through that RSSM:
       - Posterior: encoder + obs_step (uses real observations at every step)
       - Prior:     img_step only (open-loop, uses real actions but no observations)
  4. Decode both posterior and prior latents and compute:
       - Per-step image reconstruction loss (negative log-prob from the decoder)
       - Per-step KL divergence between prior and posterior
       - Per-step reward prediction loss (negative log-prob from task_heads.reward)
  5. Save:
       - Side-by-side GIFs (ground-truth | posterior recon | prior recon)
       - Combined bar chart of all three metrics across all tasks

Usage:
    python evaluate_rssm.py \
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 metaworld_compo-draweropen-pickplace \
        --configs metaworld_visual_200M_heavy_long_speedup ... \
        --logdir ../logdir/sequential/mw_sequential_drawerpnp_0330 \
        --outdir ../logdir/sequential/mw_sequential_drawerpnp_0330/rssm_eval \
        --num-demos 10 \
        --max-attempts 50
"""

import argparse
import pathlib
import sys
import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["EGL_LOG_LEVEL"] = "fatal"

import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

import numpy as np
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import models
import tools
import envs.wrappers as wrappers
import envs.metaworld_wrappers as metaworld_wrappers

import gymnasium
gymnasium.logger.min_level = gymnasium.logger.ERROR

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper


# ── helpers ──────────────────────────────────────────────────────────────────

def build_task_config(task_name, config_name, configs_yaml, remaining_args):
    """Build a config namespace for a task (mirrors dreamer_sequential.py)."""
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", config_name]:
        recursive_update(defaults, configs_yaml[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining_args)

    config.task = task_name
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    return config


def make_env(task_name, config):
    """Create a single evaluation environment."""
    suite, task = task_name.split("_", 1)
    if suite != "metaworld":
        raise NotImplementedError(f"Suite '{suite}' not supported.")
    env = gymnasium.make(
        "Meta-World/MT1", env_name=task, render_mode="rgb_array",
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


def multicam_frame(image):
    """HxWx(3*C) -> side-by-side HxCW x 3 uint8 frame."""
    h, w, c = image.shape
    num_cameras = c // 3
    parts = [image[:, :, i * 3:(i + 1) * 3] for i in range(num_cameras)]
    frame = np.concatenate(parts, axis=1)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    return frame


# ── expert rollout collection ────────────────────────────────────────────────

def collect_expert_demos(actor, rssm, env, config, num_demos, max_attempts):
    """Roll out the actor in the env, returning only *successful* episodes.

    Each demo is a dict with keys:
        obs_images: list of HxWxC numpy arrays (raw env images, float [0,1])
        actions:    list of numpy arrays (action_dim,)
        rewards:    list of float
        obs_dicts:  list of raw obs dicts (for feeding into the RSSM)
    """
    demos = []
    attempts = 0

    while len(demos) < num_demos and attempts < max_attempts:
        attempts += 1
        obs = env.reset()
        latent = action = None
        done = False
        ep_obs_dicts = []
        ep_actions = []
        ep_rewards = []
        ep_images = []

        while not done:
            ep_obs_dicts.append(obs)
            if "image" in obs:
                img = obs["image"].copy()
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                ep_images.append(img)

            obs_batch = {k: np.stack([v]) for k, v in obs.items()}
            with torch.no_grad():
                obs_proc = rssm.preprocess(obs_batch)
                embed = rssm.encoder(obs_proc)
                latent, _ = rssm.dynamics.obs_step(
                    latent, action, embed, obs_proc["is_first"]
                )
                if config.eval_state_mean:
                    latent["stoch"] = latent["mean"]
                feat = rssm.dynamics.get_feat(latent)
                actor_dist = actor(feat)
                action = actor_dist.mode()
                latent = {k: v.detach() for k, v in latent.items()}
                action = action.detach()
                if config.actor["dist"] == "onehot_gumble":
                    action = torch.one_hot(
                        torch.argmax(action, dim=-1), config.num_actions
                    )

            action_np = action.cpu().numpy()[0]
            ep_actions.append(action_np)
            obs, reward, done, info = env.step({"action": action_np})
            ep_rewards.append(float(reward))

        is_success = info.get("success", 0.0) > 0.5
        tag = "SUCCESS" if is_success else "fail"
        print(f"    Attempt {attempts}: {tag}  reward={sum(ep_rewards):.1f}  "
              f"steps={len(ep_actions)}  ({len(demos)}/{num_demos} collected)")

        if is_success:
            demos.append(dict(
                obs_images=ep_images,
                actions=ep_actions,
                rewards=ep_rewards,
                obs_dicts=ep_obs_dicts,
            ))

    if len(demos) < num_demos:
        print(f"    WARNING: only collected {len(demos)}/{num_demos} successful "
              f"demos in {max_attempts} attempts")
    return demos


# ── RSSM evaluation on a demo ───────────────────────────────────────────────

@torch.no_grad()
def evaluate_rssm_on_demo(rssm, task_heads, demo, config):
    """Feed one expert demo through an RSSM and return per-step metrics + decoded frames.

    Args:
        rssm:       RSSMWorldModel checkpoint to evaluate.
        task_heads: TaskHeads from the demo's own task (for reward prediction loss).
        demo:       dict with obs_dicts, actions, rewards.
        config:     task config namespace.

    Returns:
        posterior_frames:   list of HxWx3 uint8
        prior_frames:       list of HxWx3 uint8
        gt_frames:          list of HxWx3 uint8
        recon_losses:       numpy (T,) — per-step image reconstruction NLL
        kl_values:          numpy (T,) — per-step KL(posterior || prior)
        reward_pred_losses: numpy (T,) — per-step reward prediction NLL
    """
    obs_dicts = demo["obs_dicts"]
    actions = demo["actions"]
    rewards = demo["rewards"]
    T = len(obs_dicts)

    posterior_frames = []
    prior_frames = []
    gt_frames = []
    recon_losses = []
    kl_values = []
    reward_pred_losses = []

    latent = None
    action = None
    prior_latent = None

    for t in range(T):
        obs = obs_dicts[t]
        obs_batch = {k: np.stack([v]) for k, v in obs.items()}
        obs_proc = rssm.preprocess(obs_batch)
        embed = rssm.encoder(obs_proc)

        # Posterior update (uses real observation)
        post, prior = rssm.dynamics.obs_step(
            latent, action, embed, obs_proc["is_first"]
        )
        if config.eval_state_mean:
            post["stoch"] = post["mean"]

        # Prior (open-loop): after t=0 use img_step with real action
        if t == 0:
            prior_latent = {k: v.detach().clone() for k, v in post.items()}
        else:
            prior_latent = rssm.dynamics.img_step(prior_latent, action)
            if config.eval_state_mean:
                prior_latent["stoch"] = prior_latent["mean"]

        # Decode posterior
        post_feat = rssm.dynamics.get_feat(post)
        post_recon = rssm.decoder(post_feat.unsqueeze(1))["image"]
        post_img = post_recon.mode()[0, 0].cpu().numpy()
        posterior_frames.append(multicam_frame(post_img))

        # Decode prior
        prior_feat = rssm.dynamics.get_feat(prior_latent)
        prior_recon = rssm.decoder(prior_feat.unsqueeze(1))["image"]
        prior_img = prior_recon.mode()[0, 0].cpu().numpy()
        prior_frames.append(multicam_frame(prior_img))

        # Ground truth
        if "image" in obs:
            gt_img = obs["image"].astype(np.float32)
            if gt_img.max() > 1.0:
                gt_img = gt_img / 255.0
            gt_frames.append(multicam_frame(gt_img))

        # Reconstruction loss (image NLL under the posterior)
        target = obs_proc["image"]  # (1, H, W, C) normalised
        nll = -post_recon.log_prob(target[:, None, :, :, :])  # (1, 1)
        recon_losses.append(float(nll.mean().cpu()))

        # KL divergence between posterior and prior at this step
        post_dist = rssm.dynamics.get_dist(post)
        prior_dist = rssm.dynamics.get_dist(prior_latent)
        if rssm._config.dyn_discrete:
            kl = torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        else:
            kl = torch.distributions.kl.kl_divergence(
                post_dist._dist, prior_dist._dist
            )
        kl_values.append(float(kl.mean().cpu()))

        # Reward prediction loss (NLL of the reward head on posterior features)
        reward_target = torch.tensor(
            [[rewards[t]]], device=config.device, dtype=torch.float32
        )
        reward_pred = task_heads.reward(post_feat)
        reward_nll = -reward_pred.log_prob(reward_target)
        reward_pred_losses.append(float(reward_nll.mean().cpu()))

        # Advance latent / action for next step
        latent = {k: v.detach() for k, v in post.items()}
        if t < len(actions):
            action = torch.tensor(
                actions[t], device=config.device, dtype=torch.float32
            ).unsqueeze(0)

    return (
        posterior_frames, prior_frames, gt_frames,
        np.array(recon_losses), np.array(kl_values),
        np.array(reward_pred_losses),
    )


# ── visualisation ────────────────────────────────────────────────────────────

def save_recon_gif(gt_frames, post_frames, prior_frames, path, fps=15):
    """Save a side-by-side GIF: GT | Posterior | Prior."""
    combined = []
    for gt, post, pri in zip(gt_frames, post_frames, prior_frames):
        h = max(gt.shape[0], post.shape[0], pri.shape[0])
        w = max(gt.shape[1], post.shape[1], pri.shape[1])

        def pad(f):
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[:f.shape[0], :f.shape[1]] = f
            return out

        row = np.concatenate([pad(gt), pad(post), pad(pri)], axis=1)
        combined.append(row)

    imageio.mimsave(str(path), combined, fps=fps, loop=0)


def plot_task_metrics(task_results, task_idx, task_name, path):
    """Single figure for one task with 3 subplots (recon_loss, kl, reward_pred).

    Each subplot shows bars across RSSM checkpoints (after task m >= task_idx).

    task_results: list of (label, dict(recon_loss=arr, kl=arr, reward_pred=arr))
    """
    if not task_results:
        return

    metric_keys = ["recon_loss", "kl", "reward_pred"]
    metric_labels = [
        "Reconstruction Loss (NLL)",
        "KL Divergence",
        "Reward Prediction Loss (NLL)",
    ]

    labels = [r[0] for r in task_results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(
        1, len(metric_keys),
        figsize=(5 * len(metric_keys), 4.5),
        sharey=False,
    )
    if len(metric_keys) == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A467"]

    for ax, mkey, mlabel, color in zip(axes, metric_keys, metric_labels, colors):
        means = [np.mean(r[1][mkey]) for r in task_results]
        stds = [np.std(r[1][mkey]) for r in task_results]
        ax.bar(x, means, yerr=stds, capsize=6, color=color,
               edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Task {task_idx+1} ({task_name}) — Metrics by RSSM Checkpoint",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"    Saved plot: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RSSM fidelity across sequential tasks",
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--logdir", type=str, required=True,
                        help="Base logdir of the sequential training run")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: <logdir>/rssm_eval)")
    parser.add_argument("--num-demos", type=int, default=10,
                        help="Number of successful expert demos per task")
    parser.add_argument("--max-attempts", type=int, default=50,
                        help="Max env episodes to try per task when collecting demos")
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)

    main_args, remaining = parser.parse_known_args()

    tasks = main_args.tasks
    config_names = main_args.configs
    num_tasks = len(tasks)
    assert len(config_names) == num_tasks

    logdir = pathlib.Path(main_args.logdir).expanduser()
    outdir = pathlib.Path(main_args.outdir) if main_args.outdir else logdir / "rssm_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    configs_yaml = yaml.safe_load(
        (pathlib.Path(__file__).parent.parent / "configs.yaml").read_text()
    )

    # Build per-task configs
    task_configs = []
    for i in range(num_tasks):
        cfg = build_task_config(tasks[i], config_names[i], configs_yaml, remaining)
        task_configs.append(cfg)

    # Resolve device
    device = main_args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg in task_configs:
        cfg.device = device

    # Resolve num_actions from first env
    tmp_env = make_env(tasks[0], task_configs[0])
    acts = tmp_env.action_space
    num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    tmp_env.close()
    for cfg in task_configs:
        cfg.num_actions = num_actions

    print("=" * 60)
    print(">>> RSSM EVALUATION")
    print("=" * 60)
    for i, t in enumerate(tasks):
        print(f"  Task {i+1}: {t} (config: {config_names[i]})")
    print(f"  Logdir:    {logdir}")
    print(f"  Outdir:    {outdir}")
    print(f"  Demos:     {main_args.num_demos}")
    print(f"  Device:    {device}")
    print("=" * 60)

    # ── per-task evaluation ──────────────────────────────────────────────
    # Collect all results for a summary at the end
    all_results = {}  # (task_n, rssm_m) -> dict of arrays

    for n in range(num_tasks):
        task_name = tasks[n]
        config = task_configs[n]
        task_dir = logdir / f"task{n+1}_{task_name}"

        print(f"\n{'='*60}")
        print(f">>> Task {n+1}/{num_tasks}: {task_name}")
        print(f"{'='*60}")

        # ── Load actor for task n (the expert policy) ────────────────────
        ac_path = task_dir / "actor_critic.pt"
        if not ac_path.exists():
            print(f"  ERROR: {ac_path} not found, skipping task {n+1}")
            continue

        # We need an RSSM to run the actor (for encoding observations).
        # Use task n's own RSSM for demo collection (the actor was trained with it).
        rssm_n_path = task_dir / "rssm.pt"
        if not rssm_n_path.exists():
            print(f"  ERROR: {rssm_n_path} not found, skipping task {n+1}")
            continue

        env = make_env(task_name, config)

        # Instantiate RSSM + ActorCritic to load weights into
        rssm_for_collection = models.RSSMWorldModel(
            env.observation_space, env.action_space, 0, config,
        ).to(device)
        rssm_for_collection.requires_grad_(False)
        tools.load_component(rssm_for_collection, rssm_n_path,
                             load_optimizers=False, device=device)

        actor_critic = models.ActorCritic(config).to(device)
        actor_critic.requires_grad_(False)
        tools.load_component(actor_critic, ac_path,
                             load_optimizers=False, device=device)

        # ── Collect successful expert demonstrations ─────────────────────
        print(f"  Collecting {main_args.num_demos} successful demos ...")
        demos = collect_expert_demos(
            actor_critic.actor, rssm_for_collection, env, config,
            main_args.num_demos, main_args.max_attempts,
        )
        if not demos:
            print(f"  No successful demos collected, skipping task {n+1}")
            env.close()
            del rssm_for_collection, actor_critic
            continue

        del rssm_for_collection, actor_critic
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Load task n's TaskHeads (for reward prediction loss) ─────────
        task_heads_path = task_dir / "task_heads.pt"
        if task_heads_path.exists():
            task_heads_n = models.TaskHeads(config).to(device)
            task_heads_n.requires_grad_(False)
            tools.load_component(task_heads_n, task_heads_path,
                                 load_optimizers=False, device=device)
            print(f"  Loaded TaskHeads for task {n+1}")
        else:
            task_heads_n = None
            print(f"  WARNING: task_heads.pt not found for task {n+1}, "
                  f"reward prediction loss will be skipped")

        # ── Evaluate with RSSM from each task m >= n ─────────────────────
        task_n_results = []  # list of (label, {recon_loss, kl, reward_pred})

        for m in range(n, num_tasks):
            rssm_m_name = tasks[m]
            rssm_m_dir = logdir / f"task{m+1}_{rssm_m_name}"
            rssm_m_path = rssm_m_dir / "rssm.pt"
            if not rssm_m_path.exists():
                print(f"  WARNING: RSSM for task {m+1} not found ({rssm_m_path}), skipping")
                continue

            label = f"After Task {m+1}"
            print(f"\n  --- Evaluating with RSSM from {label} ({rssm_m_name}) ---")

            rssm = models.RSSMWorldModel(
                env.observation_space, env.action_space, 0, config,
            ).to(device)
            rssm.requires_grad_(False)
            tools.load_component(rssm, rssm_m_path,
                                 load_optimizers=False, device=device)

            demo_recon_losses = []
            demo_kl_values = []
            demo_reward_pred_losses = []

            gif_dir = outdir / f"task{n+1}_{task_name}" / f"rssm_after_task{m+1}"
            gif_dir.mkdir(parents=True, exist_ok=True)

            for d_idx, demo in enumerate(demos):
                post_frames, prior_frames, gt_frames, recon_loss, kl_val, reward_pred_loss = \
                    evaluate_rssm_on_demo(rssm, task_heads_n, demo, config)

                demo_recon_losses.append(recon_loss.mean())
                demo_kl_values.append(kl_val.mean())
                demo_reward_pred_losses.append(reward_pred_loss.mean())

                # Save GIF for first few demos
                if d_idx < 3:
                    gif_path = gif_dir / f"demo{d_idx+1}.gif"
                    save_recon_gif(gt_frames, post_frames, prior_frames,
                                   gif_path, fps=main_args.gif_fps)
                    print(f"    Saved GIF: {gif_path}")

                print(f"    Demo {d_idx+1}/{len(demos)}: "
                      f"recon_loss={recon_loss.mean():.3f}  "
                      f"kl={kl_val.mean():.3f}  "
                      f"reward_pred={reward_pred_loss.mean():.3f}")

            demo_recon_losses = np.array(demo_recon_losses)
            demo_kl_values = np.array(demo_kl_values)
            demo_reward_pred_losses = np.array(demo_reward_pred_losses)

            metric_dict = dict(
                recon_loss=demo_recon_losses,
                kl=demo_kl_values,
                reward_pred=demo_reward_pred_losses,
            )
            all_results[(n, m)] = metric_dict
            task_n_results.append((label, metric_dict))

            print(f"  {label}: recon_loss={demo_recon_losses.mean():.3f} "
                  f"+/- {demo_recon_losses.std():.3f}  "
                  f"kl={demo_kl_values.mean():.3f} +/- {demo_kl_values.std():.3f}  "
                  f"reward_pred={demo_reward_pred_losses.mean():.3f} "
                  f"+/- {demo_reward_pred_losses.std():.3f}")

            del rssm
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Save plot for this task (all three metrics side-by-side) ─────
        task_out = outdir / f"task{n+1}_{task_name}"
        task_out.mkdir(parents=True, exist_ok=True)
        plot_task_metrics(
            task_n_results, n, task_name,
            path=task_out / "metrics_bar.png",
        )

        del task_heads_n
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        env.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(">>> SUMMARY")
    print(f"{'='*60}")
    for (n, m), res in sorted(all_results.items()):
        print(f"  Task {n+1} demos | RSSM after task {m+1}: "
              f"recon={res['recon_loss'].mean():.3f}+/-{res['recon_loss'].std():.3f}  "
              f"kl={res['kl'].mean():.3f}+/-{res['kl'].std():.3f}  "
              f"reward_pred={res['reward_pred'].mean():.3f}+/-{res['reward_pred'].std():.3f}")

    # Save numeric results as a simple text file
    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("task_n\trssm_m\trecon_mean\trecon_std\tkl_mean\tkl_std\t"
                "reward_pred_mean\treward_pred_std\n")
        for (n, m), res in sorted(all_results.items()):
            f.write(f"{n+1}\t{m+1}\t{res['recon_loss'].mean():.4f}\t"
                    f"{res['recon_loss'].std():.4f}\t"
                    f"{res['kl'].mean():.4f}\t{res['kl'].std():.4f}\t"
                    f"{res['reward_pred'].mean():.4f}\t"
                    f"{res['reward_pred'].std():.4f}\n")
    print(f"  Results saved to {summary_path}")
    print(">>> Done.")


if __name__ == "__main__":
    main()
