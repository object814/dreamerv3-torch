"""Standalone evaluation for Sequential-ER trained DreamerV3 checkpoints.

This script loads the separated checkpoints saved by ``dreamer_sequential_er.py``
and evaluates the agent on real Metaworld environments.

Checkpoint layout produced by training::

    logdir/
        rssm_final.pt                      # shared RSSM after all tasks
        task0_drawer-open-v3/
            rssm_task0.pt
            heads_task0.pt                  # reward + continue heads
            actor_critic_task0.pt           # actor + value + slow_value
        task1_pick-place-v3/
            rssm_task1.pt
            heads_task1.pt
            actor_critic_task1.pt
        ...

This evaluator:
  1. Reconstructs a full Dreamer agent for each task by loading the shared RSSM
     and the task-specific reward/cont heads + actor-critic.
  2. Runs real-environment rollouts, reporting success rate, avg reward, and
     avg episode length.
  3. Optionally saves evaluation videos.

Usage:
    python er_training/eval_sequential_er.py \\
        --logdir ./logdir/sequential_er_run \\
        --tasks drawer-open-v3 pick-place-v3 \\
        --configs metaworld_visual_heavy_long \\
        --episodes 50 \\
        --outdir ./eval_results \\
        --device cuda

    # Or evaluate a single task with explicit checkpoint paths:
    python er_training/eval_sequential_er.py \\
        --rssm-checkpoint ./logdir/rssm_final.pt \\
        --heads-checkpoints ./logdir/task0/heads_task0.pt \\
        --ac-checkpoints ./logdir/task0/actor_critic_task0.pt \\
        --tasks drawer-open-v3 \\
        --configs metaworld_visual_heavy_long \\
        --episodes 50
"""

import argparse
import json
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import torch
import ruamel.yaml as yaml
import imageio
import gymnasium

# ---------------------------------------------------------------------------
# Path setup
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


# ---------------------------------------------------------------------------
# Key normalisation (torch.compile inserts _orig_mod. segments)
# ---------------------------------------------------------------------------

def _normalise_key(key: str) -> str:
    """Strip ``._orig_mod.`` and ``_orig_mod.`` segments inserted by torch.compile."""
    return key.replace("._orig_mod.", ".").replace("_orig_mod.", "")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_names):
    """Load and merge configuration presets from configs.yaml."""
    configs = yaml.safe_load((DREAMER_DIR / "configs.yaml").read_text())

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
# Environment creation
# ---------------------------------------------------------------------------

def make_eval_env(task_name, config):
    """Create a Metaworld environment wrapped for DreamerV3 evaluation."""
    print(f"  Creating eval env for task: {task_name}")
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
# Observation-space inference from checkpoint
# ---------------------------------------------------------------------------

def _infer_obs_shapes(state_dict):
    """Infer observation space shapes from a full agent state dict.

    Mirrors the logic in ``actor_critic_training/train_actor_critic.py``.
    """
    import gymnasium.spaces as spaces

    shapes = {}

    for k, v in state_dict.items():
        clean = _normalise_key(k)

        # CNN decoder final conv bias → number of output channels
        if (clean.startswith("_wm.heads.decoder._cnn.layers.")
                and clean.endswith(".bias") and v.dim() == 1):
            shapes["_cnn_out_channels"] = v.shape[0]

        # MLP decoder per-key output layers → vector obs shapes
        if "_wm.heads.decoder._mlp.mean_layer." in clean and clean.endswith(".weight"):
            parts = clean.split("_wm.heads.decoder._mlp.mean_layer.")[-1]
            obs_key = parts.replace(".weight", "")
            shapes[obs_key] = (v.shape[0],)

    # Reconstruct image shape from CNN decoder
    cnn_out_channels = shapes.pop("_cnn_out_channels", None)
    if cnn_out_channels is not None:
        # Find number of input channels from encoder first conv
        c_in = cnn_out_channels  # fallback
        for k, v in state_dict.items():
            clean = _normalise_key(k)
            if clean == "_wm.encoder._cnn.layers.0.weight":
                c_in = v.shape[1]
                break

        # Count conv layers to determine downsampling factor
        conv_keys = sorted(
            k for k in state_dict
            if "_wm.encoder._cnn.layers." in _normalise_key(k)
            and k.endswith(".weight") and state_dict[k].dim() == 4
        )
        n_conv = len(conv_keys)

        # Decoder linear projects features → spatial
        for k, v in state_dict.items():
            clean = _normalise_key(k)
            if clean == "_wm.heads.decoder._cnn._linear_layer.weight":
                spatial_flat = v.shape[0]
                break

        cnn_depth = state_dict[conv_keys[0]].shape[0]
        top_depth = cnn_depth * (2 ** (n_conv - 1))
        minres_sq = spatial_flat // top_depth
        minres = int(round(minres_sq ** 0.5))
        h = w = minres * (2 ** n_conv)

        shapes["image"] = (h, w, c_in)

    return shapes


# ---------------------------------------------------------------------------
# Agent reconstruction from separated checkpoints
# ---------------------------------------------------------------------------

def _build_agent(obs_space, act_space, config, logdir):
    """Create a fresh Dreamer agent (uncompiled) for evaluation."""
    from dreamer_sequential_er import Dreamer

    logger = tools.Logger(logdir, 0)
    dataset = iter([])
    # Temporarily disable compile for clean state dict keys
    orig_compile = config.compile
    config.compile = False
    agent = Dreamer(obs_space, act_space, config, logger, dataset)
    config.compile = orig_compile
    return agent


def load_agent_from_separated_checkpoints(
    rssm_path, heads_path, ac_path, config, device="cuda",
):
    """Reconstruct a full Dreamer agent from separated .pt checkpoints.

    Parameters
    ----------
    rssm_path : str or Path
        Path to the RSSM checkpoint (rssm_taskN.pt or rssm_final.pt).
    heads_path : str or Path
        Path to the heads checkpoint (heads_taskN.pt).
    ac_path : str or Path
        Path to the actor-critic checkpoint (actor_critic_taskN.pt).
    config : argparse.Namespace
        DreamerV3 config.
    device : str
        Device to load onto.

    Returns
    -------
    agent : Dreamer
        Fully assembled agent ready for evaluation.
    """
    rssm_sd = torch.load(rssm_path, map_location="cpu")
    heads_sd = torch.load(heads_path, map_location="cpu")
    ac_sd = torch.load(ac_path, map_location="cpu")

    # Merge into a single agent state dict
    merged_sd = {}
    merged_sd.update(rssm_sd)
    merged_sd.update(heads_sd)
    merged_sd.update(ac_sd)

    # Infer observation space from the merged state dict
    obs_shapes = _infer_obs_shapes(merged_sd)
    import gymnasium.spaces as spaces
    obs_space = spaces.Dict(
        {k: spaces.Box(low=-np.inf, high=np.inf, shape=v)
         for k, v in obs_shapes.items()}
    )

    # Infer num_actions from actor mean_layer weight
    for k, v in merged_sd.items():
        clean = _normalise_key(k)
        if "_task_behavior.actor.mean_layer.weight" in clean:
            config.num_actions = v.shape[0]
            break
    else:
        # Try the Dreamer _policy action shape
        for k, v in merged_sd.items():
            clean = _normalise_key(k)
            if "actor" in clean and "weight" in clean and v.dim() == 2:
                config.num_actions = v.shape[0]
                break

    print(f"  Inferred num_actions = {config.num_actions}")

    # Create a dummy action space
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(config.num_actions,))

    # Build a fresh un-compiled agent
    tmpdir = pathlib.Path("/tmp/eval_sequential_er_dummy")
    tmpdir.mkdir(parents=True, exist_ok=True)
    agent = _build_agent(obs_space, act_space, config, tmpdir)

    # Load merged weights using normalised-key matching
    fresh_sd = agent.state_dict()
    norm_to_fresh = {_normalise_key(fk): fk for fk in fresh_sd}

    loaded = 0
    for k, v in merged_sd.items():
        if k in fresh_sd:
            fresh_sd[k] = v
            loaded += 1
        else:
            norm_k = _normalise_key(k)
            if norm_k in norm_to_fresh:
                fresh_sd[norm_to_fresh[norm_k]] = v
                loaded += 1

    agent.load_state_dict(fresh_sd)
    agent = agent.to(device)
    agent.eval()

    total_params = sum(v.numel() for v in merged_sd.values())
    print(f"  Loaded {loaded} tensors ({total_params:,} params) into agent")

    return agent


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def eval_task(agent, env, task_name, episodes, save_video_path=None):
    """Run evaluation episodes and return metrics.

    Parameters
    ----------
    agent : Dreamer
        The assembled agent.
    env : wrapped Metaworld env
        Evaluation environment.
    task_name : str
        Task name (for logging / video filenames).
    episodes : int
        Number of evaluation episodes.
    save_video_path : str or None
        Directory to save videos.  None → no video.

    Returns
    -------
    dict with keys: task, avg_reward, success_rate, avg_length, rewards, successes
    """
    total_rewards = []
    total_lengths = []
    success_count = 0

    for ep in range(episodes):
        obs = env.reset()
        state = None
        done = False
        episode_reward = 0.0
        video_frames = []
        step_count = 0

        while not done:
            obs_batch = {k: np.stack([v]) for k, v in obs.items()}

            with torch.no_grad():
                policy_output, state = agent(
                    obs_batch, [False], state, training=False
                )
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
        success_count += int(is_success)
        total_rewards.append(episode_reward)
        total_lengths.append(step_count)

        print(
            f"    Episode {ep + 1:3d}/{episodes} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Success: {is_success} | "
            f"Steps: {step_count}"
        )

        if save_video_path and video_frames:
            tag = "success" if is_success else "failure"
            vid_path = os.path.join(
                save_video_path, f"{task_name}_ep{ep + 1}_{tag}.mp4"
            )
            try:
                imageio.mimsave(vid_path, video_frames, fps=30)
                print(f"    Saved video → {vid_path}")
            except Exception as e:
                print(f"    Error saving video: {e}")

    avg_reward = float(np.mean(total_rewards))
    avg_length = float(np.mean(total_lengths))
    success_rate = success_count / episodes * 100.0

    return {
        "task": task_name,
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "avg_length": avg_length,
        "num_episodes": episodes,
        "rewards": [float(r) for r in total_rewards],
        "successes": [bool(total_rewards[i]) for i in range(episodes)],
    }


# ---------------------------------------------------------------------------
# Auto-discovery of checkpoints from logdir
# ---------------------------------------------------------------------------

def discover_task_checkpoints(logdir, tasks):
    """Find per-task checkpoint files from the training logdir layout.

    Parameters
    ----------
    logdir : Path
        Root training logdir.
    tasks : list[str]
        Task names in training order.

    Returns
    -------
    rssm_path : Path
        Path to rssm_final.pt (or last task's RSSM).
    task_infos : list[dict]
        Per-task dicts with keys: task, heads_path, ac_path.
    """
    logdir = pathlib.Path(logdir)

    # Find RSSM: prefer rssm_final.pt, fall back to last task's rssm
    rssm_path = logdir / "rssm_final.pt"
    if not rssm_path.exists():
        # Search for highest-numbered rssm_taskN.pt
        rssm_candidates = sorted(logdir.glob("task*/rssm_task*.pt"))
        if not rssm_candidates:
            raise FileNotFoundError(
                f"No rssm_final.pt or rssm_taskN.pt found in {logdir}"
            )
        rssm_path = rssm_candidates[-1]
    print(f"  RSSM checkpoint: {rssm_path}")

    # Discover per-task heads + actor-critic
    task_infos = []
    for i, task in enumerate(tasks):
        # Try standard naming: task{i}_{taskname}/
        task_dir_candidates = sorted(logdir.glob(f"task{i}_*"))
        if not task_dir_candidates:
            # Try with metaworld_ prefix stripped
            task_dir_candidates = sorted(logdir.glob(f"task{i}*"))
        if not task_dir_candidates:
            print(f"  [WARNING] No task directory found for task {i} ({task}), skipping")
            continue

        task_dir = task_dir_candidates[0]
        heads_path = task_dir / f"heads_task{i}.pt"
        ac_path = task_dir / f"actor_critic_task{i}.pt"

        if not heads_path.exists():
            print(f"  [WARNING] Missing {heads_path}, skipping task {i}")
            continue
        if not ac_path.exists():
            print(f"  [WARNING] Missing {ac_path}, skipping task {i}")
            continue

        task_infos.append({
            "task": task,
            "task_idx": i,
            "heads_path": heads_path,
            "ac_path": ac_path,
        })
        print(f"  Task {i} ({task}): heads={heads_path.name}, ac={ac_path.name}")

    return rssm_path, task_infos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Sequential-ER trained DreamerV3 on Metaworld.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint discovery: either --logdir (auto-discover) or explicit paths
    ckpt_group = parser.add_argument_group("Checkpoint specification")
    ckpt_group.add_argument(
        "--logdir", type=str, default=None,
        help="Training logdir — auto-discovers rssm_final.pt and per-task heads/ac.",
    )
    ckpt_group.add_argument(
        "--rssm-checkpoint", type=str, default=None,
        help="Explicit path to the shared RSSM checkpoint.",
    )
    ckpt_group.add_argument(
        "--heads-checkpoints", nargs="+", type=str, default=None,
        help="Explicit paths to per-task heads checkpoints (one per task).",
    )
    ckpt_group.add_argument(
        "--ac-checkpoints", nargs="+", type=str, default=None,
        help="Explicit paths to per-task actor-critic checkpoints (one per task).",
    )

    # Task / config
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="Metaworld task names in training order "
             "(e.g. drawer-open-v3 pick-place-v3).",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["defaults", "metaworld"],
        help="Config presets from configs.yaml.",
    )

    # Evaluation
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes per task.")
    parser.add_argument("--outdir", type=str, default="eval_results",
                        help="Directory to save results and videos.")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save evaluation videos.")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load config -------------------------------------------------------
    cfg_dict = load_config(args.configs)
    config = argparse.Namespace(**cfg_dict)
    config.device = args.device

    # ---- Resolve checkpoints -----------------------------------------------
    if args.logdir is not None:
        # Auto-discover from logdir
        rssm_path, task_infos = discover_task_checkpoints(
            pathlib.Path(args.logdir), args.tasks
        )
    elif (args.rssm_checkpoint and args.heads_checkpoints and args.ac_checkpoints):
        # Explicit paths
        rssm_path = pathlib.Path(args.rssm_checkpoint)
        if len(args.heads_checkpoints) != len(args.tasks):
            parser.error(
                f"--heads-checkpoints ({len(args.heads_checkpoints)}) must match "
                f"--tasks ({len(args.tasks)})"
            )
        if len(args.ac_checkpoints) != len(args.tasks):
            parser.error(
                f"--ac-checkpoints ({len(args.ac_checkpoints)}) must match "
                f"--tasks ({len(args.tasks)})"
            )
        task_infos = [
            {"task": t, "task_idx": i,
             "heads_path": pathlib.Path(h), "ac_path": pathlib.Path(a)}
            for i, (t, h, a) in enumerate(
                zip(args.tasks, args.heads_checkpoints, args.ac_checkpoints)
            )
        ]
    else:
        parser.error(
            "Provide either --logdir for auto-discovery, or all of "
            "--rssm-checkpoint, --heads-checkpoints, --ac-checkpoints."
        )

    if not task_infos:
        print("No task checkpoints found. Nothing to evaluate.")
        return

    # ---- Evaluate each task ------------------------------------------------
    all_results = []
    print(f"\n{'=' * 60}")
    print(f"Sequential-ER Evaluation  |  {len(task_infos)} task(s)  |  "
          f"{args.episodes} episodes each")
    print(f"{'=' * 60}\n")

    for info in task_infos:
        task_name = info["task"]
        task_idx = info["task_idx"]
        heads_path = info["heads_path"]
        ac_path = info["ac_path"]

        print(f"--- Task {task_idx}: {task_name} ---")
        print(f"  RSSM:  {rssm_path}")
        print(f"  Heads: {heads_path}")
        print(f"  AC:    {ac_path}")

        # Build agent from separated checkpoints
        agent = load_agent_from_separated_checkpoints(
            rssm_path, heads_path, ac_path, config, device=args.device,
        )

        # Create environment
        env = make_eval_env(task_name, config)

        # Evaluate
        video_dir = str(outdir / f"task{task_idx}_{task_name}") if args.save_videos else None
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        result = eval_task(agent, env, task_name, args.episodes, save_video_path=video_dir)
        result["task_idx"] = task_idx
        all_results.append(result)

        env.close()

        print(f"\n  Task {task_idx} ({task_name}) results:")
        print(f"    Success Rate : {result['success_rate']:.1f}%")
        print(f"    Avg Reward   : {result['avg_reward']:.2f}")
        print(f"    Avg Length   : {result['avg_length']:.1f}")
        print()

        # Free GPU memory before next task
        del agent
        torch.cuda.empty_cache()

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Task':<25s} {'Success%':>10s} {'AvgReward':>12s} {'AvgLen':>8s}")
    print(f"{'-' * 55}")
    for r in all_results:
        print(
            f"{r['task']:<25s} "
            f"{r['success_rate']:>9.1f}% "
            f"{r['avg_reward']:>12.2f} "
            f"{r['avg_length']:>8.1f}"
        )

    # Mean across tasks
    if len(all_results) > 1:
        mean_sr = np.mean([r["success_rate"] for r in all_results])
        mean_rw = np.mean([r["avg_reward"] for r in all_results])
        mean_len = np.mean([r["avg_length"] for r in all_results])
        print(f"{'-' * 55}")
        print(
            f"{'MEAN':<25s} "
            f"{mean_sr:>9.1f}% "
            f"{mean_rw:>12.2f} "
            f"{mean_len:>8.1f}"
        )
    print(f"{'=' * 60}\n")

    # ---- Save results to JSON ----------------------------------------------
    results_path = outdir / "eval_results.json"
    # Convert Path objects and non-serialisable types
    serialisable = []
    for r in all_results:
        sr = dict(r)
        sr.pop("rewards", None)
        sr.pop("successes", None)
        serialisable.append(sr)

    with open(results_path, "w") as f:
        json.dump({"tasks": serialisable}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
