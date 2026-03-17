"""Sequential training script for DreamerV3 with EWC (Elastic Weight Consolidation)
and separated architecture (shared RSSM + per-task reward/cont/actor-critic).

Unlike dreamer_sequential.py which trains strictly sequentially without any
continual-learning mechanism, this script:
  1. Uses EWC to protect important RSSM parameters from previous tasks.
     After each task, a diagonal Fisher Information Matrix is computed and stored,
     and a quadratic penalty is added to the world model loss during subsequent tasks.
  2. Separates the RSSM (encoder + dynamics + decoder) from task-specific components
     (reward head, continue head, actor-critic).
  3. Between tasks: keeps and continues updating the RSSM (with EWC penalty);
     resets reward head, continue head, and actor-critic from scratch.
  4. Saves separated checkpoints:
     - rssm_task{N}.pt: RSSM weights at end of task N
     - heads_task{N}.pt: reward + continue head weights for task N
     - actor_critic_task{N}.pt: actor + critic weights for task N
     - ewc_state_task{N}.pt: EWC Fisher + param snapshots up to task N
     - rssm_final.pt: final RSSM after all tasks

Usage:
    python dreamer_sequential_ewc.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --configs metaworld_default_light metaworld_default_light \\
        --task-steps 200000 200000 \\
        --dataset-sizes 400000 400000 \\
        --logdir ./logdir/sequential_ewc_run \\
        --ewc-lambda 5000.0 \\
        --ewc-fisher-batches 50 \\
        --wandb-entity my-entity \\
        --wandb-project my-project
"""

import argparse
import collections
import functools
import gc
import json
import os
import pathlib
import shutil
import sys
import time as _time

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import ruamel.yaml as yaml

# Path setup: add dreamerv3 dir to path
DREAMER_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(DREAMER_DIR))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import gymnasium

import torch
from torch import nn
from torch import distributions as torchd
from tqdm.auto import tqdm

# Metaworld import setup
import envs.metaworld_wrappers as metaworld_wrappers
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper

# EWC
from ewc import EWCManager


to_np = lambda x: x.detach().cpu().numpy()


# ============================================================================
# Architecture separation utilities
# ============================================================================

# State dict key prefixes for each component group.
# When torch.compile is used, keys gain an "_orig_mod." segment, e.g.
#   _wm._orig_mod.encoder.xxx  instead of  _wm.encoder.xxx
# We normalise keys before matching so the same prefixes work in both cases.
RSSM_PREFIXES = ("_wm.encoder.", "_wm.dynamics.", "_wm.heads.decoder.")
HEADS_PREFIXES = ("_wm.heads.reward.", "_wm.heads.cont.")
ACTOR_CRITIC_PREFIXES = ("_task_behavior.",)


def _normalise_key(key):
    """Strip '_orig_mod.' segments inserted by torch.compile."""
    return key.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def _matches_prefixes(key, prefixes):
    """Check if key (or its normalised form) starts with any prefix."""
    norm = _normalise_key(key)
    return any(norm.startswith(p) for p in prefixes)


def extract_rssm_state_dict(agent_state_dict):
    """Extract RSSM (encoder + dynamics + decoder) weights from full agent state dict."""
    return {k: v for k, v in agent_state_dict.items()
            if _matches_prefixes(k, RSSM_PREFIXES)}


def extract_heads_state_dict(agent_state_dict):
    """Extract reward + continue head weights from full agent state dict."""
    return {k: v for k, v in agent_state_dict.items()
            if _matches_prefixes(k, HEADS_PREFIXES)}


def extract_actor_critic_state_dict(agent_state_dict):
    """Extract actor-critic (actor + value + slow_value + ema) weights."""
    return {k: v for k, v in agent_state_dict.items()
            if _matches_prefixes(k, ACTOR_CRITIC_PREFIXES)}


def load_rssm_into_agent(agent, rssm_state_dict):
    """Load RSSM weights into a fresh agent, keeping other weights at random init.

    Handles torch.compile key mismatches.
    """
    fresh_sd = agent.state_dict()

    norm_to_fresh = {}
    for fk in fresh_sd:
        norm_to_fresh[_normalise_key(fk)] = fk

    loaded = 0
    for k, v in rssm_state_dict.items():
        if k in fresh_sd:
            fresh_sd[k] = v
            loaded += 1
        else:
            norm_k = _normalise_key(k)
            if norm_k in norm_to_fresh:
                fresh_sd[norm_to_fresh[norm_k]] = v
                loaded += 1

    agent.load_state_dict(fresh_sd)
    return loaded


def load_partial_state_dict(agent, partial_sd):
    """Load a partial state dict into an agent, updating only matching keys.

    Used to swap in saved task-specific weights (heads, actor-critic) while
    keeping the rest of the agent (e.g. RSSM) unchanged.

    Args:
        agent: A Dreamer agent.
        partial_sd: Dict of keys -> tensors to load (subset of full state dict).

    Returns:
        Number of parameters loaded.
    """
    current_sd = agent.state_dict()

    norm_to_current = {}
    for k in current_sd:
        norm_to_current[_normalise_key(k)] = k

    loaded = 0
    for k, v in partial_sd.items():
        if k in current_sd:
            current_sd[k] = v
            loaded += 1
        else:
            norm_k = _normalise_key(k)
            if norm_k in norm_to_current:
                current_sd[norm_to_current[norm_k]] = v
                loaded += 1

    agent.load_state_dict(current_sd)
    return loaded


# ============================================================================
# Agent
# ============================================================================

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


# ============================================================================
# Infrastructure
# ============================================================================

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(task_name, config, mode, id):
    """Create an environment for the given task using the given config."""
    suite, task = task_name.split("_", 1)
    if suite == "metaworld":
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
    raise NotImplementedError(f"Suite '{suite}' is not supported.")


class LazyParallelEnv:
    """Picklable proxy that lazily constructs the real env inside the worker process."""

    def __init__(self, task_name, config, mode, env_id):
        self._task_name = task_name
        self._config = config
        self._mode = mode
        self._env_id = env_id
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = make_env(self._task_name, self._config, self._mode, self._env_id)
        return self._env

    @property
    def observation_space(self):
        return self._ensure_env().observation_space

    @property
    def action_space(self):
        return self._ensure_env().action_space

    @property
    def id(self):
        return self._ensure_env().id

    def reset(self):
        return self._ensure_env().reset()

    def step(self, action):
        return self._ensure_env().step(action)

    def close(self):
        if self._env is not None:
            return self._env.close()


class PrefixedLogger:
    """Wraps a logger to prefix all metric names for task-specific eval logging."""

    def __init__(self, real_logger, prefix, record_video=True):
        self.real_logger = real_logger
        self.prefix = prefix
        self.record_video = record_video

    def scalar(self, name, value):
        self.real_logger.scalar(f"{self.prefix}/{name}", value)

    def image(self, name, value):
        self.real_logger.image(f"{self.prefix}/{name}", value)

    def video(self, name, value):
        if self.record_video:
            self.real_logger.video(f"{self.prefix}/{name}", value)

    def write(self, **kwargs):
        pass

    @property
    def step(self):
        return self.real_logger.step

    @step.setter
    def step(self, value):
        self.real_logger.step = value


def save_sequential_progress(base_logdir, task_idx, global_env_step_at_start=None,
                             completed=False, global_env_step_at_end=None):
    """Save sequential training progress to a JSON file for resume support."""
    progress_file = base_logdir / "sequential_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"tasks": {}}

    task_key = str(task_idx)
    if task_key not in progress["tasks"]:
        progress["tasks"][task_key] = {}

    if global_env_step_at_start is not None:
        progress["tasks"][task_key]["global_env_step_at_start"] = global_env_step_at_start
    if completed:
        progress["tasks"][task_key]["completed"] = True
    if global_env_step_at_end is not None:
        progress["tasks"][task_key]["global_env_step_at_end"] = global_env_step_at_end

    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def load_sequential_progress(base_logdir):
    """Load sequential training progress from JSON file. Returns None if not found."""
    progress_file = base_logdir / "sequential_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return None


def build_task_config(task_name, config_name, task_steps, configs_yaml, remaining_args):
    """Build a config namespace for a specific task."""
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", config_name]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs_yaml[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining_args)

    config.task = task_name
    if task_steps is not None:
        config.steps = task_steps

    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    return config


# ============================================================================
# Main
# ============================================================================

def main(args, remaining_args):
    tasks = args.tasks
    config_names = args.configs
    task_steps_list = args.task_steps
    dataset_sizes_list = args.dataset_sizes
    num_tasks = len(tasks)

    assert len(config_names) == num_tasks, (
        f"Number of configs ({len(config_names)}) must match number of tasks ({num_tasks})"
    )
    assert len(task_steps_list) == num_tasks, (
        f"Number of task-steps ({len(task_steps_list)}) must match number of tasks ({num_tasks})"
    )
    if dataset_sizes_list is not None:
        assert len(dataset_sizes_list) == num_tasks, (
            f"Number of dataset-sizes ({len(dataset_sizes_list)}) must match number of tasks ({num_tasks})"
        )

    # Load configs yaml
    configs_yaml = yaml.safe_load(
        (DREAMER_DIR / "configs.yaml").read_text()
    )

    # Build per-task configs
    task_configs = []
    for i in range(num_tasks):
        config = build_task_config(
            tasks[i], config_names[i], task_steps_list[i],
            configs_yaml, remaining_args,
        )
        if dataset_sizes_list is not None:
            config.dataset_size = int(dataset_sizes_list[i])
        task_configs.append(config)

    tools.set_seed_everywhere(task_configs[0].seed)
    if task_configs[0].deterministic_run:
        tools.enable_deterministic_run()

    base_logdir = pathlib.Path(args.logdir).expanduser()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # EWC setup
    # ================================================================
    ewc_manager = EWCManager(
        lambda_ewc=args.ewc_lambda,
        rssm_prefixes=("encoder.", "dynamics.", "heads.decoder."),
    )

    # ================================================================
    # Resume detection
    # ================================================================
    global_env_step = 0
    prev_rssm_checkpoint = None
    prev_full_checkpoint = args.from_checkpoint
    resume_from_task_idx = 0

    progress = load_sequential_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            task_info = progress.get("tasks", {}).get(str(i))
            if task_info and task_info.get("completed"):
                resume_from_task_idx = i + 1
                global_env_step = task_info["global_env_step_at_end"]
                task_logdir_i = base_logdir / f"task{i+1}_{tasks[i]}"
                rssm_file = task_logdir_i / f"rssm_task{i+1}.pt"
                if rssm_file.exists():
                    prev_rssm_checkpoint = str(rssm_file)
                else:
                    full_file = task_logdir_i / "latest.pt"
                    if full_file.exists():
                        ckpt = torch.load(full_file, map_location="cpu")
                        rssm_sd = extract_rssm_state_dict(ckpt["agent_state_dict"])
                        torch.save(rssm_sd, rssm_file)
                        prev_rssm_checkpoint = str(rssm_file)
                        del ckpt
                prev_full_checkpoint = str(task_logdir_i / "latest.pt")
                print(f">>> RESUME: Task {i+1} ({tasks[i]}) already completed "
                      f"(ended at global step {global_env_step}), skipping.")
            else:
                break
    else:
        for i in range(num_tasks):
            task_logdir_i = base_logdir / f"task{i+1}_{tasks[i]}"
            checkpoint_file = task_logdir_i / f"checkpoint_task{i+1}.pt"
            if checkpoint_file.exists():
                ckpt = torch.load(checkpoint_file, map_location="cpu")
                if "logger_step" in ckpt:
                    global_env_step = ckpt["logger_step"]
                else:
                    traindir_i = task_logdir_i / "train_eps"
                    if traindir_i.exists():
                        global_env_step += count_steps(traindir_i) * task_configs[i].action_repeat
                resume_from_task_idx = i + 1
                rssm_file = task_logdir_i / f"rssm_task{i+1}.pt"
                if rssm_file.exists():
                    prev_rssm_checkpoint = str(rssm_file)
                else:
                    rssm_sd = extract_rssm_state_dict(ckpt["agent_state_dict"])
                    torch.save(rssm_sd, rssm_file)
                    prev_rssm_checkpoint = str(rssm_file)
                prev_full_checkpoint = str(task_logdir_i / "latest.pt")
                del ckpt
                print(f">>> RESUME: Task {i+1} ({tasks[i]}) already completed "
                      f"(global step ~{global_env_step}), skipping.")
            else:
                break

    # [EWC] Resume: load EWC state if available from the latest completed task
    if resume_from_task_idx > 0:
        for j in range(resume_from_task_idx - 1, -1, -1):
            ewc_file = base_logdir / f"task{j+1}_{tasks[j]}" / f"ewc_state_task{j+1}.pt"
            if ewc_file.exists():
                print(f">>> EWC: Loading state from {ewc_file}")
                ewc_state = torch.load(ewc_file, map_location="cpu")
                ewc_manager.load_state_dict(ewc_state)
                print(f">>> EWC: Restored {ewc_manager.num_tasks_consolidated} "
                      f"consolidated task(s)")
                break

    # For within-task resume, peek at the current task's latest.pt
    initial_logger_step = global_env_step
    if resume_from_task_idx < num_tasks:
        current_task_logdir = base_logdir / f"task{resume_from_task_idx+1}_{tasks[resume_from_task_idx]}"
        current_latest = current_task_logdir / "latest.pt"
        if current_latest.exists():
            try:
                peek_ckpt = torch.load(current_latest, map_location="cpu")
                if "logger_step" in peek_ckpt:
                    initial_logger_step = peek_ckpt["logger_step"]
                    print(f">>> RESUME: Will resume within task {resume_from_task_idx+1} "
                          f"at logger_step={initial_logger_step}")
                else:
                    current_traindir = current_task_logdir / "train_eps"
                    if current_traindir.exists():
                        task_start = global_env_step
                        if progress is not None:
                            saved_start = (progress.get("tasks", {})
                                           .get(str(resume_from_task_idx), {})
                                           .get("global_env_step_at_start"))
                            if saved_start is not None:
                                task_start = saved_start
                        steps_in_dir = count_steps(current_traindir)
                        initial_logger_step = task_start + steps_in_dir * task_configs[resume_from_task_idx].action_repeat
                        print(f">>> RESUME: Estimated within-task logger_step={initial_logger_step}")
            except Exception as e:
                print(f">>> RESUME: Could not peek at checkpoint: {e}")

    is_resuming = resume_from_task_idx > 0 or initial_logger_step > 0

    if resume_from_task_idx > 0:
        print(f">>> RESUME: Will resume from task {resume_from_task_idx + 1} "
              f"(global_env_step={global_env_step})")
    if resume_from_task_idx >= num_tasks:
        print(">>> RESUME: All tasks already completed. Nothing to do.")
        return

    # Create logger
    first_config = task_configs[0]
    if args.logger == "tensorboard":
        logger = tools.Logger(base_logdir, initial_logger_step)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, first_config, base_logdir, initial_logger_step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # Print training plan
    print("=" * 60)
    print(">>> SEQUENTIAL EWC TRAINING WITH CROSS-TASK EVALUATION <<<")
    print("=" * 60)
    for i, (task, cfg_name, steps) in enumerate(zip(tasks, config_names, task_steps_list)):
        ds = int(task_configs[i].dataset_size)
        print(f"  Task {i+1}: {task} (config: {cfg_name}, steps: {steps}, dataset_size: {ds})")
    print(f"  Log directory: {base_logdir}")
    print(f"  EWC lambda: {args.ewc_lambda}")
    print(f"  EWC Fisher batches: {args.ewc_fisher_batches}")
    print(f"  Eval previous task videos: {args.eval_prev_video}")
    print("=" * 60)

    if not args.skip_config_check:
        input(">>> Press Enter to start sequential EWC training...")

    # ================================================================
    # Sequential task loop
    # ================================================================
    for task_idx in range(num_tasks):
        task_name = tasks[task_idx]
        config = task_configs[task_idx]

        task_logdir = base_logdir / f"task{task_idx+1}_{task_name}"
        traindir = task_logdir / "train_eps"
        evaldir = task_logdir / "eval_eps"
        task_logdir.mkdir(parents=True, exist_ok=True)
        traindir.mkdir(parents=True, exist_ok=True)
        evaldir.mkdir(parents=True, exist_ok=True)
        config.traindir = traindir
        config.evaldir = evaldir

        # Skip completed tasks
        if task_idx < resume_from_task_idx:
            continue

        print("=" * 60)
        print(f">>> SEQUENTIAL EWC: Starting Task {task_idx+1}/{num_tasks}: {task_name}")
        print(f">>> SEQUENTIAL EWC: Task steps: {config.steps * config.action_repeat}")
        print(f">>> SEQUENTIAL EWC: Global env step: {global_env_step}")
        print(f">>> SEQUENTIAL EWC: EWC consolidated tasks so far: "
              f"{ewc_manager.num_tasks_consolidated}")
        print("=" * 60)

        # Create train envs
        if config.parallel:
            print(f">>> SEQUENTIAL EWC: Creating parallel train envs for task {task_idx+1}...")
            train_envs = [
                Parallel(LazyParallelEnv(task_name, config, "train", i), "process")
                for i in range(config.envs)
            ]
        else:
            train_envs = [make_env(task_name, config, "train", i) for i in range(config.envs)]
            train_envs = [Damy(env) for env in train_envs]

        acts = train_envs[0].action_space
        print(f">>> SEQUENTIAL EWC: Action Space: {acts}")
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        # Prepare eval dirs and caches (envs are created lazily during eval)
        all_eval_dirs = {}
        all_eval_caches = {}
        for j in range(task_idx + 1):
            eval_dir_j = task_logdir / f"eval_eps_task{j+1}_{tasks[j]}"
            eval_dir_j.mkdir(parents=True, exist_ok=True)
            all_eval_dirs[j] = eval_dir_j
            all_eval_caches[j] = tools.load_episodes(eval_dir_j, limit=1)

        print(f">>> SEQUENTIAL EWC: Prepared eval dirs for {task_idx + 1} task(s)")

        # Set logger to global step
        logger.step = global_env_step

        # Record global step at start of this task
        task_global_env_step_at_start = global_env_step
        if progress is not None:
            saved_start = (progress.get("tasks", {})
                           .get(str(task_idx), {})
                           .get("global_env_step_at_start"))
            if saved_start is not None:
                task_global_env_step_at_start = saved_start
                global_env_step = saved_start
                logger.step = global_env_step

        # Load current task episodes
        train_eps = tools.load_episodes(traindir, limit=config.dataset_size)
        tools.erase_over_episodes(train_eps, config.dataset_size)
        if config.dataset_size:
            tools.erase_over_episode_files(traindir, train_eps)

        # Prefill replay buffer
        state = None
        if not config.offline_traindir:
            prefill = max(0, config.prefill - count_steps(traindir))
            print(f">>> SEQUENTIAL EWC: Prefill dataset ({prefill} steps).")
            if prefill > 0:
                if hasattr(acts, "discrete"):
                    random_actor = tools.OneHotDist(
                        torch.zeros(config.num_actions).repeat(config.envs, 1)
                    )
                else:
                    random_actor = torchd.independent.Independent(
                        torchd.uniform.Uniform(
                            torch.tensor(acts.low).repeat(config.envs, 1),
                            torch.tensor(acts.high).repeat(config.envs, 1),
                        ),
                        1,
                    )

                def random_agent(o, d, s):
                    action = random_actor.sample()
                    logprob = random_actor.log_prob(action)
                    return {"action": action, "logprob": logprob}, None

                state = tools.simulate(
                    random_agent,
                    train_envs,
                    train_eps,
                    traindir,
                    logger,
                    limit=config.dataset_size,
                    steps=prefill,
                )
                logger.step += prefill * config.action_repeat
                print(f">>> SEQUENTIAL EWC: Logger step after prefill: {logger.step}")

        # Create dataset and agent
        print(">>> SEQUENTIAL EWC: Creating agent.")
        train_dataset = make_dataset(train_eps, config)

        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            train_dataset,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)

        # ============================================================
        # [EWC] Attach EWC manager to world model
        # ============================================================
        agent._wm.ewc_manager = ewc_manager

        # Sanity check: verify RSSM params are found by EWC
        rssm_names = [n for n, _ in agent._wm.named_parameters()
                      if ewc_manager._is_rssm_param(n)]
        rssm_param_count = sum(
            p.numel() for n, p in agent._wm.named_parameters()
            if ewc_manager._is_rssm_param(n)
        )
        print(f">>> EWC: Tracking {len(rssm_names)} RSSM param groups "
              f"({rssm_param_count:,} params)")
        assert len(rssm_names) > 0, "No RSSM params found — check prefix matching!"

        # ============================================================
        # Load checkpoint
        # Priority: latest.pt (resume within task) > RSSM from prev task > from-checkpoint
        # ============================================================
        load_path = None
        resuming_within_task = False

        if (task_logdir / "latest.pt").exists():
            print(f">>> SEQUENTIAL EWC: Resuming from: {task_logdir / 'latest.pt'}")
            load_path = task_logdir / "latest.pt"
            resuming_within_task = True
        elif task_idx > 0 and prev_rssm_checkpoint is not None:
            if os.path.exists(prev_rssm_checkpoint):
                print(f">>> SEQUENTIAL EWC: Loading RSSM from: {prev_rssm_checkpoint}")
                rssm_sd = torch.load(prev_rssm_checkpoint, map_location=config.device)
                loaded = load_rssm_into_agent(agent, rssm_sd)
                rssm_scalar_count = sum(v.numel() for v in rssm_sd.values())
                print(f">>> SEQUENTIAL EWC: Loaded {loaded} RSSM tensors "
                      f"({rssm_scalar_count:,} params). "
                      f"Reward head, continue head, and actor-critic start FRESH.")
                del rssm_sd
            else:
                raise FileNotFoundError(f"RSSM checkpoint not found: {prev_rssm_checkpoint}")
        elif task_idx == 0 and prev_full_checkpoint is not None:
            if os.path.exists(prev_full_checkpoint):
                print(f">>> SEQUENTIAL EWC: Loading full checkpoint: {prev_full_checkpoint}")
                load_path = pathlib.Path(prev_full_checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {prev_full_checkpoint}")

        checkpoint_data = None
        if load_path:
            checkpoint_data = torch.load(load_path)
            agent.load_state_dict(checkpoint_data["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint_data["optims_state_dict"])

        # [EWC] Rebuild penalty cache now that we have the model on the correct device
        if ewc_manager.num_tasks_consolidated > 0 and not ewc_manager._penalty_cache_valid:
            for tid, reg in ewc_manager.regularization_terms.items():
                for k in reg["importance"]:
                    reg["importance"][k] = reg["importance"][k].to(config.device)
                    reg["task_param"][k] = reg["task_param"][k].to(config.device)
            ewc_manager._rebuild_penalty_cache(agent._wm)
            print(f">>> EWC: Rebuilt penalty cache on {config.device}")

        # Skip pretraining for task 2+ or resume
        if task_idx > 0 or args.skip_pretrain or resuming_within_task:
            print(">>> SEQUENTIAL EWC: Skipping pretraining (sequential continuation).")
            agent._should_pretrain._once = False

        # Restore step tracking
        if resuming_within_task and checkpoint_data and "logger_step" in checkpoint_data:
            logger.step = checkpoint_data["logger_step"]
            agent._step = logger.step // config.action_repeat
            task_start_step = checkpoint_data["task_start_step"]
            print(f">>> RESUME: Restored logger.step={logger.step}, "
                  f"agent._step={agent._step}, task_start_step={task_start_step}")
            print(f">>> RESUME: Training progress within task: "
                  f"{agent._step - task_start_step}/{config.steps} steps")
        elif resuming_within_task and checkpoint_data:
            steps_in_traindir = count_steps(traindir)
            logger.step = task_global_env_step_at_start + steps_in_traindir * config.action_repeat
            agent._step = logger.step // config.action_repeat
            task_start_step = task_global_env_step_at_start // config.action_repeat + config.prefill
            training_steps_done = agent._step - task_start_step
            print(f">>> RESUME (legacy checkpoint): Estimated from episode files:")
            print(f"    steps_in_traindir={steps_in_traindir}, "
                  f"logger.step={logger.step}, agent._step={agent._step}")
            print(f"    task_start_step={task_start_step}, "
                  f"training_progress={training_steps_done}/{config.steps}")
        else:
            task_start_step = agent._step

        print(f">>> SEQUENTIAL EWC: Agent step: {agent._step}, task_start_step: {task_start_step}")

        # Save progress: mark this task as started
        save_sequential_progress(
            base_logdir, task_idx,
            global_env_step_at_start=task_global_env_step_at_start,
        )

        # ============================================================
        # Main training loop
        # ============================================================
        items_to_save = None
        task_train_steps_done = min(max(agent._step - task_start_step, 0), config.steps)
        progress_bar = tqdm(
            total=config.steps,
            initial=task_train_steps_done,
            desc=f">>> SEQUENTIAL EWC: Task {task_idx+1}/{num_tasks} Training",
            unit="step",
        )
        try:
            while (agent._step - task_start_step) < config.steps + config.eval_every:
                logger.write()

                # === EVALUATION on all tasks seen so far ===
                if config.eval_episode_num > 0:
                    print(f">>> SEQUENTIAL EWC: Evaluation at global step {logger.step} "
                          f"(evaluating {task_idx + 1} task(s))")

                    # Save current agent weights so we can restore after
                    # evaluating previous tasks with their own heads/actor-critic
                    current_agent_sd = {k: v.clone() for k, v in agent.state_dict().items()}

                    for j in range(task_idx + 1):
                        eval_task_name = tasks[j]
                        task_label = f"eval_task{j+1}_{eval_task_name}"
                        is_current_task = (j == task_idx)
                        record_video = is_current_task or args.eval_prev_video

                        # For previous tasks: swap in their saved heads + actor-critic
                        # while keeping the current RSSM weights
                        if not is_current_task:
                            prev_task_logdir = base_logdir / f"task{j+1}_{tasks[j]}"
                            heads_path = prev_task_logdir / f"heads_task{j+1}.pt"
                            ac_path = prev_task_logdir / f"actor_critic_task{j+1}.pt"
                            if heads_path.exists() and ac_path.exists():
                                heads_sd = torch.load(heads_path, map_location=config.device)
                                ac_sd = torch.load(ac_path, map_location=config.device)
                                h_loaded = load_partial_state_dict(agent, heads_sd)
                                ac_loaded = load_partial_state_dict(agent, ac_sd)
                                print(f"    Eval {task_label}: swapped in saved heads "
                                      f"({h_loaded} tensors) + actor-critic ({ac_loaded} tensors)")
                                del heads_sd, ac_sd
                            else:
                                print(f"    WARNING: Missing saved checkpoints for task {j+1} "
                                      f"({tasks[j]}), using current agent weights for eval")

                        eval_policy = functools.partial(agent, training=False)

                        prefixed_logger = PrefixedLogger(
                            logger, task_label, record_video=record_video,
                        )

                        # Create eval envs on demand, close immediately after
                        eval_cfg = task_configs[j]
                        if config.parallel:
                            eval_envs_j = [
                                Parallel(LazyParallelEnv(tasks[j], eval_cfg, "eval", i), "process")
                                for i in range(config.envs)
                            ]
                        else:
                            eval_envs_j = [make_env(tasks[j], eval_cfg, "eval", i) for i in range(config.envs)]
                            eval_envs_j = [Damy(env) for env in eval_envs_j]

                        tools.simulate(
                            eval_policy,
                            eval_envs_j,
                            all_eval_caches[j],
                            all_eval_dirs[j],
                            prefixed_logger,
                            is_eval=True,
                            episodes=config.eval_episode_num,
                        )

                        for env in eval_envs_j:
                            try:
                                env.close()
                            except Exception:
                                pass
                        del eval_envs_j

                        # Restore current agent weights after evaluating a previous task
                        if not is_current_task:
                            agent.load_state_dict(current_agent_sd)

                        print(f"    Eval {task_label}: done")

                    del current_agent_sd

                    if config.video_pred_log:
                        eval_dataset = make_dataset(all_eval_caches[task_idx], config)
                        video_pred = agent._wm.video_pred(next(eval_dataset))
                        logger.video("eval_openl", to_np(video_pred))

                    logger.write(step=logger.step)

                # === TRAINING ===
                print(f">>> SEQUENTIAL EWC: Training task {task_idx+1} "
                      f"(step {agent._step - task_start_step}/{config.steps})")
                state = tools.simulate(
                    agent,
                    train_envs,
                    train_eps,
                    traindir,
                    logger,
                    limit=config.dataset_size,
                    steps=config.eval_every,
                    state=state,
                )
                updated_task_train_steps = min(max(agent._step - task_start_step, 0), config.steps)
                step_delta = max(0, updated_task_train_steps - task_train_steps_done)
                if step_delta:
                    progress_bar.update(step_delta)
                    task_train_steps_done = updated_task_train_steps

                # Save checkpoint (with step info for resume)
                items_to_save = {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                    "logger_step": logger.step,
                    "task_start_step": task_start_step,
                }
                torch.save(items_to_save, task_logdir / "latest.pt")
        finally:
            progress_bar.close()

        # ============================================================
        # [EWC] Compute Fisher and consolidate (one-time per task)
        # ============================================================
        if task_idx < num_tasks - 1:
            print(f">>> EWC: Computing Fisher for task {task_idx+1} ({task_name})...")
            t0 = _time.time()

            fisher_dataset = make_dataset(train_eps, config)
            importance, task_param = ewc_manager.compute_fisher(
                agent._wm, fisher_dataset, config,
                num_batches=args.ewc_fisher_batches,
                device=config.device,
            )
            ewc_manager.consolidate(agent._wm, importance, task_param, task_idx)

            elapsed = _time.time() - t0
            n_params = sum(v.numel() for v in importance.values())
            fisher_mean = sum(v.mean().item() for v in importance.values()) / max(len(importance), 1)
            fisher_max = max(v.max().item() for v in importance.values())
            print(f">>> EWC: Consolidated task {task_idx+1} in {elapsed:.1f}s: "
                  f"{len(importance)} groups, {n_params:,} params, "
                  f"mean Fisher={fisher_mean:.2e}, max Fisher={fisher_max:.2e}")

            # Log Fisher stats
            logger.scalar("ewc/fisher_mean", fisher_mean)
            logger.scalar("ewc/fisher_max", fisher_max)
            logger.scalar("ewc/fisher_compute_time_s", elapsed)
            logger.scalar("ewc/num_tasks_consolidated", ewc_manager.num_tasks_consolidated)
            logger.write(step=logger.step)

            # Save EWC state for resume
            torch.save(ewc_manager.state_dict(),
                       task_logdir / f"ewc_state_task{task_idx+1}.pt")
            print(f">>> EWC: Saved state -> ewc_state_task{task_idx+1}.pt")

        # ============================================================
        # Save separated checkpoints
        # ============================================================
        full_sd = agent.state_dict()

        if items_to_save is not None:
            torch.save(items_to_save, task_logdir / f"checkpoint_task{task_idx+1}.pt")

        def _sd_param_count(sd):
            return sum(v.numel() for v in sd.values())

        # RSSM checkpoint
        rssm_sd = extract_rssm_state_dict(full_sd)
        torch.save(rssm_sd, task_logdir / f"rssm_task{task_idx+1}.pt")
        print(f">>> SEQUENTIAL EWC: Saved RSSM checkpoint "
              f"({len(rssm_sd)} tensors, {_sd_param_count(rssm_sd):,} params) "
              f"-> rssm_task{task_idx+1}.pt")

        # Reward + continue heads
        heads_sd = extract_heads_state_dict(full_sd)
        torch.save(heads_sd, task_logdir / f"heads_task{task_idx+1}.pt")
        print(f">>> SEQUENTIAL EWC: Saved heads checkpoint "
              f"({len(heads_sd)} tensors, {_sd_param_count(heads_sd):,} params) "
              f"-> heads_task{task_idx+1}.pt")

        # Actor-critic
        ac_sd = extract_actor_critic_state_dict(full_sd)
        torch.save(ac_sd, task_logdir / f"actor_critic_task{task_idx+1}.pt")
        print(f">>> SEQUENTIAL EWC: Saved actor-critic checkpoint "
              f"({len(ac_sd)} tensors, {_sd_param_count(ac_sd):,} params) "
              f"-> actor_critic_task{task_idx+1}.pt")

        # Update global state
        global_env_step = logger.step
        prev_rssm_checkpoint = str(task_logdir / f"rssm_task{task_idx+1}.pt")

        # Save progress: mark this task as completed
        save_sequential_progress(
            base_logdir, task_idx,
            completed=True,
            global_env_step_at_end=global_env_step,
        )

        # Detach EWC manager from this world model before cleanup
        agent._wm.ewc_manager = None

        # Cleanup train envs and free memory
        for env in train_envs:
            try:
                env.close()
            except Exception:
                pass
        del train_envs, train_dataset, train_eps, agent, items_to_save
        del all_eval_dirs, all_eval_caches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f">>> SEQUENTIAL EWC: Task {task_idx+1} ({task_name}) completed "
              f"at global step {global_env_step}")
        print()

    # ============================================================
    # Save final RSSM after all tasks
    # ============================================================
    if prev_rssm_checkpoint is not None:
        final_rssm_path = base_logdir / "rssm_final.pt"
        shutil.copy2(prev_rssm_checkpoint, final_rssm_path)
        print(f">>> SEQUENTIAL EWC: Saved final RSSM -> {final_rssm_path}")

    # Finish logging
    if hasattr(logger, "finish"):
        logger.finish()

    print("=" * 60)
    print(">>> SEQUENTIAL EWC: All tasks completed successfully!")
    print(f">>> SEQUENTIAL EWC: Final global step: {global_env_step}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential DreamerV3 training with EWC and separated architecture",
    )

    # Multi-task arguments
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="List of task names (e.g., metaworld_drawer-open-v3 metaworld_pick-place-v3)",
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Config profile per task (e.g., debug metaworld_default_light)",
    )
    parser.add_argument(
        "--task-steps", nargs="+", type=int, required=True,
        help="Training steps per task in env steps (e.g., 200000 200000)",
    )

    # Logging
    parser.add_argument("--logdir", type=str, required=True, help="Base log directory")
    parser.add_argument("--logger", type=str, default="wandb", help="wandb or tensorboard")
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3_Sequential_EWC")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Checkpoint
    parser.add_argument(
        "--from-checkpoint", type=str, default=None,
        help="Path to initial full checkpoint for the first task",
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip pretraining on the first task",
    )
    parser.add_argument(
        "--skip-config-check", action="store_true",
        help="Skip interactive confirmation before training",
    )

    # Per-task dataset size overrides
    parser.add_argument(
        "--dataset-sizes", nargs="+", type=int, default=None,
        help="Override dataset_size per task (e.g., 400000 400000 800000). "
             "If omitted, each task uses the dataset_size from its config profile.",
    )

    # EWC arguments
    parser.add_argument(
        "--ewc-lambda", type=float, default=5000.0,
        help="EWC regularization strength (default: 5000.0). "
             "Sweep [500, 1000, 5000, 10000, 50000] to calibrate.",
    )
    parser.add_argument(
        "--ewc-fisher-batches", type=int, default=50,
        help="Number of mini-batches for Fisher estimation at each task boundary "
             "(default: 50, ~800 sequence samples at batch_size=16).",
    )

    # Eval options
    parser.add_argument(
        "--eval-prev-video", action="store_true", default=True,
        help="Record eval videos for previous tasks (default: True)",
    )
    parser.add_argument(
        "--no-eval-prev-video", dest="eval_prev_video", action="store_false",
        help="Disable eval videos for previous tasks",
    )

    main_args, remaining = parser.parse_known_args()
    main(main_args, remaining)
