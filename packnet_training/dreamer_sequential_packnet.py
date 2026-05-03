"""Sequential training script for DreamerV3 with PackNet (conservative / paper-faithful).

Conservative variant that matches the original PackNet paper behavior:
  - Only weight matrices (ndim >= 2) are pruned and masked per-task.
  - Bias and normalization parameters (ndim == 1) are frozen after the first
    task and shared across all subsequent tasks (NOT pruned).

Uses the disentangled model architecture from models.py:
  - RSSMWorldModel: shared backbone (encoder, dynamics, decoder) -- persists across tasks
  - TaskHeads: per-task reward & continuation heads -- fresh for each task
  - ActorCritic: per-task actor & value networks -- fresh for each task

Continual learning protocol:
  1. Init RSSM + task1 heads + task1 actor-critic
  2. Train on task1
  3. Prune RSSM weight matrices -> retrain -> freeze surviving weights + all bias/norm
  4. Save task1 checkpoints + PackNet masks
  5. Evaluate task1 with (masked RSSM + task1 modules)
  6. Init fresh task2 heads + task2 actor-critic
  7. Continue training RSSM on task2 (only free weight params get gradients;
     bias/norm stay frozen from task 1)
  8. Prune -> retrain -> freeze
  9. Evaluate task1 with (task1-masked RSSM + task1 modules),
     task2 with (task2-masked RSSM + task2 modules)
  10. ...

Usage:
    python dreamer_sequential_packnet_conservative.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --configs metaworld_visual_200M_heavy_long metaworld_visual_200M_heavy_long \\
        --task-steps 200000 500000 \\
        --logdir ./logdir/sequential_packnet_conservative_run \\
        --packnet-prune-ratio 0.75 \\
        --packnet-retrain-ratio 0.1 \\
        --wandb-entity my-entity \\
        --wandb-project my-project
"""

import argparse
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
os.environ["EGL_LOG_LEVEL"] = "fatal"
import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

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
gymnasium.logger.min_level = gymnasium.logger.ERROR

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

# PackNet (paper-faithful: prune weights only, freeze bias/norm)
from packnet import PackNetManager


to_np = lambda x: x.detach().cpu().numpy()


# ===========================================================================
#  GPU memory diagnostics
# ===========================================================================

def _gpu_mem_str():
    """Return a short string describing current GPU memory usage."""
    if not torch.cuda.is_available():
        return "GPU: N/A"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"GPU mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.1f}GB total"


def _sys_mem_str():
    """Return a short string describing current system (RAM) memory usage."""
    import psutil
    vm = psutil.virtual_memory()
    used = vm.used / 1024**3
    total = vm.total / 1024**3
    pct = vm.percent
    proc = psutil.Process().memory_info()
    rss = proc.rss / 1024**3
    return f"RAM: {used:.2f}GB / {total:.1f}GB ({pct}%) | proc RSS: {rss:.2f}GB"


def _log_mem(tag):
    """Print a tagged memory snapshot for OOM debugging."""
    print(f"  [MEM] {tag}: {_gpu_mem_str()} | {_sys_mem_str()}")


def _force_cleanup():
    """Aggressive memory cleanup: delete caches, run GC, empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===========================================================================
#  Training agent -- holds shared RSSM + current task's modules
# ===========================================================================

class SequentialDreamer(nn.Module):
    """Training agent for sequential continual learning with PackNet.

    Holds three independent nn.Modules:
      _rssm:          shared across tasks  (RSSMWorldModel)
      _task_heads:     per-task, replaced when switching tasks  (TaskHeads)
      _actor_critic:   per-task, replaced when switching tasks  (ActorCritic)
    """

    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
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

        _log_mem("Before RSSM creation")
        self._rssm = models.RSSMWorldModel(obs_space, act_space, self._step, config)
        _log_mem("After RSSM creation")

        _log_mem("Before TaskHeads creation")
        self._task_heads = models.TaskHeads(config)
        _log_mem("After TaskHeads creation")

        _log_mem("Before ActorCritic creation")
        self._actor_critic = models.ActorCritic(config)
        _log_mem("After ActorCritic creation")

        # Optional torch.compile
        if config.compile and os.name != "nt":
            self._rssm = torch.compile(self._rssm, mode="reduce-overhead")
            self._task_heads = torch.compile(self._task_heads, mode="reduce-overhead")
            self._actor_critic = torch.compile(self._actor_critic, mode="reduce-overhead")

        # Exploration behavior
        reward = lambda f, s, a: self._task_heads.reward(f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._actor_critic,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._rssm, reward),
        )[config.expl_behavior]()
        if isinstance(self._expl_behavior, nn.Module):
            self._expl_behavior = self._expl_behavior.to(self._config.device)

    # ---- called by tools.simulate during rollout --------------------------

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
                    openl = self._rssm.video_pred(next(self._dataset))
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
        obs = self._rssm.preprocess(obs)
        embed = self._rssm.encoder(obs)
        latent, _ = self._rssm.dynamics.obs_step(
            latent, action, embed, obs["is_first"]
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._rssm.dynamics.get_feat(latent)

        if not training:
            actor = self._actor_critic.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._actor_critic.actor(feat)
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

        # 1. Train RSSM backbone + task heads jointly
        post, context, mets = models.train_world_model_step(
            self._rssm, self._task_heads, data
        )
        metrics.update(mets)

        # 2. Train actor-critic on imagined trajectories
        start = post
        def reward_fn(feat, state, action):
            return self._task_heads.reward(
                self._rssm.get_feat(state)
            ).mode()

        ac_results = self._actor_critic._train(
            start, reward_fn, self._rssm, self._task_heads
        )
        metrics.update(ac_results[-1])

        # 3. Exploration (no-op for greedy)
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


# ===========================================================================
#  Lightweight eval-only agent -- no training, no weight swapping
# ===========================================================================

class EvalAgent:
    """Pairs a shared RSSM with a task-specific actor for evaluation."""

    def __init__(self, rssm, actor, config):
        self._rssm = rssm        # shared, NOT copied
        self._actor = actor       # task-specific, loaded from checkpoint
        self._config = config

    @torch.no_grad()
    def __call__(self, obs, reset, state=None, training=False):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._rssm.preprocess(obs)
        embed = self._rssm.encoder(obs)
        latent, _ = self._rssm.dynamics.obs_step(
            latent, action, embed, obs["is_first"]
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._rssm.dynamics.get_feat(latent)
        actor_dist = self._actor(feat)
        action = actor_dist.mode()
        logprob = actor_dist.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        return {"action": action, "logprob": logprob}, (latent, action)


# ===========================================================================
#  Helpers
# ===========================================================================

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(task_name, config, mode, id):
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
    raise NotImplementedError(f"Suite '{suite}' not supported in sequential training.")


class LazyParallelEnv:
    """Picklable proxy that lazily constructs the real env inside the worker."""

    def __init__(self, task_name, config, mode, env_id):
        self._task_name = task_name
        self._config = config
        self._mode = mode
        self._env_id = env_id
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = make_env(
                self._task_name, self._config, self._mode, self._env_id
            )
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
    """Wraps a logger to prefix metric names for task-specific eval logging."""

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
        pass  # suppress -- main loop flushes all eval metrics in one write

    @property
    def step(self):
        return self.real_logger.step

    @step.setter
    def step(self, value):
        self.real_logger.step = value


# ===========================================================================
#  Sequential progress tracking (for resume support)
# ===========================================================================

def save_sequential_progress(base_logdir, task_idx, global_env_step_at_start=None,
                             completed=False, global_env_step_at_end=None):
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
    progress_file = base_logdir / "sequential_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return None


# ===========================================================================
#  Per-task checkpoint save / load
# ===========================================================================

def save_task_checkpoint(agent, task_logdir, logger_step, task_start_step):
    """Save all three components + resume metadata for a task."""
    step = agent._step
    _log_mem("Before checkpoint save")

    tools.save_component(
        agent._rssm, task_logdir / "rssm.pt", step=step,
    )
    tools.save_component(
        agent._task_heads, task_logdir / "task_heads.pt", step=step,
    )
    tools.save_component(
        agent._actor_critic, task_logdir / "actor_critic.pt", step=step,
    )

    manifest = {
        "step": step,
        "logger_step": logger_step,
        "task_start_step": task_start_step,
    }
    torch.save(manifest, task_logdir / "manifest.pt")
    _log_mem("After checkpoint save")


def load_task_checkpoint(agent, task_logdir, load_optimizers=True, device=None):
    """Load all three components + resume manifest for a task."""
    _log_mem("Before checkpoint load")
    tools.load_component(
        agent._rssm, task_logdir / "rssm.pt",
        load_optimizers=load_optimizers, device=device,
    )
    tools.load_component(
        agent._task_heads, task_logdir / "task_heads.pt",
        load_optimizers=load_optimizers, device=device,
    )
    tools.load_component(
        agent._actor_critic, task_logdir / "actor_critic.pt",
        load_optimizers=load_optimizers, device=device,
    )
    manifest = torch.load(task_logdir / "manifest.pt", map_location=device)
    _log_mem("After checkpoint load")
    return manifest


# ===========================================================================
#  Config builder
# ===========================================================================

def build_task_config(task_name, config_name, task_steps, configs_yaml, remaining_args):
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


# ===========================================================================
#  Evaluation helpers
# ===========================================================================

def _create_eval_envs(task_name, config, num_envs, parallel):
    _log_mem(f"Before creating eval envs for {task_name}")
    if parallel:
        envs = [
            Parallel(LazyParallelEnv(task_name, config, "eval", i), "process")
            for i in range(num_envs)
        ]
    else:
        envs = [
            Damy(make_env(task_name, config, "eval", i))
            for i in range(num_envs)
        ]
    _log_mem(f"After creating eval envs for {task_name}")
    return envs


def _close_envs(envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def evaluate_task(
    task_idx, task_name, rssm, actor, config, eval_cache, eval_dir,
    logger, record_video, eval_episodes,
):
    """Evaluate a single task using the shared RSSM + a task-specific actor."""
    task_label = f"eval_task{task_idx+1}_{task_name}"
    prefixed_logger = PrefixedLogger(logger, task_label, record_video=record_video)

    eval_agent = EvalAgent(rssm, actor, config)

    _log_mem(f"Before eval envs for task {task_idx+1} ({task_name})")
    # We manually constrain eval envs to 2 or fewer to avoid OOM issues during parallel evaluation
    eval_envs = _create_eval_envs(
        task_name, config, min(config.envs, 2), config.parallel,
    )
    _log_mem(f"After eval envs for task {task_idx+1} ({task_name})")

    try:
        tools.simulate(
            eval_agent,
            eval_envs,
            eval_cache,
            eval_dir,
            prefixed_logger,
            is_eval=True,
            episodes=eval_episodes,
        )
    finally:
        _close_envs(eval_envs)
        del eval_envs
        _log_mem(f"After eval cleanup for task {task_idx+1} ({task_name})")

    print(f"    Eval {task_label}: done")


def run_cross_task_evaluation(
    agent, task_idx, tasks, task_configs, base_logdir,
    all_eval_dirs, all_eval_caches, logger, args,
    packnet_manager,
):
    """Evaluate ALL tasks seen so far (current + previous).

    For the current task: uses the agent's current RSSM + actor-critic.
    For previous tasks: applies PackNet task mask to RSSM, loads saved
    actor-critic, evaluates, then restores RSSM weights.
    """
    config = task_configs[task_idx]
    print(f">>> SEQUENTIAL PackNet: Evaluation at global step {logger.step} "
          f"(evaluating {task_idx + 1} task(s))")
    _log_mem("Before cross-task evaluation")

    for j in range(task_idx + 1):
        is_current_task = (j == task_idx)
        record_video = is_current_task or args.eval_prev_video

        if is_current_task:
            # Current task: use the training agent's actor directly
            # If task has been packed (pruned+frozen), apply its mask
            if j in packnet_manager.task_masks:
                saved_weights = packnet_manager.save_rssm_weights(agent._rssm)
                packnet_manager.apply_eval_mask(agent._rssm, j)
                try:
                    evaluate_task(
                        j, tasks[j], agent._rssm, agent._actor_critic.actor,
                        config, all_eval_caches[j], all_eval_dirs[j],
                        logger, record_video, config.eval_episode_num,
                    )
                finally:
                    packnet_manager.restore_rssm_weights(agent._rssm, saved_weights)
                    del saved_weights
            else:
                # Not yet packed (during training before prune), use as-is
                evaluate_task(
                    j, tasks[j], agent._rssm, agent._actor_critic.actor,
                    config, all_eval_caches[j], all_eval_dirs[j],
                    logger, record_video, config.eval_episode_num,
                )
        else:
            # Previous task: apply PackNet mask + load saved actor-critic
            prev_task_logdir = base_logdir / f"task{j+1}_{tasks[j]}"
            ac_path = prev_task_logdir / "actor_critic.pt"

            if ac_path.exists() and j in packnet_manager.task_masks:
                _log_mem(f"Before loading prev task {j+1} actor-critic")
                prev_ac = models.ActorCritic(task_configs[j]).to(config.device)
                prev_ac.requires_grad_(False)
                tools.load_component(prev_ac, ac_path, load_optimizers=False)
                _log_mem(f"After loading prev task {j+1} actor-critic")

                # Apply PackNet mask for this task's RSSM view
                saved_weights = packnet_manager.save_rssm_weights(agent._rssm)
                packnet_manager.apply_eval_mask(agent._rssm, j)

                try:
                    evaluate_task(
                        j, tasks[j], agent._rssm, prev_ac.actor,
                        task_configs[j], all_eval_caches[j], all_eval_dirs[j],
                        logger, record_video, config.eval_episode_num,
                    )
                finally:
                    packnet_manager.restore_rssm_weights(agent._rssm, saved_weights)
                    del saved_weights
                    del prev_ac
                    _force_cleanup()
                    _log_mem(f"After freeing prev task {j+1} actor-critic")
            else:
                print(f"    WARNING: No actor_critic.pt or PackNet mask for "
                      f"task {j+1} ({tasks[j]}), skipping eval")

    # Video prediction for current task
    if config.video_pred_log and len(all_eval_caches.get(task_idx, {})) > 0:
        eval_dataset = make_dataset(all_eval_caches[task_idx], config)
        video_pred = agent._rssm.video_pred(next(eval_dataset))
        logger.video("eval_openl", to_np(video_pred))

    logger.write(step=logger.step)
    _log_mem("After cross-task evaluation complete")


# ===========================================================================
#  Main sequential training loop
# ===========================================================================

def main(args, remaining_args):
    tasks = args.tasks
    config_names = args.configs
    task_steps_list = args.task_steps
    dataset_sizes_list = args.dataset_sizes
    num_tasks = len(tasks)

    assert len(config_names) == num_tasks
    assert len(task_steps_list) == num_tasks
    if dataset_sizes_list is not None:
        assert len(dataset_sizes_list) == num_tasks

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
    # PackNet setup
    # ================================================================
    packnet_manager = PackNetManager(
        prune_ratio=args.packnet_prune_ratio,
        rssm_prefixes=("encoder.", "dynamics.", "decoder."),
    )
    retrain_ratio = args.packnet_retrain_ratio

    # ================================================================
    #  Resume detection
    # ================================================================
    global_env_step = 0
    prev_rssm_path = None
    resume_from_task_idx = 0

    progress = load_sequential_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            task_info = progress.get("tasks", {}).get(str(i))
            if task_info and task_info.get("completed"):
                resume_from_task_idx = i + 1
                global_env_step = task_info["global_env_step_at_end"]
                task_logdir_i = base_logdir / f"task{i+1}_{tasks[i]}"
                rssm_file = task_logdir_i / "rssm.pt"
                if rssm_file.exists():
                    prev_rssm_path = str(rssm_file)
                print(f">>> RESUME: Task {i+1} ({tasks[i]}) already completed "
                      f"(ended at global step {global_env_step}), skipping.")
            else:
                break

    # [PackNet] Resume: load PackNet state if available from the latest completed task
    if resume_from_task_idx > 0:
        for j in range(resume_from_task_idx - 1, -1, -1):
            pn_file = base_logdir / f"task{j+1}_{tasks[j]}" / f"packnet_state_task{j+1}.pt"
            if pn_file.exists():
                print(f">>> PackNet: Loading state from {pn_file}")
                pn_state = torch.load(pn_file, map_location="cpu")
                packnet_manager.load_state_dict(pn_state)
                print(f">>> PackNet: Restored {packnet_manager.num_tasks_packed} "
                      f"packed task(s)")
                break

    # For within-task resume, peek at manifest
    initial_logger_step = global_env_step
    if resume_from_task_idx < num_tasks:
        current_task_logdir = (
            base_logdir / f"task{resume_from_task_idx+1}_{tasks[resume_from_task_idx]}"
        )
        manifest_path = current_task_logdir / "manifest.pt"
        if manifest_path.exists():
            try:
                peek = torch.load(manifest_path, map_location="cpu")
                initial_logger_step = peek.get("logger_step", global_env_step)
                print(f">>> RESUME: Will resume within task {resume_from_task_idx+1} "
                      f"at logger_step={initial_logger_step}")
            except Exception as e:
                print(f">>> RESUME: Could not peek at manifest: {e}")

    if resume_from_task_idx > 0:
        print(f">>> RESUME: Resuming from task {resume_from_task_idx + 1} "
              f"(global_env_step={global_env_step})")
    if resume_from_task_idx >= num_tasks:
        print(">>> RESUME: All tasks already completed. Nothing to do.")
        return

    # ================================================================
    #  Logger
    # ================================================================
    first_config = task_configs[0]
    if args.logger == "tensorboard":
        logger = tools.Logger(base_logdir, initial_logger_step)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, first_config, base_logdir, initial_logger_step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # Print plan
    print("=" * 60)
    print(">>> SEQUENTIAL PackNet TRAINING WITH CROSS-TASK EVAL <<<")
    print("=" * 60)
    for i, (task, cfg_name, steps) in enumerate(
        zip(tasks, config_names, task_steps_list)
    ):
        ds = int(task_configs[i].dataset_size)
        marker = " <-- resume here" if i == resume_from_task_idx else ""
        print(f"  Task {i+1}: {task} (config: {cfg_name}, "
              f"steps: {steps}, dataset_size: {ds}){marker}")
    print(f"  Log directory: {base_logdir}")
    print(f"  PackNet prune ratio: {args.packnet_prune_ratio}")
    print(f"  PackNet retrain ratio: {retrain_ratio} (fraction of task steps)")
    print(f"  Eval previous task videos: {args.eval_prev_video}")
    print("=" * 60)
    _log_mem("Before training starts")

    if not args.skip_config_check:
        input(">>> Press Enter to start sequential PackNet training...")

    # ================================================================
    #  Task loop
    # ================================================================
    for task_idx in range(num_tasks):
        if task_idx < resume_from_task_idx:
            continue

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

        print("=" * 60)
        print(f">>> SEQUENTIAL PackNet: Starting Task {task_idx+1}/{num_tasks}: {task_name}")
        print(f">>> SEQUENTIAL PackNet: Task steps: {config.steps * config.action_repeat}")
        print(f">>> SEQUENTIAL PackNet: Global env step: {global_env_step}")
        print(f">>> SEQUENTIAL PackNet: Tasks packed so far: "
              f"{packnet_manager.num_tasks_packed}")
        print("=" * 60)

        # ---- Create train environments ----
        _log_mem("Before creating train envs")
        if config.parallel:
            train_envs = [
                Parallel(LazyParallelEnv(task_name, config, "train", i), "process")
                for i in range(config.envs)
            ]
        else:
            train_envs = [
                Damy(make_env(task_name, config, "train", i))
                for i in range(config.envs)
            ]
        _log_mem("After creating train envs")

        acts = train_envs[0].action_space
        print(f">>> SEQUENTIAL PackNet: Action Space: {acts}")
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        for tc in task_configs:
            tc.num_actions = config.num_actions

        # ---- Prepare eval dirs/caches ----
        all_eval_dirs = {}
        all_eval_caches = {}
        for j in range(task_idx + 1):
            eval_dir_j = task_logdir / f"eval_eps_task{j+1}_{tasks[j]}"
            eval_dir_j.mkdir(parents=True, exist_ok=True)
            all_eval_dirs[j] = eval_dir_j
            all_eval_caches[j] = tools.load_episodes(eval_dir_j, limit=1)

        # ---- Logger step ----
        logger.step = global_env_step
        task_global_env_step_at_start = global_env_step
        if progress is not None:
            saved_start = (progress.get("tasks", {})
                           .get(str(task_idx), {})
                           .get("global_env_step_at_start"))
            if saved_start is not None:
                task_global_env_step_at_start = saved_start
                global_env_step = saved_start
                logger.step = global_env_step

        # ---- Load train episodes ----
        _log_mem("Before loading train episodes")
        train_eps = tools.load_episodes(traindir, limit=config.dataset_size)
        tools.erase_over_episodes(train_eps, config.dataset_size)
        if config.dataset_size:
            tools.erase_over_episode_files(traindir, train_eps)
        _log_mem("After loading train episodes")

        # ---- Prefill replay buffer ----
        state = None
        if not config.offline_traindir:
            prefill = max(0, config.prefill - count_steps(traindir))
            print(f">>> SEQUENTIAL PackNet: Prefill dataset ({prefill} steps).")
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
                    random_agent, train_envs, train_eps, traindir, logger,
                    limit=config.dataset_size, steps=prefill,
                )
                logger.step += prefill * config.action_repeat
                print(f">>> SEQUENTIAL PackNet: Logger step after prefill: {logger.step}")

        # ---- Create agent ----
        _log_mem("Before agent creation")
        print(">>> SEQUENTIAL PackNet: Creating agent.")
        train_dataset = make_dataset(train_eps, config)
        agent = SequentialDreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config, logger, train_dataset,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        _log_mem("After agent creation + .to(device)")

        # ============================================================
        # [PackNet] Attach PackNet manager to RSSM
        # ============================================================
        agent._rssm.packnet_manager = packnet_manager

        # Move masks to device
        packnet_manager.to_device(config.device)

        # Register param categories (prunable weights vs shared bias/norm)
        packnet_manager.register_rssm_params(agent._rssm)

        if packnet_manager.num_tasks_packed > 0:
            n_frozen = sum(v.sum().item() for v in packnet_manager.frozen_mask.values())
            n_total_w = sum(
                p.numel() for n, p in agent._rssm.named_parameters()
                if packnet_manager._is_prunable(n, p)
            )
            print(f">>> PackNet: {int(n_frozen):,}/{n_total_w:,} weight params frozen "
                  f"from {packnet_manager.num_tasks_packed} previous task(s), "
                  f"shared params frozen={packnet_manager._shared_params_frozen}")

        # ---- Load checkpoint ----
        resuming_within_task = False
        if (task_logdir / "manifest.pt").exists():
            print(f">>> SEQUENTIAL PackNet: Resuming from: {task_logdir}")
            manifest = load_task_checkpoint(agent, task_logdir, load_optimizers=True)
            resuming_within_task = True
        elif task_idx > 0 and prev_rssm_path is not None:
            if os.path.exists(prev_rssm_path):
                print(f">>> SEQUENTIAL PackNet: Loading RSSM from previous task: {prev_rssm_path}")
                _log_mem("Before RSSM load from previous task")
                tools.load_component(
                    agent._rssm, prev_rssm_path, load_optimizers=True,
                )
                _log_mem("After RSSM load from previous task")
                print(f">>> SEQUENTIAL PackNet: TaskHeads and ActorCritic start FRESH for task {task_idx+1}.")
            else:
                raise FileNotFoundError(f"RSSM checkpoint not found: {prev_rssm_path}")
            manifest = None
        elif task_idx == 0 and args.from_checkpoint is not None:
            ckpt_dir = pathlib.Path(args.from_checkpoint)
            if (ckpt_dir / "manifest.pt").exists():
                print(f">>> SEQUENTIAL PackNet: Loading full checkpoint from: {ckpt_dir}")
                manifest = load_task_checkpoint(agent, ckpt_dir, load_optimizers=True)
            else:
                raise FileNotFoundError(f"No manifest.pt in {ckpt_dir}")
        else:
            manifest = None

        # Skip pretraining for task 2+ or resume
        if task_idx > 0 or args.skip_pretrain or resuming_within_task:
            print(">>> SEQUENTIAL PackNet: Skipping pretraining (sequential continuation).")
            agent._should_pretrain._once = False

        # Restore step tracking
        if resuming_within_task and manifest is not None:
            logger.step = manifest["logger_step"]
            agent._step = logger.step // config.action_repeat
            task_start_step = manifest["task_start_step"]
            print(f">>> RESUME: Restored logger.step={logger.step}, "
                  f"agent._step={agent._step}, task_start_step={task_start_step}")
        else:
            task_start_step = agent._step

        print(f">>> SEQUENTIAL PackNet: Agent step: {agent._step}, task_start_step: {task_start_step}")

        # Mark task as started
        save_sequential_progress(
            base_logdir, task_idx,
            global_env_step_at_start=task_global_env_step_at_start,
        )

        # ---- Main training loop ----
        task_train_steps_done = min(max(agent._step - task_start_step, 0), config.steps)
        progress_bar = tqdm(
            total=config.steps,
            initial=task_train_steps_done,
            desc=f">>> Task {task_idx+1}/{num_tasks} PackNet Training",
            unit="step",
        )
        try:
            while (agent._step - task_start_step) < config.steps + config.eval_every:
                logger.write()

                # === EVALUATION ===
                if config.eval_episode_num > 0:
                    run_cross_task_evaluation(
                        agent, task_idx, tasks, task_configs, base_logdir,
                        all_eval_dirs, all_eval_caches, logger, args,
                        packnet_manager,
                    )

                # === TRAINING ===
                print(f">>> SEQUENTIAL PackNet: Training task {task_idx+1} "
                      f"(step {agent._step - task_start_step}/{config.steps})")
                _log_mem("Before training simulate")
                state = tools.simulate(
                    agent, train_envs, train_eps, traindir, logger,
                    limit=config.dataset_size, steps=config.eval_every,
                    state=state,
                )
                _log_mem("After training simulate")

                # Update progress bar
                updated = min(max(agent._step - task_start_step, 0), config.steps)
                delta = max(0, updated - task_train_steps_done)
                if delta:
                    progress_bar.update(delta)
                    task_train_steps_done = updated

                # Save checkpoint
                save_task_checkpoint(agent, task_logdir, logger.step, task_start_step)

        finally:
            progress_bar.close()

        # ============================================================
        # [PackNet] Prune, retrain, freeze (one-time per task)
        # ============================================================
        # Skip pruning for the last task (no more tasks to add)
        if task_idx < num_tasks - 1:
            print(f">>> PackNet: Pruning task {task_idx+1} ({task_name})...")
            t0 = _time.time()

            # Step 1: Prune
            task_mask = packnet_manager.prune(agent._rssm, task_idx)

            # Step 2: Retrain with pruned weights frozen
            # Compute retrain steps as a fraction of this task's training steps
            task_env_steps = task_steps_list[task_idx]
            retrain_steps = int(task_env_steps * retrain_ratio)
            if retrain_steps > 0:
                retrain_steps_action = retrain_steps // config.action_repeat
                print(f">>> PackNet: Retraining for {retrain_steps} env steps "
                      f"({retrain_ratio*100:.0f}% of {task_env_steps} task steps, "
                      f"{retrain_steps_action} action steps)...")
                packnet_manager.start_retrain(task_mask)

                retrain_start_step = agent._step
                retrain_bar = tqdm(
                    total=retrain_steps_action,
                    desc=f">>> Task {task_idx+1} PackNet Retrain",
                    unit="step",
                )
                retrain_done = 0
                try:
                    while (agent._step - retrain_start_step) < retrain_steps_action:
                        state = tools.simulate(
                            agent, train_envs, train_eps, traindir, logger,
                            limit=config.dataset_size,
                            steps=min(config.eval_every, retrain_steps_action - retrain_done),
                            state=state,
                        )
                        new_done = min(agent._step - retrain_start_step, retrain_steps_action)
                        delta = max(0, new_done - retrain_done)
                        if delta:
                            retrain_bar.update(delta)
                            retrain_done = new_done
                finally:
                    retrain_bar.close()

                packnet_manager.end_retrain()

            # Step 3: Freeze surviving weights
            packnet_manager.freeze_task(task_mask, task_idx)

            elapsed = _time.time() - t0
            print(f">>> PackNet: Pruned + retrained + frozen task {task_idx+1} "
                  f"in {elapsed:.1f}s")

            # Log stats
            n_frozen = sum(v.sum().item() for v in packnet_manager.frozen_mask.values())
            n_total = sum(v.numel() for v in packnet_manager.frozen_mask.values())
            logger.scalar("packnet/frozen_ratio", n_frozen / max(n_total, 1))
            logger.scalar("packnet/num_tasks_packed", packnet_manager.num_tasks_packed)
            logger.scalar("packnet/prune_retrain_time_s", elapsed)
            logger.write(step=logger.step)

            # Save PackNet state for resume
            torch.save(packnet_manager.state_dict(),
                       task_logdir / f"packnet_state_task{task_idx+1}.pt")
            print(f">>> PackNet: Saved state -> packnet_state_task{task_idx+1}.pt")

        else:
            # Last task: still create a task mask for eval (no pruning needed)
            # The task mask = everything active (frozen + free)
            task_mask = {}
            for name, param in agent._rssm.named_parameters():
                if packnet_manager._is_rssm_param(name):
                    # All weights are active for the last task
                    task_mask[name] = torch.ones_like(param.data)
            packnet_manager.task_masks[task_idx] = task_mask
            print(f">>> PackNet: Last task {task_idx+1} -- stored full mask (no pruning)")

            # Save PackNet state
            torch.save(packnet_manager.state_dict(),
                       task_logdir / f"packnet_state_task{task_idx+1}.pt")

        # ---- Save final per-task checkpoints ----
        save_task_checkpoint(agent, task_logdir, logger.step, task_start_step)

        for src_name in ["rssm.pt", "task_heads.pt", "actor_critic.pt"]:
            src = task_logdir / src_name
            if src.exists():
                dst = task_logdir / src_name.replace(
                    ".pt", f"_task{task_idx+1}.pt"
                )
                shutil.copy2(src, dst)

        def _pt_param_count(path):
            ckpt = torch.load(path, map_location="cpu")
            sd = ckpt.get("model_state_dict", {})
            return sum(v.numel() for v in sd.values())

        print(f">>> SEQUENTIAL PackNet: Saved checkpoints for task {task_idx+1}:")
        print(f"    rssm.pt ({_pt_param_count(task_logdir / 'rssm.pt'):,} params)")
        print(f"    task_heads.pt ({_pt_param_count(task_logdir / 'task_heads.pt'):,} params)")
        print(f"    actor_critic.pt ({_pt_param_count(task_logdir / 'actor_critic.pt'):,} params)")

        # ---- Update global state ----
        global_env_step = logger.step
        prev_rssm_path = str(task_logdir / "rssm.pt")

        save_sequential_progress(
            base_logdir, task_idx,
            completed=True,
            global_env_step_at_end=global_env_step,
        )

        # ---- Cleanup ----
        agent._rssm.packnet_manager = None

        _log_mem("Before end-of-task cleanup")
        _close_envs(train_envs)
        del train_envs, train_dataset, train_eps, agent
        del all_eval_dirs, all_eval_caches
        _force_cleanup()
        _log_mem("After end-of-task cleanup")

        print(f">>> SEQUENTIAL PackNet: Task {task_idx+1} ({task_name}) completed "
              f"at global step {global_env_step}")
        print()

    # ---- Final RSSM copy ----
    if prev_rssm_path is not None:
        final_rssm = base_logdir / "rssm_final.pt"
        shutil.copy2(prev_rssm_path, final_rssm)
        print(f">>> SEQUENTIAL PackNet: Saved final RSSM -> {final_rssm}")

    if hasattr(logger, "finish"):
        logger.finish()

    print("=" * 60)
    print(">>> SEQUENTIAL PackNet: All tasks completed successfully!")
    print(f">>> SEQUENTIAL PackNet: Final global step: {global_env_step}")
    print("=" * 60)


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential DreamerV3 training with PackNet (conservative: weight-only pruning)",
    )

    # Multi-task arguments
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="Task names (e.g., metaworld_drawer-open-v3 metaworld_pick-place-v3)",
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Config profile per task",
    )
    parser.add_argument(
        "--task-steps", nargs="+", type=int, required=True,
        help="Training steps per task in env steps",
    )

    # Logging
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3_Sequential_PackNet_Conservative")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Existing WandB run id to resume with resume='must'.")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="WandB group; multiple seeds of one (method, taskset) share a group.")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="WandB tags (space-separated).")

    # Checkpoint
    parser.add_argument("--from-checkpoint", type=str, default=None)
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-config-check", action="store_true")

    # Per-task dataset size overrides
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=None)

    # PackNet arguments
    parser.add_argument(
        "--packnet-prune-ratio", type=float, default=0.75,
        help="Fraction of free weights to prune after each task (default: 0.75). "
             "Higher = more aggressive pruning, more room for future tasks.",
    )
    parser.add_argument(
        "--packnet-retrain-ratio", type=float, default=0.1,
        help="Fraction of each task's training steps used for retraining after "
             "pruning (default: 0.1 = 10%%). Set to 0 to skip retrain phase.",
    )

    # Eval options
    parser.add_argument("--eval-prev-video", action="store_true", default=True)
    parser.add_argument(
        "--no-eval-prev-video", dest="eval_prev_video", action="store_false",
    )

    main_args, remaining = parser.parse_known_args()
    main(main_args, remaining)
