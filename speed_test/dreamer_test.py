"""
DreamerV3 Training Time Analysis Script
=========================================
Benchmarks wall-clock time for each phase of the DreamerV3 training loop
across different numbers of environments and parallel/sequential modes.

Outputs a structured log file: dreamer_time_analysis.txt

Usage:
    python dreamer_test.py [--config metaworld_default_light] [--steps 1000]
"""

import argparse
import functools
import os
import pathlib
import platform
import subprocess
import sys
import time
import json
from datetime import datetime

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import gymnasium

import torch
from torch import nn
from torch import distributions as torchd

# Metaworld import setup
import envs.metaworld_wrappers as metaworld_wrappers
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper


to_np = lambda x: x.detach().cpu().numpy()


# ============================================================================
# System info gathering
# ============================================================================

def get_system_info():
    """Gather system hardware and software specs."""
    info = {}

    # OS
    info["os"] = platform.platform()
    info["python"] = platform.python_version()

    # CPU
    info["cpu_arch"] = platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        model_names = [
            line.split(":")[1].strip()
            for line in cpuinfo.splitlines()
            if "model name" in line
        ]
        info["cpu_model"] = model_names[0] if model_names else "Unknown"
        info["cpu_cores_logical"] = len(model_names)
    except Exception:
        info["cpu_model"] = "Unknown"
        info["cpu_cores_logical"] = os.cpu_count()

    try:
        info["cpu_cores_physical"] = int(
            subprocess.check_output(
                "lscpu | grep '^CPU(s):' | awk '{print $2}'", shell=True
            ).decode().strip()
        )
    except Exception:
        info["cpu_cores_physical"] = "Unknown"

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = "Unknown"

    # GPU (NVIDIA)
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        gpus = []
        for line in result.splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "memory_mb": parts[2],
                "driver": parts[3],
                "cuda": parts[4],
            })
        info["gpus"] = gpus
    except Exception:
        info["gpus"] = []

    # PyTorch
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_device_count"] = torch.cuda.device_count()
        info["torch_current_device"] = torch.cuda.get_device_name(0)
    else:
        info["torch_cuda_version"] = "N/A"
        info["torch_device_count"] = 0
        info["torch_current_device"] = "CPU"

    return info


def format_system_info(info):
    """Format system info into human-readable text."""
    lines = []
    lines.append("=" * 72)
    lines.append("SYSTEM SPECIFICATION")
    lines.append("=" * 72)
    lines.append(f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  OS:                {info['os']}")
    lines.append(f"  Python:            {info['python']}")
    lines.append(f"  CPU Model:         {info['cpu_model']}")
    lines.append(f"  CPU Cores (phys):  {info['cpu_cores_physical']}")
    lines.append(f"  CPU Cores (logic): {info['cpu_cores_logical']}")
    lines.append(f"  RAM:               {info['ram_gb']} GB")
    lines.append("")
    lines.append("  PyTorch:           " + info["torch_version"])
    lines.append("  CUDA Available:    " + str(info["torch_cuda_available"]))
    lines.append("  CUDA Version:      " + str(info["torch_cuda_version"]))
    lines.append("  GPU Count:         " + str(info["torch_device_count"]))
    if info["gpus"]:
        for gpu in info["gpus"]:
            lines.append(f"  GPU [{gpu['index']}]:          {gpu['name']}  |  {gpu['memory_mb']} MB  |  Driver {gpu['driver']}")
    else:
        lines.append("  GPU:               None detected")
    lines.append("=" * 72)
    return "\n".join(lines)


# ============================================================================
# Environment / Model setup (reused from dreamer.py)
# ============================================================================

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
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
    raise NotImplementedError(f"Suite '{suite}' not supported in test script.")


class LazyParallelEnv:
    """Picklable proxy that lazily constructs the real env inside the worker process."""

    def __init__(self, config, mode, env_id):
        self._config = config
        self._mode = mode
        self._env_id = env_id
        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = make_env(self._config, self._mode, self._env_id)
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


class DreamerBench(nn.Module):
    """Dreamer agent instrumented with timing hooks."""

    def __init__(self, obs_space, act_space, config, dataset):
        super().__init__()
        self._config = config
        self._should_train = tools.Every(
            config.batch_size * config.batch_length / config.train_ratio
        )
        self._should_pretrain = tools.Once()
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = 0
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            print("Compiling models with torch.compile...")
            t1 = time.time()
            self._wm = torch.compile(self._wm)
            print(f"  World model compiled in {time.time() - t1:.2f}s")
            print("Compiling task behavior with torch.compile...")
            t2 = time.time()
            self._task_behavior = torch.compile(self._task_behavior)
            print(f"  Task behavior compiled in {time.time() - t2:.2f}s")
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        # Timing accumulators (seconds)
        self.time_dataset_sample = 0.0
        self.time_model_train = 0.0
        self.time_policy_inference = 0.0
        self.train_update_count = 0

    def __call__(self, obs, reset, state=None, training=True):
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(self._step)
            )
            for _ in range(steps):
                t0 = time.perf_counter()
                data = next(self._dataset)
                t1 = time.perf_counter()
                self._train(data)
                if self._config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                self.time_dataset_sample += t1 - t0
                self.time_model_train += t2 - t1
                self._update_count += 1
                self.train_update_count += 1

        t0 = time.perf_counter()
        policy_output, state = self._policy(obs, state, training)
        if self._config.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.time_policy_inference += t1 - t0

        if training:
            self._step += len(reset)
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


def convert(value):
    """Convert observation values to numpy."""
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value
    return value


def simulate_timed(agent, envs, steps):
    """
    Run the simulate loop for a fixed number of env steps, timing each phase.

    Returns a dict of timing breakdowns.
    """
    step = 0
    done = np.ones(len(envs), bool)
    obs = [None] * len(envs)
    agent_state = None

    time_env_reset = 0.0
    time_env_step_dispatch = 0.0
    time_env_step_collect = 0.0
    time_obs_stack = 0.0
    time_agent_call = 0.0  # includes policy + training
    time_action_process = 0.0
    time_cache_misc = 0.0

    while step < steps:
        # Reset envs if needed
        if done.any():
            indices = [i for i, d in enumerate(done) if d]
            t0 = time.perf_counter()
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            t1 = time.perf_counter()
            time_env_reset += t1 - t0

            for index, result in zip(indices, results):
                obs[index] = result

        # Stack observations
        t0 = time.perf_counter()
        obs_batch = {
            k: np.stack([o[k] for o in obs])
            for k in obs[0] if "log_" not in k
        }
        t1 = time.perf_counter()
        time_obs_stack += t1 - t0

        # Agent call (policy inference + model training)
        t0 = time.perf_counter()
        action, agent_state = agent(obs_batch, done, agent_state)
        t1 = time.perf_counter()
        time_agent_call += t1 - t0

        # Process actions
        t0 = time.perf_counter()
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        t1 = time.perf_counter()
        time_action_process += t1 - t0

        # Step envs (dispatch)
        t0 = time.perf_counter()
        results = [e.step(a) for e, a in zip(envs, action)]
        t1 = time.perf_counter()
        time_env_step_dispatch += t1 - t0

        # Step envs (collect results)
        t0 = time.perf_counter()
        results = [r() for r in results]
        t1 = time.perf_counter()
        time_env_step_collect += t1 - t0

        # Unpack
        t0 = time.perf_counter()
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        done = np.stack(done)
        step += len(envs)
        t1 = time.perf_counter()
        time_cache_misc += t1 - t0

    return {
        "env_reset": time_env_reset,
        "obs_stacking": time_obs_stack,
        "agent_call_total": time_agent_call,
        "action_processing": time_action_process,
        "env_step_dispatch": time_env_step_dispatch,
        "env_step_collect": time_env_step_collect,
        "cache_misc": time_cache_misc,
    }


def build_config(config_name, task, steps, configs_yaml, num_envs, use_parallel):
    """Build a config namespace for a benchmark run."""
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", config_name]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs_yaml[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=value)
    config = parser.parse_args([])

    # Override for benchmark
    config.task = task
    config.steps = steps
    config.envs = num_envs
    config.parallel = use_parallel
    config.eval_every = steps + 1  # no eval during benchmark
    config.log_every = steps + 1
    config.video_pred_log = False
    config.compile = True
    config.prefill = 0
    config.expl_until = 0
    config.batch_size = 2

    # Apply action_repeat divisions
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    return config


def run_benchmark(config, num_envs, use_parallel, logdir):
    """Run a single benchmark configuration and return timing results."""
    import tempfile
    import shutil

    # Create temp directories for train/eval episodes
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="dreamer_bench_"))
    traindir = tmpdir / "train_eps"
    traindir.mkdir(parents=True, exist_ok=True)
    config.traindir = traindir
    config.evaldir = tmpdir / "eval_eps"
    config.evaldir.mkdir(parents=True, exist_ok=True)

    results = {"num_envs": num_envs, "parallel": use_parallel}

    # ---- Phase 1: Environment creation ----
    t0 = time.perf_counter()
    if use_parallel:
        train_envs = [
            Parallel(LazyParallelEnv(config, "train", i), "process")
            for i in range(num_envs)
        ]
    else:
        train_envs = [make_env(config, "train", i) for i in range(num_envs)]
        train_envs = [Damy(env) for env in train_envs]
    t1 = time.perf_counter()
    results["env_creation"] = t1 - t0

    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ---- Phase 2: Prefill / initial data collection ----
    # We need some data in the replay buffer before training.
    # Collect a small amount with random actions.
    t0 = time.perf_counter()
    prefill_eps = {}
    prefill_steps = max(config.batch_size * config.batch_length * 2, 500)
    done = np.ones(len(train_envs), bool)
    obs_list = [None] * len(train_envs)
    prefill_count = 0

    while prefill_count < prefill_steps:
        if done.any():
            indices = [i for i, d in enumerate(done) if d]
            resets = [train_envs[i].reset() for i in indices]
            resets = [r() for r in resets]
            for idx, result in zip(indices, resets):
                obs_list[idx] = result
                t = {k: convert(v) for k, v in result.items()}
                t["reward"] = 0.0
                t["discount"] = 1.0
                tools.add_to_cache(prefill_eps, train_envs[idx].id, t)

        obs_batch = {
            k: np.stack([o[k] for o in obs_list])
            for k in obs_list[0] if "log_" not in k
        }
        # Random actions — wrapped in dict for SelectAction wrapper
        action = np.random.uniform(-1, 1, (len(train_envs), config.num_actions)).astype(np.float32)
        step_results = [e.step({"action": a}) for e, a in zip(train_envs, action)]
        step_results = [r() for r in step_results]
        obs_new, reward, done = zip(*[p[:3] for p in step_results])
        obs_list = list(obs_new)
        done = np.stack(done)
        prefill_count += len(train_envs)

        for a, result, env in zip(action, step_results, train_envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            tools.add_to_cache(prefill_eps, env.id, transition)

    t1 = time.perf_counter()
    results["data_prefill"] = t1 - t0
    results["prefill_steps"] = prefill_count

    # ---- Phase 3: Dataset creation ----
    t0 = time.perf_counter()
    train_dataset = tools.PrefetchIterator(
        tools.from_generator(
            tools.sample_episodes(prefill_eps, config.batch_length),
            config.batch_size,
        ),
        prefetch_count=2,
    )
    t1 = time.perf_counter()
    results["dataset_creation"] = t1 - t0

    # ---- Phase 4: Agent creation ----
    t0 = time.perf_counter()
    agent = DreamerBench(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    results["agent_creation"] = t1 - t0

    # ---- Phase 5: Timed training loop ----
    total_env_steps = config.steps  # already divided by action_repeat
    t0 = time.perf_counter()
    sim_timings = simulate_timed(agent, train_envs, total_env_steps)
    t1 = time.perf_counter()
    results["total_training_loop"] = t1 - t0
    results.update(sim_timings)

    # Extract agent-internal timings
    results["gpu_model_training"] = agent.time_model_train
    results["gpu_dataset_sampling"] = agent.time_dataset_sample
    results["gpu_policy_inference"] = agent.time_policy_inference
    results["model_update_count"] = agent.train_update_count

    # Derived: env stepping total
    results["env_stepping_total"] = (
        results["env_reset"]
        + results["env_step_dispatch"]
        + results["env_step_collect"]
    )

    # SPS
    results["env_steps_per_sec"] = total_env_steps / results["total_training_loop"] if results["total_training_loop"] > 0 else 0

    # ---- Cleanup ----
    for env in train_envs:
        try:
            env.close()
        except Exception:
            pass
    shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def format_results(results):
    """Format a single benchmark result block."""
    lines = []
    par = "PARALLEL" if results["parallel"] else "SEQUENTIAL"
    lines.append(f"  Envs: {results['num_envs']:>3}  |  Mode: {par}")
    lines.append(f"  {'─' * 56}")

    total = results["total_training_loop"]

    def row(label, val, show_pct=True):
        pct = f"({100 * val / total:5.1f}%)" if total > 0 and show_pct else ""
        return f"    {label:<32s}  {val:8.3f}s  {pct}"

    lines.append(row("Env creation", results["env_creation"], show_pct=False))
    lines.append(row("Data prefill", results["data_prefill"], show_pct=False))
    lines.append(f"    {'─' * 54}")
    lines.append(row("TOTAL training loop", total, show_pct=False))
    lines.append(f"    {'─' * 54}")
    lines.append(row("  Env reset", results["env_reset"]))
    lines.append(row("  Env step (dispatch)", results["env_step_dispatch"]))
    lines.append(row("  Env step (collect/wait)", results["env_step_collect"]))
    lines.append(row("  Env stepping SUBTOTAL", results["env_stepping_total"]))
    lines.append(f"    {'─' * 54}")
    lines.append(row("  Obs stacking", results["obs_stacking"]))
    lines.append(row("  Action processing", results["action_processing"]))
    lines.append(row("  Cache / misc", results["cache_misc"]))
    lines.append(f"    {'─' * 54}")
    lines.append(row("  Agent call TOTAL", results["agent_call_total"]))
    lines.append(row("    ├─ Dataset sampling", results["gpu_dataset_sampling"]))
    lines.append(row("    ├─ Model training (GPU)", results["gpu_model_training"]))
    lines.append(row("    └─ Policy inference (GPU)", results["gpu_policy_inference"]))
    lines.append(f"    {'─' * 54}")
    lines.append(f"    {'Model updates:':<32s}  {results['model_update_count']:>8d}")
    lines.append(f"    {'Env steps/sec (SPS):':<32s}  {results['env_steps_per_sec']:>8.1f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="DreamerV3 Training Time Analysis",
    )
    parser.add_argument(
        "--config", type=str, default="metaworld_default_light",
        help="Base config profile from configs.yaml",
    )
    parser.add_argument(
        "--task", type=str, default="metaworld_pick-place-v3",
        help="Task name",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of env steps per configuration (before action_repeat division)",
    )
    parser.add_argument(
        "--env-counts", nargs="+", type=int, default=[1, 8, 32, 64],
        help="List of env counts to benchmark",
    )
    parser.add_argument(
        "--output", type=str, default="dreamer_time_analysis.txt",
        help="Output log file name",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (default: from config, usually cuda:0)",
    )
    parser.add_argument(
        "--skip-parallel", action="store_true",
        help="Skip parallel mode benchmarks (only run sequential)",
    )
    parser.add_argument(
        "--skip-sequential", action="store_true",
        help="Skip sequential mode benchmarks (only run parallel)",
    )
    args = parser.parse_args()

    # Load configs yaml
    configs_yaml = yaml.safe_load(
        (pathlib.Path(__file__).parent.parent / "configs.yaml").read_text()
    )

    # Gather system info
    sys_info = get_system_info()
    header = format_system_info(sys_info)

    output_lines = []
    output_lines.append(header)
    output_lines.append("")
    output_lines.append("=" * 72)
    output_lines.append("BENCHMARK CONFIGURATION")
    output_lines.append("=" * 72)
    output_lines.append(f"  Task:              {args.task}")
    output_lines.append(f"  Config profile:    {args.config}")
    output_lines.append(f"  Steps per run:     {args.steps} env steps")
    output_lines.append(f"  Env counts:        {args.env_counts}")
    output_lines.append(f"  Device:            {args.device or 'from config'}")
    output_lines.append("=" * 72)
    output_lines.append("")

    all_results = []

    modes = []
    if not args.skip_sequential:
        modes.append(False)
    if not args.skip_parallel:
        modes.append(True)

    total_runs = len(args.env_counts) * len(modes)
    run_idx = 0

    for use_parallel in modes:
        for num_envs in args.env_counts:
            run_idx += 1
            par_str = "parallel" if use_parallel else "sequential"
            print(f"\n{'=' * 60}")
            print(f"[{run_idx}/{total_runs}] Benchmarking: {num_envs} envs, {par_str}")
            print(f"{'=' * 60}")

            config = build_config(
                args.config, args.task, args.steps,
                configs_yaml, num_envs, use_parallel,
            )
            if args.device:
                config.device = args.device

            # Temporary logdir
            config.logdir = f"/tmp/dreamer_bench_{num_envs}_{par_str}"

            try:
                results = run_benchmark(config, num_envs, use_parallel, config.logdir)
                all_results.append(results)

                block = format_results(results)
                output_lines.append(f"── Run {run_idx}/{total_runs} " + "─" * 50)
                output_lines.append(block)

                print(f"  Done in {results['total_training_loop']:.2f}s  "
                      f"({results['env_steps_per_sec']:.1f} SPS)")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  FAILED: {e}")
                output_lines.append(f"── Run {run_idx}/{total_runs} " + "─" * 50)
                output_lines.append(f"  Envs: {num_envs}  |  Mode: {par_str}")
                output_lines.append(f"  FAILED: {e}")
                output_lines.append(f"  {tb}")
                output_lines.append("")

    # ---- Summary table ----
    output_lines.append("")
    output_lines.append("=" * 72)
    output_lines.append("SUMMARY TABLE")
    output_lines.append("=" * 72)

    header_row = f"  {'Envs':>4s}  {'Mode':<12s}  {'Total(s)':>9s}  {'EnvStep(s)':>10s}  {'Train(s)':>9s}  {'Policy(s)':>10s}  {'SPS':>8s}"
    output_lines.append(header_row)
    output_lines.append("  " + "─" * 68)

    for r in all_results:
        mode_str = "parallel" if r["parallel"] else "sequential"
        output_lines.append(
            f"  {r['num_envs']:>4d}  {mode_str:<12s}  "
            f"{r['total_training_loop']:>9.2f}  "
            f"{r['env_stepping_total']:>10.2f}  "
            f"{r['gpu_model_training']:>9.2f}  "
            f"{r['gpu_policy_inference']:>10.2f}  "
            f"{r['env_steps_per_sec']:>8.1f}"
        )

    output_lines.append("")
    output_lines.append("=" * 72)
    output_lines.append(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("=" * 72)

    # Write output
    output_path = pathlib.Path(args.output)
    output_text = "\n".join(output_lines)
    output_path.write_text(output_text)
    print(f"\n{'=' * 60}")
    print(f"Results written to: {output_path.resolve()}")
    print(f"{'=' * 60}")
    print(output_text)

    # Also save raw JSON
    json_path = output_path.with_suffix(".json")
    json_data = {
        "system_info": sys_info,
        "benchmark_config": {
            "task": args.task,
            "config_profile": args.config,
            "steps_per_run": args.steps,
            "env_counts": args.env_counts,
        },
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Raw JSON saved to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
