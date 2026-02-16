"""
Sequential DreamerV3 Training for Continual Learning

This script trains DreamerV3 on a sequence of tasks following the continual learning setting:
- No access to previous replay buffer/data from earlier tasks
- Train starting on a new task from the last checkpoint of the previous task
- Separate checkpoints and episode storage for each task

Usage:
    python sequential_dreamer.py --configs metaworld_default_light \
        --tasks metaworld_drawer-open-v3 metaworld_door-open-v3 metaworld_button-press-v3 \
        --logdir ./logs/sequential_training
"""

import argparse
import functools
import gc
import os
import pathlib
import sys
import shutil
from typing import List, Optional

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

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
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    """
    DreamerV3 Agent for sequential continual learning.
    
    The agent consists of:
    1. WorldModel: Learns environment dynamics via RSSM (encoder, dynamics, decoder, reward/cont heads)
    2. ImagBehavior: Learns policy via imagination (actor, critic)
    
    Training flow:
    1. Collect real transitions via rollouts
    2. Train WorldModel on replay buffer to predict observations, rewards, and dynamics
    3. Train actor/critic by imagining trajectories in learned world model
    """
    
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
        if config.compile and os.name != "nt":
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        """
        Main agent call during simulation.
        
        When training=True:
        1. Sample batch and train world model + behavior
        2. Get action from policy
        
        When training=False (evaluation):
        1. Only get action from policy (mode, not sampled)
        """
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
        """
        Get action from policy.
        
        1. Encode observation
        2. Update latent state via RSSM obs_step
        3. Get action from actor (mode if eval, sample if train)
        """
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
        """
        Single training step.
        
        1. Train WorldModel:
           - Encode observations
           - Run RSSM dynamics (observe)
           - Train decoder, reward head, continue head
           - KL loss between prior and posterior
        
        2. Train ImagBehavior:
           - Start from posterior states
           - Imagine future trajectories in world model
           - Compute lambda returns
           - Train actor and critic
        """
        metrics = {}
        # Train world model on real data
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        
        # Train actor/critic via imagination starting from posterior
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        
        # Optional exploration behavior training
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def update_dataset(self, new_dataset):
        """Update the dataset reference for new task."""
        self._dataset = new_dataset

    def reset_step_counter(self, new_step=0):
        """Reset step counter for new task."""
        self._step = new_step

    def reset_schedulers(self):
        """Reset training schedulers for new task."""
        self._should_pretrain = tools.Once()
        batch_steps = self._config.batch_size * self._config.batch_length
        self._should_train = tools.Every(batch_steps / self._config.train_ratio)
        self._should_log = tools.Every(self._config.log_every)
        self._should_expl = tools.Until(int(self._config.expl_until / self._config.action_repeat))
        self._metrics = {}
        self._update_count = 0


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(task_name, config, mode, id):
    """Create environment for a specific task."""
    suite, task = task_name.split("_", 1)
    if suite == "metaworld":
        env = gymnasium.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", max_episode_steps=config.time_limit)
        env = ProprioMultiImageObsWrapper(env,
                                        image_height=config.size[0],
                                        image_width=config.size[1],
                                        camera_names=["topview", "front", "gripperPOV"])
        env = metaworld_wrappers.FirstTerminalObs(env)
        env = metaworld_wrappers.RewardTuningWrapperV2(env)
        env = metaworld_wrappers.Gymnasium2Gym(env)
        env = wrappers.NormalizeActions(env)
        env = wrappers.RewardObs(env)
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env
    else:
        raise NotImplementedError(f"Suite {suite} not implemented for sequential training.")


def train_single_task(
    task_name: str,
    task_idx: int,
    config,
    args,
    base_logdir: pathlib.Path,
    prev_checkpoint_path: Optional[pathlib.Path] = None,
):
    """
    Train on a single task following continual learning protocol.
    
    Args:
        task_name: Name of the task (e.g., "metaworld_drawer-open-v3")
        task_idx: Index of the task in the sequence (0-based)
        config: Training configuration
        args: Command line arguments
        base_logdir: Base directory for all logging
        prev_checkpoint_path: Path to checkpoint from previous task (None for first task)
    
    Returns:
        Path to the saved checkpoint for this task
    """
    tools.set_seed_everywhere(config.seed + task_idx)  # Different seed per task
    
    # Create task-specific directories
    task_logdir = base_logdir / f"task_{task_idx}_{task_name.replace('/', '_')}"
    task_logdir.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: Fresh directories for each task - no access to previous data
    train_eps_dir = task_logdir / "train_eps"
    eval_eps_dir = task_logdir / "eval_eps"
    train_eps_dir.mkdir(parents=True, exist_ok=True)
    eval_eps_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure for this task
    task_config = argparse.Namespace(**vars(config))
    task_config.task = task_name
    task_config.traindir = train_eps_dir
    task_config.evaldir = eval_eps_dir
    
    # Adjust steps for this task
    task_config.steps = config.steps // task_config.action_repeat
    task_config.eval_every = config.eval_every // task_config.action_repeat
    task_config.log_every = config.log_every // task_config.action_repeat
    task_config.time_limit = task_config.time_limit // task_config.action_repeat
    
    print("=" * 60)
    print(f"TASK {task_idx + 1}/{len(args.tasks)}: {task_name}")
    print("=" * 60)
    print(f"Task log directory: {task_logdir}")
    print(f"Training episodes directory: {train_eps_dir}")
    print(f"Evaluation episodes directory: {eval_eps_dir}")
    if prev_checkpoint_path:
        print(f"Loading checkpoint from: {prev_checkpoint_path}")
    else:
        print("Starting from scratch (first task)")
    print("=" * 60)
    
    # Initialize step counter from zero for this task
    step = 0
    
    # Create logger
    if args.logger == "tensorboard":
        logger = tools.Logger(task_logdir, step)
    elif args.logger == "wandb":
        # Create a unique run name for this task
        wandb_args = argparse.Namespace(**vars(args))
        if args.wandb_run_name:
            wandb_args.wandb_run_name = f"{args.wandb_run_name}_task{task_idx}_{task_name}"
        else:
            wandb_args.wandb_run_name = f"sequential_task{task_idx}_{task_name}"
        logger = tools.WandBLogger(wandb_args, task_config, task_logdir, step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")
    
    # Print task configuration
    print(">>> Task Setup Configuration: <<<")
    print(f"Task: {task_name}")
    print(f"Image observation size: {task_config.size}")
    print(f"Action repeat: {task_config.action_repeat}")
    print(f"Time limit (in env step): {task_config.time_limit * task_config.action_repeat}")
    print("================================")
    print(">>> Training Configuration: <<<")
    print(f"Steps for this task (in env step): {config.steps}")
    print(f"Evaluation every (in env step): {config.eval_every}")
    print(f"Logging every (in env step): {config.log_every}")
    print(f"Number of parallel environments: {task_config.envs}")
    print(f"Batch size: {task_config.batch_size}")
    print(f"Previous checkpoint: {prev_checkpoint_path}")
    print("================================")
    
    if prev_checkpoint_path is None:
        input("Press Enter to start training")
    
    # Create environments for this task
    print("Creating environments...")
    make = lambda mode, id: make_env(task_name, task_config, mode, id)
    train_envs = [make("train", i) for i in range(task_config.envs)]
    eval_envs = [make("eval", i) for i in range(task_config.envs)]
    
    if task_config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    
    acts = train_envs[0].action_space
    print("Action Space", acts)
    task_config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    
    # Initialize fresh episode caches for this task (continual learning: no previous data)
    train_eps = tools.load_episodes(train_eps_dir, limit=task_config.dataset_size)  # Will be empty
    eval_eps = tools.load_episodes(eval_eps_dir, limit=1)  # Will be empty
    
    state = None
    
    # Prefill dataset with random actions for this task
    prefill = max(0, task_config.prefill - count_steps(train_eps_dir))
    print(f"Prefilling dataset with {prefill} random steps...")
    
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(
            torch.zeros(task_config.num_actions).repeat(task_config.envs, 1)
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(task_config.envs, 1),
                torch.tensor(acts.high).repeat(task_config.envs, 1),
            ),
            1,
        )
    
    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None
    
    if prefill > 0:
        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            train_eps_dir,
            logger,
            limit=task_config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * task_config.action_repeat
        print(f"Prefill complete. Logger step: {logger.step}")
    
    # Create datasets and agent
    print("Creating agent...")
    train_dataset = make_dataset(train_eps, task_config)
    eval_dataset = make_dataset(eval_eps, task_config)
    
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        task_config,
        logger,
        train_dataset,
    ).to(task_config.device)
    agent.requires_grad_(requires_grad=False)
    
    # Load checkpoint from previous task if available (CONTINUAL LEARNING: model transfer only)
    if prev_checkpoint_path is not None and prev_checkpoint_path.exists():
        print(f"Loading model weights from previous task: {prev_checkpoint_path}")
        checkpoint = torch.load(prev_checkpoint_path, map_location=task_config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # Skip pretraining since we have a pretrained model
        agent._should_pretrain._once = False
        print("Loaded checkpoint successfully. Pretraining skipped.")
    else:
        print("No previous checkpoint. Training from scratch.")
    
    # Main training loop for this task
    print("Starting training...")
    while agent._step < task_config.steps + task_config.eval_every:
        logger.write()
        
        # Evaluation
        if task_config.eval_episode_num > 0:
            print(f"Evaluation at step {agent._step * task_config.action_repeat}...")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                eval_eps_dir,
                logger,
                is_eval=True,
                episodes=task_config.eval_episode_num,
            )
            if task_config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        
        # Training
        print(f"Training step {agent._step * task_config.action_repeat}/{config.steps}...")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            train_eps_dir,
            logger,
            limit=task_config.dataset_size,
            steps=task_config.eval_every,
            state=state,
        )
        
        # Save intermediate checkpoint
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "task_name": task_name,
            "task_idx": task_idx,
            "step": agent._step,
        }
        torch.save(items_to_save, task_logdir / "latest.pt")
    
    # Save final checkpoint for this task
    checkpoint_path = base_logdir / f"checkpoint_task_{task_idx}.pt"
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "task_name": task_name,
        "task_idx": task_idx,
        "step": agent._step,
        "final": True,
    }
    torch.save(items_to_save, checkpoint_path)
    print(f"Saved final checkpoint for task {task_idx} to: {checkpoint_path}")
    
    # Cleanup environments
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    
    # Finish wandb run for this task (if using wandb)
    if hasattr(logger, 'finish'):
        logger.finish()
        print(f"Closed logger for task {task_idx}: {task_name}")
    
    # ========== MEMORY CLEANUP ==========
    # Explicitly free memory to avoid OOM when starting next task
    print("Cleaning up memory...")
    
    # Clear episode data (main RAM consumer)
    train_eps.clear()
    eval_eps.clear()
    
    # Delete large objects
    del train_dataset, eval_dataset
    del agent
    del train_envs, eval_envs
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print(f"Memory cleanup complete for task {task_idx}.")
    # =====================================
    
    return checkpoint_path


def main(args, config):
    """
    Main function for sequential continual learning training.
    
    Trains on a sequence of tasks where:
    - Each task starts from the checkpoint of the previous task
    - No access to replay data from previous tasks
    - Separate episode storage for each task
    """
    print("=" * 60)
    print("SEQUENTIAL DREAMERV3 - CONTINUAL LEARNING")
    print("=" * 60)
    print(f"Tasks to train: {args.tasks}")
    print(f"Steps per task: {config.steps}")
    print(f"Base log directory: {args.logdir}")
    print(f"Resume from task: {args.resume_from_task}")
    print("=" * 60)
    
    # Create base log directory
    base_logdir = pathlib.Path(args.logdir).expanduser()
    base_logdir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_save_path = base_logdir / "sequential_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.YAML().dump({
            "tasks": args.tasks,
            "steps": config.steps,
            "eval_every": config.eval_every,
            "log_every": config.log_every,
            "config": vars(config),
        }, f)
    print(f"Saved configuration to: {config_save_path}")
    
    # Determine starting point
    start_task_idx = args.resume_from_task
    prev_checkpoint = None
    
    # If resuming, find previous checkpoint
    if start_task_idx > 0:
        prev_checkpoint = base_logdir / f"checkpoint_task_{start_task_idx - 1}.pt"
        if not prev_checkpoint.exists():
            raise FileNotFoundError(
                f"Cannot resume from task {start_task_idx}: "
                f"Previous checkpoint {prev_checkpoint} not found."
            )
        print(f"Resuming from task {start_task_idx}, loading checkpoint: {prev_checkpoint}")
    
    # Train on each task sequentially
    for task_idx in range(start_task_idx, len(args.tasks)):
        task_name = args.tasks[task_idx]
        
        # Train on this task
        checkpoint_path = train_single_task(
            task_name=task_name,
            task_idx=task_idx,
            config=config,
            args=args,
            base_logdir=base_logdir,
            prev_checkpoint_path=prev_checkpoint,
        )
        
        # Update checkpoint for next task
        prev_checkpoint = checkpoint_path
        
        print(f"\nCompleted task {task_idx + 1}/{len(args.tasks)}: {task_name}")
        print(f"Checkpoint saved to: {checkpoint_path}\n")
    
    print("=" * 60)
    print("SEQUENTIAL TRAINING COMPLETE")
    print("=" * 60)
    print(f"All checkpoints saved in: {base_logdir}")
    for task_idx, task_name in enumerate(args.tasks):
        print(f"  Task {task_idx}: {task_name} -> checkpoint_task_{task_idx}.pt")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential DreamerV3 for Continual Learning")
    
    # Base config
    parser.add_argument("--configs", nargs="+", help="Config presets to load")
    
    # Sequential training arguments
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        required=True,
        help="List of tasks to train sequentially (e.g., metaworld_drawer-open-v3 metaworld_door-open-v3)"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs/sequential_dreamer",
        help="Base directory for logging and checkpoints"
    )
    parser.add_argument(
        "--resume_from_task",
        type=int,
        default=0,
        help="Task index to resume from (0-indexed). Resume will load checkpoint from task-1."
    )
    
    # Logger arguments
    parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb"])
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    args, remaining = parser.parse_known_args()
    
    # Load config from YAML
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    
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
    
    # Parse remaining arguments as config overrides
    config_parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        config_parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    
    config = config_parser.parse_args(remaining)
    
    main(args, config)
