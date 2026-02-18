"""Sequential training script for DreamerV3 with cross-task evaluation.

Unlike the standard dreamer.py which trains on a single task, this script
handles the entire sequential training pipeline: training on multiple tasks
one after another, while evaluating the current model on ALL tasks seen so far
at every eval interval. This enables tracking catastrophic forgetting.

Usage:
    python dreamer_sequential.py \
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \
        --configs metaworld_default_light metaworld_default_light \
        --task-steps 200000 200000 \
        --logdir ./logdir/sequential_run \
        --wandb-entity my-entity \
        --wandb-project my-project
"""

import argparse
import functools
import os
import pathlib
import sys

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
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
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
    raise NotImplementedError(f"Suite '{suite}' is not supported in sequential training.")


class PrefixedLogger:
    """Wraps a logger to prefix all metric names for task-specific eval logging.

    When evaluating previous tasks, metrics are logged with a task-specific prefix
    (e.g., 'eval_task1_drawer-open/eval_return') so they appear as separate curves
    in WandB. The write() call is suppressed so the main loop can flush all
    eval metrics from all tasks in a single write.
    """

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


def build_task_config(task_name, config_name, task_steps, configs_yaml, remaining_args):
    """Build a config namespace for a specific task.

    Loads the yaml profile, applies CLI overrides from remaining_args,
    overrides task name and steps, and applies action_repeat divisions.
    """
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

    # Parse remaining CLI args as config overrides
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(remaining_args)

    # Override task and steps
    config.task = task_name
    if task_steps is not None:
        config.steps = task_steps

    # Apply action_repeat divisions (same as dreamer.py main)
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    return config


def main(args, remaining_args):
    tasks = args.tasks
    config_names = args.configs
    task_steps_list = args.task_steps
    num_tasks = len(tasks)

    assert len(config_names) == num_tasks, (
        f"Number of configs ({len(config_names)}) must match number of tasks ({num_tasks})"
    )
    assert len(task_steps_list) == num_tasks, (
        f"Number of task-steps ({len(task_steps_list)}) must match number of tasks ({num_tasks})"
    )

    # Load configs yaml
    configs_yaml = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )

    # Build per-task configs
    task_configs = []
    for i in range(num_tasks):
        config = build_task_config(
            tasks[i], config_names[i], task_steps_list[i],
            configs_yaml, remaining_args,
        )
        task_configs.append(config)

    tools.set_seed_everywhere(task_configs[0].seed)
    if task_configs[0].deterministic_run:
        tools.enable_deterministic_run()

    base_logdir = pathlib.Path(args.logdir).expanduser()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # Create single logger for entire sequential training
    first_config = task_configs[0]
    if args.logger == "tensorboard":
        logger = tools.Logger(base_logdir, 0)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, first_config, base_logdir, 0)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # Print sequential training plan
    print("=" * 60)
    print(">>> SEQUENTIAL TRAINING WITH CROSS-TASK EVALUATION <<<")
    print("=" * 60)
    for i, (task, cfg_name, steps) in enumerate(zip(tasks, config_names, task_steps_list)):
        print(f"  Task {i+1}: {task} (config: {cfg_name}, steps: {steps})")
    print(f"  Log directory: {base_logdir}")
    print(f"  Eval previous task videos: {args.eval_prev_video}")
    print("=" * 60)

    if not args.skip_config_check:
        input(">>> Press Enter to start sequential training...")

    global_env_step = 0
    prev_checkpoint = args.from_checkpoint

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

        print("=" * 60)
        print(f">>> SEQUENTIAL: Starting Task {task_idx+1}/{num_tasks}: {task_name}")
        print(f">>> SEQUENTIAL: Task steps: {config.steps * config.action_repeat}")
        print(f">>> SEQUENTIAL: Global env step: {global_env_step}")
        print("=" * 60)

        # Create train envs
        train_envs = [make_env(task_name, config, "train", i) for i in range(config.envs)]
        if config.parallel:
            train_envs = [Parallel(env, "process") for env in train_envs]
        else:
            train_envs = [Damy(env) for env in train_envs]

        acts = train_envs[0].action_space
        print(f">>> SEQUENTIAL: Action Space: {acts}")
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        # Create eval envs for ALL tasks seen so far (0..task_idx inclusive)
        all_eval_envs = {}
        all_eval_dirs = {}
        all_eval_caches = {}
        for j in range(task_idx + 1):
            eval_cfg = task_configs[j]
            eval_envs_j = [make_env(tasks[j], eval_cfg, "eval", i) for i in range(config.envs)]
            if config.parallel:
                eval_envs_j = [Parallel(env, "process") for env in eval_envs_j]
            else:
                eval_envs_j = [Damy(env) for env in eval_envs_j]
            all_eval_envs[j] = eval_envs_j

            eval_dir_j = task_logdir / f"eval_eps_task{j+1}_{tasks[j]}"
            eval_dir_j.mkdir(parents=True, exist_ok=True)
            all_eval_dirs[j] = eval_dir_j
            all_eval_caches[j] = tools.load_episodes(eval_dir_j, limit=1)

        print(f">>> SEQUENTIAL: Created eval envs for {task_idx + 1} task(s)")

        # Set logger to global step
        logger.step = global_env_step

        # Load train episodes
        train_eps = tools.load_episodes(traindir, limit=config.dataset_size)

        # Prefill replay buffer
        state = None
        if not config.offline_traindir:
            prefill = max(0, config.prefill - count_steps(traindir))
            print(f">>> SEQUENTIAL: Prefill dataset ({prefill} steps).")
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
                print(f">>> SEQUENTIAL: Logger step after prefill: {logger.step}")

        # Create dataset and agent
        print(">>> SEQUENTIAL: Creating agent.")
        train_dataset = make_dataset(train_eps, config)
        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            train_dataset,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)

        # Load checkpoint
        load_path = None
        if prev_checkpoint is not None:
            if os.path.exists(prev_checkpoint):
                print(f">>> SEQUENTIAL: Loading checkpoint: {prev_checkpoint}")
                load_path = pathlib.Path(prev_checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {prev_checkpoint}")
        elif (task_logdir / "latest.pt").exists():
            # Resume interrupted training for this task
            print(f">>> SEQUENTIAL: Resuming from: {task_logdir / 'latest.pt'}")
            load_path = task_logdir / "latest.pt"

        if load_path:
            checkpoint = torch.load(load_path)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            if task_idx > 0 or args.skip_pretrain:
                print(">>> SEQUENTIAL: Skipping pretraining (sequential continuation).")
                agent._should_pretrain._once = False

        task_start_step = agent._step
        print(f">>> SEQUENTIAL: Agent step: {agent._step}, task_start_step: {task_start_step}")

        # Main training loop
        items_to_save = None
        while (agent._step - task_start_step) < config.steps + config.eval_every:
            logger.write()

            # === EVALUATION on all tasks seen so far ===
            if config.eval_episode_num > 0:
                print(f">>> SEQUENTIAL: Evaluation at global step {logger.step} "
                      f"(evaluating {task_idx + 1} task(s))")
                eval_policy = functools.partial(agent, training=False)

                for j in range(task_idx + 1):
                    eval_task_name = tasks[j]
                    task_label = f"eval_task{j+1}_{eval_task_name}"
                    is_current_task = (j == task_idx)
                    record_video = is_current_task or args.eval_prev_video

                    prefixed_logger = PrefixedLogger(
                        logger, task_label, record_video=record_video,
                    )

                    tools.simulate(
                        eval_policy,
                        all_eval_envs[j],
                        all_eval_caches[j],
                        all_eval_dirs[j],
                        prefixed_logger,
                        is_eval=True,
                        episodes=config.eval_episode_num,
                    )

                    print(f"    Eval {task_label}: done")

                # Video prediction for current task
                if config.video_pred_log:
                    eval_dataset = make_dataset(all_eval_caches[task_idx], config)
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(video_pred))

                # Flush all eval metrics at once
                logger.write(step=logger.step)

            # === TRAINING ===
            print(f">>> SEQUENTIAL: Training task {task_idx+1} "
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

            # Save checkpoint
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, task_logdir / "latest.pt")

        # Save named checkpoint for this task
        if items_to_save is not None:
            torch.save(items_to_save, task_logdir / f"checkpoint_task{task_idx+1}.pt")

        # Update global state
        global_env_step = logger.step
        prev_checkpoint = str(task_logdir / "latest.pt")

        # Cleanup all envs for this task phase
        for env in train_envs:
            try:
                env.close()
            except Exception:
                pass
        for j in range(task_idx + 1):
            for env in all_eval_envs[j]:
                try:
                    env.close()
                except Exception:
                    pass

        print(f">>> SEQUENTIAL: Task {task_idx+1} ({task_name}) completed "
              f"at global step {global_env_step}")
        print()

    # Finish logging
    if hasattr(logger, "finish"):
        logger.finish()

    print("=" * 60)
    print(">>> SEQUENTIAL: All tasks completed successfully!")
    print(f">>> SEQUENTIAL: Final global step: {global_env_step}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential DreamerV3 training with cross-task evaluation",
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
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3_Sequential")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Checkpoint
    parser.add_argument(
        "--from-checkpoint", type=str, default=None,
        help="Path to initial checkpoint for the first task",
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Skip pretraining on the first task",
    )
    parser.add_argument(
        "--skip-config-check", action="store_true",
        help="Skip interactive confirmation before training",
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
