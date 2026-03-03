"""
dreamer_boost/dreamer_boost.py

DreamerV3 training with expert demonstration boosting for Metaworld tasks.

This script extends the standard DreamerV3 training loop by injecting expert
demonstration episodes (collected via hand-crafted Metaworld policies) into
the replay buffer. This dramatically accelerates early training by providing
high-quality transitions for the world model to learn from.

Usage (same as dreamer.py, with extra arguments):
    python dreamer_boost/dreamer_boost.py \
        --configs metaworld_default_light \
        --task metaworld_pick-place-v3 \
        --logdir ./logdir/boost_test \
        --expert-episodes 10 \
        --expert-inject-mode prefill \
        --logger tensorboard \
        --skip-config-check

Expert injection modes:
    prefill  : Generate all expert episodes before training starts and add to
               the replay buffer. Training proceeds normally afterwards.
    periodic : Inject a batch of expert episodes at regular intervals during
               training (controlled by --expert-inject-every).

Author: Auto-generated for Metaworld baselines.
"""

import argparse
import functools
import os
import pathlib
import sys
import datetime
import uuid
import time

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import ruamel.yaml as yaml

# Add dreamerv3 root to path so we can import its modules
DREAMER_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DREAMER_DIR))

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

# Expert policy imports
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy
from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy

# Compositional task policies
from metaworld.policies.compo_draweropen_pickplace_policy import CompoDrawerOpenPickPlacePolicy
from metaworld.policies.compo_dooropen_doorclose_policy import CompoDoorOpenDoorClosePolicy

to_np = lambda x: x.detach().cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Expert policy registry
# Maps Metaworld env-name (the part after "metaworld_") to a policy class.
# ──────────────────────────────────────────────────────────────────────────────

EXPERT_POLICY_REGISTRY = {
    "pick-place-v3": SawyerPickPlaceV3Policy,
    "drawer-open-v3": SawyerDrawerOpenV3Policy,
    "door-open-v3": SawyerDoorOpenV3Policy,
    "door-close-v3": SawyerDoorCloseV3Policy,
    "door-unlock-v3": SawyerDoorUnlockV3Policy,
    "door-lock-v3": SawyerDoorLockV3Policy,
    "assembly-v3": SawyerAssemblyV3Policy,
    "disassemble-v3": SawyerDisassembleV3Policy,
    "compo-draweropen-pickplace": CompoDrawerOpenPickPlacePolicy,
    "compo-dooropen-doorclose": CompoDoorOpenDoorClosePolicy,
}


def get_expert_policy(task_name):
    """Get expert policy instance for the given Metaworld task name."""
    if task_name not in EXPERT_POLICY_REGISTRY:
        raise ValueError(
            f"No expert policy registered for task '{task_name}'. "
            f"Available tasks: {list(EXPERT_POLICY_REGISTRY.keys())}"
        )
    policy_cls = EXPERT_POLICY_REGISTRY[task_name]
    policy = policy_cls()
    return policy


# ──────────────────────────────────────────────────────────────────────────────
# Expert episode generation
# ──────────────────────────────────────────────────────────────────────────────

def make_expert_env(config):
    """
    Create a Metaworld environment with the FULL Dreamer wrapper stack so that
    expert episodes are stored in exactly the same format as online rollouts.
    
    Returns (env, task_name, native_action_low, native_action_high).
    native_action_low/high are the action bounds BEFORE NormalizeActions is applied,
    needed to properly normalize expert actions into [-1, 1].
    """
    suite, task = config.task.split("_", 1)
    assert suite == "metaworld", (
        f"Expert boosting only supports metaworld tasks, got suite='{suite}'."
    )

    env = gymnasium.make(
        "Meta-World/MT1",
        env_name=task,
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
    # Record native action space bounds before NormalizeActions
    native_action_low = env.action_space.low.copy()
    native_action_high = env.action_space.high.copy()
    env = wrappers.NormalizeActions(env)
    env = wrappers.RewardObs(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    # NOTE: We do NOT add SelectAction or UUID — we control actions directly
    # and generate our own episode IDs.
    return env, task, native_action_low, native_action_high


def generate_expert_episode(env, policy, config, native_action_low, native_action_high):
    """
    Roll out one expert episode and return a dict of numpy arrays in the
    exact same format Dreamer's ``save_episodes`` / ``load_episodes`` expects.

    Keys: image, proprio, original_obs, is_first, is_terminal, obs_reward,
          reward, discount, action, logprob
    """
    episode = {
        "image": [],
        "proprio": [],
        "original_obs": [],
        "is_first": [],
        "is_terminal": [],
        "obs_reward": [],
        "reward": [],
        "discount": [],
        "action": [],
        "logprob": [],
    }

    obs = env.reset()

    # Reset stateful policies (compositional policies need this)
    if hasattr(policy, "reset"):
        policy.reset()

    # Record initial observation (t=0)
    _record_step(episode, obs, action=np.zeros(env.action_space.shape, dtype=np.float32),
                 reward=0.0, discount=1.0)

    done = False
    for t in range(config.time_limit):
        # Expert policies expect the raw 39-dim Metaworld observation
        raw_obs = obs["original_obs"]
        expert_action = policy.get_action(raw_obs)

        # Normalise action from native range to [-1, 1] for NormalizeActions wrapper
        # NormalizeActions.step does: original = (action + 1)/2 * (high - low) + low
        # So we need: normalised = 2 * (original - low) / (high - low) - 1
        normalised_action = 2.0 * (expert_action - native_action_low) / (native_action_high - native_action_low) - 1.0
        normalised_action = np.clip(normalised_action, -1.0, 1.0).astype(np.float32)

        obs, reward, done, info = env.step(normalised_action)
        discount = info.get("discount", np.array(1.0 - float(done), dtype=np.float32))
        _record_step(episode, obs, normalised_action, reward, discount)

        if done:
            break

    # Stack all lists into arrays
    episode = {k: np.array(v) for k, v in episode.items()}
    return episode


def _record_step(episode, obs, action, reward, discount):
    """Append one transition to the episode dict."""
    episode["image"].append(obs["image"])
    episode["proprio"].append(obs["proprio"])
    episode["original_obs"].append(obs["original_obs"])
    episode["is_first"].append(obs["is_first"])
    episode["is_terminal"].append(obs["is_terminal"])
    episode["obs_reward"].append(obs.get("obs_reward", np.array([0.0], dtype=np.float32)))
    episode["reward"].append(np.float32(reward))
    episode["discount"].append(np.float32(discount))
    episode["action"].append(action)
    # Expert actions don't have a meaningful logprob, use 0.0
    episode["logprob"].append(np.float32(0.0))


def generate_and_save_expert_episodes(config, directory, cache, num_episodes, verbose=True):
    """
    Generate ``num_episodes`` expert demonstrations and save them to
    ``directory`` in Dreamer's .npz format. Also loads them into ``cache``
    so they are immediately available for sampling.

    Returns the total number of expert steps added.
    """
    if num_episodes <= 0:
        return 0

    env, task_name, native_low, native_high = make_expert_env(config)
    policy = get_expert_policy(task_name)

    total_steps = 0

    for i in range(num_episodes):
        ep = generate_expert_episode(env, policy, config, native_low, native_high)
        ep_len = len(ep["reward"])
        total_steps += ep_len - 1  # first step is initial obs, not a transition

        # Create a unique ID consistent with Dreamer's UUID wrapper
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_id = f"{timestamp}-{uuid.uuid4().hex}"

        # Save to disk
        tools.save_episodes(directory, {ep_id: ep})

        # Load into in-memory cache
        for key, val in ep.items():
            if ep_id not in cache:
                cache[ep_id] = {}
            cache[ep_id][key] = val

        ep_reward = float(ep["reward"].sum())
        if verbose:
            print(
                f"  Expert episode {i + 1}/{num_episodes}: "
                f"length={ep_len - 1}, reward={ep_reward:.2f}"
            )

        # Small sleep to ensure unique timestamps
        time.sleep(0.01)

    env.close()

    if verbose:
        print(
            f">>> BOOST: Injected {num_episodes} expert episodes "
            f"({total_steps} steps) into {directory}"
        )

    return total_steps


# ──────────────────────────────────────────────────────────────────────────────
# Dreamer agent (unchanged from dreamer.py)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Unchanged helpers from dreamer.py
# ──────────────────────────────────────────────────────────────────────────────

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "metaworld":
        env = gymnasium.make(
            "Meta-World/MT1",
            env_name=task,
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

    raise NotImplementedError(
        f"dreamer_boost only supports metaworld tasks, got suite='{suite}'. "
        f"For other suites, use the original dreamer.py."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop (with expert boosting)
# ──────────────────────────────────────────────────────────────────────────────

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)

    # Logger
    if args.logger == "tensorboard":
        logger = tools.Logger(logdir, config.action_repeat * step)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, config, logdir, config.action_repeat * step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    # ── Print configuration ───────────────────────────────────────────────
    print("=" * 60)
    print(">>> DREAMER BOOST: Task Setup Configuration <<<")
    print(f"Task: {config.task}")
    print(f"Image observation size: {config.size}")
    print(f"Action repeat: {config.action_repeat}")
    print(f"Time limit (in env step): {config.time_limit}")
    print("-" * 60)
    print(">>> DREAMER BOOST: Training Configuration <<<")
    print(f"Total training steps (in env step): {config.steps * config.action_repeat}")
    print(f"Evaluation every (in env step): {config.eval_every * config.action_repeat}")
    print(f"Logging every (in env step): {config.log_every * config.action_repeat}")
    print(f"Number of parallel environments: {config.envs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Train ratio: {config.train_ratio}")
    print(f"Video prediction logging: {config.video_pred_log}")
    print(f"Pretraining steps (in env step): {config.pretrain * config.action_repeat}")
    print(f"Exploration until (in env step): {config.expl_until * config.action_repeat}")
    print(f"Exploration behavior: {config.expl_behavior}")
    print(f"Evaluation episodes: {config.eval_episode_num}")
    print("-" * 60)
    print(">>> DREAMER BOOST: Expert Boosting Configuration <<<")
    print(f"Expert injection mode: {config.expert_inject_mode}")
    print(f"Expert episodes: {config.expert_episodes}")
    if config.expert_inject_mode == "periodic":
        print(f"Expert inject every (in env step): {config.expert_inject_every}")
        print(f"Expert episodes per injection: {config.expert_episodes_per_inject}")
    print("=" * 60)

    if config.skip_config_check:
        print(">>> DREAMER BOOST: Start training...")
    else:
        input(">>> DREAMER BOOST: Press Enter to start training...")

    # ── Load existing episodes ────────────────────────────────────────────
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    # ── EXPERT BOOST: Prefill mode ───────────────────────────────────────
    if config.expert_inject_mode == "prefill" and config.expert_episodes > 0:
        print(f">>> BOOST: Generating {config.expert_episodes} expert episodes (prefill mode)...")
        expert_steps = generate_and_save_expert_episodes(
            config, config.traindir, train_eps, config.expert_episodes
        )
        logger.step += expert_steps * config.action_repeat
        print(f">>> BOOST: Expert prefill complete. Logger step: {logger.step}")

    # ── Create environments ───────────────────────────────────────────────
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print(">>> DREAMER BOOST: Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ── Random prefill (standard Dreamer prefill) ────────────────────────
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f">>> DREAMER BOOST: Prefill dataset ({prefill} steps).")
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
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f">>> DREAMER BOOST: Logger: ({logger.step} steps).")

    # ── Create agent ──────────────────────────────────────────────────────
    print(">>> DREAMER BOOST: Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # ── Load checkpoint ───────────────────────────────────────────────────
    load_path = None
    if config.from_checkpoint is not None:
        if os.path.exists(config.from_checkpoint):
            print(f">>> DREAMER BOOST: Loading from specified checkpoint: {config.from_checkpoint}")
            load_path = pathlib.Path(config.from_checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint path {config.from_checkpoint} does not exist.")
    else:
        if (logdir / "latest.pt").exists():
            print(f">>> DREAMER BOOST: Resuming from current logdir: {logdir / 'latest.pt'}")
            load_path = logdir / "latest.pt"
    if load_path:
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        if config.skip_pretrain:
            print(">>> DREAMER BOOST: Skipping pretraining.")
            agent._should_pretrain._once = False

    # ── Periodic injection state ──────────────────────────────────────────
    if config.expert_inject_mode == "periodic":
        inject_every = config.expert_inject_every // config.action_repeat
        next_inject_step = inject_every
        total_expert_injected = 0
        print(
            f">>> BOOST: Periodic expert injection enabled. "
            f"Injecting {config.expert_episodes_per_inject} episodes every "
            f"{config.expert_inject_every} env steps."
        )

    # ── Main training loop ────────────────────────────────────────────────
    while agent._step < config.steps + config.eval_every:
        logger.write()

        # Evaluation
        if config.eval_episode_num > 0:
            print(">>> DREAMER BOOST: Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        # Periodic expert injection
        if config.expert_inject_mode == "periodic":
            if agent._step >= next_inject_step:
                print(
                    f">>> BOOST: Periodic injection at step {agent._step * config.action_repeat}. "
                    f"Generating {config.expert_episodes_per_inject} expert episodes..."
                )
                expert_steps = generate_and_save_expert_episodes(
                    config,
                    config.traindir,
                    train_eps,
                    config.expert_episodes_per_inject,
                )
                total_expert_injected += config.expert_episodes_per_inject
                next_inject_step += inject_every
                logger.scalar("expert_episodes_injected", total_expert_injected)
                logger.scalar("expert_steps_total", expert_steps)
                # Rebuild dataset so new episodes are available for sampling
                train_dataset = make_dataset(train_eps, config)
                agent._dataset = train_dataset
                print(
                    f">>> BOOST: Injection complete. "
                    f"Total expert episodes injected: {total_expert_injected}"
                )

        # Training
        print(">>> DREAMER BOOST: Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
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
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DreamerV3 training with expert demonstration boosting."
    )
    parser.add_argument("--configs", nargs="+")
    # Logger
    parser.add_argument("--logger", type=str, default="wandb")
    # Wandb arguments
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (DREAMER_DIR / "configs.yaml").read_text()
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

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    # Standard dreamer extra args
    parser.add_argument(
        "--from-checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint to load weights from.",
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Whether to skip pretraining.",
    )
    parser.add_argument(
        "--skip-config-check", action="store_true",
        help="Skip configuration consistency check.",
    )

    # ── Expert boosting arguments ─────────────────────────────────────────
    parser.add_argument(
        "--expert-episodes", type=int, default=200,
        help=(
            "Number of expert demonstration episodes to inject. "
            "In 'prefill' mode this is the total count injected before training. "
            "In 'periodic' mode this argument is ignored (use --expert-episodes-per-inject)."
        ),
    )
    parser.add_argument(
        "--expert-inject-mode",
        type=str,
        default="prefill",
        choices=["prefill", "periodic"],
        help=(
            "How to inject expert data. "
            "'prefill': generate all episodes once before training. "
            "'periodic': inject batches at regular intervals during training."
        ),
    )
    parser.add_argument(
        "--expert-inject-every",
        type=int,
        default=5000,
        help=(
            "[periodic mode only] Inject expert episodes every N env steps."
        ),
    )
    parser.add_argument(
        "--expert-episodes-per-inject",
        type=int,
        default=5,
        help=(
            "[periodic mode only] Number of expert episodes per injection."
        ),
    )

    main(parser.parse_args(remaining))
