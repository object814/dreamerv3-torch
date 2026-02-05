"""
In this branch, we write detailed comments for the code to explain the logic and flow of the DreamerV3 implementation.
"""

import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp" # avoid video recording error in headless server

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
        """
        number of environment steps in one update step.
        batch_size is the number of sequences in one batch, batch_length is the length of each sequence (temporal length).
        e.g. shape of an image batch is (batch_size, batch_length, H, W, C).
        """
        batch_steps = config.batch_size * config.batch_length
        """
        train_ratio controls how many environment steps to take in one update step.
        e.g. if train_ratio=1, then one training update will need to wait for 1 batch_steps environment steps to be collected;
             if train_ration=512, then one training update will need to wait for 512 batch_steps environment steps to be collected.
        bigger train_ratio means less frequent training updates comparing to environment steps, which will lead to more stable training but slower learning;
        smaller train_ratio means more frequent training updates comparing to environment steps, which will lead to faster learning but less stable training.
        """
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        """
        pretrain means how many update steps to train before using the agent to interact with environment.
        e.g. if pretrain=1000, then the agent will be trained for 1000 update steps before it is used to interact with environment.
        during pretraining, the agent 
        """
        self._should_pretrain = tools.Once()
        """
        dreamer resets the agent every reset_every steps.
        if reset_every=0, then the agent will never reset during training.
        """
        self._should_reset = tools.Every(config.reset_every)
        """
        dreamer does exploration for self._config.expl_until environment steps.
        after that, the agent will use the task behavior policy for exploration.
        """
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        """
        self._step is the total number of environment steps to take.
        """
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
        """
        What to do everytime Dreamer agent is called to interact with environment.
        Called in tools.simulate() function in main loop.
        obs: current observation from environment.
        reset: boolean array indicating which environments are reset.
        state: current latent state of the agent.
        training: boolean indicating whether the agent is in training mode or evaluation mode.
        returns:
            policy_output: action and logprob from the agent's policy.
            state: updated latent state of the agent.
        """
        step = self._step
        if training:
            """
            logic of steps:
            if self._should_pretrain() is True:
                steps = self._config.pretrain
            else:
                steps = self._should_train(step)
            self._should_pretrain() is True only once at the beginning of training, so the agent will first do pretraining for self._config.pretrain update steps.
            after that, the agent will do training updates every self._should_train(step) steps.
            if self._should_train(step) is True, then it returns 1, so the agent will do 1 training update.
            if self._should_train(step) is False or not should_train step, then it returns 0, so the agent will not do any training update.
            """
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                """
                each training update will use one batch of data from the dataset, and each batch of data contains batch_size * batch_length environment steps.
                """
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                """
                self._metrics is a dictionary that stores the metrics to be logged.
                you can find the definition of the metrics in the _train() function.
                """
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
        """
        this is the function that defines what to do in one training update.
        data: a batch of data from the dataset, containing batch_size * batch_length environment steps.
        returns:
            post: the posterior latent state after observing the data.
            context: the context latent state before observing the data.
            mets: the metrics from training the world model.
        """
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


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "metaworld":
        print("Training DreamerV3 on Metaworld task:", task)
        env = gymnasium.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", max_episode_steps=config.time_limit)
        env = ProprioMultiImageObsWrapper(env,
                                        image_height=args.image_size,
                                        image_width=args.image_size,
                                        camera_names=["topview", "front", "gripperPOV"])
        # Converting to Dreamer compatible environment
        env = metaworld_wrappers.FirstTerminalObs(env) # Add is_first and is_terminal flags in observation for Dreamer
        env = metaworld_wrappers.RewardTuningWrapperV2(env) # Tune rewards and termination for Dreamer
        env = metaworld_wrappers.Gymnasium2Gym(env) # Convert Gymnasium env to Gym env for Dreamer
        # Apply standard Dreamer wrappers
        env = wrappers.NormalizeActions(env) # Normalize action to [-1, 1] for Dreamer, it will rescale back to original range before env.step()
        env = wrappers.RewardObs(env) # Add previous reward as 'obs_reward' in observation for Dreamer reward prediction
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env) # This wrapper must be put at the last in order to let UUID generated after all other wrappers
        return env
    
    print("You are running original DreamerV3 code, not using Metaworld.")

    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


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

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    if args.logger == "tensorboard":
        logger = tools.Logger(logdir, config.action_repeat * step)
    elif args.logger == "wandb":
        logger = tools.WandBLogger(args, config, logdir, config.action_repeat * step)
    else:
        raise NotImplementedError(f"Logger {args.logger} is not implemented.")

    print("Create envs.")
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
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
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
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
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
    """Original loading checkpoint in dreamer code for reference"""
    # if (logdir / "latest.pt").exists():
    #     checkpoint = torch.load(logdir / "latest.pt")
    #     agent.load_state_dict(checkpoint["agent_state_dict"])
    #     tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    #     agent._should_pretrain._once = False

    # Determine which checkpoint to load
    load_path = None
    if config.from_checkpoint is not None:
        if os.path.exists(config.from_checkpoint):
            print(f"Loading from specified checkpoint: {config.from_checkpoint}")
            load_path = pathlib.Path(config.from_checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint path {config.from_checkpoint} does not exist.")
    else:
        if (logdir / "latest.pt").exists():
            print(f"Resuming from current logdir: {logdir / 'latest.pt'}")
            load_path = logdir / "latest.pt"
    # Load checkpoint if specified
    if load_path:
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
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
        print("Start training.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    # Specify logger
    parser.add_argument("--logger", type=str, default="wandb") # options: wandb, tensorboard
    # Wandb arguments
    parser.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    parser.add_argument("--wandb-project", type=str, default="CCLB_Dreamerv3")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    # Input image resolution for Metaworld
    parser.add_argument("--image-size", type=int, default=64, help="Input image size for Metaworld environments.")
    args, remaining = parser.parse_known_args()
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
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    # Add from_checkpoint argument
    parser.add_argument("--from_checkpoint", type=str, default=None, help="Path to a .pt checkpoint to load weights from.")
    main(parser.parse_args(remaining))