# SAC.py
import os
import re
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import torch

# import your env factory
from pick_and_place import PickAndPlaceGymEnv
from bimanual_suite.mjc.cubes import OneCubeAssembleEnvironment

def make_env(seed: int = 0, render: bool = False):
    def _init():
        backend = OneCubeAssembleEnvironment(seed=seed)
        env = PickAndPlaceGymEnv(env=backend, render=render)
        # Monitor records episode reward/length for SB3
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default="", help="Path to SAC checkpoint zip to resume from (optional).")
    p.add_argument("--log-dir", type=str, default="./logs_sb3", help="Path to save logs and models")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # training config
    num_envs = 1
    total_timesteps = 1_000_000
    seed = 42
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Make vectorized env
    env_fns = [make_env(seed + i, render=False) for i in range(num_envs)]
    vecenv = DummyVecEnv(env_fns)
    # images HWC -> CHW
    # vecenv = VecTransposeImage(vecenv)
    vecenv = VecMonitor(vecenv, log_dir)

    # choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create or load SAC model
    model = None
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            # parse a number if present (optional)
            m = re.search(r"(\d+)", os.path.basename(resume_path))
            if m:
                loaded_steps = int(m.group(1))
            else:
                loaded_steps = 0

            print(f"Loading model from checkpoint: {resume_path} (parsed loaded_steps={loaded_steps})")
            model = SAC.load(resume_path, env=vecenv, device=device)
            remaining_timesteps = total_timesteps - loaded_steps
            if remaining_timesteps <= 0:
                print(f"Checkpoint already at or past target total_timesteps ({loaded_steps} >= {total_timesteps}). Nothing to do.")
                vecenv.close()
                exit(0)
        else:
            raise FileNotFoundError(f"Resume path specified but file does not exist: {resume_path}")

    if model is None:
        model = SAC(
            policy="MultiInputPolicy",
            env=vecenv,
            verbose=1,
            seed=seed,
            device=device,
            buffer_size=200_000,    # replay buffer
            learning_rate=3e-4,
            batch_size=256,         # mini-batch for updates
            tau=0.005,              # target smoothing coefficient
            gamma=0.99,
            train_freq=1,           # train every step
            gradient_steps=1,       # gradient updates per train call
            ent_coef="auto",        # automatic entropy tuning
            tensorboard_log=log_dir,
        )

    # Checkpoint callback every N steps
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=log_dir,
                                             name_prefix="sac_pick_and_place")

    # Basic training
    if args.resume and 'remaining_timesteps' in locals():
        print(f"Resuming training for {remaining_timesteps} timesteps (to reach total {total_timesteps}).")
        model.learn(total_timesteps=remaining_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
    else:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model (this will also save replay buffer)
    model.save(os.path.join(log_dir, "sac_pick_and_place_final"))

    vecenv.close()
