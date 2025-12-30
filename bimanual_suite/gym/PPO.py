# PPO.py
import os
import re
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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
    p.add_argument("--resume", type=str, default="", help="Path to PPO checkpoint zip to resume from (optional).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # training config
    num_envs = 1                      # keep 1 for sanity checking / debugging
    total_timesteps = 1_000_000       # total training timesteps (final target)
    seed = 42
    log_dir = "./logs_sb3"
    os.makedirs(log_dir, exist_ok=True)

    # Make vectorized env
    env_fns = [make_env(seed + i, render=False) for i in range(num_envs)]
    vecenv = DummyVecEnv(env_fns)
    # Your observations include image (H,W,3) — SB3's CNN expects channels-first, so transpose:
    vecenv = VecTransposeImage(vecenv)   # converts image (H,W,C) -> (C,H,W) automatically
    vecenv = VecMonitor(vecenv, log_dir) # wraps for logging

    # choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create or load PPO model
    model = None
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            # Try to parse the number of timesteps from filename (e.g. ppo_pick_and_place_300000_steps.zip)
            m = re.search(r"(\d+)", os.path.basename(resume_path))
            if m:
                loaded_steps = int(m.group(1))
            else:
                loaded_steps = 0

            print(f"Loading model from checkpoint: {resume_path} (parsed loaded_steps={loaded_steps})")
            # Load model and attach current env
            model = PPO.load(resume_path, env=vecenv, device=device)
            # Compute remaining timesteps to reach total_timesteps
            remaining_timesteps = total_timesteps - loaded_steps
            if remaining_timesteps <= 0:
                print(f"Checkpoint already at or past target total_timesteps ({loaded_steps} >= {total_timesteps}). Nothing to do.")
                vecenv.close()
                exit(0)
        else:
            raise FileNotFoundError(f"Resume path specified but file does not exist: {resume_path}")

    if model is None:
        # Create fresh PPO model (same as before)
        model = PPO(
            policy="MultiInputPolicy",
            env=vecenv,
            verbose=1,
            seed=seed,
            device=device,
            batch_size=64,
            n_steps=2048,
            learning_rate=3e-4,
            tensorboard_log=log_dir,
        )

    # Checkpoint callback every N steps
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=log_dir,
                                             name_prefix="ppo_pick_and_place")

    # Basic training
    if args.resume and 'remaining_timesteps' in locals():
        # Continue training from the loaded checkpoint until final total_timesteps.
        # Keep reset_num_timesteps=False so the model's internal timestep counter is not reset.
        print(f"Resuming training for {remaining_timesteps} timesteps (to reach total {total_timesteps}).")
        model.learn(total_timesteps=remaining_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)
    else:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model
    model.save(os.path.join(log_dir, "ppo_pick_and_place_final"))

    # Clean up
    vecenv.close()
