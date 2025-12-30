# eval.py
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# import your env factory
from pick_and_place import PickAndPlaceGymEnv
from bimanual_suite.mjc.cubes import OneCubeAssembleEnvironment

def make_env(seed: int = 0, render: bool = False):
    def _init():
        backend = OneCubeAssembleEnvironment(seed=seed)
        env = PickAndPlaceGymEnv(env=backend, render=render)
        # Wrap in Monitor for consistent info/logging
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to SB3 checkpoint zip (e.g. ./logs_sb3/ppo_pick_and_place_300000_steps.zip)")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=42, help="Base seed for env(s)")
    p.add_argument("--render", action="store_true", help="Render environment (slow)")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions from policy (default: True behavior in this script)")
    return p.parse_args()

def load_model(checkpoint_path: str, env, device: str = "cpu"):
    """
    Load model based on the checkpoint file. Tries PPO and SAC.
    Returns the loaded model.
    Raises FileNotFoundError if checkpoint not found.
    Raises RuntimeError if loading fails for all known algorithms.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_errors = []
    for Algo in (PPO, SAC):
        try:
            model = Algo.load(checkpoint_path, env=env, device=device)
            print(f"Successfully loaded checkpoint as {Algo.__name__}.")
            return model
        except Exception as e:
            load_errors.append((Algo.__name__, str(e)))
            print(f"Could not load with {Algo.__name__}: {e}")

    # If none loaded, raise informative error
    msg = "Failed to load checkpoint. Attempts:\n"
    for name, err in load_errors:
        msg += f"- {name}: {err}\n"
    raise RuntimeError(msg)

def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the (single) vectorized environment the same way as training
    num_envs = 1
    env_fns = [make_env(args.seed + i, render=args.render) for i in range(num_envs)]
    vecenv = DummyVecEnv(env_fns)
    vecenv = VecTransposeImage(vecenv)   # image H,W,C -> C,H,W
    # Use a temporary monitor directory for evaluation logging (optional)
    eval_log_dir = "./eval_logs"
    os.makedirs(eval_log_dir, exist_ok=True)
    vecenv = VecMonitor(vecenv, eval_log_dir)

    # Load model and attach env
    model = load_model(checkpoint_path, env=vecenv, device=device)
    print(f"Loaded model from {checkpoint_path} on device={device}")

    episode_returns = []
    episode_lengths = []

    # Evaluate episodes (vectorized env but n_envs==1)
    for ep in range(args.episodes):
        ep_reward = 0.0
        ep_len = 0

        # Reset at the start of each episode (vec env returns batch-shaped obs)
        obs = vecenv.reset()

        while True:
            # model.predict handles vectorized obs; for deterministic use the flag
            action, _states = model.predict(obs, deterministic=args.deterministic if hasattr(args, "deterministic") else True)
            # Step the env
            obs, rewards, dones, infos = vecenv.step(action)
            # rewards, dones are arrays (length num_envs)
            # Sum rewards across envs (here num_envs==1, so this picks the single env reward)
            if isinstance(rewards, (list, tuple, np.ndarray)):
                r = float(np.asarray(rewards).sum())
            else:
                r = float(rewards)
            ep_reward += r
            ep_len += 1

            # Determine done for the first env
            done_flag = False
            if isinstance(dones, dict):
                # gymnasium style dict, check keys
                done_flag = any(dones.values())
            else:
                if isinstance(dones, (list, tuple, np.ndarray)):
                    done_flag = bool(dones[0])
                else:
                    done_flag = bool(dones)

            # Extract Monitor episode info if available (VecMonitor places it in infos[0]['episode'])
            info0 = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos
            if info0 and isinstance(info0, dict) and "episode" in info0:
                # Use Monitor's episode return/length if present
                ep_info = info0["episode"]
                ep_reward = float(ep_info.get("r", ep_reward))
                ep_len = int(ep_info.get("l", ep_len))

            if args.render:
                # Slow down for rendering so viewer updates
                try:
                    import time
                    time.sleep(0.01)
                except Exception:
                    pass

            if done_flag:
                break

        print(f"Episode {ep+1}/{args.episodes}: return={ep_reward:.3f}, length={ep_len}")
        episode_returns.append(ep_reward)
        episode_lengths.append(ep_len)

    # Summary statistics
    returns = np.array(episode_returns, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.int32)
    print("=== Evaluation summary ===")
    print(f"episodes: {len(returns)}")
    print(f"return mean: {returns.mean():.3f}, std: {returns.std():.3f}, min: {returns.min():.3f}, max: {returns.max():.3f}")
    print(f"length mean: {lengths.mean():.1f}, std: {lengths.std():.1f}")

    # cleanup
    vecenv.close()

if __name__ == "__main__":
    main()
