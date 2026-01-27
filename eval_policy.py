import argparse
import pathlib
import sys
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import torch
import ruamel.yaml as yaml
import imageio
import gymnasium

sys.path.append(str(pathlib.Path(__file__).parent))
import envs.wrappers as wrappers
from dreamer import Dreamer
import tools

import envs.metaworld_wrappers as metaworld_wrappers
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper

def load_config(config_names):
    """
    Loads and updates configuration from configs.yaml.
    Adapted from dreamer.py
    """
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    
    return argparse.Namespace(**defaults)

def make_eval_env(task_name, config):
    """
    Creates the Metaworld environment with Dreamer wrappers.
    """
    print("Evaluating DreamerV3 on Metaworld task:", task_name)
    
    env = gymnasium.make("Meta-World/MT1", env_name=task_name, render_mode="rgb_array", max_episode_steps=config.time_limit)
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=64,
        image_width=64,
        camera_names=["topview", "front", "gripperPOV"]
    )
    # Converting to Dreamer compatible environment
    env = metaworld_wrappers.FirstTerminalObs(env)
    env = metaworld_wrappers.RewardTuningWrapperV2(env)
    env = metaworld_wrappers.Gymnasium2Gym(env)
    env = wrappers.NormalizeActions(env)
    env = wrappers.RewardObs(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action") 
    env = wrappers.UUID(env)
    
    return env

def eval_policy(agent, env, args, config, save_video_path=None):
    """
    Runs the evaluation loop with corrected video extraction logic.
    """
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    total_rewards = []
    success_counts = 0
    
    for episode in range(args.episodes):
        obs = env.reset()
        agent_state = None
        done = False
        episode_reward = 0
        video_frames = []
        
        step_count = 0
        while not done:
            # Dreamer expects a batch dimension (B, ...)
            obs_batch = {k: np.stack([v]) for k, v in obs.items()}
            
            with torch.no_grad():
                policy_output, agent_state = agent(obs_batch, [False], agent_state, training=False)
                action_values = policy_output['action'].cpu().numpy()[0]

            action_dict = {'action': action_values} # Dreamer SelectAction wrapper expects dict
            obs, reward, done, info = env.step(action_dict)
            
            episode_reward += reward
            step_count += 1
            
            if save_video_path and "image" in obs:
                # obs["image"] (H, W, 3*N) comes from ProprioMultiImageObsWrapper
                img = obs["image"]
                h, w, c = img.shape
                num_cameras = c // 3
                camera_frames = []
                for i in range(num_cameras):
                    camera_frame = img[:, :, i*3:(i+1)*3]
                    camera_frames.append(camera_frame)
                
                if camera_frames:
                    # Stitch cameras horizontally
                    combined_frame = np.concatenate(camera_frames, axis=1)
                    video_frames.append(combined_frame)

        is_success = info.get('success', 0.0) > 0.5
        success_counts += int(is_success)
        total_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}/{args.episodes} | Reward: {episode_reward:.2f} | Success: {is_success} | Steps: {step_count}")

        if save_video_path and video_frames:
            if is_success:
                vid_filename = os.path.join(save_video_path, f"{args.task}_ep{episode+1}_success.mp4")
            else:
                vid_filename = os.path.join(save_video_path, f"{args.task}_ep{episode+1}_failure.mp4")
            try:
                # FPS=30 is standard for Metaworld
                imageio.mimsave(vid_filename, video_frames, fps=30)
                print(f"Saved video to {vid_filename}")
            except Exception as e:
                print(f"Error saving video: {e}. (Ensure `pip install imageio[ffmpeg]` is run)")
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_counts / args.episodes * 100
    
    print("-" * 10)
    print(f"Evaluation Complete.")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate:   {success_rate:.2f}%")
    print("-" * 10)

def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 on Metaworld")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pt checkpoint")
    parser.add_argument("--task", type=str, default="button-press-v2", help="Metaworld task name (e.g., button-press-v2)")
    parser.add_argument("--configs", nargs="+", default=["defaults", "metaworld"], help="Config names to load from configs.yaml")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--outdir", type=str, default="eval_results", help="Directory to save videos")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load Dreamer Config
    config = load_config(args.configs)

    # Initialize model
    print("Initializing DreamerV3 model...")
    dummy_dataset = iter([]) # Dummy dataset for agent initialization
    # Create Environment
    env = make_eval_env(args.task, config)
    # Update config based on env
    act_space = env.action_space
    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config,
        tools.Logger(outdir, 0),
        dummy_dataset,
    ).to(args.device)

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
        
    agent.eval()

    # Run Evaluation
    eval_policy(agent, env, args, config, save_video_path=outdir)
    
    env.close()

if __name__ == "__main__":
    main()