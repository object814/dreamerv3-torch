import torch
import numpy as np
import cv2
from pathlib import Path
import pathlib
import gymnasium
import envs.wrappers as wrappers
import tools
import models
import argparse
import ruamel.yaml as yaml
import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp" # avoid video recording error in headless server

# Metaworld import setup
import sys
import envs.metaworld_wrappers as metaworld_wrappers
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper

def draw_reward(img, reward):
    """
    img: (H, W, 3), values in [0, 1]
    reward: scalar
    """
    img = (img * 255).astype(np.uint8)
    text = f"r = {reward:.2f}"
    cv2.putText(
        img,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img

@torch.no_grad()
def imagine_rollout(
    obs,
    preprocess,
    encoder,
    rssm,
    decoder,
    reward_head,
    action_dim,
    horizon=30,
    device="cuda",
    out_dir="imagined_rollout",
):
    """
    obs: (H, W, 3) numpy array in uint8
    """

    Path(out_dir).mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # 1. Preprocess observation
    # ------------------------------------------------------------
    # obs = torch.tensor(obs, device=device).float() / 255.0
    # obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 3)

    # data = {
    #     "image": obs,
    #     "is_first": torch.ones(1, 1, device=device),
    #     "is_terminal": torch.zeros(1, 1, device=device),
    #     "obs_reward": torch.zeros(1, 1, device=device),
    #     "action": torch.zeros(1, 1, action_dim, device=device),
    # }

    # ------------------------------------------------------------
    # 2. Encode observation
    # ------------------------------------------------------------
    # Convert single-env obs dict → Dreamer format
    obs = {k: np.expand_dims(v, 0) for k, v in obs.items() if "log_" not in k}

    # Let WorldModel handle tensor conversion + normalization
    obs = preprocess(obs)

    embed = encoder(obs)
    embed = embed.unsqueeze(1)  # add time dimension: (batch, time=1, embed_dim)

    # ------------------------------------------------------------
    # 3. Infer initial latent state (posterior)
    # ------------------------------------------------------------
    post, _ = rssm.observe(
        embed=embed,
        action=torch.zeros(1, 1, action_dim, device=device),
        is_first=torch.ones(1, 1, device=device),
    )

    # Take last timestep
    state = {k: v[:, -1] for k, v in post.items()}

    imagined_states = []
    imagined_rewards = []

    # ------------------------------------------------------------
    # 4. Imagination rollout
    # ------------------------------------------------------------
    for t in range(horizon):
        # Random action
        action = torch.rand(1, action_dim, device=device) * 2 - 1

        # Dynamics step
        state = rssm.img_step(state, action)

        imagined_states.append(state)

        # Predict reward
        feat = rssm.get_feat(state)
        reward = reward_head(feat).mode()
        imagined_rewards.append(reward.item())

    # ------------------------------------------------------------
    # 5. Decode imagined states
    # ------------------------------------------------------------
    feats = torch.stack([rssm.get_feat(s) for s in imagined_states], 1)
    recon = decoder(feats)["image"].mode()  # (1, T, H, W, 3)

    recon = recon[0].cpu().numpy()

    # ------------------------------------------------------------
    # 6. Save frames
    # ------------------------------------------------------------
    for t in range(horizon):
        img = recon[t]  # (H, W, 3*num_cameras), float in [0, 1]

        # Handle multi-camera images exactly like before
        h, w, c = img.shape
        if c > 3:
            num_cameras = c // 3
            camera_frames = []
            for i in range(num_cameras):
                camera_frame = img[:, :, i*3:(i+1)*3]
                camera_frames.append(camera_frame)
            img = np.concatenate(camera_frames, axis=1)  # (H, W*num_cameras, 3)

        # Now safe to draw text
        img = draw_reward(img, imagined_rewards[t])

        # Img has a reverse color channel order due to OpenCV, convert it back before saving
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{out_dir}/frame_{t:03d}.png", img)

    print(f"Saved imagination rollout to {out_dir}")

def make_env(config):
    suite, task = config.task.split("_", 1)
    print("Running DreamerV3 on Metaworld task:", task)
    env = gymnasium.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", max_episode_steps=config.time_limit)
    env = ProprioMultiImageObsWrapper(env,
                                    image_height=config.size[0],
                                    image_width=config.size[1],
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

def main(config):
    device = "cuda"
    tools.set_seed_everywhere(config.seed)

    # Create environment
    env = make_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = (
        act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    )

    # Create models
    wm = models.WorldModel(
        obs_space=obs_space,
        act_space=act_space,
        step=0,
        config=config,
    ).to(device)
    wm.requires_grad_(False)
    wm.eval()

    # Load trained checkpoint
    assert config.checkpoint is not None
    checkpoint = torch.load(config.checkpoint, map_location=device)

    # Extract only WorldModel parameters from the agent checkpoint
    agent_state = checkpoint["agent_state_dict"]

    wm_state = {}
    for k, v in agent_state.items():
        if k.startswith("_wm._orig_mod."):
            new_key = k.replace("_wm._orig_mod.", "")
            wm_state[new_key] = v

    missing, unexpected = wm.load_state_dict(wm_state, strict=False)

    print("Missing:", missing)
    print("Unexpected:", unexpected)

    # Reset environment
    obs = env.reset()

    # Run imagination rollout
    imagine_rollout(
        obs=obs,
        preprocess=wm.preprocess,
        encoder=wm.encoder,
        rssm=wm.dynamics,
        decoder=wm.heads["decoder"],
        reward_head=wm.heads["reward"],
        action_dim=config.num_actions,
        horizon=30,
        device=device,
        out_dir="imagined_rollout",
    )

    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
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

    # Add checkpoint argument
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint to load weights from.")
    main(parser.parse_args(remaining))