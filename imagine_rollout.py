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
import imageio

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# Metaworld setup
import sys
import envs.metaworld_wrappers as metaworld_wrappers
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper


# ------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------

def format_multicamera_image(img):
    """
    img: (H, W, 3*num_cameras), float in [0,1]
    returns: (H, W*num_cameras, 3), float
    """
    h, w, c = img.shape
    if c > 3:
        num_cameras = c // 3
        frames = [img[:, :, i*3:(i+1)*3] for i in range(num_cameras)]
        img = np.concatenate(frames, axis=1)
    return img


def draw_reward(img, reward):
    """
    img: (H, W, 3), float in [0,1]
    reward: scalar
    """
    img = (img * 255).astype(np.uint8)
    cv2.putText(
        img,
        f"Reward: {reward:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


# ------------------------------------------------------------
# Imagination rollout
# ------------------------------------------------------------

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
):
    """
    obs: single environment observation dict
    returns: list of BGR uint8 frames
    """

    # ---- prepare obs for encoder ----
    obs = {k: np.expand_dims(v, 0) for k, v in obs.items() if "log_" not in k}
    obs = preprocess(obs)

    embed = encoder(obs)
    embed = embed.unsqueeze(1)  # (B=1, T=1, E)

    # ---- initial posterior ----
    post, _ = rssm.observe(
        embed=embed,
        action=torch.zeros(1, 1, action_dim, device=device),
        is_first=torch.ones(1, 1, device=device),
    )

    state = {k: v[:, -1] for k, v in post.items()}

    imagined_states = []
    imagined_rewards = []

    # ---- imagination loop ----
    for _ in range(horizon):
        action = torch.rand(1, action_dim, device=device) * 2 - 1
        state = rssm.img_step(state, action)
        imagined_states.append(state)

        feat = rssm.get_feat(state)
        reward = reward_head(feat).mode()
        imagined_rewards.append(reward.item())

    # ---- decode ----
    feats = torch.stack([rssm.get_feat(s) for s in imagined_states], 1)
    recon = decoder(feats)["image"].mode()[0].cpu().numpy()

    # ---- build frames ----
    frames = []
    for t in range(horizon):
        img = recon[t]
        img = format_multicamera_image(img)
        img = draw_reward(img, imagined_rewards[t])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frames.append(img)

    return frames


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

def make_env(config):
    suite, task = config.task.split("_", 1)
    print("Running DreamerV3 on Metaworld task:", task)

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(config):
    device = "cuda"
    tools.set_seed_everywhere(config.seed)

    env = make_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = act_space.shape[0]

    # ---- world model ----
    wm = models.WorldModel(
        obs_space=obs_space,
        act_space=act_space,
        step=0,
        config=config,
    ).to(device)

    wm.requires_grad_(False)
    wm.eval()

    # ---- load checkpoint ----
    checkpoint = torch.load(config.checkpoint, map_location=device)
    agent_state = checkpoint["agent_state_dict"]

    wm_state = {}
    for k, v in agent_state.items():
        if k.startswith("_wm._orig_mod."):
            wm_state[k.replace("_wm._orig_mod.", "")] = v

    wm.load_state_dict(wm_state, strict=False)
    print("Loaded world model.")

    # ---- reset env ----
    obs = env.reset()

    # ---- real frame ----
    real_img = obs["image"].astype(np.float32) / 255.0
    real_img = format_multicamera_image(real_img)
    real_img = draw_reward(real_img, 0.0)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)

    # ---- imagination ----
    imagined_frames = imagine_rollout(
        obs=obs,
        preprocess=wm.preprocess,
        encoder=wm.encoder,
        rssm=wm.dynamics,
        decoder=wm.heads["decoder"],
        reward_head=wm.heads["reward"],
        action_dim=config.num_actions,
        horizon=30,
        device=device,
    )

    # ---- write video ----
    frames = [real_img] + imagined_frames
    h, w, _ = frames[0].shape

    fps = 20

    if config.video_format == "mp4":
        video_path = "imagined_rollout.mp4"
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for f in frames:
            writer.write(f)
        writer.release()

    elif config.video_format == "gif":
        video_path = "imagined_rollout.gif"

        # Convert BGR → RGB for imageio
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        imageio.mimsave(
            video_path,
            rgb_frames,
            fps=fps,
            loop=0,        # infinite loop
        )

    print(f"Saved imagination video: {video_path}")

    env.close()



# ------------------------------------------------------------
# Config parsing
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    defaults = {}
    for name in ["defaults", *(args.configs or [])]:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for k, v in sorted(defaults.items()):
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video_format", type=str, default="gif", choices=["mp4", "gif"], help="Output video format")
    main(parser.parse_args(remaining))
