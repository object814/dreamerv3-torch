import gymnasium as gym
import numpy as np
import cv2


from bimanual_suite.mjc.cubes import OneCubeAssembleEnvironment
from bimanual_suite.gym import PickAndPlaceGymEnv

class BimanualControl:
    metadata = {}

    def __init__(self, name, image_size=(64, 64), action_repeat=1, seed=42):
        """
        Initialise a Mujuco environment for DreamerV3 training.

        Args:
            name (str): The name of the Mujuco environment.
            image_size (tuple): The size of the rendered images.
            action_repeat (int): The number of times to repeat each action.
            seed (int): Random seed for environment initialization.
        """
        if name == "pick_and_place":
            pass
        else:
            raise NotImplementedError(f"MuJoco environment '{name}' is not implemented.")
        self._action_repeat = action_repeat
        self._size = image_size
        self.reward_range = [-np.inf, np.inf]

        backend = OneCubeAssembleEnvironment(seed=seed)
        self._env = PickAndPlaceGymEnv(env=backend, init_with_pregrasp=True, render=False)
        self._build_observation_space()

    def _build_observation_space(self):
        spaces = {}
        if isinstance(self._env.observation_space, gym.spaces.Dict):
            spaces.update(self._env.observation_space.spaces)
        else:
            spaces["obs"] = self._env.observation_space
        
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, shape=(), dtype=np.bool_)
        spaces["is_first"] = gym.spaces.Box(0, 1, shape=(), dtype=np.bool_)
        self._obs_space = gym.spaces.Dict(spaces)

    @property
    def observation_space(self):
        return self._obs_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def _make_image(self):
        raw_img = self._env.mjc_obs["front_camera"] # (H, W, 3)
        if raw_img is None:
            raise RuntimeError("No front_camera image found in the mujoco environment.")
        if raw_img.dtype == np.uint8:
            img = raw_img
        else:
            print("Warning: Converting raw image to uint8.")
            img = (np.clip(raw_img, 0, 1) * 255).astype(np.uint8)


        resized = cv2.resize(img, (self._size[1], self._size[0]), interpolation=cv2.INTER_AREA)
        return resized
    
    def step(self, action):
        assert np.isfinite(action).all(), "Action contains non-finite values."
        
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            done = bool(terminated or truncated)

            if done:
                break

        # Build observation
        out = dict(obs)
        out["image"] = self._make_image()
        out["is_terminal"] = done
        out["is_first"] = False
        return out, total_reward, done, info
    
    def reset(self):
        obs, info = self._env.reset()
        out = dict(obs)
        out["image"] = self._make_image()
        out["is_terminal"] = False
        out["is_first"] = True
        return out
    
    def render(self, mode="rgb_array"):
        try:
            if getattr(self._env, "_viewer", None) is None:
                self._env._start_viewer()
            # sync the viewer (may throw)
            if getattr(self._env, "_viewer", None) is not None:
                self._env._viewer.sync()
        except Exception:
            print("Warning: Failed to sync the viewer for rendering.")
            pass

    