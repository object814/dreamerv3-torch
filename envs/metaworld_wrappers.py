"""
metaworld_wrappers.py

Wrappers for Metaworld environments to be used with DreamerV3, alongside dreamer wrappers.
Main wrappers:
    - Gymnasium2Gym: Converts Gymnasium environments to be compatible with Gym-based DreamerV3.
    - FirstandTerminalObs: Adds is_first and is_terminal observation flags to the observation space as DreamerV3 expects.

Make sure to wrap your Metaworld environments with these wrappers before using them with DreamerV3, and with Gymnasium2Gym at last.
"""
import gymnasium
import gym
import numpy as np

class Gymnasium2Gym(gymnasium.Wrapper):
    """
    Adapts a Gymnasium environment to the classic Gym API.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        # Convert observation space from Gymnasium to Gym
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, gymnasium.spaces.Box):
                low = space.low
                high = space.high
                shape = space.shape
                dtype = space.dtype
                env.observation_space.spaces[key] = gym.spaces.Box(low, high, shape, dtype)
            elif isinstance(space, gymnasium.spaces.Discrete):
                n = space.n
                env.observation_space.spaces[key] = gym.spaces.Discrete(n)
            else:
                raise NotImplementedError(f"Space type {type(space)} not supported in Gymnasium2Gym wrapper.")
        self.observation_space = env.observation_space
        # Convert action space from Gymnasium to Gym
        space = env.action_space
        if isinstance(space, gymnasium.spaces.Box):
            low = space.low
            high = space.high
            shape = space.shape
            dtype = space.dtype
            self.action_space = gym.spaces.Box(low, high, shape, dtype)
        elif isinstance(space, gymnasium.spaces.Discrete):
            n = space.n
            self.action_space = gym.spaces.Discrete(n)
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported in Gymnasium2Gym wrapper.")

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        return obs, reward, done, info
    
class FirstTerminalObs(gymnasium.Wrapper):
    """
    Add 'is_first' and 'is_terminal' flags to the observation space.
    """
    def __init__(self, env):
        super().__init__(env)
        spaces = dict(self.env.observation_space.spaces)

        spaces["is_first"] = gymnasium.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
        spaces["is_terminal"] = gymnasium.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Dict(spaces)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        obs["is_first"] = np.array(1.0, dtype=np.float32)
        obs["is_terminal"] = np.array(0.0, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs["is_first"] = np.array(0.0, dtype=np.float32)
        obs["is_terminal"] = np.array(1.0 if terminated else 0.0, dtype=np.float32)
        return obs, reward, terminated, truncated, info