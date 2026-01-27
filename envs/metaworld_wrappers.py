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
        done = bool(terminated or truncated)

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
    
class RewardTuningWrapper(gymnasium.Wrapper):
    """
    Wrapper to tune rewards for Metaworld environments used with DreamerV3.
    
    Originally, Metaworld environments do not terminate episodes when tasks are completed.
    Instead, they provide a constant positive reward for staying at the goal state.
    That is ok for fixed-horizon training setups,
    but not for DreamerV3 which relies on episode termination signals and emphasises learning longer-term rewards through imagination.

    Reward tuning:
        - Add step-based penalties to encourage faster task completion.
        - Scale rewards to balance between task completion bonuses and dense rewards.
    """
    def __init__(
        self,
        env: gymnasium.Env,
        success_bonus: float = 200.0,
        step_penalty: float = 0.2,
        success_key: str = "success",
    ):
        """
        Args:
            env (gymnasium.Env): The Metaworld environment to wrap.
            success_bonus (float): The bonus reward to give upon task completion.
            step_penalty (float): The penalty to subtract at each step to encourage faster completion.
            success_key (str): The key in the metaworld info dictionary that indicates task success.
        """
        super().__init__(env)

        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.success_key = success_key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Per-step penalty
        if self.step_penalty != 0.0:
            reward -= self.step_penalty

        # Success bonus
        success = bool(info.get(self.success_key, False))
        if success:
            print(">>> DEBUG: Success detected in RewardTuningWrapper.")
            print(f">>> DEBUG: Previous reward: {reward}")
            reward += self.success_bonus
            print(f">>> DEBUG: New reward: {reward}")

        return obs, reward, terminated, truncated, info
    
class RewardTuningWrapperV2(gymnasium.Wrapper):
    """
    Wrapper to tune rewards for Metaworld environments used with DreamerV3.
    
    Originally, Metaworld environments do not terminate episodes when tasks are completed.
    Instead, they provide a constant positive reward for staying at the goal state.
    That is ok for fixed-horizon training setups,
    but not for DreamerV3 which relies on episode termination signals and emphasises learning longer-term rewards through imagination.

    Reward tuning:
        - Scale rewards from original range to target range.
    """
    def __init__(
        self,
        env: gymnasium.Env,
        original_reward_range: tuple = (-1.0, 1.0),
        target_reward_range: tuple = (-1.0, 0.0),
    ):
        """
        Args:
            env (gymnasium.Env): The Metaworld environment to wrap.
            original_reward_range (tuple): The original reward range of the environment.
            target_reward_range (tuple): The desired target reward range after tuning.
        """
        super().__init__(env)
        
        self.orig_min, self.orig_max = original_reward_range
        self.target_min, self.target_max = target_reward_range

        print(">>> DEBUG: RewardTuningWrapperV2 initialized.")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        normed_reward = (reward - self.orig_min) / (self.orig_max - self.orig_min) # [0, 1]
        scaled_reward = self.target_min + normed_reward * (self.target_max - self.target_min) # [target_min, target_max]

        return obs, scaled_reward, terminated, truncated, info