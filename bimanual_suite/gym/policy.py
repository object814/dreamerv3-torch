from gym import spaces
import numpy as np
from typing import Any, Dict

obs_space = spaces.Dict({
    "image": spaces.Box(0.0, 1.0, shape=(84, 84, 3), dtype=np.float32),
    "proprio": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),   # [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    # "percep": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),           # e.g. [obj_x, obj_y, obj_z, obj_, tgt_v, tgt_z]
})
action_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)  # [left_arm_ee delta (xyzrotvec), right_arm_ee delta (xyzrotvec), left_gripper, right_gripper]


class RandomPolicy:
    def __init__(self):
        self.action_space = action_space

    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        return self.action_space.sample()
    
class PPOPolicy:
    def __init__(self):
        self.action_space = action_space

    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        # Placeholder for PPO policy action selection
        # In a real implementation, this would use a trained model to select actions
        return self.action_space.sample()