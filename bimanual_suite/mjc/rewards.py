from bimanual_suite.mjc import cubes, control
import mink
import numpy as np

class RewardFunction:
    def __init__(self,
                 env: cubes.OneCubeAssembleEnvironment,
                 dist_weight: float = 5.0,
                 lift_weight: float = 2.0,
                 grasp_bonus: float = 20.0,
                 place_bonus: float = 200.0,
                 passive_stable_weight: float = 0.5,
                 sep_penalty_weight: float = 0.1,
                 joint_force_penalty: float = 0.05,
                 time_penalty: float = -0.01,
                 grasp_dist_thresh: float = 0.02,
                 place_dist_thresh: float = 0.02,
                 block_init_pos: np.ndarray = None,
                 verbose: bool = False):
        """
        Reward function for pick and place task.

        Args:
            env: The pick and place environment.
            dist_weight: Weight for distance to target.
            lift_weight: Weight for lifting the object.
            grasp_bonus: Bonus for successful grasp.
            place_bonus: Bonus for successful placement.
            passive_stable_weight: Weight for passive arm stability.
            sep_penalty_weight: Penalty weight for distance between two arms.
            joint_force_penalty: Penalty for joint forces.
            time_penalty: Penalty per time step.
            grasp_dist_thresh: Distance threshold for considering a successful grasp.
            place_dist_thresh: Distance threshold for considering a successful placement.
            block_init_pos: Initial position of the block.
            verbose: Whether to print debug information.
        """
        self.env = env
        self.cube_width = env.cube_width
        
        # Weights and bonuses
        self.dist_weight = dist_weight # distance weight
        self.lift_weight = lift_weight
        self.grasp_bonus = grasp_bonus
        self.place_bonus = place_bonus
        self.passive_stable_weight = passive_stable_weight
        self.sep_penalty_weight = sep_penalty_weight # penalty for distance between two arms
        self.joint_force_penalty = joint_force_penalty
        self.time_penalty = time_penalty
        self.grasp_dist_thresh = grasp_dist_thresh if grasp_dist_thresh is not None else max(0.05, 0.6 * self.cube_width)
        self.place_dist_thresh = place_dist_thresh

        # Internal state
        self.prev_grasped = False
        self.prev_block_pos = None
        self.prev_active_pose = None # previous active end-effector pose
        self.prev_passive_pose = None # previous passive end-effector pose
        self.is_placed = False
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.block_init_pos = block_init_pos
        self.verbose = verbose

    def compute_reward(self,
                       active_arm: str,
                       passive_arm: str,
                       left_pose: mink.SE3,
                       right_pose: mink.SE3,
                       block_pose: mink.SE3,
                       target_pose: mink.SE3
                       ):
        """
        Compute reward based on the current state and observations.

        Args:
            active_arm: Which arm is active (left or right).
            passive_arm: Which arm is passive (left or right).
            left_pose: Pose of the left end-effector.
            right_pose: Pose of the right end-effector.
            block_pose: Pose of the block being manipulated.
            target_pose: Target pose for placing the block.
        Returns:
            reward: Computed reward.
        """
        self.step_count += 1

        # Get end-effector poses
        ee_pose = {
            'left': left_pose,
            'right': right_pose
        }
        active_pose = ee_pose[active_arm]
        passive_pose = ee_pose[passive_arm]
        t_active = active_pose.translation().copy()
        t_passive = passive_pose.translation().copy()
        t_block = block_pose.translation().copy()
        t_target = target_pose.translation().copy()

        # State comprehension
        gripper_pos = self.env.get_gripper_state(active_arm) # 0 if closed
        if gripper_pos < 0.25:
            gripper_closed = False
        else:
            gripper_closed = True
        dist_to_block = np.linalg.norm(t_active - t_block)
        dist_block_to_target = np.linalg.norm(t_block - t_target)
        
        has_grasped = (dist_to_block < self.grasp_dist_thresh) and gripper_closed
        is_placed = (dist_block_to_target < self.place_dist_thresh) and not gripper_closed
        
        # Active arm reward
        r_active = 0.0
        if not has_grasped:
            # Encourage approaching the block
            r_active = - self.dist_weight * dist_to_block
        else:
            # Encourage lifting and approaching the target
            # TODO: consider normalising the height by subtracting table height or initial block height
            t_lift = t_active[2] - (self.block_init_pos[2] if self.block_init_pos is not None else 0.0)
            r_lift = self.lift_weight * t_lift
            r_target = - self.dist_weight * dist_block_to_target
            r_active = r_lift + r_target

        # Passive arm reward
        r_passive = 0.0
        # Currently only encourage passive arm to stay where it is
        r_passive -= self.passive_stable_weight * np.linalg.norm(t_passive - (self.prev_passive_pose.translation() if self.prev_passive_pose is not None else t_passive))

        # Separation penalty, penalise strong proximity between arms
        sep_dist = np.linalg.norm(t_active - t_passive)
        r_sep = - self.sep_penalty_weight * np.exp(-10.0 * sep_dist)
        r_passive += r_sep

        # Sparse task rewards
        r_task = 0.0
        if has_grasped and not self.prev_grasped:
            if self.verbose:
                print(f"Grasped at step {self.step_count}!")
            r_task += self.grasp_bonus
        if is_placed and not self.is_placed:
            if self.verbose:
                print(f"Placed at step {self.step_count}!")
            r_task += self.place_bonus

        # Penalties and time cost
        r_penalty = 0.0
        # Joint force penalty
        left_f = self.env.get_joint_forces("left")
        right_f = self.env.get_joint_forces("right")
        max_force = max(np.max(np.abs(left_f)), np.max(np.abs(right_f)))
        force_scale = np.mean([np.std(left_f), np.std(right_f)]) + 1e-6  # soft normalization
        normalized_force = max_force / force_scale
        r_force = - self.joint_force_penalty * normalized_force**2
        # Time penalty
        r_time = self.time_penalty

        r_penalty += r_force + r_time

        # Sum up rewards and penalties
        reward = r_active + r_passive + r_task + r_penalty

        self.prev_grasped = has_grasped
        self.is_placed = is_placed
        self.prev_passive_pose = passive_pose
        self.prev_active_pose = active_pose # Temporarily not used
        self.prev_block_pos = t_block.copy() # Temporarily not used

        # Debugging info
        if self.verbose:
            print(f"Step Reward: {reward:.3f}")
        return reward