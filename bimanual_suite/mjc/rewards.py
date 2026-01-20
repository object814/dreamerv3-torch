from bimanual_suite.mjc import cubes
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

class RewardFunction:
    def __init__(self,
                 env: cubes.OneCubeAssembleEnvironment,
                 # Weights
                 reach_weight: float = 3.0, # L2 distance
                 place_weight: float = 2.0, # for L2 distance
                 gripper_weight: float = 0.5,
                 
                 # Config
                 grasp_penalty_dist_thresh: float = 0.03, # distance beyond which early close penalty applies
                 place_dist_thresh: float = 0.03,
                 verbose: bool = False,
                ):
        self.env = env
        self.reach_weight = reach_weight
        self.place_weight = place_weight
        self.gripper_weight = gripper_weight
        self.grasp_penalty_dist_thresh = grasp_penalty_dist_thresh
        self.place_dist_thresh = place_dist_thresh
        self.verbose = verbose
        
        # Internal State
        self.has_grasped = False
        self.has_placed = False
        print("=== Reward Function Initialised ===")

    def reset(self):
        self.has_grasped = False
        self.has_placed = False

    def compute_reward(self,
                       active_arm: str,
                       passive_arm: str,
                       left_pose: mink.SE3,
                       right_pose: mink.SE3,
                       block_pose: mink.SE3,
                       target_pose: mink.SE3
                       ):
        # ==========
        # Calculations
        # ==========
        ee_pose = {'left': left_pose, 'right': right_pose}
        t_active = ee_pose[active_arm].translation()
        t_block = block_pose.translation()
        t_target = target_pose.translation()

        # Calculate L2 distances
        dist_gripper_block = np.linalg.norm(t_active - t_block)
        dist_block_target = np.linalg.norm(t_block - t_target)

        # Gripper open/close state
        gripper_val = self.env.get_gripper_state(active_arm)
        if gripper_val <= 0.25:
            gripper_is_closed = False
        else:
            gripper_is_closed = True

        # ==========
        # Grasping detection
        # ==========
        if active_arm == 'left':
            active_gripper_id = [mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_gripper_left_finger"
            ), mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_gripper_right_finger"
            )]
        else:
            active_gripper_id = [mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_gripper_left_finger"
            ), mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_gripper_right_finger"
            )]
        block_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "orange_cube_geom"
        )
        contact_list = set()
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            geom1 = c.geom1
            geom2 = c.geom2
            if geom1 == block_id:
                contact_list.add(geom2)
            elif geom2 == block_id:
                contact_list.add(geom1)
        is_grasping_now = all(gid in contact_list for gid in active_gripper_id)
        if is_grasping_now and not self.has_grasped:
            self.has_grasped = True
            print(">>> Grasp detected!")
        
        # ==========
        # Placing detection
        # ==========
        is_placed_now = dist_block_target < self.place_dist_thresh
        if is_placed_now and not self.has_placed:
            self.has_placed = True
            print(">>> Place detected!")

        # ==========
        # Reward
        # ==========
        # Reach reward
        r_reach = - dist_gripper_block

        # Place reward
        r_place = - dist_block_target

        # Gripper reward and penalty
        if (dist_gripper_block < self.grasp_penalty_dist_thresh and dist_block_target > self.place_dist_thresh):
            # Encourage closing
            r_gripper = 1.0 if gripper_is_closed else 0.0
        else:
            # Encourage opening
            r_gripper = 0.0 if gripper_is_closed else 1.0

        # ==========
        # Total Reward
        # ==========
        reward = self.reach_weight * r_reach + \
                 self.place_weight * r_place + \
                 self.gripper_weight * r_gripper
                #  self.hold_weight * r_hold + \
                #  self.grasp_weight * r_grasp + \
                #  self.lift_weight * r_lift + \
                #  self.passive_drift_penalty_weight * r_passive_penalty + \

        # Populate info keys
        info = {}
        info['reward/reach'] = self.reach_weight * r_reach
        info['reward/place'] = self.place_weight * r_place
        info['reward/gripper'] = self.gripper_weight * r_gripper
        info['reward/total'] = reward

        if self.verbose:
            print(f"  Reach: {r_reach*100:.2f}%, Place: {r_place*100:.2f}%, Gripper: {r_gripper:.2f}%")
            print(f"  Total Reward: {reward:.3f}\n")
        
        return reward, info