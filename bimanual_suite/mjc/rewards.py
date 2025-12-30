from bimanual_suite.mjc import cubes
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

class RewardFunction:
    def __init__(self,
                 env: cubes.OneCubeAssembleEnvironment,
                 # Weights
                 reach_weight: float = 0.1,
                 grasp_weight: float = 0.2,
                 hold_weight: float = 0.5,
                 place_weight: float = 0.5,
                 lift_weight: float = 0.02,
                 passive_drift_penalty_weight: float = 0.1,
                 
                 # Sparse Bonuses
                 grasp_bonus: float = 10.0,
                 success_bonus: float = 20.0,
                 
                 # Penalties
                 early_close_weight: float = 0.1,
                 time_penalty: float = 0.01,
                 
                 # Scales
                 reach_scale: float = 4.0,
                 place_scale: float = 4.0,
                 lift_scale: float = 4.0,
                 
                 # Config
                 use_orientation: bool = True,
                 grasp_hold_steps_required: int = 10,
                 orient_weight: float = 0.0,
                 grasp_penalty_dist_thresh: float = 0.05, # distance beyond which early close penalty applies
                 passive_drift_dist_thresh: float = 0.08, # distance beyond which passive penalty starts
                 place_dist_thresh: float = 0.03,
                 table_height: float = 0.795,
                 lift_height_target: float = 0.10,
                 verbose: bool = False,
                ):
        """
        Reward function for bimanual pick-and-place task with:
          - passive arm -> penalty if drifts beyond deadzone
          - place reward computed as relative progress from initial distance
          - increased sparse grasp/success bonuses
        """
        self.env = env
        self.reach_weight = reach_weight
        self.grasp_weight = grasp_weight
        self.hold_weight = hold_weight
        self.place_weight = place_weight
        self.lift_weight = lift_weight
        self.passive_drift_penalty_weight = passive_drift_penalty_weight
        self.early_close_weight = early_close_weight
        self.time_penalty = time_penalty
        self.grasp_bonus = grasp_bonus
        self.success_bonus = success_bonus
        self.reach_scale = reach_scale
        self.place_scale = place_scale
        self.lift_scale = lift_scale
        self.passive_drift_dist_thresh = passive_drift_dist_thresh
        self.table_height = table_height
        self.lift_height_target = lift_height_target
        self.use_orientation = use_orientation
        self.orient_weight = orient_weight
        self.grasp_penalty_dist_thresh = grasp_penalty_dist_thresh
        self.place_dist_thresh = place_dist_thresh
        self.verbose = verbose

        self.grasp_hold_steps_required = grasp_hold_steps_required
        
        # Internal State
        self.has_grasped = False
        self.has_placed = False
        self._initial_gripper_block_dist = None
        self._initial_passive_pos = None
        self._initial_block_height = None
        self._initial_block_target_dist = None
        self._grasp_hold_counter = 0

        print("=== Reward Function Initialised ===")

    def reset(self):
        self.has_grasped = False
        self.has_placed = False
        self._initial_gripper_block_dist = None
        self._initial_passive_pos = None
        self._initial_block_height = None
        self._initial_block_target_dist = None
        self._grasp_hold_counter = 0

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
        t_passive = ee_pose[passive_arm].translation()
        t_block = block_pose.translation()
        t_target = target_pose.translation()

        # Calculate distances
        dist_gripper_block = np.linalg.norm(t_active - t_block)
        dist_block_target = np.linalg.norm(t_block - t_target)
        # Initial passive hand position
        if self._initial_passive_pos is None:
            self._initial_passive_pos = t_passive.copy()
        dist_passive_drift = np.linalg.norm(t_passive - self._initial_passive_pos)
        # Initial gripper-block distance
        if self._initial_gripper_block_dist is None:
            # Save initial distance
            self._initial_gripper_block_dist = float(dist_gripper_block + 1e-8)
        # Initial block-target distance
        if self._initial_block_target_dist is None:
            # Save initial distance
            self._initial_block_target_dist = float(dist_block_target + 1e-8)
        if self._initial_block_height is None:
            self._initial_block_height = t_block[2]
                
        # Grasp Detection
        # Find active arm gripper geometry names
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
        # Check for contact, grasped if both gripper geoms in contact with block
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

        # ==========
        # Reward
        # ==========
        # Reach reward (0,1)
        reach_progress = (self._initial_gripper_block_dist - dist_gripper_block) / self._initial_gripper_block_dist
        r_reach = np.tanh(self.reach_scale * reach_progress)
        r_reach = max(0.0, r_reach)  # No negative reach reward

        # Place reward (0,1)
        place_progress = (self._initial_block_target_dist - dist_block_target) / self._initial_block_target_dist
        r_place = np.tanh(self.place_scale * place_progress)
        r_place = max(0.0, r_place)  # No negative place reward

        # Grasp reward =1
        r_grasp = 1.0 if is_grasping_now else 0.0

        # Consecutive-grasp reward =1
        if is_grasping_now:
            self._grasp_hold_counter += 1
        else:
            self._grasp_hold_counter = 0

        if self._grasp_hold_counter >= self.grasp_hold_steps_required:
            r_hold = 1.0
        else:
            r_hold = 0.0

        # Lift reward (0,1)
        if self._initial_block_height is not None:
            lift_progress = t_block[2] - self._initial_block_height
            lift_progress = max(0.0, lift_progress)
            r_lift = np.tanh(self.lift_scale * lift_progress)
        else:
            r_lift = 0.0

        # ==========
        # Penalties
        # ==========
        # Gripper close penalty
        gripper_val = self.env.get_gripper_state(active_arm)
        gripper_closed = (gripper_val > 0.25)  # Threshold for "closed" state
        if dist_gripper_block > self.grasp_penalty_dist_thresh and gripper_closed:
            r_early_close_penalty = -1.0
        else:
            r_early_close_penalty = 0.0
        
        # Passive hand drift penalty (-1,0)
        drift_dist = max(0.0, dist_passive_drift - self.passive_drift_dist_thresh)
        r_passive_penalty = -0.5 * np.tanh(drift_dist)
        
        # ==========
        # Sparse Bonuses
        # ==========
        # Grasp Bonus =1
        if (self._grasp_hold_counter >= self.grasp_hold_steps_required) and (not self.has_grasped):
            r_grasp_bonus = 1.0
            self.has_grasped = True
            print("*** OBJECT GRASPED! ***")
        else:
            r_grasp_bonus = 0.0
            
        # Placed Bonus =1
        if dist_block_target < self.place_dist_thresh and not self.has_placed:
            r_success_bonus = 1.0
            self.has_placed = True
            print("*** OBJECT PLACED SUCCESSFULLY! ***")
        else:
            r_success_bonus = 0.0

        # ==========
        # Total Reward
        # ==========
        reward = self.reach_weight * r_reach + \
                 self.grasp_weight * r_grasp + \
                 self.hold_weight * r_hold + \
                 self.place_weight * r_place + \
                 self.lift_weight * r_lift + \
                 self.early_close_weight * r_early_close_penalty + \
                 self.grasp_bonus * r_grasp_bonus + \
                 self.success_bonus * r_success_bonus + \
                 self.time_penalty * (-1.0)
                #  self.passive_drift_penalty_weight * r_passive_penalty + \

        # Populate info keys
        info = {}
        info['reward/reach'] = self.reach_weight * r_reach
        info['reward/grasp'] = self.grasp_weight * r_grasp
        info['reward/hold'] = self.hold_weight * r_hold
        info['reward/place'] = self.place_weight * r_place
        info['reward/lift'] = self.lift_weight * r_lift
        info['reward/early_close'] = self.early_close_weight * r_early_close_penalty
        # info['reward/passive'] = self.passive_drift_penalty_weight * r_passive_penalty
        info['reward/grasp_bonus'] = self.grasp_bonus * r_grasp_bonus
        info['reward/success_bonus'] = self.success_bonus * r_success_bonus
        info['reward/time_penalty'] = self.time_penalty * (-1.0)
        info['reward/total'] = reward

        if self.verbose:
            print(f"  Reach: {r_reach*100:.2f}%, Grasp: {r_grasp*100:.2f}%, Hold: {r_hold*100:.2f}%, Place: {r_place*100:.2f}%")
            print(f"  Early Close: {r_early_close_penalty:.1f}, Passive: {r_passive_penalty:.3f}")
            print(f"  Grasp Bonus: {r_grasp_bonus:.3f}, Success Bonus: {r_success_bonus:.3f}")
            print(f"  Total Reward: {reward:.3f}\n")
        
        return reward, info