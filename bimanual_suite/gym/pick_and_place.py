"""
pick_and_place.py
A Gym environment wrapper for the pick and place bimanual task using mujoco as simulation.
"""
from random import seed
from typing import Any, Dict, Tuple, Optional
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from scipy.spatial.transform import Rotation as R
import mink

from bimanual_suite.mjc import control
from bimanual_suite.mjc.cubes import OneCubeAssembleEnvironment
from bimanual_suite.mjc.rewards import RewardFunction
from bimanual_suite.gym.policy import RandomPolicy

obs_space = spaces.Dict({
    # "image": spaces.Box(0.0, 1.0, shape=(128, 128, 3), dtype=np.float32),
    "qpos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),   # [active_arm, passive_arm]
    "qvel": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),  # [active_arm, passive_arm]
    "ee_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),  # [active_arm (xyz), passive_arm(xyz)]
    "ee_quat": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),  # [active_arm (xyzw), passive_arm(xyzw)]
    "gripper": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),  # [active_arm, passive_arm]
    "obj_pos": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),           # e.g. [obj_x, obj_y, obj_z]
    "obj_quat": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),         # e.g. [obj_xyzw]
    "target_pos": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),        # e.g. [target_x, target_y, target_z]
    "target_quat": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),      # e.g. [target_xyzw]
    "active_arm": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),  # one-hot: [1,0]=left active, [0,1]=right
})
action_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)  # [active_arm_ee delta (xyzrpy), passive_arm_ee delta (xyzrpy), active_gripper, passive_gripper]
class PickAndPlaceGymEnv(gym.Env):
    def __init__(self,
                 env: OneCubeAssembleEnvironment,
                 dt: float=0.01,
                 init_with_pregrasp: bool=True,
                 solver: str="osqp",
                 max_episode_steps: int=800,
                 seed: Optional[int]=None,
                 verbose: bool=False,
                 render_mode: str = None):
        """
        PickAndPlaceGymEnv
        A Gym environment wrapper for the pick and place bimanual task using mujoco as simulation.

        Args:
            env: Backend mujoco environment.
            dt: Timestep, currently for action delta scaling and ik solver time step.
            init_with_pregrasp: Whether to initialise the environment to a pre-grasp pose on reset.
            solver: IK solver to use.
            max_episode_steps: Maximum number of steps per episode.
            seed: Random seed.
            self.verbose: Whether to print debug information.
            render: Use display rendering or rgb array rendering.
        """
        super().__init__()
        self.env = env # backend mujoco environment
        self.action_space = action_space
        self.observation_space = obs_space
        self.dt = dt
        self.init_with_pregrasp = init_with_pregrasp
        self.solver = solver
        self.seed = seed
        self.verbose = verbose
        if self.verbose:
            print("Logging debug information in PickAndPlaceGymEnv")
        self.render_mode = render_mode
        self._viewer_ctx = None
        self._viewer = None
        if self.render_mode == "display":
            self._start_viewer()
        if self.verbose:
            print(f"Render mode: {self.render_mode}")
        
        # Initialise environment
        self.max_episode_steps = max_episode_steps
        self.active_arm = None
        self.passive_arm = None

        # Initialise reward function
        self.reward_fn = RewardFunction(
            env=self.env,
            verbose=self.verbose
        )

    def _toGymObs(self, mjc_obs: Dict) -> Dict:
        """
        Convert backend (mujoco) observation to Gym observation format.
        """ 
        # mjc_img = mjc_obs["front_camera"]  # (H, W, 3)
        # Determine if normalisation is needed based on range of pixel values
        # if mjc_img.dtype == np.uint8:
            # mjc_img = mjc_img.astype(np.float32) / 255.0
        # Resize image to (128, 128)
        # mjc_img_resized = cv2.resize(mjc_img, (128, 128))

        # Proprioception
        # qpos
        mjc_left_pose = mjc_obs["left_pos"][:7]  # (7,)
        mjc_right_pose = mjc_obs["right_pos"][:7]  # (7,)

        # gripper
        mjc_left_gripper = mjc_obs["left_pos"][7] # (1,)
        mjc_right_gripper = mjc_obs["right_pos"][7] # (1,)

        # qvel
        mjc_left_vel = mjc_obs["left_vel"][:7]  # (7,)
        mjc_right_vel = mjc_obs["right_vel"][:7]  # (7,)

        # ee_pos
        mjc_left_ee_pos = mjc_obs["left_ee_pos"]  # (3,)
        mjc_left_ee_quat = mjc_obs["left_ee_quat"]  # (4,)
        mjc_right_ee_pos = mjc_obs["right_ee_pos"]  # (3,)
        mjc_right_ee_quat = mjc_obs["right_ee_quat"]  # (4,)

        # object and target pos
        obj_pos = mjc_obs["orange_pos"]  # (3,)
        obj_quat = mjc_obs["orange_quat"]  # (4,)
        target_pos = mjc_obs["target_pos"]  # (3,)
        target_quat = mjc_obs["target_quat"]  # (4,)

        # Active arm one-hot
        if self.active_arm == "left":
            active_onehot = np.array([1.0, 0.0], dtype=np.float32)
        else:
            active_onehot = np.array([0.0, 1.0], dtype=np.float32)

        # Re-order proprio to [active_arm, passive_arm]
        if self.active_arm == "left":
            qpos = np.concatenate([mjc_left_pose, mjc_right_pose], axis=0)  # (14,)
            qvel = np.concatenate([mjc_left_vel, mjc_right_vel], axis=0)  # (14,)
            ee_pos = np.concatenate([mjc_left_ee_pos, mjc_right_ee_pos], axis=0)  # (6,)
            ee_quat =  np.concatenate([mjc_left_ee_quat, mjc_right_ee_quat], axis=0)  # (8,)
            gripper = np.array([mjc_left_gripper, mjc_right_gripper], dtype=np.float32)  # (2,)
        else:
            qpos = np.concatenate([mjc_right_pose, mjc_left_pose], axis=0)  # (14,)
            qvel = np.concatenate([mjc_right_vel, mjc_left_vel], axis=0)  # (14,)
            ee_pos = np.concatenate([mjc_right_ee_pos, mjc_left_ee_pos], axis=0)  # (6,)
            ee_quat =  np.concatenate([mjc_right_ee_quat, mjc_left_ee_quat], axis=0)  # (8,)
            gripper = np.array([mjc_right_gripper, mjc_left_gripper], dtype=np.float32)  # (2,)

        gym_obs = {
            # "image": mjc_img_resized.astype(np.float32),
            "qpos": qpos,   # [active_arm, passive_arm]
            "qvel": qvel,  # [active_arm, passive_arm]
            "ee_pos": ee_pos,  # [active_arm (xyz), passive_arm(xyz)]
            "ee_quat": ee_quat,  # [active_arm (xyzw), passive_arm(xyzw)]
            "gripper": gripper,  # [active_arm, passive_arm]
            "obj_pos": obj_pos,           # e.g. [obj_x, obj_y, obj_z]
            "obj_quat": obj_quat,         # e.g. [obj_xyzw]
            "target_pos": target_pos,        # e.g. [target_x, target_y, target_z]
            "target_quat": target_quat,      # e.g. [target_xyzw]
            "active_arm": active_onehot,  # one-hot: [1,0]=left active, [0,1]=right
        }
        return gym_obs
    
    def _getGymPoseForArm(self, side: str) -> np.ndarray:
        """
        Get the gym pose for the specified arm. (xyzrpy)

        Args:
            side: "active" or "passive".

        Returns:
            gym_action: gym pose for the specified arm.
        """
        if side == "active":
            arm = self.active_arm
        elif side == "passive":
            arm = self.passive_arm
        else:
            raise ValueError(f"Invalid side: {side}, must be 'active' or 'passive'")
        
        ee_pos = self.mjc_obs[f"{arm}_ee_pos"].copy()  # (3,)
        ee_quat = self.mjc_obs[f"{arm}_ee_quat"].copy()  # (4,)
        ee_rpy = R.from_quat(ee_quat).as_euler('xyz', degrees=True)  # (3,)
        gym_pose = np.concatenate([ee_pos, ee_rpy], axis=0)  # (6,)
        return gym_pose
    
    def _toMjcAction(self, gym_action: np.ndarray, mjc_env: OneCubeAssembleEnvironment, mjc_obs: Dict) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
            """
            Convert a 14-dim EE-delta into a full environment action.

            Args:
                gym_action: (14,) array of end-effector deltas + gripper commands.
                mjc_obs: Current observation from the environment (used to get current EE poses).

            Returns:
                action (16,) or (None, info) on failure. info dictionary contains debug fields.
            """
            if gym_action.shape != (14,):
                raise ValueError(f"Expected gym_action shape (14,), got {gym_action.shape}")

            # Parse gym_action parts
            delta_active_ee = gym_action[0:6]
            delta_active_ee_t = delta_active_ee[0:3]
            delta_active_ee_r = R.from_euler('xyz', delta_active_ee[3:6], degrees=True)

            delta_passive_ee = gym_action[6:12]
            delta_passive_ee_t = delta_passive_ee[0:3]
            delta_passive_ee_r = R.from_euler('xyz', delta_passive_ee[3:6], degrees=True)
            active_gripper_cmd = float(gym_action[12])
            passive_gripper_cmd = float(gym_action[13])

            # Map to left/right based on active arm
            if self.active_arm == "left":
                delta_left_ee_t = delta_active_ee_t
                delta_left_ee_r = delta_active_ee_r
                delta_right_ee_t = delta_passive_ee_t
                delta_right_ee_r = delta_passive_ee_r
                left_gripper_cmd = active_gripper_cmd
                right_gripper_cmd = passive_gripper_cmd
            else:
                delta_left_ee_t = delta_passive_ee_t
                delta_left_ee_r = delta_passive_ee_r
                delta_right_ee_t = delta_active_ee_t
                delta_right_ee_r = delta_active_ee_r
                left_gripper_cmd = passive_gripper_cmd
                right_gripper_cmd = active_gripper_cmd

            # Get current (reported) end-effector poses from obs
            current_left_ee_t = mjc_obs["left_ee_pos"].copy()
            current_left_ee_quat = mjc_obs["left_ee_quat"].copy()
            current_right_ee_t = mjc_obs["right_ee_pos"].copy()
            current_right_ee_quat = mjc_obs["right_ee_quat"].copy()

            # Compute target EE poses by applying delta in world frame as you've done
            new_left_ee_t = current_left_ee_t + delta_left_ee_t
            new_left_ee_r = delta_left_ee_r * R.from_quat(current_left_ee_quat)

            new_right_ee_t = current_right_ee_t + delta_right_ee_t
            new_right_ee_r = delta_right_ee_r * R.from_quat(current_right_ee_quat)

            # Build mink.SE3 targets
            target_left_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(new_left_ee_r.as_quat()), np.array(new_left_ee_t)
            )
            target_right_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(new_right_ee_r.as_quat()), np.array(new_right_ee_t)
            )

            # Prepare tasks (keep the same costs you used in the original)
            static_lift_cost = np.zeros((mjc_env.model.nv,))
            lift_idx = mjc_env.model.jnt_dofadr[
                mujoco.mj_name2id(
                    mjc_env.model, mujoco.mjtObj.mjOBJ_JOINT, "ewellix_lift_top_joint"
                )
            ]
            static_lift_cost[lift_idx] = 100.0
            left_ee_task = mink.FrameTask(
                frame_name="left_ee",
                frame_type="site",
                position_cost=np.array([1.0, 1.0, 1.0]),
                orientation_cost=1.0,
            )
            right_ee_task = mink.FrameTask(
                frame_name="right_ee",
                frame_type="site",
                position_cost=np.array([1.0, 1.0, 1.0]),
                orientation_cost=1.0,
            )
            damping_task = mink.DampingTask(mjc_env.model, static_lift_cost)
            tasks = [left_ee_task, right_ee_task, damping_task]
            left_ee_task.set_target(target_left_pose)
            right_ee_task.set_target(target_right_pose)

            # Build collision / limit objects exactly as in your original file
            left_arm_geoms = mink.get_subtree_geom_ids(mjc_env.model, mjc_env.model.body("left_kinova_arm_shoulder_link").id)
            right_arm_geoms = mink.get_subtree_geom_ids(mjc_env.model, mjc_env.model.body("right_kinova_arm_shoulder_link").id)
            left_gripper_geoms = mink.get_subtree_geom_ids(mjc_env.model, mjc_env.model.body("left_kinova_arm_bracelet_link").id)
            right_gripper_geoms = mink.get_subtree_geom_ids(mjc_env.model, mjc_env.model.body("right_kinova_arm_bracelet_link").id)
            cube_geoms = ["orange_cube_geom", "orange_cube_target_geom"]
            env_geoms = cube_geoms + ["table"]
            collision_pairs = [
                (left_arm_geoms, left_gripper_geoms),
                (right_arm_geoms, right_gripper_geoms),
                (
                    env_geoms,
                    left_arm_geoms + right_arm_geoms + left_gripper_geoms + right_gripper_geoms,
                ),
            ]
            limits = [
                mink.ConfigurationLimit(model=mjc_env.model),
                mink.CollisionAvoidanceLimit(
                    model=mjc_env.model,
                    geom_pairs=collision_pairs,
                    minimum_distance_from_collisions=0.01,
                    collision_detection_distance=0.2,
                ),
            ]

            # Create configuration and call IK
            configuration = mink.Configuration(mjc_env.model, q=mjc_env.data.qpos.copy())

            ik_succeeded = False
            vel = None
            ik_dampings = [1e-2, 1e-3, 1e-4, 0.0]
            for damping in ik_dampings:
                try:
                    configuration.update(q=mjc_env.data.qpos.copy())
                    vel_solution = mink.solve_ik(
                        configuration,
                        tasks,
                        self.dt,
                        self.solver,
                        damping=damping,
                        safety_break=False,
                        limits=limits,
                    )
                    vel = vel_solution
                    ik_succeeded = True
                    break
                except mink.exceptions.NoSolutionFound:
                    continue

            if not ik_succeeded:
                return None, {"ik_succeeded": False, "reason": "NoSolutionFound"}

            # vel might be given per-state; re-order/concatenate like in original script
            try:
                vel = np.concatenate((vel[mjc_env.left_joint_state_to_vel], vel[mjc_env.right_joint_state_to_vel]))
            except Exception:
                print("Warning: vel indexing failed, using raw vel output")
                vel = np.asarray(vel).ravel()

            # Normalize/group clipping same as original
            max_norm = getattr(control, "MAX_VEL_NORM", 1.0)
            left_group_norm = np.linalg.norm(vel[:6])
            if left_group_norm > 0:
                vel[:6] = max_norm * vel[:6] / max(max_norm, left_group_norm)
            vel[6] = np.clip(vel[6], -2.0, 2.0)
            right_group_norm = np.linalg.norm(vel[7:13])
            if right_group_norm > 0:
                vel[7:13] = max_norm * vel[7:13] / max(max_norm, right_group_norm)
            vel[13] = np.clip(vel[13], -2.0, 2.0)

            # --- IMPORTANT: Build current_joint_pos from the FRESH obs (caller must provide fresh obs) ---
            # We will print a small warning if mjc_env.data site positions disagree with obs to catch stale-observations,
            # but use obs as the source of truth for current_joint_pos because the caller reads obs from mjc_env.step.
            pos_diff = 0.0
            try:
                left_site_id = mjc_env.model.site("left_ee").id
                right_site_id = mjc_env.model.site("right_ee").id
                env_left_pos = mjc_env.data.site_xpos[left_site_id].copy()
                env_right_pos = mjc_env.data.site_xpos[right_site_id].copy()
                pos_diff = np.linalg.norm(env_left_pos - mjc_obs["left_ee_pos"]) + np.linalg.norm(env_right_pos - mjc_obs["right_ee_pos"])
            except Exception:
                # If anything goes wrong, skip mismatch check (we'll still use obs)
                pos_diff = 0.0

            if pos_diff > 1e-6 and self.verbose:
                print(f"WARNING: obs appears stale or out-of-sync (EE pos diff {pos_diff:.6e}). "
                    f"Ensure caller uses `obs = env.step(...)` and passes the returned obs to this function.")

            # Build current_joint_pos from obs
            current_joint_pos = np.concatenate((mjc_obs["left_pos"][:7], mjc_obs["right_pos"][:7]))

            # integrate velocity to create joint position setpoint
            action_joint = current_joint_pos + self.dt * vel

            # assemble final action: joints + grippers
            action = np.concatenate([action_joint, np.array([left_gripper_cmd, right_gripper_cmd])])

            # Sanity check
            if action.size != 16:
                raise RuntimeError(f"Output action length {action.size} != 16")

            info = {"ik_succeeded": True, "vel": vel, "pos_diff": pos_diff}
            return action, info
        
    def _getReward(self, mjc_obs: Dict) -> Tuple[float, Dict]:
        """
        Compute reward using the reward function.
        """ 
        left_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3(mjc_obs["left_ee_quat"]), mjc_obs["left_ee_pos"]
        )
        right_pose = mink.SE3.from_rotation_and_translation(
            mink.SO3(mjc_obs["right_ee_quat"]), mjc_obs["right_ee_pos"]
        )
        block_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(self.env.get_cube_poses()["orange_quat"]), self.env.get_cube_poses()["orange_pos"]
        )
        target_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(self.env.get_cube_poses()["target_quat"]), self.env.get_cube_poses()["target_pos"]
        )
        reward, reward_info = self.reward_fn.compute_reward(
            self.active_arm,
            self.passive_arm,
            left_pose,
            right_pose,
            block_pose,
            target_pose
        )
        return reward, reward_info
    
    def _start_viewer(self):
        if self._viewer is not None:
            return
        self._viewer_ctx = mujoco.viewer.launch_passive(model=self.env.model, data=self.env.data,
                                                        show_left_ui=False, show_right_ui=False)
        self._viewer = self._viewer_ctx.__enter__()

    def _stop_viewer(self):
        if self._viewer_ctx is None:
            return
        try:
            self._viewer_ctx.__exit__(None, None, None)
        finally:
            self._viewer_ctx = None
            self._viewer = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return initial observation.

        Returns:
            obs: Observation.
        """
        super().reset(seed=seed)  # important for Gymnasium compatibility

        # Reset reward function
        if hasattr(self, "reward_fn"):
            self.reward_fn.reset()

        self.mjc_obs = self.env.reset()

        # Determine active and passive arms
        average_y = 0.5 * (self.env.get_cube_poses()["orange_pos"][1]) + 0.5 * (self.env.get_cube_poses()["target_pos"][1])
        self.active_arm = "left" if average_y > 0.0 else "right"
        self.passive_arm = "right" if self.active_arm == "left" else "left"

        for _ in range(5):
            gym_stable_action = np.array([0.0]*14, dtype=np.float32)  # zero action for stability
            mjc_stable_action, _ = self._toMjcAction(gym_stable_action, self.env, self.mjc_obs)
            self.mjc_obs = self.env.step(mjc_stable_action)  # take a few step
            if self.render_mode and self._viewer is not None:
                try:
                    self._viewer.sync()
                except Exception as e:
                    print("Viewer sync failed, closing viewer:", e)
                    self._stop_viewer()

        self.step_count = 0
        self.done = False
        gym_obs = self._toGymObs(self.mjc_obs)
        info = {"active_arm": self.active_arm}

        if self.init_with_pregrasp:
            # Manually set to pre-grasp pose
            # Set translation
            if self.active_arm == "left":
                pregrasp_ee_t = np.array([0.2, 0.2, 1.0])  # x,y,z
            else:
                pregrasp_ee_t = np.array([0.2, -0.2, 1.0])  # x,y,z
            # Set orientation
            pregrasp_ee_rpy = R.from_euler('xyz', [0, 0, 0], degrees=True).as_euler('xyz', degrees=True)
            pregrasp_ee_pose = np.concatenate([pregrasp_ee_t, pregrasp_ee_rpy], axis=0)  # (6,) xyzrpy
            # Move to pre-grasp pose
            reached = False
            current_ee_pose = self._getGymPoseForArm("active")  # (6,) xyzrpy
            while not reached:
                # Generate interpolated gym action towards pre-grasp pose
                # Convert to gym action (delta)
                delta_ee_t = pregrasp_ee_pose[:3] - current_ee_pose[:3]
                delta_ee_r = R.from_euler('xyz', pregrasp_ee_pose[3:6], degrees=True) *  R.from_euler('xyz', current_ee_pose[3:6], degrees=True).inv()
                delta_ee_rpy = delta_ee_r.as_euler('xyz', degrees=True)
                # Scale delta to be within reasonable step size
                max_step_size = 0.05  # max translation step size per axis
                delta_norm = np.linalg.norm(delta_ee_t)
                if delta_norm > max_step_size:
                    delta_ee_t = (delta_ee_t / delta_norm) * max_step_size
                active_arm_action = np.concatenate([delta_ee_t, delta_ee_rpy], axis=0)  # (6,)
                passive_arm_action = np.zeros((6,), dtype=np.float32)  # no movement for passive arm
                gripper_action = np.array([0.0, 0.0], dtype=np.float32)  # keep grippers unchanged
                gym_action = np.concatenate([active_arm_action, passive_arm_action, gripper_action], axis=0)  # (14,)

                # Convert to mjc action
                mjc_actions, _ = self._toMjcAction(gym_action, self.env, self.mjc_obs)

                # # Force no movement for passive arm by directly passing current joint positions
                # if self.active_arm == "left":
                #     right_joint_pos = self.mjc_obs["right_pos"][:7]
                #     mjc_actions = np.concatenate([mjc_actions[:7], right_joint_pos, np.array([0.0, 0.0])], axis=0)
                # else:
                #     left_joint_pos = self.mjc_obs["left_pos"][:7]
                #     mjc_actions = np.concatenate([left_joint_pos, mjc_actions[7:14], np.array([0.0, 0.0])], axis=0)

                # Temp debug
                # right_joint_pos = self.mjc_obs["right_pos"][:7]
                # debug_mjc_actions = np.concatenate([mjc_actions[:7], right_joint_pos, np.array([0.0, 0.0])], axis=0)
                # mjc_actions = debug_mjc_actions

                # Step environment
                self.mjc_obs = self.env.step(mjc_actions)
                if self.render_mode == "display" and self._viewer:
                    self._viewer.sync()
                current_ee_pose = self._getGymPoseForArm("active")  # (6,) xyzrpy
                if np.linalg.norm(current_ee_pose[:3] - pregrasp_ee_pose[:3]) < 0.02:
                    reached = True

            # Update gym_obs
            gym_obs = self._toGymObs(self.mjc_obs)
        
        return gym_obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment using the provided action.
        """
        self.step_count += 1
        # Convert action
        mjc_action, info = self._toMjcAction(action, self.env, self.mjc_obs)
        # Handle IK failure
        if mjc_action is None:
            try:
                # Stay in place
                current_left_pos = self.mjc_obs["left_pos"][:7]
                current_right_pos = self.mjc_obs["right_pos"][:7]
                current_joint_pos = np.concatenate((current_left_pos, current_right_pos))
                fallback_action_joint = current_joint_pos.copy()  # hold position
                fallback_grippers = np.array([0.0, 0.0], dtype=np.float32)
                mjc_action = np.concatenate([fallback_action_joint, fallback_grippers])
                if self.verbose:
                    print("IK Failed: Fallback action (hold position):", mjc_action)
            except Exception:
                # Last-resort fallback: zeros of length 16
                mjc_action = np.zeros(16, dtype=np.float32)
                if self.verbose:
                    print("IK Failed: Fallback action (zeros):", mjc_action)

        # Force no movement for passive arm
        if self.active_arm == "left":
            right_joint_pos = self.mjc_obs["right_pos"][:7]
            mjc_action = np.concatenate([mjc_action[:7], right_joint_pos, np.array([mjc_action[14], 0.0])], axis=0)
        else:
            left_joint_pos = self.mjc_obs["left_pos"][:7]
            mjc_action = np.concatenate([left_joint_pos, mjc_action[7:14], np.array([0.0, mjc_action[15]])], axis=0)

        # Get observation
        self.mjc_obs = self.env.step(mjc_action)
        # Convert to Gym observation
        gym_obs = self._toGymObs(self.mjc_obs)

        # Info for matrics
        is_success = self.env.success()
        if hasattr(self.reward_fn, 'has_grasped'):
            has_grasped = self.reward_fn.has_grasped

        # Done if max steps reached or success
        terminated = bool(self.env.success())
        truncated = self.step_count >= self.max_episode_steps
        # Get reward
        reward, reward_info = self._getReward(self.mjc_obs)
        # Render if needed
        if self.render_mode == "display" and self._viewer:
            self._viewer.sync()
        # Info dictionary
        info = {
            "step_count": self.step_count,
            "is_success": is_success,
            "has_grasped": has_grasped,
            "active_arm": self.active_arm,
            **reward_info,
        }
        return gym_obs, reward, terminated, truncated, info

    def render(self):
        """
        Gym render function.
        """
        if self.render_mode == "rgb_array":
            # Ensure we have an observation to grab the image from
            if hasattr(self, 'mjc_obs') and "front_camera" in self.mjc_obs:
                frame = self.mjc_obs["front_camera"]
                # Convert to uint8 if it isn't already
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                return frame
            else:
                # Return black frame if no obs yet (e.g. before reset)
                return np.zeros((128, 128, 3), dtype=np.uint8)
                
        elif self.render_mode == "display":
            # Just sync the viewer if it exists
            if self._viewer:
                self._viewer.sync()        

    def close(self):
        self._stop_viewer()
        self.env.close()

if __name__ == "__main__":
    # Simple test of the environment with a random policy
    backend_env = OneCubeAssembleEnvironment(seed=42)
    env = PickAndPlaceGymEnv(env=backend_env, render_mode="display", dt=0.02, init_with_pregrasp=True)
    policy = RandomPolicy()
    done = False
    obs, info = env.reset()
    while not done:
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step reward: {reward}, done: {done}, info: {info}")
    env.close()
