import cv2
import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
import pathlib
import math
import scipy.spatial as spatial 

CTRL_FREQ = 20

GRIPPER_RANGE: float = 0.8
LIFT_HEIGHT: float = 0.4
PTU_TILT: float = -1.0
PTU_PAN: float = 0.0

CALIBRATION_POSE = {
    "ewellix_lift_top_joint": LIFT_HEIGHT,
    "ptu_pan": PTU_PAN,
    "ptu_tilt": PTU_TILT,
    "left_kinova_arm_joint_1": -1.60,
    "left_kinova_arm_joint_2": 0.55,
    "left_kinova_arm_joint_3": -1.63,
    "left_kinova_arm_joint_4": -1.63,
    "left_kinova_arm_joint_5": -0.50,
    "left_kinova_arm_joint_6": -1.57,
    "left_kinova_arm_joint_7": -0.08,
    "right_kinova_arm_joint_1": -1.60,
    "right_kinova_arm_joint_2": -0.55,
    "right_kinova_arm_joint_3": 1.63,
    "right_kinova_arm_joint_4": 1.63,
    "right_kinova_arm_joint_5": 0.50,
    "right_kinova_arm_joint_6": 1.57,
    "right_kinova_arm_joint_7": 0.08,
}
CALIBRATION_ACTIONS = np.zeros((16,))
CALIBRATION_ACTIONS[:14] = np.asarray(list(CALIBRATION_POSE.values()))[3:]

HOME_POSE = {
    "ewellix_lift_top_joint": LIFT_HEIGHT,
    "ptu_pan": PTU_PAN,
    "ptu_tilt": PTU_TILT,
    "left_kinova_arm_joint_1": -2.23,
    "left_kinova_arm_joint_2": 0.392,
    "left_kinova_arm_joint_3": -1.43,
    "left_kinova_arm_joint_4": -2.32,
    "left_kinova_arm_joint_5": 0.352,
    "left_kinova_arm_joint_6": -0.258,
    "left_kinova_arm_joint_7": -1.02,
    "right_kinova_arm_joint_1": -0.87,
    "right_kinova_arm_joint_2": -0.48,
    "right_kinova_arm_joint_3": -1.66,
    "right_kinova_arm_joint_4": -2.32,
    "right_kinova_arm_joint_5": -0.110,
    "right_kinova_arm_joint_6": -0.249,
    "right_kinova_arm_joint_7": -2.28,
}


def crop_wrist(image: np.ndarray):
    h, w, c = image.shape
    g = w // 4
    f = h // 2
    image = image[-f:, g:-g, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=[w, h], interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class CubeBaseEnvironment:

    default_actions = CALIBRATION_ACTIONS

    def __init__(self, seed: int, render_height: int = 144, render_width: int = 256):
        self.model = mujoco.MjModel.from_xml_path(
            str(pathlib.Path(__file__).parents[0] / self.xml_file)
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.dt = 1 / CTRL_FREQ
        self.substeps = int(1 / (CTRL_FREQ * self.model.opt.timestep))

        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        self.data.ctrl[self.model.actuator("lift").id] = LIFT_HEIGHT
        self.data.ctrl[self.model.actuator("ptu_tilt").id] = PTU_TILT
        self.data.ctrl[self.model.actuator("ptu_pan").id] = PTU_PAN

        left_joint_names = [f"left_kinova_arm_joint_{i}" for i in range(1, 8)]
        right_joint_names = [f"right_kinova_arm_joint_{i}" for i in range(1, 8)]
        self.dof_ids = np.array([self.model.joint(name).dofadr[0] for name in left_joint_names + right_joint_names])
        self.joint_action_map = np.array(
            [
                self.model.actuator(name).id
                for name in (left_joint_names + right_joint_names)
            ]
        ).astype(np.int32)

        for name, value in CALIBRATION_POSE.items():
            idx = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            self.data.qpos[idx] = value
            idx = self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            self.data.qvel[idx] = 0.0

        self.left_joint_state_to_pos = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in left_joint_names
            ]
        )
        self.right_joint_state_to_pos = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in right_joint_names
            ]
        )
        self.left_joint_state_to_vel = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in left_joint_names
            ]
        )
        self.right_joint_state_to_vel = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in right_joint_names
            ]
        )
        self.left_gripper_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_kinova_arm_finger_joint"
            )
        ]
        self.right_gripper_idx = self.model.jnt_qposadr[
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_kinova_arm_finger_joint"
            )
        ]

        self.left_kinova_arm_finger_1_id = self.model.actuator(
            "left_kinova_arm_finger_1"
        ).id
        self.left_kinova_arm_finger_2_id = self.model.actuator(
            "left_kinova_arm_finger_2"
        ).id
        self.right_kinova_arm_finger_1_id = self.model.actuator(
            "right_kinova_arm_finger_1"
        ).id
        self.right_kinova_arm_finger_2_id = self.model.actuator(
            "right_kinova_arm_finger_2"
        ).id

        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "black_cube_geom")
        half_width = self.model.geom_size[cube_geom_id][0]  # 0.0275
        self.cube_width = half_width * 2  # 0.055

    def name_to_qpos_adr(self, name: str):
        part_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = self.model.body_jntadr[part_id]
        return self.model.jnt_qposadr[joint_id]

    def cube_init(self, seed: int):
        raise NotImplementedError

    def success(self):
        raise NotImplementedError

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.calibration_qpos[:]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            self.step(CALIBRATION_ACTIONS)
        return self.observation

    def step(self, actions: np.ndarray):
        self.data.ctrl[self.model.actuator("lift").id] = LIFT_HEIGHT
        self.data.ctrl[self.model.actuator("ptu_tilt").id] = PTU_TILT
        self.data.ctrl[self.model.actuator("ptu_pan").id] = PTU_PAN

        self.data.ctrl[self.joint_action_map] = actions[:14]

        self.data.qfrc_applied[self.dof_ids] = self.data.qfrc_bias[self.dof_ids]

        # actions are [0, 1]
        left_gripper, right_gripper = GRIPPER_RANGE * actions[14:16]
        self.data.ctrl[self.left_kinova_arm_finger_1_id] = left_gripper
        self.data.ctrl[self.left_kinova_arm_finger_2_id] = left_gripper
        self.data.ctrl[self.right_kinova_arm_finger_1_id] = right_gripper
        self.data.ctrl[self.right_kinova_arm_finger_2_id] = right_gripper

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        return self.observation

    def get_site_pose(self, id: int):
        pos = self.data.site_xpos[id]
        # Orientation as rotation matrix (3x3)
        mat = self.data.site_xmat[id].reshape(3, 3)
        # Orientation as quaternion (4D vector: w, x, y, z)
        quat = spatial.transform.Rotation.from_matrix(mat).as_quat(scalar_first=True)
        return pos, quat
    
    def get_gripper_state(self, side: str):
        """
        Get the gripper current state given the side.
        
        Args:
            side (str): "left" or "right" to specify which gripper.
        """
        if side == "left":
            gripper_pos = np.clip(
                self.data.qpos[self.left_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
            ) * np.ones((1,))
        elif side == "right":
            gripper_pos = np.clip(
                self.data.qpos[self.right_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
            ) * np.ones((1,))
        else:
            raise ValueError("side must be 'left' or 'right'")
        return gripper_pos

    def get_joint_forces(self, side: str):
        """
        Get the joint forces for the specified side.

        Args:
            side (str): "left" or "right" to specify which arm.
        """
        if side == "left":
            return self.data.qfrc_applied[self.dof_ids][0:7].copy()
        elif side == "right":
            return self.data.qfrc_applied[self.dof_ids][7:14].copy()
        else:
            raise ValueError("side must be 'left' or 'right'")

    def get_cube_poses(self):
        raise NotImplementedError

    @property
    def observation(self):
        self.renderer.update_scene(self.data, camera="overhead_cam")
        overhead_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="left_wrist_cam")
        left_wrist_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="right_wrist_cam")
        right_wrist_cam = self.renderer.render()
        self.renderer.update_scene(self.data, camera="back")
        back = self.renderer.render()
        self.renderer.update_scene(self.data, camera="front")
        front = self.renderer.render()
        left_wrist_crop = crop_wrist(left_wrist_cam)
        right_wrist_crop = crop_wrist(right_wrist_cam)
        # Create user friendly camera view containing all views
        # Combination of back, front, and overhead cameras on top row
        user_camera = np.concatenate((back, front, overhead_cam), axis=1)
        # Combination of left and right wrist cameras on bottom row
        wrist_camera = np.concatenate((left_wrist_crop, right_wrist_crop), axis=1)
        # Resolve dimension mismatch by padding
        target_width = user_camera.shape[1] 
        current_width = wrist_camera.shape[1]
        if current_width < target_width:
            padding_width = target_width - current_width
            padding = np.zeros((wrist_camera.shape[0], padding_width, wrist_camera.shape[2]), dtype=wrist_camera.dtype)
            wrist_camera = np.concatenate((wrist_camera, padding), axis=1)
        # Final user camera view
        user_camera = np.concatenate((user_camera, wrist_camera), axis=0)

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        left_gripper_pos = np.clip(
            qpos[self.left_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
        ) * np.ones((1,))
        right_gripper_pos = np.clip(
            qpos[self.right_gripper_idx] / GRIPPER_RANGE, a_min=0.0, a_max=1.0
        ) * np.ones((1,))

        left_ee_pos, left_ee_quat = self.get_site_pose(self.model.site("left_ee").id)
        right_ee_pos, right_ee_quat = self.get_site_pose(self.model.site("right_ee").id)
        overhead_cam_pos, overhead_cam_quat = self.get_site_pose(self.model.site("overhead_cam").id)
        left_cam_pos, left_cam_quat = self.get_site_pose(self.model.site("left_wrist_cam").id)
        right_cam_pos, right_cam_quat = self.get_site_pose(self.model.site("right_wrist_cam").id)

        cube_pose = self.get_cube_poses()
        observations = {
            "left_pos": np.concatenate(
                (qpos[self.left_joint_state_to_pos], left_gripper_pos)
            ),
            "right_pos": np.concatenate(
                (qpos[self.right_joint_state_to_pos], right_gripper_pos)
            ),
            "left_vel": qvel[self.left_joint_state_to_vel],
            "right_vel": qvel[self.right_joint_state_to_pos],
            "left_ee_pos": left_ee_pos.copy(),
            "left_ee_quat": left_ee_quat.copy(),
            "right_ee_pos": right_ee_pos.copy(),
            "right_ee_quat": right_ee_quat.copy(),
            "overhead_cam_pos": overhead_cam_pos.copy(),
            "overhead_cam_quat": overhead_cam_quat.copy(),
            "left_cam_pos": left_cam_pos.copy(),
            "left_cam_quat": left_cam_quat.copy(),
            "right_cam_pos": right_cam_pos.copy(),
            "right_cam_quat": right_cam_quat.copy(),
            "overhead_camera": overhead_cam,
            "left_camera": left_wrist_cam,
            "right_camera": right_wrist_cam,
            "front_camera": front,
            "back_camera": back,
            "user_camera": user_camera,
            "left_camera_crop": left_wrist_crop,
            "right_camera_crop": right_wrist_crop,
        }
        return {**observations, **cube_pose}

    def close(self):
        self.renderer.close()

class ThreeCubeBaseEnvironment(CubeBaseEnvironment):

    xml_file = "three_cubes.xml"
    black_qpos = None
    blue_qpos = None
    orange_qpos = None

    def get_cube_poses(self):
        cubes = {
            "black": self.black_qpos,
            "blue": self.blue_qpos,
            "orange": self.orange_qpos,
        }
        cube_pos = {
            f"{k}_pos": self.data.qpos[cube_idx: cube_idx + 3].copy()
            for k, cube_idx in cubes.items()
        }
        cube_quat = {
            f"{k}_quat":  self.data.qpos[cube_idx + 3: cube_idx + 7].copy()
            for k, cube_idx in cubes.items()
        }
        return {**cube_pos, **cube_quat}


class CubeAssembleEnvironment(ThreeCubeBaseEnvironment):
    def __init__(self, seed, render_height = 144, render_width = 256):
        super().__init__(seed, render_height, render_width)
        self.cube_init(seed)
        self.calibration_qpos = np.copy(self.data.qpos)

    def cube_init(self, seed: int):
        self.black_qpos = self.name_to_qpos_adr("black_cube")
        self.blue_qpos = self.name_to_qpos_adr("blue_cube")
        self.orange_qpos = self.name_to_qpos_adr("orange_cube")
        cubes = [self.black_qpos, self.blue_qpos, self.orange_qpos]
        center = np.array([0.35, 0.0])
        cube_pos = np.zeros((len(cubes), 2))
        for i, cube_idx in enumerate(cubes):
            np.random.seed(seed + i)
            scale = np.random.uniform()
            angle = 2 * np.pi * np.random.uniform()
            ori = 2 * np.pi * np.random.uniform()
            radius = scale * np.array([0.2 * np.sin(angle), 0.3 * np.cos(angle)])
            quat = np.array([np.cos(ori / 2), 0.0, 0.0, np.sin(ori / 2)])
            cube_pos[i, :] = center + radius
            self.data.qpos[cube_idx + 3 : cube_idx + 7] = quat

        n_resolve = 10
        radius = math.sqrt(2 * 0.055 ** 2)
        for _ in range(n_resolve):
            diff = cube_pos[None, :, :] - cube_pos[:, None, :]
            l2 = np.sqrt((diff ** 2).sum(axis=2))
            inside = 1.0 * (l2 < radius)
            vec = diff / (l2[..., None] + 1e-6)
            delta = 0.5 * (inside[:, :, None] * (radius - l2[:, :, None]) * vec).sum(axis=0)
            cube_pos = cube_pos + delta

        for i, cube_idx in enumerate(cubes):
            self.data.qpos[cube_idx : cube_idx + 2] = cube_pos[i, :]

    def success(self, xy_tol = 0.055 / 2, z_tol = 0.01):
        cube_pose = self.get_cube_poses()
        blue_xy = np.linalg.norm(cube_pose["blue_pos"][:2] - cube_pose["black_pos"][:2]) < xy_tol
        orange_xy = np.linalg.norm(cube_pose["orange_pos"][:2] - cube_pose["black_pos"][:2]) < xy_tol
        blue_z = np.abs(cube_pose["blue_pos"][2] - cube_pose["black_pos"][2] - 0.055) < z_tol
        orange_z = np.abs(cube_pose["orange_pos"][2] - cube_pose["black_pos"][2] - 2 * 0.055) < z_tol
        return blue_xy and blue_z and orange_xy and orange_z


class CubeDissassembleEnvironment(ThreeCubeBaseEnvironment):
    def __init__(self, seed, render_height = 144, render_width = 256):
        super().__init__(seed, render_height, render_width)
        self.cube_init(seed)
        self.calibration_qpos = np.copy(self.data.qpos)

    def cube_init(self, seed: int):
        self.black_qpos = self.name_to_qpos_adr("black_cube")
        self.blue_qpos = self.name_to_qpos_adr("blue_cube")
        self.orange_qpos = self.name_to_qpos_adr("orange_cube")
        mujoco.mj_forward(self.model, self.data)
        left_ee_pos = self.data.site_xpos[self.model.site("left_ee").id]
        right_ee_pos = self.data.site_xpos[self.model.site("right_ee").id]
        center_y = 0.5 * left_ee_pos[1] + 0.5 * right_ee_pos[1]
        center_x = 0.5 * left_ee_pos[0] + 0.5 * right_ee_pos[0]
        range_y = abs(left_ee_pos[1] - right_ee_pos[1])
        np.random.seed(seed)
        rand_pos = np.random.uniform() - 0.5  # [-0.5, 0.5]
        rand_ori = 2. * np.pi * np.random.uniform()
        cubes = [self.black_qpos, self.blue_qpos, self.orange_qpos]
        for i, cube_idx in enumerate(cubes):
            self.data.qpos[cube_idx : cube_idx + 3] = np.array(
                [center_x, center_y + 0.5 * range_y * rand_pos, 0.79 + 0.055 * i]
            )
            self.data.qpos[cube_idx + 3 : cube_idx + 7] = np.array(
                [np.cos(rand_ori / 2), 0.0, 0.0, np.sin(rand_ori / 2)]
            )

class OneCubeBaseEnvironment(CubeBaseEnvironment):
    xml_file = "pick_and_place.xml"
    orange_qpos = None
    target_qpos = None

    def get_cube_poses(self):
        cubes = {
            "orange": self.orange_qpos,
            "target": self.target_qpos
        }
        cube_pos = {
            f"{k}_pos": self.data.qpos[cube_idx: cube_idx + 3].copy()
            for k, cube_idx in cubes.items()
        }
        cube_quat = {
            f"{k}_quat": self.data.qpos[cube_idx + 3: cube_idx + 7].copy()
            for k, cube_idx in cubes.items()
        }
        
        return {**cube_pos, **cube_quat}
    
class OneCubeAssembleEnvironment(OneCubeBaseEnvironment):
    def __init__(self, seed, render_height = 144, render_width = 256):
        super().__init__(seed, render_height, render_width)
        self.np_random = np.random.default_rng(seed)
        self.cube_init()
        self.calibration_qpos = np.copy(self.data.qpos)

    def reset(self):
        super().reset()
        self.cube_init()
        return self.observation

    def cube_init(self):
        self.orange_qpos = self.name_to_qpos_adr("orange_cube")
        self.target_qpos = self.name_to_qpos_adr("orange_cube_target")
    
        cube_pos = np.array([0.2, 0.0])
        target_pos = np.array([0.4, 0.0, 0.79])
        rand1 = self.np_random.uniform(-0.15, 0.15)
        # Ensure target and current position are on same side
        if rand1 > 0:
            rand2 = self.np_random.uniform(0.0, 0.15)
        else:
            rand2 = self.np_random.uniform(-0.15, 0.0)
        cube_pos[1] = rand1
        target_pos[1] = rand2

        # Set orange block location to a randomised location according to seed
        self.data.qpos[self.orange_qpos : self.orange_qpos + 2] = cube_pos

        # Set target block location
        self.data.qpos[self.target_qpos : self.target_qpos + 3] = target_pos


    def success(self, xy_tol = 0.01, z_tol = 0.01):
        cube_pose = self.get_cube_poses()
        orange_xy = np.linalg.norm(cube_pose["orange_pos"][:2] - cube_pose["target_pos"][:2]) < xy_tol
        orange_z = np.abs(cube_pose["orange_pos"][2] - cube_pose["target_pos"][2]) < z_tol
        return orange_xy and orange_z

if __name__ == "__main__":
    import time
    import imageio
    import matplotlib.pyplot as plt

    n_steps = 100
    fig, axs = plt.subplots(7)
    actions_ = np.zeros((n_steps, 7))
    states_ = np.zeros((n_steps, 7))

    env = OneCubeAssembleEnvironment(0, render_height=144*4, render_width=256*4)
    actions = np.zeros((16,))
    actions[:14] = np.asarray(list(CALIBRATION_POSE.values()))[3:]
    env.reset()
    images = []
    time_ = time.time()
    if True:
        for _ in range(20):
            obs = env.step(actions)
            images.append(obs["user_camera"])
        actions[14] = 1.0
        actions[15] = 1.0
        for _ in range(20):
            obs = env.step(actions)
            images.append(obs["user_camera"])
        for _ in range(100):
            actions[1] = actions[1] * 0.9 + -0.4 * 0.1
            obs = env.step(actions)
            images.append(obs["user_camera"])
        for _ in range(40):
            actions[4] = actions[4] * 0.9
            obs = env.step(actions)
            images.append(obs["user_camera"])
        for _ in range(40):
            actions[5] = actions[5] * 0.9
            obs = env.step(actions)
            images.append(obs["user_camera"])

    print(len(images) / (time.time() - time_))
    writer = imageio.get_writer(pathlib.Path(__file__).parent / "frank.mp4", fps=20)

    # for i, ax in enumerate(axs):
    #     ax.plot(states_[:, i], 'b')
    #     ax.plot(actions_[:, i], 'k')

    for img in images:
        writer.append_data(img.astype(np.uint8))
    writer.close()
    # plt.show()
