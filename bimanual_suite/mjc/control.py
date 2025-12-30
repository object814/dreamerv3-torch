from typing import Any, Dict, Tuple
import enum
import mujoco
import mink
import numpy as np
from scipy import spatial

CTRL_HZ = 20
DT = 1.0 / CTRL_HZ
MAX_VEL_NORM = 2.0

GRASP_HEIGHT = 0.01
DELTA_POS_NORM = 0.2
DELTA_POS_NORM_PREGRASP = 0.02

class Arm(enum.Enum):
    LEFT = "left"
    RIGHT = "right"

def calculate_pointing_pose(target: np.ndarray,
                            end_effector_pos: np.ndarray,
                            up_vector: np.ndarray = np.array([0, 0, 1])
    ) -> mink.SO3:
    "Compute the end effector pose that points to a desired target position."
    # Calculate pointing direction (z-axis)
    z_axis = target - end_effector_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return mink.SO3.from_matrix(R)

def calculate_cube_grasp_pose(current_quat: np.ndarray,
                              target_quat: np.ndarray
    ) -> np.ndarray:
    "Simplify target orientation by taking 90deg rotational symmetry into account."
    current_rot = spatial.transform.Rotation.from_quat(current_quat, scalar_first=True)
    target_rot = spatial.transform.Rotation.from_quat(target_quat, scalar_first=True)
    target_z_in_current = target_rot.apply([0, 0, 1])
    # target = relative x current
    relative_rot = current_rot.inv() * target_rot
    rotvec = relative_rot.as_rotvec()
    axis = rotvec / np.linalg.norm(rotvec)
    angle = np.linalg.norm(rotvec)
    # Component of rotation around target's Z-axis [0, 1]
    z_scale = np.dot(axis, target_z_in_current)
    z_angle = angle * z_scale
    z_angle_adjusted = np.mod(z_angle, np.pi / 2)
    z_delta = z_angle_adjusted - z_angle
    delta_rot = spatial.transform.Rotation.from_rotvec(z_delta * np.array([0, 0, 1]))
    new_target_rot = target_rot * delta_rot
    return new_target_rot.as_quat(scalar_first=True)

def compute_complete(
    target_pose: mink.SE3,
    actual_pose: mink.SE3,
    previous_error: float,
    x_threshold: float = 0.01,
    y_threshold: float = 0.055 / 2,
    z_threshold: float = 0.055,
    quat_threshold: float = 5e-2,
    rate_threshold: float = 1e-3,
) -> Tuple[bool, float, Dict[str, Any]]:
    target_quat, target_pos = target_pose.wxyz_xyz[:4], target_pose.wxyz_xyz[4:]
    actual_quat, actual_pos = actual_pose.wxyz_xyz[:4], actual_pose.wxyz_xyz[4:]
    diff_pos = target_pos - actual_pos
    grasp_rot = target_pose.rotation().as_matrix()
    relative_pos_local = grasp_rot.T @ diff_pos
    x_error, y_error, z_error = np.abs(relative_pos_local)
    x_ok = x_error < x_threshold
    y_ok = y_error < y_threshold
    z_ok = z_error < z_threshold
    pos_error = np.linalg.norm(target_pos - actual_pos)
    quat_error = min(
        np.linalg.norm(target_quat - actual_quat),
        np.linalg.norm(target_quat + actual_quat),
    )
    quat_ok = quat_error < quat_threshold
    error = pos_error + quat_error
    rate = abs(previous_error - error)
    rate_ok = rate < rate_threshold
    info = {
        "x_ok": x_ok,
        "x_error": x_error,
        "y_ok": y_ok,
        "y_error": y_error,
        "z_ok": z_ok,
        "z_error": z_error,
        "quat_ok": quat_ok,
        'quat_error': quat_error,
        "rate_ok": rate_ok,
        'rate': rate,
    }
    return rate_ok and x_ok and y_ok and z_ok and quat_ok, error, info

def project_pose(target_pose: mink.SE3,
                 previous_target_pose: mink.SE3,
                 current_pose: mink.SE3,
                 bound: float, eps: float = 1e-6
    ) -> mink.SE3:
    """Project the current pose to the nearest pose on a linear interpolation between two target poses."""
    target_pos = target_pose.translation()
    delta_target = target_pose.translation() - previous_target_pose.translation()
    delta_pos = target_pose.translation() - current_pose.translation()

    if np.linalg.norm(delta_pos) < bound:
        # within bound, so just set it directly without bounding the update
        new_pos = target_pos + delta_pos
    else:  # we have to do some geometry to get he best setpoint on the linear interpolation
        # Vector from sphere center to line point
        v = previous_target_pose.translation() - current_pose.translation()
        # Coefficients of quadratic equation at² + bt + c = 0
        a = np.dot(delta_target, delta_target) + eps
        b = 2 * np.dot(v, delta_target) + eps
        c = np.dot(v, v) - bound ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            # No intersection, find closet on sphere
            t = np.dot(v, delta_target) / a
            closest_on_line = previous_target_pose.translation() + t * delta_target
            new_pos = closest_on_line
        elif discriminant == 0:
            # One intersection (tangent), so easy
            t = -b / (2 * a)
            intersection = previous_target_pose.translation() + t * delta_target
            new_pos = intersection
        else:
            # Two intersections, so find the closest to our target
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            intersection1 = previous_target_pose.translation() + t1 * delta_target
            intersection2 = previous_target_pose.translation() + t2 * delta_target
            new_pos = intersection1 if np.linalg.norm(intersection1 - target_pos) < np.linalg.norm(intersection2 - target_pos) else intersection2

    new_wyz_xyz = target_pose.wxyz_xyz.copy()
    new_wyz_xyz[4:] = new_pos
    return mink.SE3(new_wyz_xyz)
