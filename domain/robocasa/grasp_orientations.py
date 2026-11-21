"""Five discrete root-frame grasp rotations about an unchanged target TCP."""
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation


GRASP_ORIENTATION_YAW_PITCH_DEG = ((0., 0.), (-15., 0.), (15., 0.), (-45., 30.), (45., 30.))


def validate_grasp_orientation_id(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or int(value) != value
            or not 0 <= int(value) < len(GRASP_ORIENTATION_YAW_PITCH_DEG)):
        raise ValueError("orientation_id must be an integer in [0, 4]")
    return int(value)


def grasp_orientation_quaternion(reset_wxyz, orientation_id):
    """Return Rz(root yaw) @ Ry(root pitch) @ R_reset as a wxyz quaternion."""
    index = validate_grasp_orientation_id(orientation_id)
    reset = np.asarray(reset_wxyz, dtype=float)
    if reset.shape != (4,) or not np.isfinite(reset).all() or np.linalg.norm(reset) <= 0:
        raise ValueError("Grasp orientation requires a finite nonzero reset wxyz quaternion")
    yaw, pitch = GRASP_ORIENTATION_YAW_PITCH_DEG[index]
    rotation = (Rotation.from_euler("z", yaw, degrees=True)
                * Rotation.from_euler("y", pitch, degrees=True)
                * Rotation.from_quat(np.roll(reset, -1)))
    return np.roll(rotation.as_quat(), 1)
