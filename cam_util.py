"""
Utility for computing the start camera pose with respect to the target pose.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np


class Transform:
    """
    A minimal Transform class for required transform math.
    """

    def __init__(self):
        self.rotation = R.identity()
        self.position = np.array([0, 0, 0])

    @classmethod
    def from_matrix(cls, m):
        """
        Construct a Transform from a 4x4 matrix.
        """
        T = cls()
        T.rotation = R.from_matrix([m[0:3], m[4:7], m[8:11]])
        T.position = np.array([m[3], m[7], m[11]])
        return T

    def __mul__(self, other):
        T = Transform()
        T.rotation = self.rotation * other.rotation
        T.position = self.rotation.apply(other.position) + self.position
        return T


def shift_world_T_camera(world_T_camera, rot_shift, direction):
    """
    From the reference world_T_camera pose, shift in the down, up, right
    # or left direction by rot_shift radians, to create a starting camera pose.
    """
    x = world_T_camera.rotation.apply(np.array([1, 0, 0]))

    # compute transform op to rotate world_T_camera down/up/right or left
    # by rot_shift radians
    op = Transform()
    if direction == 0:
        # rotate down
        op.rotation = R.from_rotvec(rot_shift * x)
    elif direction == 1:
        # rotate up
        op.rotation = R.from_rotvec(-rot_shift * x)
    elif direction == 2:
        # rotate right
        op.rotation = R.from_euler("z", rot_shift)
    else:
        # rotate left
        op.rotation = R.from_euler("z", -rot_shift)

    world_T_shifted_camera = op * world_T_camera
    return (
        world_T_shifted_camera.rotation.as_euler("zyx"),
        world_T_shifted_camera.position,
    )
