from isaacgym import gymutil, gymtorch, gymapi
from scipy.spatial.transform import Rotation as R


class Pose:
    def __init__(self, pos, quat):
        # pos:  LIST [FLOAT * 3] xyz  format
        # quat: LIST [FLOAT * 4] xyzw format
        self.pos = pos
        self.quat = quat

    def to_isaacgym_pose(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.pos)
        pose.r = gymapi.Quat(*self.quat)
        return pose

    def to_list(self):
        return self.pos + self.quat

    def post_multiply_quat(self, quat):
        """
        self.quat = quat * self.quat
        Fortunately, scipy also uses xyzw format.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
        """
        r = R.from_quat(self.quat)
        q = R.from_quat(quat)
        r = q * r
        self.quat = list(r.as_quat().reshape(-1))
        return self

    def post_multiply_euler(self, mode="zyx", angle=[0, 0, 0]):
        """
        Transform the pose with an Eulerian
        """
        r = R.from_quat(self.quat)
        q = R.from_euler(mode, angle, degrees=True)
        r = q * r
        self.quat = list(r.as_quat().reshape(-1))
        return self


class HumanoidEnvInitializer:
    def __init__(self, **kwargs):
        self.dof_velocity = 3.14
        # Pose [px, py, pz, qx, qy, qz, qw]
        self.robot_init_pose = Pose([0.0, 0.0, 1.0], [0, 0, 0, 1])
        self.object_init_pose = Pose([0.01, 0.0, 0.51], [0, -0.7071068, 0, 0.7071068])

    def initialize_object_dof(self, object_dof_props):
        for i in range(1):
            object_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            object_dof_props["velocity"][i] = 3.0
            object_dof_props["effort"][i] = 0.0
            object_dof_props["stiffness"][i] = 0.01
            object_dof_props["damping"][i] = 0.0
            object_dof_props["friction"][i] = 2000.0
            object_dof_props["armature"][i] = 0.0

    def get_dof_velocity(self):
        return self.dof_velocity

    def get_robot_init_pose(self):
        return self.robot_init_pose

    def get_object_init_pose(self):
        return self.object_init_pose

    def get_hand_info(self):
        info = {"stiffness": 0.0, "damping": 0.0, "armature": 0.001}
        return info

    def get_arm_info(self):
        info = {"stiffness": 1000.0, "damping": 200.0, "armature": 0.001}
        return info


def build(**kwargs):
    return HumanoidEnvInitializer(**kwargs)
