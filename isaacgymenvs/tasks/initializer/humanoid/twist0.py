"""Most stats follow Jan8 initializer"""

from isaacgym import gymutil, gymtorch, gymapi
from scipy.spatial.transform import Rotation as R
from isaacgymenvs.tasks.initializer.humanoid.base import HumanoidEnvInitializer, Pose

import numpy as np


class Twist0HumanoidEnvInitializer(HumanoidEnvInitializer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = kwargs.get("cfg")

        self.dof_velocity = 1.0

        self.table_init_pos = [0.7, 0.0, 0.9]

        self.object_target_pos = [0.7, -0.03, 1.24]
        self.object_cap_target_pos = [0.0, 0.0, 0.0]  # TODO: update

        # Pose [px, py, pz, qx, qy, qz, qw]
        self.robot_init_pose = Pose([0.0, 0.0, 1.0], [0, 0, 0, 1])
        self.object_init_pose = Pose([0.40, 0, 1.20], [0, 0, 0, 1])

    def initialize_object_dof(self, object_dof_props):
        # print(object_dof_props)
        if len(object_dof_props["driveMode"]) == 2:
            # print("ENTER!")
            object_dof_props["driveMode"][0] = gymapi.DOF_MODE_EFFORT
            object_dof_props["velocity"][0] = 3.0
            object_dof_props["effort"][0] = 1.0
            object_dof_props["stiffness"][0] = 0.0
            object_dof_props["damping"][0] = 0.0
            object_dof_props["friction"][0] = 0.0
            object_dof_props["armature"][0] = 0.0001

            object_dof_props["driveMode"][1] = gymapi.DOF_MODE_NONE
            object_dof_props["velocity"][1] = 3.0
            object_dof_props["effort"][1] = 1.0
            object_dof_props["stiffness"][1] = 0.00
            object_dof_props["damping"][1] = 0.1
            object_dof_props["friction"][1] = self.cfg["obj_dof_friction"]
            object_dof_props["armature"][1] = 0.0001
        else:
            for i in range(1):
                # print("INIT!", self.cfg["obj_dof_friction"])
                object_dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
                object_dof_props["velocity"][i] = 3.0
                object_dof_props["effort"][i] = 10.0
                object_dof_props["stiffness"][i] = 0.01
                object_dof_props["damping"][i] = 20.0
                object_dof_props["friction"][i] = self.cfg["obj_dof_friction"]
                object_dof_props["armature"][i] = 0.0001


def build(**kwargs):
    return Twist0HumanoidEnvInitializer(**kwargs)
