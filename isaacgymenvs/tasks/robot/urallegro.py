from isaacgymenvs.tasks.robot.base_robot import Robot


class URAllegroFunction(Robot):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._asset_file_path = (
            "urdf/ur5e_allegro/robots/dual_ur5e_allegro_real_v2.urdf"
        )
        self._single_arm_dof = 6
        self._single_hand_dof = 16

        self._left_arm_dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._right_arm_dof_names = [dof + "_r" for dof in self._left_arm_dof_names]

        self._left_arm_link_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        self._right_arm_link_names = [link + "_r" for link in self._left_arm_link_names]

        self._left_hand_dof_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
        ]
        self._right_hand_dof_names = [dof + "_r" for dof in self._left_hand_dof_names]
        self._left_hand_link_names = ["link_" + str(i) + ".0" for i in range(16)]
        self._right_hand_link_names = ["link_" + str(i) + ".0_r" for i in range(16)]

        self._left_hand_nonthumb_tip_names = [
            "link_7.0_tip",
            "link_3.0_tip",
            "link_11.0_tip",
        ]
        self._right_hand_nonthumb_tip_names = [
            "link_7.0_tip_r",
            "link_3.0_tip_r",
            "link_11.0_tip_r",
        ]
        self._left_hand_thumb_tip_names = ["link_15.0_tip"]
        self._right_hand_thumb_tip_names = ["link_15.0_tip_r"]

        self._wrist_link_names = ["wrist_base", "wrist_base_r"]

    def get_arm_init_pos(self):
        return [
            # 830
            -1.567662541066305,
            -2.4176141224303187,
            -1.470444917678833,
            -0.8341446679881592,
            0.894737720489502,
            0.08133087307214737,
            # 828
            -4.674656931553976,
            -0.6805991691401978,
            1.5093582312213343,
            -2.377801080743307,
            -0.8824575583087366,
            -0.06327754655946904,
        ]


def build(**kwargs):
    return URAllegroFunction(**kwargs)
