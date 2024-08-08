from isaacgymenvs.tasks.robot.base_robot import Robot


class H1Function(Robot):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._asset_file_path = "h1/h1_5.urdf"
        self._torso_dof = 13
        self._single_arm_dof = 7
        self._single_hand_dof = 12

        self._torso_dof_names = [
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "torso_joint",
        ]
        assert len(self._torso_dof_names) == self._torso_dof

        self._left_hand_dof_names = [
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
        ]
        self._right_hand_dof_names = [
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
        ]
        assert len(self._left_hand_dof_names) == self._single_hand_dof
        assert len(self._right_hand_dof_names) == self._single_hand_dof

        self._left_arm_dof_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        self._right_arm_dof_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        assert len(self._left_arm_dof_names) == self._single_arm_dof
        assert len(self._right_arm_dof_names) == self._single_arm_dof

        self._left_hand_link_names = [
            "L_index_proximal",
            "L_index_intermediate",
            "L_middle_proximal",
            "L_middle_intermediate",
            "L_pinky_proximal",
            "L_pinky_intermediate",
            "L_ring_proximal",
            "L_ring_intermediate",
            "L_thumb_proximal_base",
            "L_thumb_proximal",
            "L_thumb_intermediate",
            "L_thumb_distal",
        ]
        self._right_hand_link_names = [
            "R_index_proximal",
            "R_index_intermediate",
            "R_middle_proximal",
            "R_middle_intermediate",
            "R_pinky_proximal",
            "R_pinky_intermediate",
            "R_ring_proximal",
            "R_ring_intermediate",
            "R_thumb_proximal_base",
            "R_thumb_proximal",
            "R_thumb_intermediate",
            "R_thumb_distal",
        ]

        self._left_arm_link_names = [
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_pitch_link",
            "left_elbow_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
        ]
        self._right_arm_link_names = [
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_pitch_link",
            "right_elbow_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]

        self._wrist_link_names = [
            "L_hand_base_link",
            "R_hand_base_link",
        ]


def build(**kwargs):
    return H1Function(**kwargs)
