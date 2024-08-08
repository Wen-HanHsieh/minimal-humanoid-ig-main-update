class Robot:
    def __init__(self, **kwargs) -> None:
        self._asset_file_path = ""
        self._num_wrist_joints = kwargs.get("num_wrist_joints", 2)
        self._fix_torso = kwargs.get("fix_torso", True)  # TODO: add this config

        self._torso_dof = 0
        self._single_arm_dof = 0
        self._single_hand_dof = 0

        self._torso_dof_names = []
        self._left_hand_dof_names = []
        self._right_hand_dof_names = []
        self._left_arm_dof_names = []
        self._right_arm_dof_names = []
        self._left_hand_link_names = []
        self._right_hand_link_names = []
        self._left_arm_link_names = []
        self._right_arm_link_names = []

        self._left_hand_nonthumb_tip_names = []
        self._right_hand_nonthumb_tip_names = []
        self._left_hand_thumb_tip_names = []
        self._right_hand_thumb_tip_names = []

        self._wrist_link_names = []

    @property
    def asset_file_path(self):
        return self._asset_file_path

    @property
    def num_wrist_joints(self):
        return self._num_wrist_joints

    @property
    def total_torso_dof(self):
        return self._torso_dof

    @property
    def total_hand_dof(self):
        return self._single_hand_dof * self._num_wrist_joints

    @property
    def total_arm_dof(self):
        return self._single_arm_dof * self._num_wrist_joints

    @property
    def torso_dof_names(self):
        return self._torso_dof_names

    @property
    def hand_dof_names(self):
        return self._left_hand_dof_names + self._right_hand_dof_names

    @property
    def arm_dof_names(self):
        return self._left_arm_dof_names + self._right_arm_dof_names
    
    @property
    def hand_link_names(self):
        return self._left_hand_link_names + self._right_hand_link_names
    
    @property
    def arm_link_names(self):
        return self._left_arm_link_names + self._right_arm_link_names
    
    @property
    def left_hand_tip_names(self):
        return self._left_hand_nonthumb_tip_names + self._left_hand_thumb_tip_names
    
    @property
    def right_hand_tip_names(self):
        return self._right_hand_nonthumb_tip_names + self._right_hand_thumb_tip_names
    
    @property
    def wrist_link_names(self):
        return self._wrist_link_names

    @property
    def act_dim(self):
        return self.total_arm_dof + self.total_hand_dof + self.total_torso_dof

    def get_torso_init_pos(self):
        return [0.0] * self.total_torso_dof

    def get_arm_init_pos(self):
        return [0.0] * self.total_arm_dof

    def get_hand_init_pos(self):
        return [0.0] * self.total_hand_dof


def build(**kwargs):
    return Robot(**kwargs)
