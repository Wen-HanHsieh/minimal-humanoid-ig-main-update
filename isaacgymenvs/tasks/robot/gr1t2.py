from isaacgymenvs.tasks.robot.base_robot import Robot


class GR1T2Function(Robot):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._asset_file_path = "GR1T2/urdf/GR1T2.urdf"


def build(**kwargs):
    return GR1T2Function(**kwargs)
