# Object manipulation task

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask
import torch
import torch.nn.functional as F

from .utils.asset_manager import AssetManager
from .utils.module_builder import build_module


class Humanoid(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        # A general robot interface
        self.robot_module = build_module(
            f'isaacgymenvs.tasks.robot.{self.cfg["env"]["robot_type"]}', {}
        )

        # Hand specs
        self.action_moving_average = self.cfg["env"]["controller"]["actionEMA"]
        self.controller_action_scale = self.cfg["env"]["controller"][
            "controllerActionScale"
        ]
        self.p_gain_val = self.cfg["env"]["controller"]["kp"]
        self.d_gain_val = self.cfg["env"]["controller"]["kd"]

        # Configs to pass to reward function
        self.reward_settings = self.cfg["env"]["reward_setup"]
        print("REWARD SETUP", self.reward_settings)

        # Count the number of assets
        self.num_objects = len(
            AssetManager(
                asset_root=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../../assets"
                ),
                model_root_path=cfg["env"]["object_asset_root_folder"],
            )
        )
        print("Total Training Objects", self.num_objects)

        # Camera specs
        self.cam_w = self.cfg["cam"]["cam_w"]
        self.cam_h = self.cfg["cam"]["cam_h"]
        self.im_size = self.cfg["cam"]["im_size"]

        # Setup a fake reward function
        reward_build_cfg = {
            "device": "cpu",
            "num_envs": 4,
            "reward_settings": self.reward_settings,
        }
        reward_function = build_module(
            f'isaacgymenvs.tasks.rewarder.humanoid.{self.cfg["env"]["rewarder"]}',
            reward_build_cfg,
        )

        self.dt = 1 / 60.0  # self.cfg["sim"]["dt"]

        self.n_stack_frame = self.cfg["env"]["n_stack_frame"]

        # TODO: remove hardcoding
        self.single_frame_obs_dim = (
            self.robot_module.act_dim * 3 + reward_function.obs_dim() + self.num_objects
        )
        self.cfg["env"]["numObservations"] = int(
            self.single_frame_obs_dim * self.n_stack_frame
        )
        self.full_state = self.cfg["env"]["fullState"]
        if self.cfg["env"]["computeState"]:
            if self.cfg["env"]["fullState"]:
                self.cfg["env"]["numStates"] = 500
            else:
                if self.num_objects > 20:
                    self.cfg["env"]["numStates"] = 280  # TODO(): Compute this.
                else:
                    self.cfg["env"]["numStates"] = 200
        else:
            self.cfg["env"]["numStates"] = 0

        if reward_function.obs_dim() > 0:
            self.use_reward_obs = True
        else:
            self.use_reward_obs = False

        self.cfg["env"]["numActions"] = self.robot_module.act_dim
        self.total_hand_dof = self.robot_module.total_hand_dof
        self.total_arm_dof = self.robot_module.total_arm_dof

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._init_object_state = None  # Initial state of object for the current env
        self.object_id = None  # Actor ID corresponding to object for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, n_actors, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof, 2)
        self._rigid_body_state = (
            None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        )
        self.franka_dof_state = None  # State of franka joints (n_envs, n_frank_dof, 2)
        self.franka_dof_pos = None  # Joint positions           (n_envs, n_frank_dof)
        self.franka_dof_vel = None  # Joint velocities          (n_envs, n_frank_dof)
        self._contact_forces = None  # Contact forces in sim
        self.eef_state = None  # end effector state (at left arm grasping point)
        self.eef_lf_state = None  # end effector state (at left arm left fingertip)
        self.eef_rf_state = None  # end effector state (at left arm right fingertip)
        self.eef_r_state = None  # end effector state (at right arm grasping point)
        self.eef_lf_r_state = None  # end effector state (at right arm left fingertip)
        self.eef_rf_r_state = None  # end effector state (at right arm right fingertip)
        self.lf_force = None  # left finger force (left arm)
        self.rf_force = None  # right finger force (left arm)
        self.lf_r_force = None  # left finger force (right arm)
        self.rf_r_force = None  # right finger force (right arm)
        self.object_state = None  # object state (n_envs, 13)

        self.franka_dof_targets = None  # Position targets (n_envs, n_frank_dof)
        self._global_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Import the initializer
        self.initializer = build_module(
            f'isaacgymenvs.tasks.initializer.humanoid.{self.cfg["env"]["initializer"]}',
            {"cfg": self.cfg["env"]["init_setup"]},
        )

        # Import the randomizer
        self.randomizer = build_module(
            f'isaacgymenvs.tasks.randomizer.humanoid.{self.cfg["env"]["randomizer"]}',
            {"cfg": self.cfg["env"]["randomization_setup"]},
        )

        # Important for Sim2Real
        self.disable_gravity = self.cfg["env"].get("disable_gravity", True)

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Setup the actual reward function
        self.reward_function = build_module(
            f'isaacgymenvs.tasks.rewarder.humanoid.{self.cfg["env"]["rewarder"]}',
            {
                "device": self.device,
                "num_envs": self.num_envs,
                "reward_settings": self.reward_settings,
            },
        )

        # set up default hand, arm, object target pos initialization.
        self._pos_init()

        # num_actions == self.cfg["env"]["numActions"] == self.robot_module.act_dim

        # set tensors and buffers.
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float,
            device=self.device,
        )

        self.p_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.p_gain_val
        )
        self.d_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.d_gain_val
        )
        self.last_object_dof_pos = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

        # setup random forces.
        self.num_bodies = self._rigid_body_state.shape[1]
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.left_control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.right_control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

        # object brake joint torque
        self.brake_torque = self.cfg["env"]["brake_torque"]
        self.object_brake_torque = torch.full(
            (self.num_envs,), self.brake_torque, dtype=torch.float, device=self.device
        )

        self.force_prob = self.cfg["env"]["randomization"]["force_prob"]
        self.force_decay = self.cfg["env"]["randomization"]["force_decay"]
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )

        self.force_decay_interval = self.cfg["env"]["randomization"][
            "force_decay_interval"
        ]

        self.force_scale = self.cfg["env"]["randomization"]["force_scale"]
        self.force_scale_x = self.cfg["env"]["randomization"]["force_scale_x"]
        self.force_scale_y = self.cfg["env"]["randomization"]["force_scale_y"]
        self.force_scale_z = self.cfg["env"]["randomization"]["force_scale_z"]
        self.force_horizon_decay = self.cfg["env"]["randomization"][
            "force_horizon_decay"
        ]
        self.force_progress_buf = torch.zeros_like(self.progress_buf)

        # refresh
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.obs_setting = self.cfg["env"]["obs_setting"]
        # refresh tensors
        self._refresh()
       

    def _pos_init(self):
        self.torso_default_dof_pos = to_torch(
            self.robot_module.get_torso_init_pos(),
            device=self.device,
        )

        self.arm_default_dof_pos = to_torch(
            self.robot_module.get_arm_init_pos(),
            device=self.device,
        )
        self.hand_default_dof_pos = to_torch(
            self.robot_module.get_hand_init_pos(),
            device=self.device,
        )

        # TODO: use name-based initialization
        # for i, (idx, qpos) in enumerate(
        #     self.hand_default_qpos_info[: self.total_hand_dof]
        # ):
        #     print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
        #     self.hand_default_dof_pos[i] = qpos

        self.object_init_pose = to_torch(
            self.initializer.get_object_init_pose().to_list(),
            dtype=torch.float,
            device=self.device,
        )

    def _update_states(self):
        self.object_state = self._env_root_state[:, self.object_handle, :]

        self.states.update(
            {
                # robot
                "dof_pos": self.hand_dof_pos[:, :],
                "dof_vel": self.hand_dof_vel[:, :],
                "arm_dof_pos": self.arm_dof_pos[:, :],
                "arm_dof_vel": self.arm_dof_vel[:, :],
                "object_pos": self.object_state[:, :3],
                "object_quat": self.object_state[:, 3:7],
                "object_vel": self.object_state[:, 7:10],
                "prev_object_quat": self.prev_object_state[:, 3:7],
                "object_angvel": self.object_state[:, 10:13],
                "left_tips_pos": self._env_rigid_body_state[
                    :, self.left_tip_handles, :3
                ],
                "right_tips_pos": self._env_rigid_body_state[
                    :, self.right_tip_handles, :3
                ],
                # we need two separate groups.
                "left_thumb_tips_pos": self._env_rigid_body_state[
                    :, self.left_thumb_tip_handles, :3
                ],
                "right_thumb_tips_pos": self._env_rigid_body_state[
                    :, self.right_thumb_tip_handles, :3
                ],
                "left_nonthumb_tips_pos": self._env_rigid_body_state[
                    :, self.left_nonthumb_tip_handles, :3
                ],
                "right_nonthumb_tips_pos": self._env_rigid_body_state[
                    :, self.right_nonthumb_tip_handles, :3
                ],
                "object_dof_pos": self.all_dof_pos[:, self.object_joint_id],
                "object_dof_vel": self.all_dof_vel[:, self.object_joint_id],
                "last_object_dof_pos": self.last_object_dof_pos.clone(),
                "object_base_pos": self._env_rigid_body_state[
                    :, self.bottle_base_handle, :3
                ],
                "object_cap_pos": self._env_rigid_body_state[
                    :, self.bottle_cap_handle, :3
                ],
                "object_base_marker_pos": self._env_rigid_body_state[
                    :, self.bottle_base_marker_handles, :3
                ],
                "object_cap_marker_pos": self._env_rigid_body_state[
                    :, self.bottle_cap_marker_handles, :3
                ],
                "left_work": self.left_control_work,
                "right_work": self.right_control_work,
                "work": self.control_work,
                "object_init_pos": self.object_init_pose,
                # "cam": self._env_rigid_body_state[:, self.wrist_cam_handles, :7],
                "eef_pos": self._env_rigid_body_state[:, self.eef_handles, :3],
                "eef_quat": self._env_rigid_body_state[:, self.eef_handles, 3:7],
                "contact_forces": self._contact_forces,
                "init_hand_dof_pos": self.hand_default_dof_pos,
                "init_ur_dof_pos": self.arm_default_dof_pos,
                "all_finger_pos": self._env_rigid_body_state[
                    :, self.all_hand_link_handles, :3
                ],
                "all_arm_pos": self._env_rigid_body_state[:, self.arm_link_handles, :3],
                "table_height": self._root_state[self.all_table_indices, 2],
                # "marker_corners": self._env_rigid_body_state[
                #     :, self.visual_marker_handles, :3
                # ],
                # "shoulder": self._env_rigid_body_state[:, self.shoulder_handles, :3],
            }
        )
        # print("LEFT TIPS POS", self.states["left_tips_pos"])
        # print("RIGHT TIPS POS", self.states["right_tips_pos"])
        # print("OBJ POS", self.states["object_pos"])
        # print("BASE", self.states["object_base_pos"])
        # print("CAP", self.states["object_cap_pos"])
        # print("CAM", self.states["cam"])
        # print(self.states["eef_pos"])
        # print(self.states["eef_quat"])
        # print(self.states["marker_corners"])
        # print(self.states["shoulder"][:, 0] - self.states["shoulder"][:, 1])

    def get_state(self):
        # For asymmetric training.
        cursor = 0
        self.states_buf = torch.zeros_like(self.states_buf)  # Clear this first
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["dof_pos"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["dof_vel"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["arm_dof_pos"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["arm_dof_vel"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["object_pos"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["object_quat"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["left_tips_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["right_tips_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["object_base_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["object_cap_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["object_base_marker_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["object_cap_marker_pos"].reshape(self.num_envs, -1),
            cursor,
        )

        # TODO: add more states
        self.states_buf, cursor = self._fill(
            self.states_buf, self.object_shape_id.reshape(self.num_envs, -1), cursor
        )
        self.states_buf = torch.nan_to_num(
            self.states_buf, nan=0.0, posinf=1.0, neginf=-1.0
        )

        # print("Buffer Effective Width", cursor)
        return self.states_buf

    @staticmethod
    def _fill(buf, x, start_pos):
        width = x.size(1)
        buf[:, start_pos : start_pos + width] = x
        return buf, start_pos + width

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.hand_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :,
            self.hand_dof_handles,
        ]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.arm_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :, self.arm_dof_handles
        ]
        self.arm_dof_pos = self.arm_dof_state[..., 0]
        self.arm_dof_vel = self.arm_dof_state[..., 1]

        # refresh states
        self._update_states()

    # def _create_camera(self, env_ptr):
    #     cam_props = gymapi.CameraProperties()
    #     cam_props.width = self.cam_w
    #     cam_props.height = self.cam_h
    #     cam_props.supersampling_vertical = self.cam_ss
    #     cam_props.enable_tensors = True
    #     cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
    #     cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
    #     cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    #     self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
    #     # self.cams.append(cam_handle)
    #     cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
    #     cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
    #     assert cam_tensor_th.shape == (cam_props.height, cam_props.width, 4)
    #     self.cam_tensors.append(cam_tensor_th)

    def _create_camera(self, env_ptr):
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.cam_w
        cam_props.height = self.cam_h
        # print("CAMERA_PROP", self.cam_h, self.cam_w)
        # cam_props.supersampling_vertical = self.cam_ss
        # cam_props.horizontal_fov = 0.6
        # cam_props.enable_tensors = True

        cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)

        self.env_camera_handles.append(cam_handle)
        # # overhead view
        # cam_pos = gymapi.Vec3(0.8, 0, 1.5)
        # cam_target = gymapi.Vec3(0.2, 0, 0.4)
        cam_pos = gymapi.Vec3(1.0, 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0.4)
        result = self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)
        # self.cams.append(cam_handle)

        # cam_tensor = self.gym.get_camera_image_gpu_tensor(
        #     self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR
        # )
        # cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
        # assert cam_tensor_th.shape == (cam_props.height, cam_props.width, 4)
        # self.cam_tensors.append(cam_tensor_th)
        # print("Camera tensor created.")

    # def _create_camera_wrist(self, env_ptr, franka_actor):
    #     cam_props = gymapi.CameraProperties()
    #     cam_props.width = self.cam_w
    #     cam_props.height = self.cam_h
    #     cam_props.supersampling_vertical = self.cam_ss
    #     cam_props.enable_tensors = True
    #     cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)

    #     wrist_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "wrist_1_link")

    #     local_t = gymapi.Transform()
    #     local_t.p = gymapi.Vec3(*self.cam_loc_p)
    #     xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
    #     local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
    #     self.gym.attach_camera_to_body(
    #         cam_handle, env_ptr, wrist_handle,
    #         local_t, gymapi.FOLLOW_TRANSFORM
    #     )

    #     cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
    #     cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
    #     assert cam_tensor_th.shape == (cam_props.height, cam_props.width, 4)
    #     self.cam_tensors_wrist.append(cam_tensor_th)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z  # self.cfg["sim"]["up_axis"]
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        print("GRAPHICS_ID", self.graphics_device_id)
        print("DEVICE ID", self.device_id)
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # --------------------------------------------------------------------------------------
        #                                   Load Assets
        # --------------------------------------------------------------------------------------
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )
        robot_asset_file = self.robot_module.asset_file_path

        object_asset_manager = AssetManager(
            asset_root=asset_root,
            model_root_path=self.cfg["env"]["object_asset_root_folder"],
        )
        self.object_asset_manager = object_asset_manager

        # Create table asset
        table_pos = self.initializer.table_init_pos
        self.table_init_pos = torch.tensor(table_pos, device=self.device)
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, *[1.0, 1.2, table_thickness], table_opts
        )

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, table_thickness / 2]
        )
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = self.disable_gravity
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, robot_asset_file, asset_options
        )

        hand_dof_names = self.robot_module.hand_dof_names
        arm_dof_names = self.robot_module.arm_dof_names
        self.hand_dof_handles = to_torch(
            [
                self.gym.find_asset_dof_index(robot_asset, hand_dof_name)
                for hand_dof_name in hand_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_dof_handles = to_torch(
            [
                self.gym.find_asset_dof_index(robot_asset, arm_dof_name)
                for arm_dof_name in arm_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        # Load Object Assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.flip_visual_attachments = False
        object_asset_options.collapse_fixed_joints = False
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            object_asset_options.use_physx_armature = True
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        # TODO() We may need this in the future.
        # hand_dof_stiffness = to_torch([
        #     400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
        #     400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
        # ], dtype=torch.float, device=self.device)

        # hand_dof_damping = to_torch([
        #     80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
        #     80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
        # ], dtype=torch.float, device=self.device)

        # --------------------------------------------------------------------------------------
        #                                   Setup Both Hands
        # --------------------------------------------------------------------------------------

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num Robot Bodies: ", self.num_robot_bodies)
        print("num Robot Dofs: ", self.num_robot_dofs)

        # ### debugging info
        # joint_count = self.gym.get_asset_joint_count(robot_asset)
        # print("joint_count:", joint_count)
        # for i in range(joint_count):
        #     joint_name = self.gym.get_asset_joint_name(robot_asset, i)
        #     print(f'"{joint_name}",')
        # print("dof_count:", self.num_robot_dofs)
        # for i in range(self.num_robot_dofs):
        #     dof_name = self.gym.get_asset_dof_name(robot_asset, i)
        #     print(f'"{dof_name}",')
        # print("rigid_body_count:", self.num_robot_bodies)
        # for i in range(self.num_robot_bodies):
        #     body_name = self.gym.get_asset_rigid_body_name(robot_asset, i)
        #     print(f'"{body_name}",')
        # breakpoint()

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        self.hand_dof_upper_limits = []
        self.hand_dof_lower_limits = []
        self.arm_dof_lower_limits = []
        self.arm_dof_upper_limits = []

        # we only set hand dof properties, arm property is inheritied from URDF
        for i in self.arm_dof_handles:
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            self.arm_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_dof_upper_limits.append(robot_dof_props["upper"][i])

            # TODO: check value correctness
            robot_dof_props["stiffness"][i] = self.initializer.get_arm_info()[
                "stiffness"
            ]
            robot_dof_props["damping"][i] = self.initializer.get_arm_info()["damping"]

        self.arm_dof_upper_limits = to_torch(
            self.arm_dof_upper_limits, device=self.device
        )
        self.arm_dof_lower_limits = to_torch(
            self.arm_dof_lower_limits, device=self.device
        )

        for i in self.hand_dof_handles:
            robot_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            robot_dof_props["velocity"][i] = self.initializer.get_dof_velocity()
            robot_dof_props["effort"][i] = 0.5
            robot_dof_props["stiffness"][i] = self.initializer.get_hand_info()[
                "stiffness"
            ]  # 0.0# 2
            robot_dof_props["damping"][i] = self.initializer.get_hand_info()[
                "damping"
            ]  # 0.0 #0.1
            robot_dof_props["friction"][i] = 0.01
            robot_dof_props["armature"][i] = self.initializer.get_hand_info()[
                "armature"
            ]  # 0.001 #0.002

            self.hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(robot_dof_props["upper"][i])

        self.hand_dof_upper_limits = to_torch(
            self.hand_dof_upper_limits, device=self.device
        )
        self.hand_dof_lower_limits = to_torch(
            self.hand_dof_lower_limits, device=self.device
        )
        self.allegro_dof_speed_scales = torch.ones_like(self.hand_dof_lower_limits)

        robot_start_pose = self.initializer.get_robot_init_pose().to_isaacgym_pose()

        # --------------------------------------------------------------------------------------
        #                                   Setup Object.
        # --------------------------------------------------------------------------------------

        object_asset_manager.load(self, object_asset_options, self.initializer)

        object_start_pose = self.initializer.get_object_init_pose().to_isaacgym_pose()

        # --------------------------------------------------------------------------------------
        #                              Initialize Vec Environment
        # --------------------------------------------------------------------------------------

        # compute aggregate size
        num_hand_bodies = (
            self.gym.get_asset_rigid_body_count(robot_asset)
            + object_asset_manager.get_asset_rigid_body_count()
        )
        num_hand_shapes = (
            self.gym.get_asset_rigid_shape_count(robot_asset)
            + object_asset_manager.get_asset_rigid_shape_count()
        )
        max_agg_bodies = num_hand_bodies + 3  # 1 for table, table stand, object
        max_agg_shapes = num_hand_shapes + 3  # 1 for table, table stand, object

        self.robot = []
        self.object = []
        self.envs = []
        self.all_table_indices = []

        self.hand_indices = []
        self.all_object_indices = []

        self.object_shape_id = []

        if self.enable_camera_sensors:
            # self.cams = []
            self.cam_tensors = []
            self.cam_tensors_wrist = []

        self.env_camera_handles = []
        self.env_physics_setup = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Begin Routine
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 0, 0
            )
            self.all_table_indices.append(
                self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
            )

            env_hand_actor_indices = []

            # Create hand actor
            robot_actor = self.gym.create_actor(
                env_ptr,
                robot_asset,
                robot_start_pose,
                "robot",
                i,
                -1,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            prop = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_actor)
            # prop[0].restitution = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_actor, prop)

            hand_idx = self.gym.get_actor_index(env_ptr, robot_actor, gymapi.DOMAIN_SIM)
            env_hand_actor_indices.append(hand_idx)
            self.hand_indices.append(env_hand_actor_indices)

            # Create object actor
            (
                object_shape_idx,
                object_asset,
                object_dof_props,
            ) = object_asset_manager.get_random_asset()
            self.object_shape_id.append(object_shape_idx)

            mass_scaling = self.randomizer.get_random_object_scaling("mass")
            object_actor = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 1, 0
            )
            object_body_prop = self.gym.get_actor_rigid_body_properties(
                env_ptr, object_actor
            )

            mass_dict = self.randomizer.get_random_bottle_mass(
                object_body_prop[0].mass, object_body_prop[1].mass
            )
            object_body_prop[0].mass = mass_dict["body_mass"]
            object_body_prop[1].mass = mass_dict["cap_mass"]
            mass_scaling = mass_dict["mass_scaling"]
            # print("MASS", object_body_prop[0].mass,  object_body_prop[1].mass)

            # Random Object Physics
            friction_rescaling = self.randomizer.get_random_object_scaling("friction")

            object_scaling = self.randomizer.get_random_object_scaling("scale")

            object_prop = self.gym.get_actor_rigid_shape_properties(
                env_ptr, object_actor
            )
            object_prop[0].restitution = 0.0
            object_prop[0].rolling_friction = 1.0
            object_prop[0].friction = 1.5 * friction_rescaling

            object_prop[1].restitution = 0.0
            object_prop[1].rolling_friction = 1.0
            object_prop[1].friction = 1.5 * friction_rescaling

            # Random Object Scale
            object_scale = self.cfg["env"]["object_scale"] * object_scaling

            self.gym.set_actor_rigid_body_properties(
                env_ptr, object_actor, object_body_prop
            )
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, object_actor, object_prop
            )
            self.gym.set_actor_scale(env_ptr, object_actor, object_scale)
            self.gym.set_actor_dof_properties(env_ptr, object_actor, object_dof_props)

            self.object.append(object_actor)
            object_idx = self.gym.get_actor_index(
                env_ptr, object_actor, gymapi.DOMAIN_SIM
            )
            self.all_object_indices.append(object_idx)

            self.env_physics_setup.append(
                [mass_scaling, friction_rescaling, object_scaling]
            )
            # End Routine
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robot.append(robot_actor)

            if self.enable_camera_sensors:
                self._create_camera(env_ptr)
                # self._create_camera_wrist(env_ptr, franka_actor)
            # print("BREAK_JOINT", self.gym.find_actor_dof_handle(env_ptr, object_actor, "brake_joint"))

        self.brake_joint_id = self.gym.find_actor_dof_handle(
            env_ptr, object_actor, "brake_joint"
        )
        self.object_joint_id = self.gym.find_actor_dof_handle(
            env_ptr, object_actor, "rev_joint"
        )

        # TODO: refactor the following code

        self.left_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module.left_hand_tip_names
        ]
        self.left_tip_handles = to_torch(
            self.left_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module.right_hand_tip_names
        ]
        self.right_tip_handles = to_torch(
            self.right_tip_handles, dtype=torch.long, device=self.device
        )

        self.left_thumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module._left_hand_thumb_tip_names
        ]
        self.left_thumb_tip_handles = to_torch(
            self.left_thumb_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_thumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module._right_hand_thumb_tip_names
        ]
        self.right_thumb_tip_handles = to_torch(
            self.right_thumb_tip_handles, dtype=torch.long, device=self.device
        )

        self.left_nonthumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module._left_hand_nonthumb_tip_names
        ]
        self.left_nonthumb_tip_handles = to_torch(
            self.left_nonthumb_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_nonthumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, tip_name)
            for tip_name in self.robot_module._right_hand_nonthumb_tip_names
        ]
        self.right_nonthumb_tip_handles = to_torch(
            self.right_nonthumb_tip_handles, dtype=torch.long, device=self.device
        )

        self.arm_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
            for name in self.robot_module.arm_dof_names  # ignore fixed joints
        ]
        self.arm_handles = to_torch(
            self.arm_handles, dtype=torch.long, device=self.device
        )

        self.all_hand_link_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
            for name in self.robot_module.hand_link_names
        ]
        self.all_hand_link_handles = to_torch(
            self.all_hand_link_handles, dtype=torch.long, device=self.device
        )

        self.arm_link_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
            for name in self.robot_module.arm_link_names
        ]
        self.arm_link_handles = to_torch(
            self.arm_link_handles, dtype=torch.long, device=self.device
        )

        self.env_physics_setup = to_torch(
            self.env_physics_setup, dtype=torch.float, device=self.device
        )

        # wrist_cam_handle_names = [
        #     "wrist_base_cam",
        #     "wrist_base_cam_r",
        # ]
        # self.wrist_cam_handles = [
        #     self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
        #     for name in wrist_cam_handle_names
        # ]
        # self.wrist_cam_handles = to_torch(
        #     self.wrist_cam_handles, dtype=torch.long, device=self.device
        # )
        # print("WRIST_CAM_HANDLES", self.wrist_cam_handles)

        # visual_marker_handle_names = [
        #     "wrist_base_marker_c0",
        #     "wrist_base_marker_c1",
        #     "wrist_base_marker_c2",
        #     "wrist_base_marker_c3",
        #     "wrist_base_marker_c0_r",
        #     "wrist_base_marker_c1_r",
        #     "wrist_base_marker_c2_r",
        #     "wrist_base_marker_c3_r",
        #     "wrist_palm_marker_c0",
        #     "wrist_palm_marker_c1",
        #     "wrist_palm_marker_c2",
        #     "wrist_palm_marker_c3",
        #     "wrist_palm_marker_c0_r",
        #     "wrist_palm_marker_c1_r",
        #     "wrist_palm_marker_c2_r",
        #     "wrist_palm_marker_c3_r",
        # ]
        # self.visual_marker_handles = [
        #     self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
        #     for name in visual_marker_handle_names
        # ]
        # self.visual_marker_handles = to_torch(
        #     self.visual_marker_handles, dtype=torch.long, device=self.device
        # )
        # print("VISUAL_MARKER_HANDLES", self.visual_marker_handles)

        eef_handle_names = self.robot_module.wrist_link_names
        self.eef_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, name)
            for name in eef_handle_names
        ]
        self.eef_handles = to_torch(
            self.eef_handles, dtype=torch.long, device=self.device
        )
        # print("EEF_HANDLES", self.eef_handles)

        # Set handles
        self.bottle_cap_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, object_actor, "link1"
        )
        # self.bottle_cap_handles = to_torch(self.bottle_cap_handles, dtype=torch.long, device=self.device)
        self.bottle_base_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, object_actor, "link2"
        )
        print(
            self.bottle_cap_handle,
            self.bottle_base_handle,
            self.gym.find_actor_rigid_body_handle(env_ptr, object_actor, "l10"),
        )

        (
            cap_marker_handle_names,
            base_marker_handle_names,
        ) = self.object_asset_manager.get_markers()
        self.bottle_cap_marker_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, object_actor, name)
            for name in cap_marker_handle_names
        ]
        self.bottle_cap_marker_handles = to_torch(
            self.bottle_cap_marker_handles, dtype=torch.long, device=self.device
        )
        self.bottle_base_marker_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, object_actor, name)
            for name in base_marker_handle_names
        ]
        self.bottle_base_marker_handles = to_torch(
            self.bottle_base_marker_handles, dtype=torch.long, device=self.device
        )

        print("CAP_MARKER_HANDLES", self.bottle_cap_marker_handles)
        print("BASE_MARKER_HANDLES", self.bottle_base_marker_handles)

        # Set shape_idx observation
        self.object_shape_id = to_torch(
            self.object_shape_id, dtype=torch.long, device=self.device
        )
        self.object_shape_id = F.one_hot(
            self.object_shape_id, num_classes=self.num_objects
        )

        # Set handles
        self.object_handle = object_actor  # this is the local handle idx in one env. not the global one. Global one should be determined by get_actor_index()

        # TODO: use name-based init
        # # Set the default qpos
        # hand_qpos_default_dict = self.initializer.get_hand_init_qpos()
        # self.hand_default_qpos_info = [
        #     (
        #         self.gym.find_actor_dof_handle(env_ptr, robot_actor, finger_name),
        #         hand_qpos_default_dict[finger_name],
        #     )
        #     for finger_name in hand_qpos_default_dict
        # ]
        # # [print(finger_name, self.gym.find_actor_dof_handle(env_ptr, allegro_right_actor, finger_name)) for finger_name in right_hand_qpos_default_dict]

        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )  # shape = [num_envs, 2]. It stores the global index of left_hand & right_hand.
        self.all_object_indices = to_torch(
            self.all_object_indices, dtype=torch.long, device=self.device
        )  # shape = [num_envs]. It stores the global index of the object.
        self.all_table_indices = to_torch(
            self.all_table_indices, dtype=torch.long, device=self.device
        )  # shape = [num_envs]. It stores the global index of the table.

        self.reach_first_goal = torch.tensor(
            [False] * self.num_envs, device=self.device
        )
        # Setup init state buffer
        # self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)
        # self._target_object_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # self.prev_targets = torch.zeros(
        #     (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        # )
        # self.cur_targets = torch.zeros(
        #     (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        # )

        # change for both hands and arms
        self.prev_targets = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"]),
            dtype=torch.float,
            device=self.device,
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"]),
            dtype=torch.float,
            device=self.device,
        )

        # get gym GPU state tensors
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # setup tensor buffers
        self._root_state = gymtorch.wrap_tensor(
            _actor_root_state_tensor
        ).view(
            -1, 13
        )  # TODO()??? gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._contact_forces = gymtorch.wrap_tensor(_contact_forces).view(
            self.num_envs, -1, 3
        )

        self._env_root_state = self._root_state.view(self.num_envs, -1, 13)
        self._env_rigid_body_state = self._rigid_body_state.view(self.num_envs, -1, 13)

        self.all_dof_pos = self._dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.all_dof_vel = self._dof_state.view(self.num_envs, -1, 2)[..., 1]

        self.hand_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :,
            self.hand_dof_handles,
        ]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]

        self.arm_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :, self.arm_dof_handles
        ]
        self.arm_dof_pos = self.arm_dof_state[..., 0]
        self.arm_dof_vel = self.arm_dof_state[..., 1]

        # initialize prev object states.
        self.prev_object_state = self._env_root_state[:, self.object_handle, :].clone()

        # initialize actions
        self.hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        # initialize indices
        self._global_indices = torch.arange(
            self.num_envs * 4, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def compute_reward(self, actions):
        # print(self.states["object_pos"].shape)
        self.rew_buf[:], self.reset_buf[:], info = self.reward_function.forward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.reward_settings,
            self.max_episode_length,
        )
        return info

    def compute_observations(self):
        self._refresh()

        # hand dofs [6:22], [28:44]
        # arm dofs [0:6], [22:28]
        # TODO: refactor - use this for training too
        self.extras["dof_pos"] = self._dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_actions, 0
        ].clone()
        # print("Left UR", self.extras["dof_pos"][:1, :6])
        # print("Right UR", self.extras["dof_pos"][:1, 22:28])
        # print("Left Hand", self.extras["dof_pos"][:1, 6:22])
        # print("Right Hand", self.extras["dof_pos"][:1, 28:44])

        # Normalized hand dof position. [0:32]
        dof_pos_scaled = (
            2.0
            * (
                self.randomizer.randomize_dofpos(self.states["dof_pos"])
                - self.hand_dof_lower_limits
            )
            / (self.hand_dof_upper_limits - self.hand_dof_lower_limits)
            - 1.0
        )

        # Hand dof velocity. [32:64]
        dof_vel_scaled = self.states["dof_vel"] * self.dof_vel_scale

        arm_dof_pos_scaled = (
            2.0
            * (
                self.randomizer.randomize_dofpos(self.states["arm_dof_pos"], hand=False)
                - self.arm_dof_lower_limits
            )
            / (self.arm_dof_upper_limits - self.arm_dof_lower_limits)
            - 1.0
        )
        arm_dof_vel_scaled = self.states["arm_dof_vel"] * self.dof_vel_scale

        if self.obs_setting["no_hand_dof_vel"]:
            dof_vel_scaled = torch.zeros_like(dof_vel_scaled)
        if self.obs_setting["no_arm_dof_vel"]:
            arm_dof_vel_scaled = torch.zeros_like(arm_dof_vel_scaled)

        # Bottle Body Pos. [64:67]
        # Bottle Cap Pos. [71:74]
        self.cap_base_pos = self.states["object_cap_pos"].reshape(self.num_envs, -1)

        # old_states = [self.states["object_base_pos"].clone(), self.cap_base_pos.clone()]
        # print("Bottle", old_states)
        object_pos, self.cap_base_pos = self.randomizer.randomize_bottle_observation(
            self.states["object_base_pos"], self.cap_base_pos
        )
        object_pos = self.randomizer.randomize_object_observation(
            self.states["object_base_pos"]
        )
        if self.obs_setting["no_obj_pos"]:
            object_pos = torch.zeros_like(object_pos)

        if self.obs_setting["no_cap_base"]:
            self.cap_base_pos = torch.zeros_like(self.cap_base_pos)
        # print(“OBJ ACTOR ROOT POS”， self.states["object_pos"])
        # print(“OBJ CAP & BODY POS”， self.states["object_base_pos"], self.states["object_cap_pos"])

        # Object Quat. [67:71]
        object_quat = self.states["object_quat"]

        if self.obs_setting["no_obj_quat"]:
            object_quat = torch.zeros_like(object_quat)

        obs_prev_target = self.prev_targets.clone()
        # randomized_prev_target = self.randomizer.randomize_prev_target(obs_prev_target)
        hand_obs_prev_target = obs_prev_target[:, self.hand_dof_handles]
        arm_obs_prev_target = obs_prev_target[:, self.arm_dof_handles]
        randomized_hand_prev_target = self.randomizer.randomize_prev_target(
            hand_obs_prev_target, hand=True
        )
        randomized_arm_prev_target = self.randomizer.randomize_prev_target(
            arm_obs_prev_target, hand=False
        )
        obs_prev_target[:, self.hand_dof_handles] = randomized_hand_prev_target
        obs_prev_target[:, self.arm_dof_handles] = randomized_arm_prev_target
        randomized_prev_target = obs_prev_target

        # Prev_target. [74:106]
        frame_obs_buf = torch.cat(
            (
                dof_pos_scaled,  # 32 [0:32]
                dof_vel_scaled,  # 32 [32:64]
                arm_dof_pos_scaled,  # 12 [64:76]
                arm_dof_vel_scaled,  # 12 [76:88]
                object_pos,  # 3  [88:91]
                object_quat,  # 4  [91:95]
                self.cap_base_pos,  # 3  [95:98]
                randomized_prev_target,  # 44 [98:142]
            ),
            dim=-1,
        )

        if self.use_reward_obs:
            frame_obs_buf = torch.cat(
                (frame_obs_buf, self.reward_function.get_observation()), dim=-1
            )

        # Concatenate object id.
        if self.obs_setting["no_obj_id"]:
            object_id_obs = torch.zeros_like(self.object_shape_id)
        else:
            object_id_obs = self.object_shape_id

        frame_obs_buf = torch.cat((frame_obs_buf, object_id_obs), dim=-1)

        # # save obs as numpy
        # print("OBJ POS", object_pos)
        # print("OBJ CAP POS", self.cap_base_pos)
        # with open("obs_pos.npy", "ab") as f:
        #     np.save(f, dof_pos_scaled.cpu().numpy())
        # with open("obs_cap_pos.npy", "ab") as f:
        #     np.save(f, self.cap_base_pos.cpu().numpy())
        # with open("obs_body_pos.npy", "ab") as f:
        #     np.save(f, object_pos.cpu().numpy())
        # with open("obs_target.npy", "ab") as f:
        #     np.save(f, obs_prev_target.cpu().numpy())

        if torch.isnan(frame_obs_buf).int().sum() > 0:
            print("Nan Detected in IsaacGym simulation.")

        # TODO(): Fix this
        frame_obs_buf = torch.nan_to_num(
            frame_obs_buf, nan=0.0, posinf=1.0, neginf=-1.0
        )
        frame_obs_buf = torch.clamp(frame_obs_buf, -100.0, 100.0)

        frame_obs_buf = self.randomizer.randomize_frame_obs_buffer(frame_obs_buf)

        # print(self.obs_buf[0])
        # frame_obs_buf = self.randomizer.randomize_observation(frame_obs_buf)
        if self.n_stack_frame == 1:
            self.obs_buf = frame_obs_buf.clone()
        else:
            self.obs_buf = torch.cat(
                (
                    frame_obs_buf[:, : self.single_frame_obs_dim],
                    self.obs_buf[:, : -self.single_frame_obs_dim],
                ),
                dim=-1,
            )
        if self.enable_camera_sensors:
            self.compute_pixel_obs()
        # print(obs_buf.shape, self.obs_buf.shape)
        # self.obs_buf[:, :obs_buf.shape[1]] = obs_buf
        return self.obs_buf

    def compute_pixel_obs(self):
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # TODO: these are for wrist camera but not third-person views
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            image = self.gym.get_camera_image(
                self.sim, self.envs[i], self.env_camera_handles[i], gymapi.IMAGE_COLOR
            )

            image = image.reshape(self.cam_h, self.cam_w, -1)[:, crop_l:crop_r, :3]

            tensor_image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            self.pix_obs_buf[i] = tensor_image
            # self.pix_obs_wrist_buf[i] = self.cam_tensors_wrist[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
            # self.pix_obs_buf[i] = (self.pix_obs_buf[i] - self.im_mean) / self.im_std
            if i == 0 and self.viewer:
                import cv2

                cv2.imshow("image", image[:, :, ::-1])
                cv2.waitKey(10)

        self.gym.end_access_image_tensors(self.sim)
        # print(self.pix_obs_buf[0], torch.max(self.pix_obs_buf[0]))
        return {"third-person": self.pix_obs_buf}

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset controller setup
        p_lower, p_upper, d_lower, d_upper = self.randomizer.get_pd_gain_scaling_setup()
        self.randomize_p_gain_lower = self.p_gain_val * p_lower  # 0.30
        self.randomize_p_gain_upper = self.p_gain_val * p_upper  # 0.60
        self.randomize_d_gain_lower = self.d_gain_val * d_lower  # 0.75
        self.randomize_d_gain_upper = self.d_gain_val * d_upper  # 1.05

        self.p_gain[env_ids] = torch_rand_float(
            self.randomize_p_gain_lower,
            self.randomize_p_gain_upper,
            (len(env_ids), self.num_actions),
            device=self.device,
        ).squeeze(1)
        self.d_gain[env_ids] = torch_rand_float(
            self.randomize_d_gain_lower,
            self.randomize_d_gain_upper,
            (len(env_ids), self.num_actions),
            device=self.device,
        ).squeeze(1)

        # reset hand dof.
        self.hand_dof_pos[env_ids] = self.hand_default_dof_pos[None]
        self.arm_dof_pos[env_ids] = self.arm_default_dof_pos[None]

        # only randomize hand, fix arm
        self.hand_dof_pos[env_ids] = self.randomizer.randomize_hand_init_qpos(
            self.hand_dof_pos[env_ids]
        )
        self.hand_dof_vel[env_ids, :] = torch.zeros_like(self.hand_dof_vel[env_ids])

        # randomize table pos
        self._root_state[self.all_table_indices[env_ids], 0:3] = (
            self.table_init_pos.unsqueeze(0)[:, 0:3]
        )
        self._root_state[self.all_table_indices[env_ids], 0:3] = (
            self.randomizer.randomize_table_pos(
                self._root_state[self.all_table_indices[env_ids], 0:3]
            )
        )
        reset_table_indices = (self.all_table_indices[env_ids]).int()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(reset_table_indices),
            len(reset_table_indices),
        )

        # reset object dof
        self.all_dof_pos[env_ids, self.object_joint_id] = 0.0
        self.all_dof_vel[env_ids, self.object_joint_id] = 0.0
        self.all_dof_pos[env_ids, self.brake_joint_id] = 0.01
        self.all_dof_vel[env_ids, self.brake_joint_id] = 0.0

        # Object Cartesian Position
        self._root_state[self.all_object_indices[env_ids], 0:3] = (
            self.object_init_pose.unsqueeze(0)[:, 0:3]
        )
        # self._root_state[self.all_object_indices[env_ids], 0:3] = (
        #     self.randomizer.randomize_object_init_pos(
        #         self._root_state[self.all_object_indices[env_ids], 0:3]
        #     )
        # )
        self._root_state[self.all_object_indices[env_ids], 2] = (
            self._root_state[self.all_table_indices[env_ids], 2] + 0.1
        )

        # Object Orientation
        self._root_state[self.all_object_indices[env_ids], 3:7] = (
            self.object_init_pose.unsqueeze(0)[:, 3:7]
        )

        # print("SHAPE", self.object_init_pose.unsqueeze(0).shape, self._root_state.shape)
        # rotation = self.randomizer.randomize_object_init_quat(
        #     self.object_init_pose.repeat(len(env_ids), 1)[:, 3:7]
        # )

        # self._root_state[self.all_object_indices[env_ids], 3:7] = rotation

        # Object 6DOF Velocity
        self._root_state[self.all_object_indices[env_ids], 7:13] = torch.zeros_like(
            self._root_state[self.all_object_indices[env_ids], 7:13]
        )

        reset_object_indices = (self.all_object_indices[env_ids]).int()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(reset_object_indices),
            len(reset_object_indices),
        )

        reset_hand_indices = self.hand_indices[env_ids].to(torch.int32).reshape(-1)
        reset_actor_indices = torch.cat((reset_hand_indices, reset_object_indices))
        self._dof_state.view(self.num_envs, -1, 2)[:, self.arm_dof_handles, 0] = (
            self.arm_default_dof_pos[None]
        )
        self._dof_state.view(self.num_envs, -1, 2)[:, self.arm_dof_handles, 1] = 0
        self._dof_state.view(self.num_envs, -1, 2)[:, self.hand_dof_handles, 0] = (
            self.hand_dof_pos.clone()
        )
        self._dof_state.view(self.num_envs, -1, 2)[:, self.hand_dof_handles, 1] = (
            self.hand_dof_vel.clone()
        )
        # Reset the hands' dof and object's dof jointly.
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(reset_actor_indices),
            len(reset_actor_indices),
        )

        # Reset observation
        self.obs_buf[env_ids, :] = 0.0
        self.states_buf[env_ids, :] = 0.0

        # Reset buffer.
        self.last_actions[env_ids, :] = 0.0
        self.last_object_dof_pos[env_ids] = 0.0

        # WARNING: DO NOT CALL SET_DOF_STATE_TENSOR TWICE IN GPU PIPELINE!!! IT CAN LEAD TO ERROR!!
        # Reset the object dof.
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(reset_object_indices), len(reset_object_indices))
        hand_pos = self.hand_default_dof_pos[None]
        arm_pos = self.arm_default_dof_pos[None]
        pos = torch.zeros(
            (1, self.cfg["env"]["numActions"]), dtype=torch.float, device=self.device
        )
        pos[:, self.hand_dof_handles] = hand_pos
        pos[:, self.arm_dof_handles] = arm_pos

        # Reset controller target.
        self.prev_targets[env_ids, :] = pos
        self.cur_targets[env_ids, :] = pos

        # Rest object state recorder
        self.prev_object_state[env_ids, ...] = self._env_root_state[
            env_ids, self.object_handle, :
        ].clone()

        self.progress_buf[env_ids] = 0
        self.force_progress_buf[env_ids] = -1000
        # self.reset_buf[env_ids] = 0

        # Reset torque setup
        (
            torque_lower,
            torque_upper,
        ) = self.randomizer.get_object_dof_friction_scaling_setup()
        torque_lower, torque_upper = (
            self.brake_torque * torque_lower,
            self.brake_torque * torque_upper,
        )
        self.object_brake_torque[env_ids] = (
            torch.rand(len(env_ids)).to(self.device) * (torque_upper - torque_lower)
            + torque_lower
        )

        # Reset reward function (goal resets)
        self.reward_function.reset(env_ids)
        self.randomizer.reset(env_ids)

        self.left_control_work = torch.zeros_like(self.left_control_work)
        self.right_control_work = torch.zeros_like(self.right_control_work)
        self.control_work = torch.zeros_like(self.control_work)

    # def reset_init_step(self):
    #     """Step the physics of the environment until it reaches init condition.
    #     """
    #     input_dict = {
    #         "obs": self.running_mean_std(self.obs_dict["obs"]),
    #     }
    #     actions = self.init_model.act_inference(input_dict)

    #     # randomize actions
    #     if self.dr_randomizations.get('actions', None):
    #         actions = self.dr_randomizations['actions']['noise_lambda'](actions)

    #     action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    #     # apply actions
    #     self.pre_physics_step(action_tensor)

    #     # step physics and render each frame
    #     for i in range(self.control_freq_inv):
    #         if self.force_render:
    #             self.render()
    #         # print(i, "step physics A")
    #         self.gym.fetch_results(self.sim, True)
    #         #self.gym.fetch_results(self.sim, True)
    #         #print("step physics 1", i)
    #         self.update_controller()
    #         # print(i, "step physics B")
    #         self.gym.simulate(self.sim)

    #     # to fix!
    #     # if self.device == 'cpu':
    #     self.gym.fetch_results(self.sim, True)

    #     # compute observations, rewards, resets, ...
    #     self.post_physics_step()

    #     # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
    #     self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

    #     # randomize observations
    #     if self.dr_randomizations.get('observations', None):
    #         self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    #     self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

    #     self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    #     # asymmetric actor-critic
    #     if self.num_states > 0:
    #         self.obs_dict["states"] = self.get_state()

    #     if self.enable_camera_sensors:
    #         self.extras["camera"] = self.pix_obs_buf

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.actions = self.randomizer.randomize_action(self.actions)

        self.actions = torch.clamp(self.actions, -1, 1)
        assert torch.isnan(self.actions).int().sum() == 0, "nan detected"

        # smooth our action.
        self.actions = self.actions * self.action_moving_average + self.last_actions * (
            1.0 - self.action_moving_average
        )

        self.cur_targets[:] = (
            self.cur_targets + self.controller_action_scale * self.actions
        )

        ur_dof_pos_target = self.cur_targets[:, self.arm_dof_handles]
        self.cur_targets[:, self.arm_dof_handles] = ur_dof_pos_target

        allegro_dof_pos_target = self.cur_targets[:, self.hand_dof_handles]
        self.cur_targets[:, self.hand_dof_handles] = allegro_dof_pos_target

        # calculate hand-arm limits
        lower_limits = torch.zeros_like(self.cur_targets)
        upper_limits = torch.zeros_like(self.cur_targets)
        lower_limits[:, self.hand_dof_handles] = self.hand_dof_lower_limits
        upper_limits[:, self.hand_dof_handles] = self.hand_dof_upper_limits
        lower_limits[:, self.arm_dof_handles] = self.arm_dof_lower_limits
        upper_limits[:, self.arm_dof_handles] = self.arm_dof_upper_limits

        self.cur_targets[:] = tensor_clamp(
            self.cur_targets,
            lower_limits,
            upper_limits,
        )

        self.prev_targets = self.cur_targets.clone()
        self.last_actions = self.actions.clone().to(self.device)
        self.last_object_dof_pos = self.all_dof_pos[:, -1].clone()

        # We need some random force to perturb the object.
        # This can hopefully leads to more robust rotation behaviors.
        if self.force_scale > 0.0:
            # print("Apply force!", self.object_handle)
            self.rb_forces *= torch.pow(
                self.force_decay, self.dt / self.force_decay_interval
            )

            # New
            obj_mass = to_torch(
                [
                    self.gym.get_actor_rigid_body_properties(
                        env, self.gym.find_actor_handle(env, "object")
                    )[0].mass
                    for env in self.envs
                ],
                device=self.device,
            )
            # print("Mass", obj_mass)
            prob = self.force_prob
            force_indices_candidate = (
                torch.less(torch.rand(self.num_envs, device=self.device), prob)
            ).nonzero()
            # torch.less(torch.rand(self.num_envs, device=self.device), prob)

            # print("Force indices candidate", force_indices_candidate)
            last_force_progress = self.force_progress_buf[force_indices_candidate]
            current_progress = self.progress_buf[force_indices_candidate]
            # print(force_indices_candidate, current_progress, last_force_progress)

            valid_indices = torch.where(
                current_progress
                > last_force_progress
                + torch.randint(  # TODO: make upper & lower bounds hyperparameters
                    20, 50, (len(force_indices_candidate),), device=self.device
                ).unsqueeze(-1)
            )[0]
            # print("Valid", valid_indices)

            force_indices = force_indices_candidate[valid_indices]
            # print("Selected", force_indices)
            self.force_progress_buf[force_indices] = self.progress_buf[force_indices]

            step = self.progress_buf[force_indices]  # [N, 1]
            horizon_decay = torch.pow(self.force_horizon_decay, step)
            # print(horizon_decay)
            # print("Force scales:")
            # print(self.force_scale * self.force_scale_x)
            # print(self.force_scale * self.force_scale_y)
            # print(self.force_scale * self.force_scale_z)
            for i, axis_scale in enumerate(
                [self.force_scale_x, self.force_scale_y, self.force_scale_z]
            ):
                self.rb_forces[force_indices, self.bottle_base_handle, i] = (
                    torch.randn(
                        self.rb_forces[force_indices, self.bottle_base_handle, i].shape,
                        device=self.device,
                    )
                    * horizon_decay
                    * obj_mass[force_indices]
                    * axis_scale
                    * self.force_scale
                )
            # print("force indices", force_indices)
            # print("force val", self.rb_forces[force_indices, self.bottle_base_handle, :])
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                None,
                gymapi.ENV_SPACE,
            )
            # breakpoint()

    def update_controller(self):
        # previous_dof_pos = self.hand_dof_pos.clone()
        previous_dof_pos = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"]),
            dtype=torch.float,
            device=self.device,
        )
        previous_dof_pos[:, self.hand_dof_handles] = self.hand_dof_pos.clone()
        previous_dof_pos[:, self.arm_dof_handles] = self.arm_dof_pos.clone()
        self._refresh()

        # dof_pos = self.hand_dof_pos
        dof_pos = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"]),
            dtype=torch.float,
            device=self.device,
        )
        dof_pos[:, self.hand_dof_handles] = self.hand_dof_pos
        dof_pos[:, self.arm_dof_handles] = self.arm_dof_pos
        dof_vel = (dof_pos - previous_dof_pos) / self.dt

        self.dof_vel_finite_diff = dof_vel.clone()
        # ("vel", self.dof_vel_finite_diff)
        hand_torques = (
            self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
        )

        # print(self.cur_targets)
        # print("ERROR", self.cur_targets - dof_pos)
        torques = torch.zeros((self.num_envs, self.num_dofs)).to(self.device)
        # torques[:, self.arm_dof_handles] = hand_torques[:, self.arm_dof_handles]
        torques[:, self.arm_dof_handles] = 0
        torques[:, self.hand_dof_handles] = hand_torques[:, self.hand_dof_handles]
        # print(torques[0])
        torques = torch.clip(torques, -1.0, 1.0)

        # Brake applies the force
        torques[:, self.brake_joint_id] = (
            self.object_brake_torque
        )  # -10.0 # adjust this to apply different friction.

        all_work = (
            torques[:, self.hand_dof_handles]
            * self.dof_vel_finite_diff[:, self.hand_dof_handles]
        )
        self.left_control_work += (
            all_work[:, : len(self.hand_dof_handles) // 2].abs().sum(-1) * self.dt
        )
        self.right_control_work += (
            all_work[:, len(self.hand_dof_handles) // 2 :].abs().sum(-1) * self.dt
        )
        self.control_work += self.left_control_work + self.right_control_work
        # self.control_work += (
        #     torques[:, self.hand_dof_handles]
        #     * self.dof_vel_finite_diff[:, self.hand_dof_handles]
        # ).abs().sum(-1) * self.dt
        # TODO: add arm control work
        # self.control_work += (
        #     torques[:, self.arm_dof_handles] * self.dof_vel_finite_diff[:, self.arm_dof_handles]
        # ).abs().sum(-1) * self.dt

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torques)
        )
        ur_target = torch.zeros(
            self.num_envs, self.num_dofs, device=self.cur_targets.device
        )
        ur_target[:, self.arm_dof_handles] = self.cur_targets[:, self.arm_dof_handles]
        ur_target[:, self.hand_dof_handles] = 0
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(ur_target)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh()

        reward_info = self.compute_reward(self.actions)
        #self.extras.update(reward_info)

        self.left_control_work = torch.zeros_like(self.left_control_work)
        self.right_control_work = torch.zeros_like(self.right_control_work)
        self.control_work = torch.zeros_like(self.control_work)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # # set object pose
        # self._root_state[self.all_object_indices, 0:3] = self.object_init_pose.unsqueeze(0)[
        #     :, 0:3
        # ]
        # self._root_state[self.all_object_indices, 3:7] = self.object_init_pose.unsqueeze(0)[
        #     :, 3:7
        # ]
        # self._root_state[self.all_object_indices, 7:13] = torch.zeros_like(
        #     self._root_state[self.all_object_indices, 7:13]
        # )
        # reset_object_indices = (self.all_object_indices).int()
        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self._root_state),
        #     gymtorch.unwrap_tensor(reset_object_indices),
        #     len(reset_object_indices),
        # )
        # # Reset the hands' dof and object's dof jointly.
        # reset_hand_indices = self.hand_indices.to(torch.int32).reshape(-1)
        # reset_actor_indices = torch.cat((reset_hand_indices, reset_object_indices))
        # self._dof_state.view(self.num_envs, -1, 2)[
        #     :, self.arm_dof_handles, 0
        # ] = self.arm_default_dof_pos[None]
        # self._dof_state.view(self.num_envs, -1, 2)[:, self.arm_dof_handles, 1] = 0
        # self._dof_state.view(self.num_envs, -1, 2)[
        #     :, self.hand_dof_handles, 0
        # ] = self.hand_dof_pos.clone()
        # self._dof_state.view(self.num_envs, -1, 2)[
        #     :, self.hand_dof_handles, 1
        # ] = self.hand_dof_vel.clone()
        # # Reset the hands' dof and object's dof jointly.
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self._dof_state),
        #     gymtorch.unwrap_tensor(reset_actor_indices),
        #     len(reset_actor_indices),
        # )
        # self.gym.refresh_dof_state_tensor(self.sim)
        self.compute_observations()

        # print(self.reset_buf[0])
        # print(self.obs_buf[0][:32])
        # print(self.all_dof_pos[0][:32])
        # print(self.obs_buf[0][64:71])
        # print('-----------------------')
        self.prev_object_state = self._env_root_state[:, self.object_handle, :].clone()

        # torch.cuda.empty_cache()

        # self.extras.update(info)

        # # debug viz
        if self.viewer:  # and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # for i in range(self.num_envs):
            #     self.reward_function.render(self, self.envs[i], i)
            #     CAP_Y = 0.00
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [-1, CAP_Y, 1.24, 1, CAP_Y, 1.24],
            #         [1, 0, 0],
            #     )

            #     CAP_X = 0.70
            #     self.gym.add_lines(
            #         self.viewer,
            #         self.envs[i],
            #         1,
            #         [CAP_X, -1, 1.24, CAP_X, 1, 1.24],
            #         [1, 0, 0],
            #     )

        #         px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.franka_grasp_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

        #         px = (self.object_grasp_pos[i] + quat_apply(self.object_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.object_grasp_pos[i] + quat_apply(self.object_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.object_grasp_pos[i] + quat_apply(self.object_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.object_grasp_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

        #         px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.franka_lfinger_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

        #         px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #         py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #         pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        #         p0 = self.franka_rfinger_pos[i].cpu().numpy()
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
        #         self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

    def update_curriculum(self, steps):
        # print("update reward curriculum")
        if not self.reward_settings["use_curriculum"]:
            return

        if "screw_curriculum" in self.reward_settings:
            low, high = self.reward_settings["screw_curriculum"]
            r_low, r_high = self.reward_settings["screw_curriculum_reward_scale"]
            if steps < low:
                scale = r_low

            elif steps > high:
                scale = r_high

            else:
                scale = r_low + (r_high - r_low) * (steps - low) / (high - low)
            self.reward_settings["rotation_reward_scale"] = scale
        # print(self.reward_settings)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat(
        [
            vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
            torch.cos(angle[idx, :] / 2.0),
        ],
        dim=-1,
    )

    # Reshape and return output
    quat = quat.reshape(
        list(input_shape)
        + [
            4,
        ]
    )
    return quat
