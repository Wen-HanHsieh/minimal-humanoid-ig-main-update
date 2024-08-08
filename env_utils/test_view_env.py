# test_view_env.py
import isaacgym
import torch
import hydra
from omegaconf import DictConfig
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
import numpy as np
from h1_ik_solver import compute_ik
import time

@hydra.main(config_name="config", config_path="../minimal_stable_PPO/configs", version_base="1.1")
def main(config: DictConfig):
    config.sim_device = "cuda:0"
    config.rl_device = "cuda:0"
    config.headless = False
    config.seed = set_seed(config.seed)

    envs = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
        virtual_screen_capture=False,
        force_render=True,
    )

    print("Observation space:", envs.observation_space)
    print("Action space:", envs.action_space)
    print("Action space shape:", envs.action_space.shape)

    obs = envs.reset()

    # Define a mapping from IK solutions to DoF indices based on the printed DOF names
    ik_to_dof_mapping = {
        "left_arm": [13, 14, 15, 16, 17, 18, 19],
        "right_arm": [32, 33, 34, 35, 36, 37, 38],
        "torso": [12]
    }

    starting_pose = {
        "left_wrist": [0.0, 0.0, 0.0],
        "right_wrist": [0.0, 0.0, 0.0],
        "torso": [0.0, 0.0, 0.0]
    }

    # Improved motion sequence to move both wrists in +z, +x, +y, -y, -x, -z directions by 0.01
    motion_sequence = [
        {"left_wrist": [0.0, 0.0, 0.01], "right_wrist": [0.0, 0.0, 0.01]},
        {"left_wrist": [0.01, 0.0, 0.0], "right_wrist": [0.01, 0.0, 0.0]},
        {"left_wrist": [0.0, 0.01, 0.0], "right_wrist": [0.0, 0.01, 0.0]},
        {"left_wrist": [0.0, -0.01, 0.0], "right_wrist": [0.0, -0.01, 0.0]},
        {"left_wrist": [-0.01, 0.0, 0.0], "right_wrist": [-0.01, 0.0, 0.0]},
        {"left_wrist": [0.0, 0.0, -0.01], "right_wrist": [0.0, 0.0, -0.01]}
    ]

    for i, move in enumerate(motion_sequence):
        current_left_wrist = [
            starting_pose["left_wrist"][j] + move["left_wrist"][j]
            for j in range(3)
        ]
        current_right_wrist = [
            starting_pose["right_wrist"][j] + move["right_wrist"][j]
            for j in range(3)
        ]
        current_torso = starting_pose["torso"]

        ik_solution = compute_ik(current_left_wrist, current_right_wrist, current_torso)

        if any(np.isnan(ik_solution["left_arm"])) or any(np.isnan(ik_solution["right_arm"])):
            print(f"Iteration {i}: Invalid IK solution, skipping.")
            continue

        print(f"Iteration {i}:")
        print("Left Arm:", ik_solution["left_arm"])
        print("Right Arm:", ik_solution["right_arm"])
        print("Torso:", ik_solution["torso"])

        # Create action tensor filled with zeros
        action = torch.zeros(envs.num_envs, envs.action_space.shape[0], device=config.rl_device)

        # Map IK solutions to the correct DOFs
        for part, indices in ik_to_dof_mapping.items():
            for j, index in enumerate(indices):
                if part == "torso":
                    action[:, index] = torch.tensor(ik_solution[part], device=config.rl_device)
                else:
                    if j < len(ik_solution[part]):  # Ensure we don't go out of bounds
                        action[:, index] = torch.tensor(ik_solution[part][j], device=config.rl_device)
                    else:
                        print(f"Warning: Index {j} out of bounds for IK solution part '{part}'")

        # Clip actions to [-1.0, 1.0]
        action = torch.clamp(action, -1.0, 1.0)

        _, _, _, info = envs.step(action)

        envs.gym.refresh_actor_root_state_tensor(envs.sim)
        if envs.viewer:
            envs.gym.step_graphics(envs.sim)
            envs.gym.draw_viewer(envs.viewer, envs.sim, True)

        time.sleep(0.05)  # 50 millisecond delay

        if "success" in info and info["success"].any():
            print(f"Target reached at step {i}")
            break

    print("Simulation complete.")

if __name__ == "__main__":
    main()

