import isaacgym

import datetime
import hydra
import gym
import os
import numpy as np
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from ppo.ppo import PPO
import random
import string

def generate_random_string(length=4, characters=string.ascii_letters + string.digits):
    """
    Generate a random string of the specified length using the given characters.

    :param length: The length of the random string (default is 12).
    :param characters: The characters to choose from when generating the string
                      (default is uppercase letters, lowercase letters, and digits).
    :return: A random string of the specified length.
    """
    return ''.join(random.choice(characters) for _ in range(length))


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # automatic folder naming
    curr_time = str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '_')
    random_tag = generate_random_string()
    config.output_name = f'{config.output_name}_{curr_time}_{random_tag}'

    if config.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        config.sim_device = f'cuda:{rank}'
        config.rl_device = f'cuda:{rank}'
        # sets seed. if seed is -1 will pick a random one
        config.seed = set_seed(config.seed + rank)
    else:
        # use the same device for sim and rl
        config.sim_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.seed = set_seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
        virtual_screen_capture=False,
        force_render=True,
    )

    output_dif = os.path.join('outputs', config.output_name)
    os.makedirs(output_dif, exist_ok=True)

    agent = PPO(env, output_dif, full_config=config)
    #if config.test:
    if config.checkpoint:
        agent.restore_test(config.checkpoint)
    openloop_trajectory = agent.collect_trajectory()
    
    # each attribute in openloop_trajectory is a tensor of shape (horizon, num_envs, ...)
    horizon, num_envs = openloop_trajectory["dof_pos"].shape[:2]
    print(f"Horizon: {horizon}, Num_envs: {num_envs}")
    
    # save all info
    root_dir =  config.eval_root_dir  # "/home/toru/Developer/Bimanual-Isaac/eval_scripts/jan"
    ckpt_name = config.checkpoint.split("/")[-3]
    save_dir = os.path.join(root_dir, ckpt_name)
    os.makedirs(save_dir, exist_ok=True)

    for k, v in openloop_trajectory.items():
        arr = v.detach().cpu().numpy()
        if k in ["dof_pos", "dones", "mu", "obs", "obs_rms"]:
            for i in range(num_envs):
                save_path = os.path.join(save_dir, f"{i}_{k}.npy")
                np.save(save_path, arr[:, i, :])
        else:
            save_path = os.path.join(save_dir, f"{k}.npy")
            np.save(save_path, arr)

    # # only save dof_pos for state replay
    # for i in range(num_envs):
    #     dof_pos = openloop_trajectory['dof_pos'][:, i, :]
    #     np.save(f"openloop_env{i}", dof_pos.detach().cpu().numpy())
        

if __name__ == '__main__':
    main()
