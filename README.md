# minimal-ig

- follow the instructions [here](https://github.com/isaac-sim/IsaacGymEnvs) to set up the environment and [here](https://github.com/ToruOwO/minimal-stable-PPO) to set up the RL code
- check whether you can get access to a GPU for visualizing the simulator environment and doing some training
- you should be able to run the following command and visualize an environment with Unitree H1 humanoid

```
python ./env_utils/view_env.py task=Humanoid num_envs=4 headless=False task.env.robot_type=h1
```