# Source Code for Model-based Imitation Learning from Offline data (MILO)
Implementation of MILO, a model-based, offline imitation learning algorithm. 

Link to pdf: https://arxiv.org/abs/2106.03207

## Notes on Installation
After cloning this repository and installing the requirements, please run

`cd milo && pip install -e .`

`cd mjrl && pip install -e .`

The experiments are run using MuJoCo physics, which requires a license to install. Please follow the instructions on [MuJoCo Website](http://www.mujoco.org)

## Environments Supported
This repository supports 5 modified MuJoCo environments that can be found in `milo/milo/gym_env`. They are
1. Hopper-v4
2. Walker2d-v4
3. HalfCheetah-v4
4. Ant-v4
5. Humanoid-v4

If you would like to add an environment, register the environment in `/milo/milo/gym_env/__init__.py` according to `gym`instructions.
## Running an Experiment

