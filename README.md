# Source Code for Model-based Imitation Learning from Offline data (MILO)
Implementation of MILO, a model-based, offline imitation learning algorithm. 

![figure](https://github.com/jdchang1/milo/blob/main/humanoid_fig.png)

Link to pdf: https://arxiv.org/abs/2106.03207

## Notes on Installation
After cloning this repository and installing the requirements, please run

`cd milo && pip install -e .`

`cd mjrl && pip install -e .`

The experiments are run using MuJoCo physics, which requires a license to install. Please follow the instructions on [MuJoCo Website](http://www.mujoco.org)

## Overview
The `milo` package contains our imitation learning, model-based environment stack, and boilerplate code. We modified the `mjrl` package to interface with our cost functions when doing model-based policy gradient. This modification can be seen in `mjrl/mjrl/algos/batch_reinforce.py`. Note that we currently only support NPG/TRPO as our policy gradient algorithm; however, in principle one could replace this with other algorithms/repositories. 

## Environments Supported
This repository supports 5 modified MuJoCo environments that can be found in `milo/milo/gym_env`. They are
1. Hopper-v4
2. Walker2d-v4
3. HalfCheetah-v4
4. Ant-v4
5. Humanoid-v4

If you would like to add an environment, register the environment in `/milo/milo/gym_env/__init__.py` according to [OpenAI Gym](http://gym.openai.com/docs/#environments) instructions.

## Downloading the Datasets
Please download the datasets from this [google drive link](https://drive.google.com/drive/folders/1gG2WIgL1mdznhuel5uKRb6lepF7EVeFr?usp=sharing). Each environment will have 2 datasets: `[ENV]_expert.pt` and `[ENV]_offline.pt`.

In the `data` directory, place the expert and offline datasets in the `data/expert_data` and `data/offline_data` direcotires respectively. 

## Running an Experiment
We provide an example run script for Hopper, `example_run.sh`, that can be modified to be used with any other registered environment. To view all the possible arguments you can run please see the argparse in `milo/milo/utils/arguments.py`.

