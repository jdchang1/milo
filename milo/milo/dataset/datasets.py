import numpy as np
import gym
from gym.spaces import Discrete, Box

import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):
    """
    Pytorch Dataset class for our offline dataset. Note we return (s,a,s') triples.
    :param env_name: (str) name of gym environment
    :param state: (torch Tensor) tensor with shape (number of samples, state dimension) with state data
    :param action: (torch Tensor) tensor with shape (number of samples, action dimension) with action data
    :param next_state: (torch Tensor) tensor with shape (number of samples, state dimension) with next state data
    :param device: (torch Device) device for pytorch. Currently hardcoded to cpu
    """
    def __init__(self, env_name, state, action, next_state, device=torch.device('cpu')):
        self.device = device
        self.state = state
        self.action = action

        env = gym.make(env_name)
        if isinstance(env.action_space, Discrete):
            self.action = self.one_hot(action, env.action_space.n)
        elif isinstance(env.action_space, Box):
            self.action = action
        else:
            raise NotImplementedError(
                "Environment Action Space not yet supported")
        self.next_state = next_state
        del env

    def get_transformations(self):
        diff = self.next_state - self.state

        # Compute Means
        state_mean = self.state.mean(dim=0).float().requires_grad_(False)
        action_mean = self.action.mean(dim=0).float().requires_grad_(False)
        diff_mean = diff.mean(dim=0).float().requires_grad_(False)

        # Compute Scales
        state_scale = torch.abs(
            self.state - state_mean).mean(dim=0).float().requires_grad_(False) + 1e-8
        action_scale = torch.abs(
            self.action - action_mean).mean(dim=0).float().requires_grad_(False) + 1e-8
        diff_scale = torch.abs(
            diff - diff_mean).mean(dim=0).float().requires_grad_(False) + 1e-8

        return state_mean.to(self.device), state_scale.to(self.device), action_mean.to(self.device), \
            action_scale.to(self.device), diff_mean.to(
                self.device), diff_scale.to(self.device)

    def one_hot(self, action, action_dim):
        db_size = action.size(0)
        one_hot_action = torch.eye(action_dim)[action]
        return one_hot_action.view(db_size, action_dim)

    def __len__(self):
        return self.state.size(0)

    def __getitem__(self, idx):
        return self.state[idx].float(), self.action[idx].float(), self.next_state[idx].float()
