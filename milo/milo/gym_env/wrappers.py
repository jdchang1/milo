import math
import torch
import gym
from gym import spaces, logger, core
from gym.utils import seeding
import numpy as np
import pdb
import copy

class MujocoModelWrapper(gym.Wrapper):
    """
    NOTE: Currently supported envs - Hopper, Walker, HalfCheetah, Ant, Humanoid
    """
    def __init__(self,
                 env,
                 model,
                 horizon,
                 init_state_buffer=None,
                 norm_thresh=float('inf'),
                 device=torch.device('cpu')):
        super().__init__(env)
        self.model = model
        self.device = device
        self.horizon = horizon
        self.ob = None
        self.num_steps = 0
        self.reset_counter = 0

        # Set model to use. Default use first
        self.curr_model = model.models[0]
        self.curr_model.eval()

        # Initial State Buffer
        self.init_state_buffer = init_state_buffer

        # Prediction Norm Threshold (This is to prevent numerical blowups)
        self.norm_thresh = norm_thresh

    def is_done(self):
        horizon_done = self.num_steps >= self.horizon
        norm_done = np.linalg.norm(self.ob) >= self.norm_thresh
        env_done = self.env.get_done(self.ob)
        return bool(horizon_done or norm_done or env_done)

    def update_model(self, model):
        self.model = model
        self.reset_counter = 0

        # Set model to use. Default use first
        self.curr_model = model.models[0]
        self.curr_model.eval()

    def step(self, action):
        assert(self.ob is not None)
        self.num_steps += 1
        with torch.no_grad():
            state_input = torch.from_numpy(self.ob).float().unsqueeze(0).to(self.device)
            action_input = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            state_diff = self.curr_model.forward(state_input, action_input)
        self.ob += state_diff.squeeze(0).cpu().numpy()
        reward = self.env.get_reward(self.ob, action)
        done = self.is_done()

        return copy.deepcopy(self.ob), reward, done, {}

    def reset(self, state=None):
        self.num_steps = 0

        # Maintain state
        if self.init_state_buffer is None:
            self.ob = self.env.reset_model()
        else:
            idx = np.random.randint(0, self.init_state_buffer.shape[0])
            self.ob = copy.deepcopy(self.init_state_buffer[idx]) if self.reset_counter % 5 == 0 else self.env.reset_model()

        # Choose model round-robin style
        self.reset_counter += 1
        self.curr_model = self.model.models[self.reset_counter % len(self.model.models)]
        self.curr_model.eval()

        return copy.deepcopy(self.ob)

class HopperModelWrapper(MujocoModelWrapper):
    def __init__(self, env, model, init_state_buffer, norm_thresh, device):
        super().__init__(env, model, horizon=400, init_state_buffer=init_state_buffer,\
                         norm_thresh=norm_thresh, device=device)

class Walker2dModelWrapper(MujocoModelWrapper):
    def __init__(self, env, model, init_state_buffer, norm_thresh, device):
        super().__init__(env, model, horizon=400, init_state_buffer=init_state_buffer,\
                         norm_thresh=norm_thresh, device=device)

class HalfCheetahModelWrapper(MujocoModelWrapper):
    def __init__(self, env, model, init_state_buffer, norm_thresh, device):
        super().__init__(env, model, horizon=500, init_state_buffer=init_state_buffer,\
                         norm_thresh=norm_thresh, device=device)

    def is_done(self):
        # Half-cheetah does not have environment termination condition
        horizon_done = self.num_steps >= self.horizon
        norm_done = np.linalg.norm(self.ob) >= self.norm_thresh
        return horizon_done or norm_done

class AntModelWrapper(MujocoModelWrapper):
    def __init__(self, env, model, init_state_buffer, norm_thresh, device):
        super().__init__(env, model, horizon=500, init_state_buffer=init_state_buffer,\
                         norm_thresh=norm_thresh, device=device)

class HumanoidModelWrapper(MujocoModelWrapper):
    def __init__(self, env, model, init_state_buffer, norm_thresh, device):
        super().__init__(env, model, horizon=500, init_state_buffer=init_state_buffer,\
                         norm_thresh=norm_thresh, device=device)

# NOTE: Add to dictionary as you keep creating more wrappers
WRAPPERS = {
    'Hopper': HopperModelWrapper,
    'Walker2d': Walker2dModelWrapper,
    'HalfCheetah': HalfCheetahModelWrapper,
    'Ant': AntModelWrapper,
    'Humanoid': HumanoidModelWrapper
}

def model_based_env(env, model, init_state_buffer=None, norm_thresh=float('inf'), device=torch.device('cpu')):
    try:
        wrapper = WRAPPERS[env.spec.id.split('-')[0]]
    except:
        raise NotImplementedError("Environment not yet supported")
    return wrapper(env, model, init_state_buffer, norm_thresh, device)


