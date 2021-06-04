import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.pos_before = 0.0
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 10)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.pos_before = self.data.qpos[0].copy()
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.get_reward(obs, a)
        done = False    # no termination for this env
        return obs, reward, done, {}

    def _get_obs(self):
        delta = self.data.qpos[0] - self.pos_before
        return np.concatenate([
            [delta],
            self.sim.data.qpos.ravel()[1:],
            self.sim.data.qvel.ravel() * self.dt,
        ])

    def get_reward(self, obs, act):
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            # vel_x = obs[-9] / self.dt               # recover velocity from delta
            vel_x = obs[0] / self.dt
            power = np.square(act).sum()
        else:
            # vel_x = obs[:, :, -9] / self.dt         # recover velocity from delta
            vel_x = obs[:, :, 0] / self.dt
            power = np.square(act).sum(axis=-1)
        reward = vel_x - 0.1 * power
        reward = reward * 2.0    # to account for scaling difference (skip 5 --> 10)
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        act = paths["actions"]
        rewards = self.get_reward(obs, act)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    def get_env_state(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())
    
    def set_env_state(self, state):
        qpos = state['qpos']
        qvel = state['qvel']
        self.sim.reset()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.sim.forward()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
