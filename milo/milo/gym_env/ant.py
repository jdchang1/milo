import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.pos_before = np.array([0.0, 0.0])
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 10)
        # mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.pos_before = self.data.qpos[:2].copy()
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.get_reward(obs, a)
        done = self.get_done(obs)
        return obs, reward, done, {}

    def _get_obs(self):
        delta = self.data.qpos[:2] - self.pos_before
        return np.concatenate([
            delta,
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.ravel() * self.dt,
            # NOTE: We are throwing away contact related info, since it is often unnecessary
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def get_reward(self, obs, act):
        obs = np.clip(obs, -10.0, 10.0)
        if len(obs.shape) == 1:
            # vector obs, called when stepping the env
            vel_x = obs[0] / self.dt               # recover velocity from delta
            power = np.square(act).sum()
            # NOTE: We will use the contact force penalties for actual reward
            # to be consistent with gym results
            cfrc_ext = np.clip(self.sim.data.cfrc_ext, -1, 1).ravel()
            height = obs[2]
            reward = - 0.5 * 1e-3 * np.square(cfrc_ext).sum()   # contact cost
        else:
            # for imaginary rollouts using learned model
            vel_x = obs[:, :, 0] / self.dt         # recover velocity from delta
            power = np.square(act).sum(axis=-1)
            height = obs[:, :, 2]
            # NOTE: WE will not consider contact costs for imaginary rollouts
            reward = 0.0
        survive_reward = 1.0 * (height > 0.2) * (height < 1.0)
        ctrl_cost = 0.5 * power
        reward += vel_x - ctrl_cost + survive_reward
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
        return paths

    def get_done(self, obs):
        done = not (np.isfinite(obs).all() and (obs[2] > 0.2) and (obs[2] < 1.0))
        return done

    def truncate_paths(self, paths):
        for path in paths:
            obs = path["observations"]
            height = obs[:,2]#obs[:, 0]
            T = obs.shape[0]
            t = 0
            done = False
            while t < T and done is False:
                done = not (np.isfinite(obs[t]).all() and (height[t] > 0.2) and (height[t] < 1.0))
                T = t if done else T
                t = t + 1
            path["observations"] = path["observations"][:T]
            path["actions"] = path["actions"][:T]
            path["rewards"] = path["rewards"][:T]
            path["terminated"] = done
        return paths

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_env_state(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())

    def set_env_state(self, state):
        qpos = state['qpos']
        qvel = state['qvel']
        self.sim.reset()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.sim.forward()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
