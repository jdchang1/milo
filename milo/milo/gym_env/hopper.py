import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.pos_before = 0.0
        self.height_idx = 1
        self.ang_idx = 2
        self.ang_threshold = 1.0
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 10)
        # mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.pos_before = self.data.qpos[0].copy()
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        reward = self.get_reward(obs, a)
        done = self.get_done(obs)
        return obs, reward, done, {}

    def _get_obs(self):
        # I am using delta instead of velocity, 
        # so that all obs are of similar magnitude
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
            # vel_x = (obs[1] - obs[0]) / self.dt               # recover velocity from delta
            vel_x = obs[0] / self.dt
            power = np.square(act).sum()
            height, ang = obs[self.height_idx:(self.ang_idx+1)]
        else:
            # vel_x = (obs[:, :, 1] - obs[:, :, 0]) / self.dt   # recover velocity from delta
            vel_x = obs[:, :, 0] / self.dt
            power = np.square(act).sum(axis=-1)
            height = obs[:, :, self.height_idx]
            ang = obs[:, :, self.ang_idx]
        alive_bonus = 1.0 * (height > .7) * (np.abs(ang) < self.ang_threshold)
        reward = vel_x + alive_bonus - 1e-3*power
        reward = reward * 2.5    # to account for scaling difference (skip 4 --> 10)
        return reward

    def compute_path_rewards(self, paths):
        # path has two keys: observations and actions
        # path["observations"] : (num_traj, horizon, obs_dim)
        # path["rewards"] should have shape (num_traj, horizon)
        obs = paths["observations"]
        act = paths["actions"]
        rewards = self.get_reward(obs, act)
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()

    def get_done(self, obs):
        height, ang = obs[self.height_idx:(self.ang_idx+1)]
        done = not (np.isfinite(obs).all() and (np.abs(obs) < 100).all() and
                    (height > .7) and (np.abs(ang) < self.ang_threshold))
        return done

    def truncate_paths(self, paths):
        for path in paths:
            obs = path["observations"]
            height = obs[:, self.height_idx]
            angle = obs[:, self.ang_idx]
            T = obs.shape[0]
            t = 0
            done = False
            while t < T and done is False:
                done = not ((np.abs(obs[t]) < 100).all() and (height[t] > .7) and (np.abs(angle[t]) < self.ang_threshold))
                t = t + 1
                T = t if done else T
            path["observations"] = path["observations"][:T]
            path["actions"] = path["actions"][:T]
            path["rewards"] = path["rewards"][:T]
            path["terminated"] = done
        return paths

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
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
