import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 1,
    'distance': 4.0,
    'lookat': np.array((0.0, 0.0, 2.0)),
    'elevation': -20.0,
}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='humanoid.xml',
                 reset_noise_scale=1e-2):
        utils.EzPickle.__init__(**locals())

        self._reset_noise_scale = reset_noise_scale

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 10)

    def step(self, action):
        self.xypos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = self.get_reward(observation, action)
        done = self.get_done(observation)

        return observation, reward, done, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        # Add Difference of center of mass to get reward
        delta = mass_center(self.model, self.sim) - self.xypos_before
        
        return np.concatenate((
            delta,
            position[2:],
            velocity*self.dt,
        ))
    
    def get_reward(self, obs, action):
        obs = np.clip(obs, -10.0, 10.0)
        ctrl = np.clip(action, -0.4, 0.4)

        x_velocity, y_velocity = obs[:2]/self.dt
        z = obs[2]
        forward_reward = 1.25 * x_velocity
        alive_reward = 5.0
        ctrl_cost = 0.1 * np.sum(np.square(ctrl))
        reward = forward_reward + alive_reward - ctrl_cost
        
        return reward * 2.0

    def get_done(self, obs):
        healthy = 1.0 < obs[2] < 2.0
        return not healthy

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
