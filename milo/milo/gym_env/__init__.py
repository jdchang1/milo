from milo.gym_env.wrappers import model_based_env
from milo.gym_env.multiprocessing_env import MujocoEnvProcess
from gym.envs.registration import register

register(
    id='Hopper-v4',
    entry_point='milo.gym_env.hopper:HopperEnv',
    max_episode_steps=400,
    reward_threshold=3800.0,
)

register(
    id='Walker2d-v4',
    max_episode_steps=400,
    entry_point='milo.gym_env.walker2d:Walker2dEnv'
)

register(
    id='HalfCheetah-v4',
    entry_point='milo.gym_env.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
    reward_threshold=4800.0,
)

register(
    id='Ant-v4',
    entry_point='milo.gym_env.ant:AntEnv',
    max_episode_steps=500,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v4',
    entry_point='milo.gym_env.humanoid:HumanoidEnv',
    max_episode_steps=500,
)
