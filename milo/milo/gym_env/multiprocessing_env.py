import copy
import numpy as np
from torch.multiprocessing import Process

class MujocoEnvProcess(Process):
    """
    Process class for model based environments that are compatible with MJRL
    """
    def __init__(self, env, child_conn, seed, eval_mode=False, paths_per_process=25):
        super().__init__()
        self.daemon = True
        self.env = copy.deepcopy(env)
        self.horizon = env.horizon
        self.child_conn = child_conn
        self.paths_per_process = paths_per_process
        self.seed = seed
        self.eval_mode = eval_mode

    def run(self):
        super().run()
        while True:
            paths, ctr = [], 0
            policy = self.child_conn.recv() # Recieve policy
            for ep in range(self.paths_per_process):
                # Set new seed
                seed = self.seed + ep
                self.env.set_seed(seed)
                np.random.seed(seed)

                observations = []
                actions = []
                rewards = []
                next_observations = []
                agent_infos = []
                env_infos = []

                o = self.env.reset()
                done = False
                t = 0
                while t < self.horizon and done != True:
                    a, agent_info = policy.get_action(o)
                    if self.eval_mode:
                        a = agent_info['evaluation']
                    next_o, r, done, info = self.env.step(a) # Take step

                    observations.append(o)
                    next_observations.append(next_o)
                    actions.append(a)
                    rewards.append(r)
                    agent_infos.append(agent_info)
                    env_infos.append(info)

                    o = next_o
                    t += 1

                path = dict(
                    observations      = np.array(observations),
                    next_observations = np.array(next_observations),
                    actions           = np.array(actions),
                    rewards           = np.array(rewards),
                    agent_infos       = stack_tensor_dict_list(agent_infos),
                    env_infos         = stack_tensor_dict_list(env_infos),
                    terminated        = done
                )

                paths.append(path)
                ctr += t

            self.child_conn.send([paths, ctr]) # Return num samples

    def close(self):
        super().close()

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret
