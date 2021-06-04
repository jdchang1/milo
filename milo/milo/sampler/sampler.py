import time
from copy import deepcopy
from milo.gym_env import MujocoEnvProcess
from torch.multiprocessing import Pipe

def mb_sampler(env,
               policy,
               num_samples,
               base_seed,
               eval_mode=False,
               num_workers=4,
               paths_per_process=13,
               verbose=False):
    """
    Multiprocess sampler for model-based rollouts. Note, this is only meant for CPU usage.
    """

    # Create Pipes and spawn jobs
    jobs, parent_conns, child_conns = [], [], []
    for idx in range(num_workers):
        parent_conn, child_conn = Pipe()
        seed = 12345+base_seed*idx
        job = MujocoEnvProcess(env, child_conn, seed, eval_mode=eval_mode, paths_per_process=paths_per_process)
        job.start()
        jobs.append(job)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # Run Jobs
    start_time = time.time()
    all_paths, curr_samples = [], 0
    while curr_samples < num_samples:
        for parent_conn in parent_conns:
            parent_conn.send(deepcopy(policy))
        for parent_conn in parent_conns:
            paths, ctr = parent_conn.recv()
            all_paths.extend(paths)
            curr_samples += ctr
    if verbose:
        print(f"Collected {curr_samples} samples and {len(all_paths)} trajectories <<<<<< took {time.time()-start_time} seconds")

    for job in jobs:
        job.terminate()

    return all_paths
