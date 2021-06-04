import os
import os.path as osp
import shutil
import random
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from milo.utils import init_logger

# =======================
# ==== Dataset Utils ====
# =======================
def get_paths_mjrl_subsample(db_path, num_trajs, subsample_freq=8, concat=False, idx=None, randomize=False):
    """
    This converts our saved db -> mjrl paths with keys 'observations' and 'actions'.
    Used to create paths that can be fed into BC.
    """
    saved_db = torch.load(db_path)
    if idx is None and not randomize:
        saved_db = saved_db[:num_trajs]
    if idx is not None:
        saved_db = [saved_db[idx]]
    start_idxs = torch.randint(0, subsample_freq, size=(len(saved_db),)).long().numpy()
    paths = []
    for start_idx, traj in zip(start_idxs, saved_db):
        state, action, _, _, _ = traj['episode']
        paths.append({'observations': state[start_idx::subsample_freq], 'actions': action[start_idx::subsample_freq]})
    if concat:
        ret = dict(
            observations=np.concatenate([traj['observations'] for traj in paths], axis=0),
            actions=np.concatenate([traj['actions'] for traj in paths], axis=0)
        )
        paths = [ret]
    return paths

def get_paths_mjrl_randomize(db_path, num_samples=100):
    """
    This converts our saved db -> mjrl paths with keys 'observations' and 'actions'.
    Used to create paths that can be fed into BC.
    """
    saved_db = torch.load(db_path)
    paths = []
    for traj in saved_db:
        state, action, _, _, _ = traj['episode']
        paths.append({'observations': state, 'actions': action})
    observations=np.concatenate([traj['observations'] for traj in paths], axis=0)
    actions=np.concatenate([traj['actions'] for traj in paths], axis=0)
    idx = torch.randint(0, observations.shape[0], size=(num_samples,)).long().numpy()
    ret = dict(
        observations=observations[idx],
        actions=actions[idx]
    )
    return [ret]

def get_paths_mjrl(db_path, num_trajs, idx=None):
    """
    This converts our saved db -> mjrl paths with keys 'observations' and 'actions'.
    Used to create paths that can be fed into BC.
    """
    saved_db = torch.load(db_path)
    if num_trajs is 'all':
        num_trajs = len(saved_db)
    saved_db = saved_db[:num_trajs]
    if idx is not None:
        saved_db = [saved_db[idx]]
    paths = []
    for traj in saved_db:
        state, action, _, _, _ = traj['episode']
        paths.append({'observations': state, 'actions': action})

    return paths

def get_db_mjrl_subsample(db_path, num_trajs, subsample_freq=8, idx=None):
    """
    This converts our saved db into torch tensors for state, action, next_state
    """
    saved_db = torch.load(db_path)
    if idx is None:
        saved_db = saved_db[:num_trajs]
    else:
        saved_db = [saved_db[idx]]
    start_idxs = torch.randint(0, subsample_freq, size=(len(saved_db),)).long().numpy()
    states, actions, next_states, dones, rewards, total_reward = [], [], [], [], [], 0
    for start_idx, traj in zip(start_idxs, saved_db):
        state, action, next_state, reward, done = traj['episode']
        states.append(state[start_idx::subsample_freq])
        actions.append(action[start_idx::subsample_freq])
        next_states.append(next_state[start_idx::subsample_freq])
        rewards.append(reward[start_idx::subsample_freq])
        dones.append(done[start_idx::subsample_freq])
        total_reward += traj['ep_rew']
    states = torch.from_numpy(np.concatenate(states, axis=0)).float()
    actions = torch.from_numpy(np.concatenate(actions, axis=0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, axis=0)).float()

    db = (states, actions, next_states)

    print('DB # Samples: {}'.format(db[0].shape[0]))
    return db

def get_db_mjrl_randomize(db_path, num_samples=100):
    saved_db = torch.load(db_path)
    states, actions, next_states, dones, rewards, total_reward = [], [], [], [], [], 0
    for traj in saved_db:
        state, action, next_state, reward, done = traj['episode']
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        total_reward += traj['ep_rew']
    states = torch.from_numpy(np.concatenate(states, axis=0)).float()
    actions = torch.from_numpy(np.concatenate(actions, axis=0)).float()
    next_states = torch.from_numpy(np.concatenate(next_states, axis=0)).float()
    idx = torch.randint(0, states.shape[0], size=(num_samples,)).long()

    db = (states[idx], actions[idx], next_states[idx])
    print('DB # Samples: {}'.format(db[0].shape[0]))
    return db

def get_db_mjrl(db_path, num_trajs, idx=None):
    """
    This converts our saved db into torch tensors for state, action, next_state
    """
    saved_db = torch.load(db_path)
    if num_trajs is 'all':
        num_trajs = len(saved_db)
    saved_db = saved_db[:num_trajs]
    if idx is not None:
        saved_db = saved_db[idx]
    states, actions, next_states, dones, rewards, total_reward = [], [], [], [], [], 0
    for traj in saved_db:
        state, action, next_state, reward, done = traj['episode']
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        total_reward += traj['ep_rew']
    mean_db_reward = total_reward/num_trajs
    db = (torch.from_numpy(np.concatenate(states, axis=0)).float(),
          torch.from_numpy(np.concatenate(actions, axis=0)).float(),
          torch.from_numpy(np.concatenate(next_states, axis=0)).float())

    print('DB Mean Reward: {} | DB # Samples: {}'.format(mean_db_reward, db[0].shape[0]))
    return db

# ========================
# === Evaluation Utils ===
# ========================

def save_checkpoint(dirs, agent, cost_function, tag, agent_type='trpo'):
    save_dir = dirs['checkpoints_dir']
    checkpoint = {'policy_params': agent.policy.get_param_values(),
                  'old_policy_params': agent.policy.old_params,
                  'baseline_params': agent.baseline.model.state_dict(),
                  'baseline_optim': agent.baseline.optimizer.state_dict(),
                  'w': cost_function.w,
                  'rff': cost_function.rff.state_dict()}
    if agent_type == 'ppo':
        checkpoint['policy_optim'] = agent.optimizer.state_dict()
    torch.save(checkpoint, osp.join(save_dir, f'checkpoint_{tag}.pt'))

def save_and_plot(n_iter, args, dirs, scores, mmds):
    mb_step = n_iter*args.samples_per_step*(args.pg_iter+1)
    scores_path = osp.join(dirs['data_dir'], 'scores.pt')
    mmds_path = osp.join(dirs['data_dir'], 'mmds.pt')
    # Scores
    if not osp.exists(scores_path):
        saved_scores ={'greedy' : [scores['greedy']],
                       'sample' : [scores['sample']],
                       'mb_step': [mb_step]}
    else:
        saved_scores = torch.load(scores_path, map_location=torch.device('cpu'))
        saved_scores['greedy'].append(scores['greedy'])
        saved_scores['sample'].append(scores['sample'])
        saved_scores['mb_step'].append(mb_step)
    torch.save(saved_scores, scores_path)
    # MMDS
    if not osp.exists(mmds_path):
        saved_mmds = {'greedy' : [mmds['greedy']],
                      'sample' : [mmds['sample']],
                      'mb_step': [mb_step]}
    else:
        saved_mmds = torch.load(mmds_path, map_location=torch.device('cpu'))
        saved_mmds['greedy'].append(mmds['greedy'])
        saved_mmds['sample'].append(mmds['sample'])
        saved_mmds['mb_step'].append(mb_step)
    torch.save(saved_mmds, mmds_path)

    # Plots
    plot_data(dirs, saved_scores, labels={'ylabel': 'Score', 'title': 'Performance', 'figname': 'scores.png'})
    plot_data(dirs, saved_mmds, labels={'ylabel': 'MMD', 'title': 'MMD', 'figname': 'mmds.png'})

def plot_data(dirs, data, labels, x_type='iter'):
    y = np.array(data['greedy'])
    if x_type == 'iter':
        x = np.arange(y.shape[0])
    elif x_type == 'timestep':
        x = np.array(data['mb_step'])

    plt.plot(x, y)
    plt.xlabel(x_type)
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    plt.savefig(osp.join(dirs['plots_dir'], labels['figname']))

# =======================
# ======== Utils ========
# =======================

def set_global_seeds(seed):
    """
    Sets global seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup(args, ask_prompt=True):
    """
    Organizes the output of experiments like so:

    |- root_dir
    |   |- experiment_id
    |   |   |-logs
    |   |   |-tensorboard_logs
    |   |   |-data
    |   |   |-checkpoints
    |   |   |-plots
    |   |- dynamics_model_weights
    |   |   |-dynamics_id.pt

    - Experiment id would be unique for each experiment that we run
    - Dynamics model weights will be saved in unique dynamics model id dirs
    - logs contain main.log which saved debugging, info, warning logs from stderr
    - tensorboard_logs will store the tensorbaord event
    - data will store performance/mmd
    - plots will save performance and mmd plots
    - checkpoints will save alg checkpoints. (i.e. policy weights, reward params, rff params)
    """
    ensemble_checkpoint = None

    # Get IDs
    exp_id = get_experiment_id(args)
    dynamics_id = get_dynamics_id(args)

    print("Creating Output Directories for:")
    print(f"Experiment: {exp_id}")
    print(f"Dynamics: {dynamics_id}")

    # Create output dir
    root_dir = args.root_path

    # Dynamics path
    dynamics_dir = osp.join(root_dir, 'dynamics_model_weights')

    if not osp.isdir(dynamics_dir):
        print(">>>>>> Dynamics Directory not there. Creating...")
        os.makedirs(dynamics_dir)

    # Check for dynamics weights
    dynamics_checkpoint = osp.join(dynamics_dir, dynamics_id+'.pt')
    if osp.exists(dynamics_checkpoint):
        ensemble_checkpoint = dynamics_checkpoint

    # Experiments Path
    exp_dir = osp.join(root_dir, exp_id)
    logs_dir = osp.join(exp_dir, 'logs')
    tensorboard_dir = osp.join(exp_dir, 'tensorboard_logs')
    data_dir = osp.join(exp_dir, 'data')
    checkpoints_dir = osp.join(exp_dir, 'checkpoints')
    plots_dir = osp.join(exp_dir, 'plots')

    if osp.isdir(exp_dir):
        delete = True
        if ask_prompt:
            while True:
                reply = input("Experiment ID already exists, delete existing directory? (y/n) ").strip().lower()
                if reply not in ['y', 'n', 'yes', 'no']:
                    print('Please reply with y, n, yes, or no')
                else:
                    delete = False if reply in ['n', 'no'] else True
                    break
        if delete:
            print('Deleting existing, duplicate experiment id')
            shutil.rmtree(exp_dir)


    if not osp.isdir(exp_dir):
        print(">>>>>> Experiment Directory not there. Creating...")
        os.makedirs(exp_dir)
        os.mkdir(logs_dir)
        os.mkdir(tensorboard_dir)
        os.mkdir(data_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(plots_dir)

    # Setup Global seeds/threads/devices
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') # TODO: Hardcoded to cpu for now
    torch.set_num_threads(1)
    set_global_seeds(args.seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Setup Loggers
    logger = init_logger(logs_dir)
    writer = SummaryWriter(tensorboard_dir)

    dirs = {'data_dir': data_dir,
            'checkpoints_dir': checkpoints_dir,
            'plots_dir': plots_dir,
            'dynamics_path': dynamics_checkpoint}

    ids = {'experiment_id': exp_id, 'dynamics_id': dynamics_id}

    # Store Arguments
    log_arguments(args, logger)

    return dirs, ids, ensemble_checkpoint, logger, writer, device

def get_dynamics_id(args):
    """
    Creates an id for dynamics models. Used for Saving/Loading
    """
    env = args.env
    seed = args.seed
    n_models = args.n_models
    return f"{env}-seed={seed}-num_models={n_models}"

def get_experiment_id(args):
    """
    Creates an experiment id
    """
    env = args.env
    seed = args.seed
    lambda_b = args.lambda_b
    experiment_id = args.id
    return f"{env}-seed={seed}-lambda={lambda_b}-id={experiment_id}"

def log_arguments(args, logger):
    """
    Adds arguments used for experiment in logger
    """
    headers = ['Args', 'Value']
    table = tabulate(list(vars(args).items()), headers=headers, tablefmt='pretty')
    logger.info(">>>>> Experiment Running with Arguments >>>>>")
    logger.info("\n"+table)
