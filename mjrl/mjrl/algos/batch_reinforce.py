"""
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""
import pdb
import logging
#logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog

try:
    from milo.sampler import mb_sampler
except:
    print("MILO not installed")


class BatchREINFORCE:
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=123,
                 desired_kl=None,
                 save_logs=False,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.desired_kl = desired_kl
        if save_logs: self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    def train_step(self, N,
                   env=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   reward_kwargs=None):

        # Clean up input arguments
        env = self.env.env_id if env is None else env
        if sample_mode not in ['trajectories', 'samples', 'model_based']:
        #if sample_mode != 'trajectories' and sample_mode != 'samples':
            print("sample_mode in NPG must be either 'trajectories', 'samples', or 'model_based'")
            quit()

        ts = timer.time()
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_paths(**input_dict)
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, paths_per_call=25, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_data_batch(**input_dict)
        # Model Based sampling
        elif sample_mode == 'model_based':
            input_dict = dict(env=env, policy=self.policy, num_samples=N, base_seed=self.seed,
                              num_workers=num_cpu, paths_per_process=2, verbose=False)
            paths = mb_sampler(**input_dict)

        # Info Logging
        infos = {'int': [], 'ext': [], 'reward': [], 'ep_len': [], \
                 'mb_mmd': None, 'bonus_mmd': None, 'vf_loss_start': None, 'vf_loss_end': None, 'vf_epoch':None}
        infos['ground_truth_reward'] = [np.sum(traj['rewards']) for traj in paths] # Log the true_rewards
        #if np.mean(infos['ground_truth_reward']) > 2000:
            #torch.save(paths, f'./experiments/paths/mb_paths_{np.mean(infos["ground_truth_reward"])}.pt')

        # Injects Custom Reward Function
        if reward_kwargs is not None:
            # FIT REWARD WITH SAMPLES COLLECTED FOR TRPO
            # NOTE: this is (S,A) specific
            cost_input = np.concatenate([np.concatenate([traj['observations'], traj['actions']], axis=1) for traj in paths], axis=0)
            mb_mmd = reward_kwargs['reward_func'].fit_cost(torch.from_numpy(cost_input).float())
            del cost_input
            infos['mb_mmd'] = mb_mmd
            device = reward_kwargs['device']
            for traj in paths:
                states = torch.from_numpy(traj['observations']).float()
                next_states = torch.from_numpy(traj['next_observations']).float()
                actions = torch.from_numpy(traj['actions']).float()
                bonus_cost, cost_info = reward_kwargs['reward_func'].get_bonus_costs(states.to(device), \
                        actions.to(device), reward_kwargs['ensemble'], next_states=next_states.to(device))
                bonus_cost = bonus_cost[:,0]

                # Record values
                intrinsic_sum = -np.sum(cost_info['bonus'][:, 0].numpy())
                extrinsic_sum = -np.sum(cost_info['ipm'][:, 0].numpy())

                infos['int'].append(intrinsic_sum)
                infos['ext'].append(extrinsic_sum)
                infos['reward'].append(extrinsic_sum + intrinsic_sum)
                infos['ep_len'].append(len(traj['rewards']))

                # Replace true rewards with our rewards
                traj['rewards'] = -1.0 * bonus_cost.cpu().numpy()
            infos['bonus_mmd'] = np.concatenate([-1.0*traj['rewards'] for traj in paths], axis=0).mean() - \
                reward_kwargs['reward_func'].get_expert_cost()

        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # fit baseline
        error_before, error_after, epoch_losses = self.baseline.fit(paths, return_errors=True,\
                return_all_errors=True)
        if self.save_logs:
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        infos['vf_loss_start'] = error_before
        infos['vf_loss_end'] = error_after
        infos['vf_epoch'] = epoch_losses

        # Add all the return info to our stats
        eval_statistics.append(infos)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update with linesearch
        # ------------------------------
        if self.desired_kl is not None:
            max_ctr = 100
            alpha = self.alpha
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params, set_new=True, set_old=False)
                kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
                if kl_dist <= self.desired_kl:
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else \
                        0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score


    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)
