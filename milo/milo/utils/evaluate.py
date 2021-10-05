import torch
import numpy as np

from mjrl.samplers.core import sample_paths

# ========================
# === Evaluation Utils ===
# ========================

def evaluate(n_iter, logger, writer, args, env, policy, reward_func, num_traj=10, adroit=False):
    greedy_samples = sample_paths(num_traj=num_traj, env=env, policy=policy, \
                        num_cpu=args.num_cpu, base_seed=args.seed, eval_mode=True, suppress_print=True)
    samples = sample_paths(num_traj=num_traj, env=env, policy=policy, \
                        num_cpu=args.num_cpu, base_seed=args.seed, eval_mode=False, suppress_print=True)

    if adroit:
        greedy_success = env.evaluate_success(greedy_samples)
        sample_success = env.evaluate_success(samples)

    # Compute scores
    greedy_scores = np.array([np.sum(traj['rewards']) for traj in greedy_samples])
    sample_scores = np.array([np.sum(traj['rewards']) for traj in samples])
    greedy_mean_lengths = np.mean([len(traj['rewards']) for traj in greedy_samples])
    sample_mean_lengths = np.mean([len(traj['rewards']) for traj in samples])
    greedy_mean, greedy_max, greedy_min = greedy_scores.mean(), greedy_scores.max(), greedy_scores.min()
    sample_mean, sample_max, sample_min = sample_scores.mean(), sample_scores.max(), sample_scores.min()

    # Compute MMD (S, A)
    greedy_x = np.concatenate([np.concatenate([traj['observations'], traj['actions']], axis=1) for traj in greedy_samples], axis=0)
    sample_x = np.concatenate([np.concatenate([traj['observations'], traj['actions']], axis=1) for traj in samples], axis=0)
    greedy_x = torch.from_numpy(greedy_x).float()
    sample_x = torch.from_numpy(sample_x).float()

    greedy_diff = reward_func.get_rep(greedy_x).mean(0) - reward_func.phi_e
    sample_diff = reward_func.get_rep(sample_x).mean(0) - reward_func.phi_e

    greedy_mmd = torch.dot(greedy_diff, greedy_diff)
    sample_mmd = torch.dot(sample_diff, sample_diff)

    # Log
    logger.info(f'Greedy Evaluation Score mean (min, max): {greedy_mean:.2f} ({greedy_min:.2f}, {greedy_max:.2f})')
    logger.info(f'Greedy Evaluation Trajectory Lengths: {greedy_mean_lengths:.2f}')
    logger.info(f'Greedy MMD: {greedy_mmd}')
    if adroit:
        logger.info(f'Greedy Success %: {greedy_success}%')
    logger.info(f'Sampled Evaluation Score mean (min, max): {sample_mean:.2f} ({sample_min:.2f}, {sample_max:.2f})')
    logger.info(f'Sampled Evaluation Trajectory Lengths: {sample_mean_lengths:.2f}')
    logger.info(f'Sampled MMD: {sample_mmd}')
    if adroit:
        logger.info(f'Sampled Success %: {sample_success}%')

    # Tensorboard Logging
    writer.add_scalars('data/inf_greedy_reward', {'min_score': greedy_min,
                                              'mean_score': greedy_mean,
                                              'max_score': greedy_max}, n_iter+1)
    writer.add_scalar('data/inf_greedy_len', greedy_mean_lengths, n_iter+1)
    writer.add_scalar('data/greedy_mmd', greedy_mmd, n_iter+1)
    writer.add_scalars('data/inf_sampled_reward', {'min_score': sample_min,
                                              'mean_score': sample_mean,
                                              'max_score': sample_max}, n_iter+1)
    writer.add_scalar('data/inf_sampled_len', sample_mean_lengths, n_iter+1)
    writer.add_scalar('data/sampled_mmd', sample_mmd, n_iter+1)
    if adroit:
        writer.add_scalar('data/greedy_success_percen', greedy_success, n_iter+1)
        writer.add_scalar('data/sampled_success_percen', sample_success, n_iter+1)

    scores = {'greedy': greedy_mean, 'sample': sample_mean}
    mmds = {'greedy': greedy_mmd, 'sample': sample_mmd}

    return scores, mmds

