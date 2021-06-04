import os
import gym
import time
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths, sample_data_batch

import milo.gym_env
from milo.utils import *
from milo.gym_env import model_based_env
from milo.dataset import OfflineDataset
from milo.cost import RBFLinearCost
from milo.dynamics_model import DynamicsEnsemble

def main():
    args = get_args()
    dirs, ids, ensemble_checkpoint, logger, writer, device = setup(args, ask_prompt=True)

    # ======== Dataset Setup ==========
    offline_db_path = os.path.join(args.data_path, 'offline_data', args.offline_db)
    expert_db_path = os.path.join(args.data_path, 'expert_data', args.expert_db)
    offline_state, offline_action, offline_next_state = get_db_mjrl(offline_db_path, 'all')
    if args.subsample_expert:
        expert_state, expert_action, expert_next_state = get_db_mjrl_randomize(expert_db_path, args.num_samples) # Expert DB
    else:
        expert_state, expert_action, expert_next_state = get_db_mjrl(expert_db_path, args.num_trajs) # Expert DB
    
    offline_dataset = OfflineDataset(args.env, offline_state, offline_action, offline_next_state, device=device)

    # Compute Expert DB stats
    max_expert_norm = torch.max(torch.norm(expert_state, p=2, dim=1))

    # ========= Create Model Ensemble =========
    model_ensemble = DynamicsEnsemble(args.env, args.n_models, offline_dataset, hidden_sizes=[1024, 1024], base_seed=args.seed)
    if ensemble_checkpoint is not None:
        logger.info(f">>>>> Loading Dynamics ensemble with id: {ids['dynamics_id']}")
        model_ensemble.load(ensemble_checkpoint)
    else:
        logger.info(f">>>>> Training Dynamics ensemble with id: {ids['dynamics_id']}")
        model_ensemble.train(n_epochs=args.n_epochs, logger=logger, log_epoch=False, grad_clip=args.grad_clip)
        logger.info(f">>>>> Saving ensemble weights in {dirs['dynamics_path']}")
        model_ensemble.save(dirs['dynamics_path'])
    model_ensemble.compute_threshold()
    logger.info(f">>>>> Computed Maximum Discrepancy for Ensemble: {model_ensemble.threshold}")

    # ======== ENV SETUP ========
    logger.info(">>>>> Creating Environments")
    inf_env = GymEnv(gym.make(args.env))
    mb_env = GymEnv(model_based_env(gym.make(args.env), model_ensemble, init_state_buffer=expert_state.numpy(),\
                                    norm_thresh = args.norm_thresh_coeff*max_expert_norm, device=device))

    # ====== Cost Setup =======
    cost_function = RBFLinearCost(torch.cat([expert_state, expert_action], dim=1), feature_dim=args.feature_dim, \
            bw_quantile=args.bw_quantile, lambda_b=args.lambda_b, seed=args.seed)

    # ============= INIT AGENT =============
    policy = MLP(inf_env.spec, hidden_sizes=tuple(args.actor_model_hidden), seed=args.seed,
                 init_log_std=args.policy_init_log, min_log_std=args.policy_min_log)
    baseline = MLPBaseline(inf_env.spec, reg_coef=args.vf_reg_coef, batch_size=args.vf_batch_size, \
                           hidden_sizes=tuple(args.critic_model_hidden), epochs=args.vf_iters, learn_rate=args.vf_lr)

    # =============== BC Warmstart =================
    if args.bc_epochs > 0:
        logger.info(f">>>>> BC Warmstart for {args.bc_epochs} epochs")
        offline_paths = get_paths_mjrl(offline_db_path, 'all')
        bc_agent = BC(offline_paths, policy=policy, epochs=args.bc_epochs, batch_size=64, lr=1e-3)
        bc_agent.train()

        # Reinit Policy Std
        policy_params = policy.get_param_values()
        action_dim = inf_env.env.action_space.shape[0]
        policy_params[-1*action_dim:] = args.policy_init_log
        policy.set_param_values(policy_params, set_new=True, set_old=True)

    # ============== Policy Gradient Init =============
    if args.subsample_expert:
        expert_paths = get_paths_mjrl_randomize(expert_db_path, args.num_samples)
    else:
        expert_paths = get_paths_mjrl(expert_db_path, args.num_trajs)
    bc_reg_args = {'flag': args.do_bc_reg, 'reg_coeff': args.bc_reg_coeff, 'expert_paths': expert_paths[0]}
    if args.planner == 'trpo':
        cg_args = {'iters': args.cg_iter, 'damping': args.cg_damping}
        planner_agent = NPG(mb_env, policy, baseline, normalized_step_size=args.kl_dist, \
                    hvp_sample_frac=args.hvp_sample_frac, seed=args.seed, FIM_invert_args=cg_args, \
                    bc_args=bc_reg_args, save_logs=True)
    else:
        raise NotImplementedError('Chosen Planner not yet supported')

    # ==============================================
    # ============== MAIN LOOP START ===============
    # ==============================================

    n_iter = 0
    best_policy_score = -float('inf')
    greedy_scores, sample_scores, greedy_mmds, sample_mmds = [], [], [], []
    while n_iter<args.n_iter:
        logger.info(f"{'='*10} Main Episode {n_iter+1} {'='*10}")
        # ============= Evaluate, Save, Plot ===============
        scores, mmds = evaluate(n_iter, logger, writer, args, inf_env, \
                                planner_agent.policy, cost_function, num_traj=50)
        save_and_plot(n_iter, args, dirs, scores, mmds)

        if scores['greedy'] > best_policy_score:
            best_policy_score = scores['greedy']
            save_checkpoint(dirs, planner_agent, cost_function, 'best', agent_type=args.planner)

        if (n_iter+1) % args.save_iter == 0:
            save_checkpoint(dirs, planner_agent, cost_function, n_iter+1, agent_type=args.planner)

        # =============== DO PG STEPS =================
        logger.info('=== PG Planning Start ===')
        best_baseline_optim, best_baseline = None, None
        best_policy = None
        curr_max_reward, curr_min_vloss = -float('inf'), float('inf')
        for i in range(args.pg_iter):
            reward_kwargs = dict(reward_func=cost_function, ensemble=model_ensemble, device=device)
            planner_args = dict(N=args.samples_per_step, env=mb_env, sample_mode='model_based', \
                                gamma=args.gamma, gae_lambda=args.gae_lambda, num_cpu=4, \
                                reward_kwargs=reward_kwargs)

            r_mean, r_std, r_min, r_max, _, infos  = planner_agent.train_step(**planner_args)
            
            # Baseline Heuristic
            if infos['vf_loss_end'] < curr_min_vloss:
                curr_min_vloss = infos['vf_loss_end']
                best_baseline = planner_agent.baseline.model.state_dict()
                best_baseline_optim = planner_agent.baseline.optimizer.state_dict()

            # Stderr Logging
            reward_mean = np.array(infos['reward']).mean()
            int_mean = np.array(infos['int']).mean()
            ext_mean = np.array(infos['ext']).mean()
            len_mean = np.array(infos['ep_len']).mean()
            ground_truth_mean = np.array(infos['ground_truth_reward']).mean()
            logger.info(f'Model MMD: {infos["mb_mmd"]}')
            logger.info(f'Bonus MMD: {infos["bonus_mmd"]}')
            logger.info(f'Model Ground Truth Reward: {ground_truth_mean}')
            logger.info('PG Iteration {} reward | int | ext | ep_len ---- {:.2f} | {:.2f} | {:.2f} | {:.2f}' \
                        .format(i+1, reward_mean, int_mean, ext_mean, len_mean))

            # Tensorboard Logging
            step_count = n_iter*args.pg_iter + i
            writer.add_scalar('data/reward_mean', reward_mean, step_count)
            writer.add_scalar('data/ext_reward_mean', ext_mean, step_count)
            writer.add_scalar('data/int_reward_mean', int_mean, step_count)
            writer.add_scalar('data/ep_len_mean', len_mean, step_count)
            writer.add_scalar('data/true_reward_mean', ground_truth_mean, step_count)
            writer.add_scalar('data/value_loss', infos['vf_loss_end'], step_count)
            writer.add_scalar('data/mb_mmd', infos['mb_mmd'], step_count)
            writer.add_scalar('data/bonus_mmd', infos['bonus_mmd'], step_count)

        planner_agent.baseline.model.load_state_dict(best_baseline)
        planner_agent.baseline.optimizer.load_state_dict(best_baseline_optim)
        n_iter += 1

if __name__ == '__main__':
    main()
