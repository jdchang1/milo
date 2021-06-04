import argparse


def get_args():
    # ====== Argument Parser ======
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Logging/Environment Arguments
    parser.add_argument('--env', type=str,
                        help='environment ID', default='Hopper-v6')
    parser.add_argument('--seed', type=int, help='seed', default=100)
    parser.add_argument('--num_cpu', type=int,
                        help='number of processes used for inference', default=4)
    parser.add_argument('--num_trajs', type=int,
                        help='number of expert trajs', default=10)
    parser.add_argument('--num_samples', type=int,
                        help='number of expert samples', default=500)
    parser.add_argument('--subsample_freq', type=int,
                        help='subsample frequency', default=8)
    parser.add_argument('--norm_thresh_coeff', type=float,
                        help='Norm threshold', default=2)
    parser.add_argument('--include_expert', action='store_true',
                        help='include expert data into offline db', default=False)
    parser.add_argument('--subsample_expert', action='store_true',
                        help='subsample expert samples', default=False)
    parser.add_argument('--randomize_expert', action='store_true',
                        help='randomize expert samples', default=False)
    parser.add_argument('--save_iter', type=int,
                        help='Interval to Save checkpoints', default=10)

    # Path Arguments
    parser.add_argument('--root_path', type=str,
                        help='Root dir to save outputs', default='./experiments')
    parser.add_argument('--data_path', type=str,
                        help='Root data dir to get db', default='./data')
    parser.add_argument('--expert_db', type=str,
                        help='expert db name', default='Hopper-v6_100_3012.62.pt')
    parser.add_argument('--offline_db', type=str,
                        help='offline db name', default='Hopper-v6_100_3025.47.pt')
    parser.add_argument('--model_save_path', type=str,  help='Path to save models',
                        default='./experiments/dynamics_model_weights')
    parser.add_argument('--id', type=int,  help='Experiment id', default=0)

    # Dynamics Model Ensemble Arguments
    parser.add_argument('--n_models', type=int,
                        help='Number of dynamics models in ensemble', default=4)
    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs to train models', default=5)
    parser.add_argument('--grad_clip', type=float,
                        help='Max Gradient Norm', default=1.0)
    parser.add_argument('--dynamics_optim', type=str,
                        help='Optimizer to use [sgd, adam]', default='sgd')

    # Cost Arguments
    parser.add_argument('--feature_dim', type=int,
                        help='Feature dimension', default=512)
    parser.add_argument('--update_type', type=str,
                        help='exact, geometric, decay, decay_sqrt, ficticious', default='exact')
    parser.add_argument('--bw_quantile', type=float,
                        help='Quantile when fitting bandwidth', default=0.2)
    parser.add_argument('--lambda_b', type=float,
                        help='Bonus/Penalty weighting param', default=0.1)
    parser.add_argument('--cost_lr', type=float,
                        help='0.0 is exact update, otherwise learning rate', default=0.0)

    # Policy Gradient Arguments
    parser.add_argument('--planner', type=str,
                        help='pg alg to use (trpo, ppo)', default='trpo')
    parser.add_argument('--actor_model_hidden', type=int,
                        nargs='+', help='hidden dims for actor', default=[32, 32])
    parser.add_argument('--critic_model_hidden', type=int, nargs='+',
                        help='hidden dims for critic', default=[128, 128])
    parser.add_argument('--gamma', type=float,
                        help='discount factor for rewards (default: 0.99)', default=0.995)
    parser.add_argument('--gae_lambda', type=float,
                        help='gae lambda val', default=0.97)
    parser.add_argument('--samples_per_step', type=int,
                        help='Number of mb samples per pg step', default=512)
    parser.add_argument('--policy_init_log', type=float,
                        help='policy init log', default=-0.25)
    parser.add_argument('--policy_min_log', type=float,
                        help='policy min log', default=-2.0)
    parser.add_argument('--vf_iters', type=int,
                        help='Number of value optim steps', default=2)
    parser.add_argument('--vf_batch_size', type=int,
                        help='Critic batch size', default=64)
    parser.add_argument('--vf_lr', type=float, help='Value lr', default=1e-3)
    parser.add_argument('--vf_reg_coef', type=float,
                        help='baseline regularization coeff', default=1e-3)

    # BC regularization Arguments
    parser.add_argument('--do_bc_reg', action='store_true', help='Add bc regularization to policy gradient', default=False)
    parser.add_argument('--bc_reg_coeff', type=float, help='Regularization coefficient for policy gradient', default=0.1)

    # TRPO Arguments
    parser.add_argument('--cg_iter', type=int,
                        help='Number of CG iterations', default=10)
    parser.add_argument('--cg_damping', type=float,
                        help='CG damping coefficient', default=1e-4)
    parser.add_argument('--kl_dist', type=float,
                        help='Trust region', default=0.05)
    parser.add_argument('--hvp_sample_frac', type=float,
                        help='Fraction of samples for FIM', default=1.0)

    # PPO Arguments
    parser.add_argument('--clip_coef', type=float,
                        help='Clip Coefficient for PPO Trust region', default=0.2)
    parser.add_argument('--ppo_lr', type=float,
                        help='PPO learning rate', default=3e-4)
    parser.add_argument('--ppo_epochs', type=int,
                        help='Epochs per PPO step', default=10)
    parser.add_argument('--ppo_batch_size', type=int,
                        help='Mini-batch size for PPO', default=64)

    # BC Arguments
    parser.add_argument('--bc_epochs', type=int,
                        help='Number of BC epochs', default=3)
    parser.add_argument('--n_bc_iters', type=int, default=10,
                        help='number of times to run BC iterations')

    # General Algorithm Arguments
    parser.add_argument('--n_iter', type=int, help='Number of offline IL iterations to run', default=300)
    parser.add_argument('--pg_iter', type=int, help='Number of pg steps', default=5)
    parser.add_argument('--use_ground_truth', action='store_true', help='use ground truth rewards', default=False)
    parser.add_argument('--do_model_free', action='store_true', help='do model free policy gradient', default=False)

    args = parser.parse_args()
    return args
