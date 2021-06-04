#!/bin/bash

python run.py --env Hopper-v4 \
              --seed 100 \
              --expert_db Hopper-v6_expert.pt \
              --offline_db Hopper-v6_offline.pt \
              --n_models 4 \
              --lambda_b 0.0025 \
              --samples_per_step 40000 \
              --pg_iter 1 \
              --bw_quantile 0.1 \
              --id 1 \
              --subsample_expert \
              --n_iter 300 \
              --cg_iter 25 \
              --bc_epochs 1 \
              --do_bc_reg \
              --bc_reg_coeff 0.1
