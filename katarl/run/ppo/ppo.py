# -*- coding: utf-8 -*-
'''
@File    : ppo.py
@Time    : 2023/08/25 11:12:24
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 

Experiment commands:
1. "CartPole-v1": 'python run/PPO/ppo.py --train'
2. "Breakout-v4":
'python run/PPO/ppo.py --train --env-name Breakout-v4 --model-name tf_cnn \
    --epsilon 0.1 --actor-N 8 --frames-M 1e7 --step-T 128 --epochs 4 \
    --batch-size 256 --coef-value 0.5 --init-lr 2.5e-4'
'''

if __name__ == '__main__':
    pass

import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import katarl.agents.constants.ppo as const
from katarl.envs.gym_env import GymEnv
from katarl.utils.parser import Parser, str2bool

import wandb
from importlib import import_module

def get_args_and_writer():
    parser = Parser(algo_name='ppo_jax', env_name='CartPole-v1', model_name='mlp_jax')
    # parser = Parser(algo_name='ppo_torch', env_name='CartPole-v1', model_name='mlp_torch')
    # hyperparameters
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of the return")
    parser.add_argument("--coef-gae", type=float, default=const.coef_gae,
        help="the lambda of GAE parameter")
    parser.add_argument("--epsilon", type=float, default=const.epsilon,
        help="the clip epsilon of policy loss")
    parser.add_argument("--v-epsilon", type=float, default=const.v_epsilon,
        help="the clip epsilon of value loss (when --flag-clip-value is taggled)")
    parser.add_argument("--num-envs", type=int, default=const.num_envs,
        help="the number of actors")
    parser.add_argument("--num-steps", type=int, default=const.num_steps,
        help="the move step of each actor")
    parser.add_argument("--total-timesteps", type=int, default=const.total_timesteps,
        help="the number of frames")
    parser.add_argument("--epochs", type=int, default=const.epochs,
        help="the epoch for each training")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the batch size for each training")
    parser.add_argument("--coef-value", type=float, default=const.coef_value,
        help="the coef of value loss")
    parser.add_argument("--coef-entropy", type=float, default=const.coef_entropy,
        help="the coef of entropy loss")
    parser.add_argument("--flag-ad-normal", type=str2bool, default=const.flag_ad_normal, const=True, nargs='?',
        help="if taggled, normalize the advantage value")
    parser.add_argument("--flag-clip-value", type=str2bool, default=const.flag_clip_value, const=True, nargs='?',
        help="if taggled, the delta value will be clipped")
    parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
        help="the initial learning rate")
    parser.add_argument("--flag-anneal-lr", type=str2bool, default=const.flag_anneal_lr, const=True, nargs='?',
        help="if taggled, the learning rate will be annealed")
    parser.add_argument("--max-grad-clip-norm", type=float, default=const.max_grad_clip_norm,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--EPS", type=float, default=const.EPS,
        help="the epsilone size of accuracy error")
    
    args, writer = parser.get_args_and_writer()
    args.data_size = args.num_envs * args.num_steps  # each update datasize
    args.num_iters = (args.total_timesteps - 1) // args.data_size + 1  # total iters
    return args, writer

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    Model = getattr(import_module(f"katarl.agents.models.ppo.{args.model_name}"), "Model")
    env = GymEnv(args=args)
    model = Model(
        name='ppo-model',
        input_shape=env.state_shape,
        output_ndim=env.action_ndim,
        args=args
    )
    Agent = getattr(import_module(f"katarl.agents.{args.algo_name}"), "Agent")
    ppo = Agent(agent_name=args.run_name, env=env, model=model, writer=writer, args=args)
    
    if args.train:
        ppo.train()
    if args.evaluate:
        ppo.evaluate()

    ppo.close()
    wandb.finish()