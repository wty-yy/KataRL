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

import wandb
from agents.PPO import PPO
import agents.constants.PPO.cartpole as const
from envs.gym_env import GymEnv
from utils.parser import Parser, str2bool
from importlib import import_module

def get_args_and_writer():
    parser = Parser(algo_name='ppo', env_name='CartPole-v1', model_name='tf_mlp')
    # hyperparameters
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of the return")
    parser.add_argument("--lambda", type=float, default=const.lambda_,
        help="the lambda of GAE parameter")
    parser.add_argument("--epsilon", type=float, default=const.epsilon,
        help="the clip epsilon of policy loss")
    parser.add_argument("--v-epsilon", type=float, default=const.v_epsilon,
        help="the clip epsilon of value loss (when --flag-clip-value is taggled)")
    parser.add_argument("--actor-N", type=int, default=const.actor_N,
        help="the number of actors")
    parser.add_argument("--frames-M", type=int, default=const.frames_M,
        help="the number of frames")
    parser.add_argument("--step-T", type=int, default=const.step_T,
        help="the move step of each actor")
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
    parser.add_argument("--init-lr", type=float, default=const.init_lr,
        help="the initial learning rate")
    parser.add_argument("--flag-anneal-lr", type=str2bool, default=const.flag_anneal_lr, const=True, nargs='?',
        help="if taggled, the learning rate will be annealed")
    
    args, writer = parser.get_args_and_writer()
    args.data_size = args.actor_N * args.step_T  # each update datasize
    args.iter_nums = (args.frames_M - 1) // args.data_size + 1  # total iters
    return args, writer

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    Model = getattr(import_module(f"agents.models.PPO.{args.model_name}"), "Model")
    env = GymEnv(
        name=args.env_name, seed=args.seed,
        num_envs=args.actor_N,
        capture_video=args.capture_video
    )
    model = Model(
        name='ppo-model',
        load_name=args.load_name, load_id=args.load_id,
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args
    )
    ppo = PPO(
        agent_name=args.run_name,
        env=env, model=model, writer=writer,
        **vars(args)
    )
    
    if args.train:
        ppo.train()
    if args.evaluate:
        ppo.evaluate()

    ppo.close()
    wandb.finish()