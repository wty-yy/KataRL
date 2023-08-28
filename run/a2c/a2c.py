import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from agents.a2c import A2C
import agents.constants.a2c as const
from envs.gym_env import GymEnv

import wandb

from importlib import import_module
from utils.parser import Parser

def get_args_and_writer():
    parser = Parser(algo_name='a2c', env_name='CartPole-v1', model_name='tf_mlp')

    parser.add_argument("--episodes", type=int, default=1000,
        help="the total episodes the agent interact with the env")
    parser.add_argument("--learning-rate-v", type=float, default=const.learning_rate_v,
        help="the learning rate of v-value model")
    parser.add_argument("--learning-rate-p", type=float, default=const.learning_rate_p,
        help="the learning rate of policy model")
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of discounted return")
    parser.add_argument("--neg-rewards", type=int, default=-20,
        help="the negative rewards of the env")
    
    return parser.get_args_and_writer()  # open wandb track, if taggled

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    Model = getattr(import_module(f"agents.models.a2c.{args.model_name}"), "Model")
    env = GymEnv(name=args.env_name, capture_video=args.capture_video, neg_rewards=args.neg_rewards)
    
    value_model = Model(
        name='value-model', is_value_model=True,
        load_name=args.load_name, load_id=args.load_id,
        lr=args.learning_rate_v,
        input_shape=env.state_shape, output_ndim=env.action_ndim
    )
    policy_model = Model(
        name='policy-model', is_value_model=False,
        load_name=args.load_name, load_id=args.load_id,
        lr=args.learning_rate_p,
        input_shape=env.state_shape, output_ndim=env.action_ndim
    )
    a2c = A2C(
        agent_name=args.run_name, env=env, 
        value_model=value_model, policy_model=policy_model,
        writer=writer, **vars(args)
    )
    if args.train:
        a2c.train()
    if args.evaluate:
        a2c.evaluate()

    a2c.close()
    wandb.finish()