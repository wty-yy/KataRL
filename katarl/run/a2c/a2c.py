import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import katarl.agents.constants.a2c as const
from katarl.envs.gym_env import GymEnv
from katarl.utils.parser import Parser, str2bool

import wandb

from importlib import import_module

def get_args_and_writer():
    parser = Parser(algo_name='a2c_jax', env_name='CartPole-v1', model_name='mlp_jax')

    parser.add_argument("--total-timesteps", type=int, default=const.total_timesteps,
        help="the total timesteps the agent interact with the env")
    parser.add_argument("--learning-rate-v", type=float, default=const.learning_rate_v,
        help="the learning rate of v-value model")
    parser.add_argument("--learning-rate-p", type=float, default=const.learning_rate_p,
        help="the learning rate of policy model")
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of discounted return")
    parser.add_argument("--neg-rewards", type=int, default=const.neg_rewards,
        help="the negative rewards of the env")
    parser.add_argument("--write-logs-frequency", type=int, default=const.write_logs_frequency,
        help="the frequency of writing the logs to tensorboard")
    parser.add_argument("--anneal-lr", type=str2bool, default=const.anneal_lr, const=True, nargs='?',
        help="if taggled, the learning rate will be linear anneal by the total timesteps")
    parser.add_argument("--target-model-update-frequency", type=int, default=const.target_model_update_frequency,
        help="the frequency of the target model update")
    parser.add_argument("--tau", type=float, default=const.tau,
        help="the ratio of the current model and the target model")
    
    return parser.get_args_and_writer()  # open wandb track, if taggled

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    Model = getattr(import_module(f"katarl.agents.models.a2c.{args.model_name}"), "Model")
    env = GymEnv(args=args)
    
    value_model = Model(
        name='value-model', is_value_model=True,
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args
    )
    policy_model = Model(
        name='policy-model', is_value_model=False,
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args
    )

    Agent = getattr(import_module(f"katarl.agents.{args.algo_name}"), "Agent")
    a2c = Agent(
        agent_name=args.run_name, env=env, 
        value_model=value_model, policy_model=policy_model,
        writer=writer, args=args
    )
    if args.train:
        a2c.train()
    if args.evaluate:
        a2c.evaluate()

    a2c.close()
    wandb.finish()