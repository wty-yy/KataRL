import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from katarl.agents.dqn_tf import DQN
import katarl.agents.constants.dqn.dqn as const
from katarl.envs.gym_env import GymEnv
import wandb

import importlib
from katarl.utils.parser import Parser

def get_args_and_writer():
    parser = Parser(algo_name='dqn', env_name='CartPole-v1', model_name='mlp_tf')

    parser.add_argument("--total-timesteps", type=int, default=const.total_timesteps,
        help="the total timesteps the agents interact with the env")
    parser.add_argument("--num-envs", type=int, default=const.num_envs,
        help="the number of parallel envs")
    parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
        help="the learning rate of q-value model")
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of discounted return")
    parser.add_argument("--epsilon-max", type=float, default=const.epsilon_max,
        help="the max epsilon-choose's probability decay")
    parser.add_argument("--epsilon-min", type=float, default=const.epsilon_min,
        help="the min epsilon-choose's probability decay")
    parser.add_argument("--exporation-proportion", type=float, default=const.exporation_proportion,
        help="the proportion of exporation in the 'total_timesteps'")
    parser.add_argument("--memory-size", type=int, default=const.memory_size,
        help="the size of memory buffer")
    parser.add_argument("--start-fit-size", type=int, default=const.start_fit_size,
        help="the size of memory cache when start training the q-value model")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the sample batch size from memory cache")
    parser.add_argument("--train-frequency", type=int, default=const.train_frequency,
        help="the training frequency in total timesteps")
    parser.add_argument("--write-logs-frequency", type=int, default=const.write_logs_frequency,
        help="the frequency of writing the logs to tensorboard")
    
    return parser.get_args_and_writer()


if __name__ == '__main__':
    args, writer = get_args_and_writer()

    Model = getattr(importlib.import_module(f"agents.models.dqn.{args.model_name}"), "Model")
    env = GymEnv(
        name=args.env_name, seed=args.seed,
        num_envs=args.num_envs,
        capture_video=args.capture_video
    )
    model = Model(
        seed=args.seed, lr=args.learning_rate,
        load_name=args.load_name, load_id=args.load_id,
        input_shape=env.state_shape, output_ndim=env.action_ndim
    )
    dqn = DQN(
        agent_name=args.run_name, env=env, model=model, writer=writer, **vars(args)
    )

    if args.train:
        dqn.train()
    if args.evaluate:
        dqn.evaluate()

    dqn.close()
    wandb.finish()
