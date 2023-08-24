import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from agents.DQN import DQN
import agents.constants.DQN as const
from envs.gym_env import GymEnv
import wandb

import importlib
from utils.parser import Parser

def get_args_and_writer():
    parser = Parser(algo_name='dqn', env_name='CartPole-v1', model_name='tf_mlp')

    parser.add_argument("--episodes", type=int, default=const.episodes,
        help="the total episodes the agent interact with the env")
    parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
        help="the learning rate of q-value model")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the sample batch size from memory cache")
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of discounted return")
    parser.add_argument("--epsilon-max", type=float, default=const.epsilon_max,
        help="the max epsilon-choose's probability decay")
    parser.add_argument("--epsilon-min", type=float, default=const.epsilon_min,
        help="the min epsilon-choose's probability decay")
    parser.add_argument("--epsilon-decay", type=float, default=const.epsilon_decay,
        help="the decay epsilon-choose's probability decay")
    parser.add_argument("--memory-size", type=int, default=const.memory_size,
        help="the size of memory cache")
    parser.add_argument("--start-fit-size", type=int, default=const.start_fit_size,
        help="the size of memory cache when start training the q-value model")
    
    return parser.get_args_and_writer()


if __name__ == '__main__':
    args, writer = get_args_and_writer()

    Model = getattr(importlib.import_module(f"agents.models.DQN.{args.model_name}"), "Model")
    env = GymEnv(name=args.env_name, capture_video=args.capture_video)
    model = Model(
        lr=args.learning_rate, load_name=args.load_name, load_id=args.load_id,
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
