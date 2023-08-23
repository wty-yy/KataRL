import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from agents.DQN import DQN
import agents.constants.DQN as const
from envs.gym_env import GymEnv
from utils import get_time_str
from tensorboardX import SummaryWriter

import importlib

import argparse

def parse_args():
    str2bool = lambda x: x in ['yes', 'y', 'True', '1']
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="dqn",
        help="the name of algorithm")
    parser.add_argument("--env-name", type=str, default="CartPole-v1",
        help="the name of environment")
    parser.add_argument("--model-name", type=str, default="tf_mlp",
        help="the name of the q-value model")
    parser.add_argument("--seed", type=int, default=1,
        help="the seed of experiment")
    parser.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
        help="if taggled, this experiment will be tracked by WandB")
    parser.add_argument("--wandb-project-name", type=str, default="rl-framework-dqn",
        help="the wandb's project name")
    parser.add_argument("--train", type=str2bool, default=False, const=True, nargs='?',
        help="if taggled, the training will be started")
    parser.add_argument("--evaluate", default=False, nargs=2,
        help="if taggled, you need pass 'run_name' (in dir '/logs/') and 'weight_id', such like '--evaluate dqn-CartPole_v1-tf_mlp-1-20230823-140730 5'")

    parser.add_argument("--episodes", type=int, default=1000,
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
    
    args = parser.parse_args()
    if args.evaluate:
        args.load_name, args.load_id = args.evaluate
        args.load_id = int(args.load_id)
    else: args.load_name, args.load_id = None, None
    args.run_name = f"{args.algo_name}__{args.env_name}__{args.model_name}__{args.seed}__{get_time_str()}"
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.wandb_track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            save_code=True,
            # monitor_gym=True
        )
    writer = SummaryWriter(f"logs/{args.run_name}")
    writer.add_text(
        "hyper-parameters",
        "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    Model = getattr(importlib.import_module(f"agents.models.DQN.{args.model_name}"), "Model")
    env = GymEnv(name=args.env_name)
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
