import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import agents.constants.dqn.ddqn as const
from envs.gym_env import GymEnv
import wandb

import importlib
from utils.parser import Parser

def get_args_and_writer():
    parser = Parser(agent_name='ddqn_jax', env_name='CartPole-v1', model_name='mlp_jax')

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
    parser.add_argument("--target-model-update-frequency", type=int, default=const.target_model_update_frequency,
        help="the frequency of the target model update")
    parser.add_argument("--tau", type=float, default=const.tau,
        help="the ratio of the current model and the target model")
    parser.add_argument("--write-logs-frequency", type=int, default=const.write_logs_frequency,
        help="the frequency of writing the logs to tensorboard")

    args, writer = parser.get_args_and_writer()
    args.slope = (args.epsilon_min - args.epsilon_max) / (args.total_timesteps * args.exporation_proportion)
    args.num_iters = (args.total_timesteps - 1) // args.num_envs + 1
    return args, writer


if __name__ == '__main__':
    args, writer = get_args_and_writer()

    env = GymEnv(
        name=args.env_name, seed=args.seed,
        num_envs=args.num_envs,
        capture_video=args.capture_video
    )
    Model = getattr(importlib.import_module(f"agents.models.dqn.{args.model_name}"), "Model")
    model = Model(
        seed=args.seed, lr=args.learning_rate,
        load_name=args.load_name, load_id=args.load_id,
        input_shape=env.state_shape, output_ndim=env.action_ndim
    )
    Agent = getattr(importlib.import_module(f"agents.{args.algo_name}"), "Agent")
    ddqn = Agent(
        agent_name=args.run_name, env=env, model=model, writer=writer, args=args
    )
    print(ddqn)

    if args.train:
        ddqn.train()
    if args.evaluate:
        ddqn.evaluate()

    wandb.finish()
    ddqn.close()
