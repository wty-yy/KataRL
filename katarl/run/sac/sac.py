import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import katarl.agents.constants.sac as const
from katarl.envs.gym_env import GymEnv
from katarl.utils.parser import Parser, str2bool

import wandb

from importlib import import_module

def get_args_and_writer():
    parser = Parser(algo_name='sac_jax', env_name='CartPole-v1', model_name='mlp_jax')

    parser.add_argument("--total-timesteps", type=int, default=const.total_timesteps,
        help="the total timesteps the agent interact with the env")
    parser.add_argument("--num-envs", type=int, default=const.num_envs,
        help="the number of the env")
    parser.add_argument("--gamma", type=float, default=const.gamma,
        help="the discount rate of the reward")
    parser.add_argument("--memory-size", type=int, default=const.memory_size,
        help="the memory buffer size")
    parser.add_argument("--start-fit-size", type=int, default=const.start_fit_size,
        help="the size to start to fit the model")
    parser.add_argument("--tau", type=float, default=const.tau,
        help="the proportion of the target model when update target model")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the batch size sampling from the memory")
    parser.add_argument("--learning-rate-p", type=float, default=const.learning_rate_p,
        help="the learning rate of policy model")
    parser.add_argument("--learning-rate-q", type=float, default=const.learning_rate_q,
        help="the learning rate of q-value model")
    parser.add_argument("--target-model-update-frequency", type=int, default=const.target_model_update_frequency,
        help="the frequency to update the target model")
    parser.add_argument("--train-frequency", type=int, default=const.train_frequency,
        help="the frequency of update the models(policy an q-value)")
    parser.add_argument("--alpha", type=float, default=const.alpha,
        help="the initial coef of the entropy regularization")
    parser.add_argument("--flag-autotune-alpha", type=str2bool, default=const.flag_autotune_alpha,
        help="if taggled, the agent will automatic tuning the alpha")
    parser.add_argument("--learning-rate-alpha", type=float, default=const.learning_rate_alpha,
        help="the learning rate of alpha, if '--flag-autotune' is taggled")
    parser.add_argument("--coef-target-entropy", type=float, default=const.coef_target_entropy,
        help="the coefficient of the target entropy")
    parser.add_argument("--max-grad-clip-norm", type=float, default=const.max_grad_clip_norm,
        help="the maximum global norm for an update")
    parser.add_argument("--flag-anneal-lr", type=str2bool, default=const.flag_anneal_lr,
        help="if taggled, the learning rate will be linear annealed")
    parser.add_argument("--EPS", type=float, default=const.EPS,
        help="the epsilone size of accuracy error")
    parser.add_argument("--write-logs-frequency", type=int, default=const.write_logs_frequency,
        help="the frequency of writing the logs")

    args, writer = parser.get_args_and_writer()
    args.num_iters = (args.total_timesteps - 1) // args.num_envs + 1
    return args, writer

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    Model = getattr(import_module(f"katarl.agents.models.sac.{args.model_name}"), "Model")
    env = GymEnv(args=args)
    
    q1_model = Model(
        name='q1-model', 
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args, is_policy_model=False, seed_delta=0
    )
    q2_model = Model(
        name='q2-model', 
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args, is_policy_model=False, seed_delta=1
    )
    p_model = Model(
        name='policy-model', 
        input_shape=env.state_shape, output_ndim=env.action_ndim,
        args=args, is_policy_model=True, seed_delta=2
    )

    Agent = getattr(import_module(f"katarl.agents.{args.algo_name}"), "Agent")
    sac = Agent(
        agent_name=args.run_name, env=env, 
        q1_model=q1_model, q2_model=q2_model, p_model=p_model,
        writer=writer, args=args
    )
    if args.train:
        sac.train()
    if args.evaluate:
        sac.evaluate()

    sac.close()
    wandb.finish()