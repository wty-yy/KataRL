from katarl.utils import get_time_str

import wandb
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from typing import NamedTuple

def str2bool(x):
    return x in ['yes', 'y', 'True', '1']

class Parser(ArgumentParser):

    def __init__(self, algo_name='dqn', env_name='CartPole-v1', model_name='tf_mlp'):
        super().__init__()
        # parser = argparse.ArgumentParser()
        self.add_argument("--algo-name", type=str, default=algo_name,
            help="the name of algorithm")
        self.add_argument("--env-name", type=str, default=env_name,
            help="the name of environment")
        self.add_argument("--model-name", type=str, default=model_name,
            help="the name of the v-value and policy model")
        self.add_argument("--seed", type=int, default=1,
            help="the seed of experiment")
        self.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
            help="if taggled, this experiment will be tracked by WandB")
        self.add_argument("--wandb-project-name", type=str, default="KataRL",
            help="the wandb's project name")
        self.add_argument("--train", default=False, nargs='*',
            help="if taggled, the training will be started, you can pass 'run_name' and 'weight_id' like '--evaluate'")
        self.add_argument("--evaluate", default=False, nargs=2,
            help="if taggled, you need pass 'run_name' (in dir '/logs/') and 'weight_id', such like '--evaluate dqn-CartPole_v1-tf_mlp-1-20230823-140730 5'")
        self.add_argument("--capture-video", type=str2bool, default=False, const=True, nargs="?",
            help="if taggled, capture the video in multiples of 10")
        self.add_argument("--num-model-save", type=int, default=10,
            help="the number of saving the model, uniform distribution in global step")
    
    def get_args_and_writer(self) -> tuple[NamedTuple, SummaryWriter]:
        args = self.parse_args()

        if isinstance(args.train, list):
            if len(args.train) == 0: args.train = True
            elif len(args.train) != 2:
                raise Exception(f"Error: should only pass two args after '--train', but get '{args.train}'")

        if args.evaluate:
            args.load_name, args.load_id = args.evaluate
            args.load_id = int(args.load_id)
            args.capture_video = True
        else: args.load_name, args.load_id = None, None
        args.run_name = f"{args.algo_name}__{args.env_name}__{args.model_name}__{args.seed}__{get_time_str()}"
        self.args = args
        self.init_wandb()
        self.writer = SummaryWriter(f"logs/{args.run_name}")
        self.writer.add_text(
            "hyper-parameters",
            "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))
        )
        return self.args, self.writer
    
    def init_wandb(self):
        if self.args.wandb_track:
            wandb.init(
                project=self.args.wandb_project_name,
                sync_tensorboard=True,
                config=vars(self.args),
                name=self.args.run_name,
                save_code=True,
                monitor_gym=True
            )

""" add hyper arguments
parser = Parser(algo_name='a2c', env_name='CartPole-v1', model_name='tf_mlp')

parser.add_argument("--episodes", type=int, default=1000,
    help="the total episodes the agent interact with the env")
parser.add_argument("--learning-rate-v", type=float, default=const.learning_rate_v,
    help="the learning rate of v-value model")
parser.add_argument("--learning-rate-p", type=float, default=const.learning_rate_p,
    help="the learning rate of policy model")
parser.add_argument("--gamma", type=float, default=const.gamma,
    help="the discount rate of discounted return")
        
"""