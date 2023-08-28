import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from envs import Env
from agents.models.base import BaseModel
from agents.constants import PATH
from tensorboardX import SummaryWriter
from typing import NamedTuple

class BaseAgent:
    """
    Initialize:
    -   env: The environment that agent interact with.
    -   agent_name: The agent's name.
    -   agent_id: The agent's id (means which agent is training).
    -   episodes: Total training episodes.
    -   timestr: The timestamp when agent was created.
    Logs:
    -   The logs of the Agent will save in dir
        '/logs/agent_name/' and there will be two subdir:
        -   history: `/logs/agent_name/history-id`
            There two files in this dir:
            1.`history-timestr`:
                save each episode's log
            2.`best-timestr`:
                save the best episode's log
                during this training
        -   checkpoints: `/logs/agent_name/cp-id`
            Save the model weight at each episode end,
            look at agents.models.Model.save_weigth()
    """
    
    def __init__(
            self, 
            agent_name:str = None,
            env: Env = None,
            models: list[BaseModel] = None,
            writer: SummaryWriter = None,
            args: NamedTuple = None
        ):
        self.agent_name, self.env, self.models, self.writer, self.args = \
            agent_name, env, models, writer, args

        # setting logs path early,
        # since history and load_weights will
        # use PATH.HISTORY and PATH.CHECKPOINTS
        PATH.get_logs_path(self.agent_name)

        for model in models:
            model.load_weights()
        # _ = (self.agent_name, self.agent_id, self.timestr)
        # self.best_episode = History(*_, "best", type='npy')
        # self.history = History(*_, "history", type='json')
    
    def train(self):
        pass

    def evaluate(self):
        pass
    
    def act(self, state):
        pass

    def fit(self):
        pass
    
    def write_tensorboard(self):
        pass
    
    def close(self):
        self.env.close()
        self.writer.close()

if __name__ == '__main__':
    agent = BaseAgent()
    agent.best_episode.update_best(10, {'a':1, 'b':2})
    agent.best_episode.update_best(20, {'a':2, 'b':3})
    print(agent.best_episode.best, agent.best_episode.logs)
