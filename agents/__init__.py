from utils.history import History
from utils import get_time_str
from envs import Env
from agents.constants import PATH

class Agent:
    """
    Initialize:
    -   env: The environment that agent interact with.
    -   verbose: Whether render the env and save the frames.
    -   agent_name: The agent's name.
    -   agent_id: The agent's id (means which agent is training).
    -   model_name: Which DNN model to use in '/agents/models'.
    -   load_id: Load model weight from (if not None)
                  '/logs/agent_name/cp-agent_id/model_id'.
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
            look at agents.models.save_weigth()
    """
    
    def __init__(
            self, env:Env=None, verbose=False,
            agent_name=None, agent_id=None,
            model_name=None, load_id=None,
            episodes=None, **kwargs
        ):
        self.env, self.verbose, self.agent_name, self.agent_id, \
            self.model_name, self.load_id, self.episodes = \
            env, verbose, agent_name, agent_id, \
            model_name, load_id, episodes
        self.timestr = get_time_str()

        # setting logs path early, since history will use PATH.HISTORY
        PATH.get_logs_path(self.agent_name, self.agent_id)

        _ = (self.agent_name, self.agent_id, self.timestr)
        self.best_episode = History(*_, "best", type='npy')
        self.history = History(*_, "history", type='json')
    
    def train(self):
        pass

    def evaluate(self):
        pass
    
    def act(self, state):
        pass

    def fit(self):
        pass

if __name__ == '__main__':
    agent = Agent()
    agent.best_episode.update_best(10, {'a':1, 'b':2})
    agent.best_episode.update_best(20, {'a':2, 'b':3})
    print(agent.best_episode.best, agent.best_episode.logs)
