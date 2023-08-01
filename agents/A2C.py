from agents import Agent
from agents.constants import PATH
from agents.constants.A2C import gamma
from agents.models import Model
from envs import Env
from utils.logs_manager import Logs
from tqdm import tqdm
import tensorflow as tf
keras = tf.keras

def get_logs() -> Logs:
    return Logs(
        init_logs={
            'episode': 0,
            'step': 0,
            'v_value': keras.metrics.Mean(name='v_value'),
            'loss': keras.metrics.Mean(name='loss'),
            'frame': []
        }
    )

class A2C(Agent):
    
    def __init__(
            self, env: Env = None, verbose=False,
            agent_name='A2C', agent_id=0,
            model: Model = None,
            episodes=1000, gamma=gamma,  # constants.A2C
            **kwargs
        ):
        super().__init__(env, verbose, agent_name, agent_id, model, episodes, **kwargs)
        self.gamma = gamma
        self.logs = get_logs()
    
    def train(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                loss, v_value = self.fit(state, action, reward, state_, terminal)
                frame = self.env.render() if self.verbose else None
                self.logs.update(['v_value', 'loss', 'frame'], [v_value, loss, frame])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
    
    def act(self, state):
        pass

    def fit(self):
        pass
    
    def update
