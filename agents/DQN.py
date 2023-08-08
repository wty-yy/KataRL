from agents import Agent
from agents.constants import *
from agents.constants.DQN import *
from envs.gym_env import GymEnv
from utils import make_onehot
from utils.logs_manager import Logs
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from envs import Env
keras = tf.keras

def get_logs() -> Logs:
    return Logs(
        init_logs = {
            'episode': 0,
            'step': 0,
            'q_value': keras.metrics.Mean(name='q_value'),
            'loss': keras.metrics.Mean(name='loss'),
            'frame': []
        }
    )
    
class MemoryCache:
    """
    save the memory cache of (S,A,R,S',T):
        type name      |  shape             |  type
        ---------------|--------------------|---------
        S(state)       |  env.state_shape   |  float32
        A(action)      |  env.action_shape  |  int32
        R(reward)      |  (1,)              |  int32
        S'(next state) |  env.state_shape   |  float32
        T(terminal)    |  (1,)              |  bool
    """

    def __init__(self, state_shape, action_shape, memory_size):
        self.state_shape, self.action_shape = state_shape, action_shape
        self.count = 0
        self.total = memory_size
        # use np.ndarray could sample by indexs easily
        self.s = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.a = np.zeros([self.total, *self.action_shape], dtype='int32')
        self.r = np.zeros([self.total, 1], dtype='float32')
        self.s_ = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.t = np.zeros([self.total, 1], dtype='bool')
        self.memory = [self.s, self.a, self.r, self.s_, self.t]
    
    def update(self, item:tuple):  # (S,A,R,S',T)
        for value, array in zip(item, self.memory):
            array[self.count % self.total] = value
        self.count += 1
    
    def sample(self, num=1):
        size = min(self.count, self.total)
        indexs = np.random.choice(size, num)
        return [array[indexs] for array in self.memory]

class DQN(Agent):

    def __init__(
            self, env:Env=None, verbose=False,
            agent_name='DQN', agent_id=0,
            episodes=100,
            model=None,  # Q value model
            batch_size=batch_size,  # constant.DQN
            memory_size=memory_size,  # constant.DQN
            start_fit_size=start_fit_size,  # constant.DQN
            **kwargs
        ):
        models = [model]
        super().__init__(env, verbose, agent_name, agent_id, episodes, models, **kwargs)
        self.model = model
        self.loss_fn = keras.losses.MeanSquaredError()
        self.logs = get_logs()
        self.epsilon = epsilon_max
        self.batch_size, self.memory_size, self.start_fit_size = \
            batch_size, memory_size, start_fit_size
        self.memory = MemoryCache(env.state_shape, env.action_shape, self.memory_size)
    
    def train(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                self.remember(state, action, reward, state_, terminal)
                loss, q_value = self.fit()
                frame = self.env.render() if self.verbose else None
                self.logs.update(['q_value', 'loss', 'frame'], [q_value, loss, frame])
                # BUGFIX: forget use new state
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.update_history()
            if (episode + 1) % 100 == 0:
                self.model.save_weights()

    def evaluate(self):
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, _, terminal = self.env.step(action)
                frame = self.env.render() if self.verbose else None
                self.logs.update(['frame'], [frame])
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.update_history()

    def act(self, state):  # epsilon choice policy
        rand = np.random.rand()
        if rand <= self.epsilon:
            return np.random.randint(0, 2)
        else:
            q_pred = self.model(tf.constant([state]))[0]
            action = np.argmax(q_pred)
            return action

    def remember(self, state, action, reward, state_, terminal):
        self.memory.update((state, action, reward, state_, terminal))

    @tf.function
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            q_state = self.model(X)
            loss = self.loss_fn(y, q_state)
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.model.apply_gradients(grads)
        return loss, tf.reduce_mean(q_state)

    def fit(self):
        if self.memory.count < self.start_fit_size: return None, None
        s, a, r, s_, t = self.memory.sample(self.batch_size)
        r, t = r.squeeze(), t.squeeze()
        a_onehot = make_onehot(a, depth=self.env.action_shape[0]).astype('bool')

        td_target = r + gamma * np.max(self.model(s_), axis=-1) * (1-t)
        q_state = self.model(s)
        q_target = q_state.numpy()
        q_target[a_onehot] = td_target

        loss, q_value = self.train_step(s, q_target)
        # BUGFIX: decrease epsilon after one fit!
        self.epsilon = max(self.epsilon * epsilon_decay, epsilon_min)
        return loss, q_value

    def update_history(self):
        self.best_episode.update_best(
            now=self.logs.logs['step'], logs=self.logs.to_dict()
        )
        self.history.update_dict(self.logs.to_dict(drops=['frame']))
    
if __name__ == '__main__':
    dqn = DQN(
        env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
        verbose=True, agent_id=0, episodes=5
    )
    dqn.train()