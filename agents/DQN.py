from tensorboardX import SummaryWriter
from agents import Agent
import agents.constants.DQN as const
from envs.gym_env import GymEnv
from utils import make_onehot
from utils.logs import Logs, MeanMetric
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import time

from envs import Env
keras = tf.keras

def init_logs() -> Logs:
    return Logs(
        init_logs = {
            'episode': 0,
            'step': 0,
            'q_value': MeanMetric(),
            'loss': MeanMetric(),
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
            self, agent_name=None,
            env: Env = None,
            models: list = None,
            writer: SummaryWriter = None, 
            # following customed
            model=None,
            episodes=None,
            batch_size=const.batch_size,
            gamma=const.gamma,
            memory_size=const.memory_size,
            start_fit_size=const.start_fit_size,
            epsilon_max=const.epsilon_max,
            epsilon_min=const.epsilon_min,
            epsilon_decay=const.epsilon_decay,
            **kwargs
        ):
        models = [model]
        super().__init__(agent_name, env, models, writer, **kwargs)
        self.model, self.logs = model, init_logs()
        self.loss_fn = keras.losses.MeanSquaredError()
        self.epsilon, self.epsilon_max, self.epsilon_min, self.epsilon_decay = \
            epsilon_max, epsilon_max, epsilon_min, epsilon_decay
        self.episodes, self.batch_size, self.gamma, self.memory_size, self.start_fit_size = \
            episodes, batch_size, gamma, memory_size, start_fit_size
        self.memory = MemoryCache(env.state_shape, env.action_shape, self.memory_size)
    
    def train(self):
        self.epsilon = self.epsilon_max
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, reward, terminal = self.env.step(action)
                self.remember(state, action, reward, state_, terminal)
                loss, q_value = self.fit()
                self.logs.update(['q_value', 'loss'], [q_value, loss])
                # BUGFIX: forget use new state
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.write_tensorboard()
            if (episode + 1) % 100 == 0:
                self.model.save_weights()

    def evaluate(self):
        self.epsilon = 0
        for episode in tqdm(range(self.episodes)):
            self.logs.reset()
            state = self.env.reset()
            for step in range(self.env.max_step):
                action = self.act(state)
                state_, _, terminal = self.env.step(action)
                state = state_
                if terminal: break
            self.logs.update(['episode', 'step'], [episode, step])
            self.write_tensorboard()

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
        a_onehot = make_onehot(a, depth=self.env.action_ndim).astype('bool')

        td_target = r + self.gamma * np.max(self.model(s_), axis=-1) * (1-t)
        q_state = self.model(s)
        q_target = q_state.numpy()
        q_target[a_onehot] = td_target

        loss, q_value = self.train_step(s, q_target)
        # BUGFIX: decrease epsilon after one fit!
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return tf.reduce_mean(loss), tf.reduce_mean(q_value)
    
    def write_tensorboard(self):
        episode = self.logs.logs['episode']
        d = self.logs.to_dict(drops=['episode'])
        for key, value in d.items():
            if key in ['step']: name = 'charts/' + key
            else: name = 'metrics/' + key
            if value is not None:
                self.writer.add_scalar(name, value, episode)
        self.writer.add_scalar('charts/epsilon', self.epsilon, episode)
        if d['loss'] is not None:
            self.writer.add_scalar(
                'charts/SPS',
                int(d['step'] * self.batch_size / self.logs.get_time_length()),
                episode
            )
    
if __name__ == '__main__':
    dqn = DQN(
        env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
        agent_id=0, episodes=5
    )
    dqn.train()