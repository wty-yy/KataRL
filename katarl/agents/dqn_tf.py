# -*- coding: utf-8 -*-
'''
@File    : dqn.py
@Time    : 2023/08/26 12:02:07
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : None
2023.8.26. 优化DQN算法，加入多进程，按照total_timesteps作为训练上限，而非episodes
'''

if __name__ == '__main__':
    pass

from tensorboardX import SummaryWriter
from agents import BaseAgent
import agents.constants.dqn.dqn as const
from agents.models.base import BaseModel
from envs.gym_env import GymEnv
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
            'episode_step': MeanMetric(),
            'episode_return': MeanMetric(),
            'q_value': MeanMetric(),
            'loss': MeanMetric(),
        },
        folder2name = {
            'charts': ['episode_step', 'episode_return'],
            'metrics': ['q_value', 'loss']
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
        self.a = np.zeros([self.total], dtype='int32')
        self.r = np.zeros([self.total], dtype='float32')
        self.s_ = np.zeros([self.total, *self.state_shape], dtype='float32')
        self.t = np.zeros([self.total], dtype='bool')
        self.memory = [self.s, self.a, self.r, self.s_, self.t]
    
    def update(self, item:tuple):  # (S,A,R,S',T)
        start = self.count % self.total
        N = item[0].shape[0]  # actor_N
        for value, array in zip(item, self.memory):
            len1 = min(self.total - start, N)
            len2 = N - len1
            array[start:start+len1] = value[:len1]
            if len2 != 0:
                array[:len2] = value[-len2:]
        self.count += N
    
    def sample(self, num=1):
        size = min(self.count, self.total)
        indexs = np.random.choice(size, num)
        return [array[indexs] for array in self.memory]

class DQN(BaseAgent):

    def __init__(
            self, agent_name=None,
            env: Env = None,
            models: list = None,
            writer: SummaryWriter = None, 
            seed: int = 1,
            # hyper-parameters
            model: BaseModel = None,
            total_timesteps=const.total_timesteps,
            num_envs=const.num_envs,
            batch_size=const.batch_size,
            gamma=const.gamma,
            memory_size=const.memory_size,
            start_fit_size=const.start_fit_size,
            epsilon_max=const.epsilon_max,
            epsilon_min=const.epsilon_min,
            exporation_proportion=const.exporation_proportion,
            train_frequency=const.train_frequency,
            write_logs_frequency=const.write_logs_frequency,
            **kwargs
        ):
        models = [model]
        super().__init__(agent_name, env, models, writer, seed, **kwargs)
        np.random.seed(self.seed)
        self.model, self.logs = model, init_logs()
        self.loss_fn = keras.losses.MeanSquaredError()
        self.epsilon_max, self.epsilon_min, self.exporation_proportion, self.train_frequency, self.write_logs_frequency = \
            epsilon_max, epsilon_min, exporation_proportion, train_frequency, write_logs_frequency
        self.total_timesteps, self.num_envs, self.batch_size, self.gamma, self.memory_size, self.start_fit_size = \
            total_timesteps, num_envs, batch_size, gamma, memory_size, start_fit_size
        self.memory = MemoryCache(env.state_shape, env.action_shape, self.memory_size)
        self.slope = (self.epsilon_min - self.epsilon_max) / (self.total_timesteps * self.exporation_proportion)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_max + self.slope * self.global_step, self.epsilon_min)
    
    def train(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        num_iters = (self.total_timesteps-1)//self.num_envs+1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            self.update_epsilon()
            action = self.act(state)
            state_, reward, terminal = self.env.step(action)
            self.remember(state, action, reward, state_, terminal)
            self.global_step += self.num_envs

            if self.global_step > self.start_fit_size:
                if self.global_step % self.train_frequency < self.num_envs:
                    loss, q_value = self.fit()
                    self.logs.update(['q_value', 'loss'], [q_value, loss])

            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_steps(), self.env.get_terminal_reward()]
            )
            # if episode end, then we plot it
            self.logs.writer_tensorboard(self.writer, self.global_step, drops=['q_value', 'loss'])

            if (i+1) % self.write_logs_frequency == 0:
                self.write_tensorboard()
            if (self.global_step + 1) % int(1e4) == 0:
                self.model.save_weights()

    def evaluate(self):
        state = self.env.reset()
        self.start_time, self.global_step = time.time(), 0
        self.epsilon = 0
        num_iters = (self.total_timesteps-1)//self.num_envs+1
        for i in tqdm(range(num_iters)):
            self.logs.reset()
            action = self.act(state)
            state_, _, _ = self.env.step(action)
            self.global_step += self.num_envs
            state = state_
            self.logs.update(
                ['episode_step', 'episode_return'],
                [self.env.get_terminal_steps(), self.env.get_terminal_reward()]
            )
            if (i+1) % self.write_logs_frequency == 0:
                self.write_tensorboard()
        
    @tf.function
    def predict(self, state):
        return self.model(state)

    def act(self, state):  # epsilon choice policy
        rand = np.random.rand()
        if rand <= self.epsilon:
            return np.random.randint(0, self.env.action_ndim, self.num_envs)
        else:
            q_pred = self.predict(state)
            action = np.argmax(q_pred, axis=-1)
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
        s, a, r, s_, t = self.memory.sample(self.batch_size)
        r, t = r.squeeze(), t.squeeze()

        td_target = r + self.gamma * np.max(self.model(s_), axis=-1) * (1-t)
        q_state = self.model(s)
        q_target = q_state.numpy()
        q_target[np.arange(self.batch_size), a] = td_target

        loss, q_value = self.train_step(s, q_target)
        return tf.reduce_mean(loss), tf.reduce_mean(q_value)
    
    def write_tensorboard(self):
        self.logs.writer_tensorboard(self.writer, self.global_step, drops=['episode_step', 'episode_return'])
        self.writer.add_scalar('charts/epsilon', self.epsilon, self.global_step)
        self.writer.add_scalar(
            'charts/SPS_avg',
            int(self.global_step / (time.time() - self.start_time)),
            self.global_step
        )
    
if __name__ == '__main__':
    dqn = DQN(
        env=GymEnv(name="CartPole-v1", render_mode="rgb_array"),
        agent_id=0, episodes=5
    )
    dqn.train()